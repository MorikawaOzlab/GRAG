

import math
import time
import warnings
from typing import Any, Callable, Dict, Protocol, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jumanji.types import TimeStep
from omegaconf import DictConfig
from typing_extensions import TypeAlias

from mava.wrappers.navix_wrapper import NavixWrapper, NavixTimestep
from mava.types import (
    Action,
    ActorApply,
    MarlEnv,
    Metrics,
    Observation,
    ObservationGlobalState,
    RecActorApply,
    State,
)

# Optional extras that are passed out of the actor and then into the actor in the next step
ActorState: TypeAlias = Dict[str, Any]
# Type of the carry for the _env_step function in the evaluator
_EvalEnvStepState: TypeAlias = Tuple[State, TimeStep, PRNGKey, ActorState]
# The function signature for the mava evaluation function (returned by `get_eval_fn`).
EvalFn: TypeAlias = Callable[[FrozenDict, PRNGKey, ActorState], Metrics]
EvalFnWithKeys: TypeAlias = Callable[[FrozenDict, PRNGKey, ActorState], tuple[Metrics, Array]]


class EvalActFn(Protocol):
    """The API for the acting function that is passed to the `EvalFn`.

    A get_action function must conform to this API in order to be used with Mava's evaluator.
    """

    def __call__(
        self,
        params: FrozenDict,
        timestep: TimeStep[Union[Observation, ObservationGlobalState]],
        key: PRNGKey,
        actor_state: ActorState,
    ) -> Tuple[Array, ActorState]: ...


def get_num_eval_envs(config: DictConfig, absolute_metric: bool) -> int:
    """Returns the number of vmapped envs/batch size during evaluation."""
    n_devices = jax.device_count() if config.arch.architecture_name == "anakin" else 1
    n_parallel_envs = config.arch.num_envs * n_devices

    if absolute_metric:
        eval_episodes = config.arch.num_absolute_metric_eval_episodes
    else:
        eval_episodes = config.arch.num_eval_episodes

    if eval_episodes <= n_parallel_envs:
        return math.ceil(eval_episodes / n_devices)  # type: ignore
    else:
        return config.arch.num_envs  # type: ignore


def get_eval_fn(env: MarlEnv, act_fn: EvalActFn, config: DictConfig, absolute_metric: bool) -> EvalFn:
    """Creates a function that can be used to evaluate agents on a given environment.
    """
    n_devices = jax.device_count()
    eval_episodes = config.arch.num_absolute_metric_eval_episodes if absolute_metric else config.arch.num_eval_episodes
    n_vmapped_envs = get_num_eval_envs(config, absolute_metric)
    n_parallel_envs = n_vmapped_envs * n_devices
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by `num_envs` * "
            f"`num_devices` ({n_parallel_envs} * {n_devices}). Some extra evaluations will be "
            f"executed. New number of evaluation episodes = {episode_loops * n_parallel_envs}",
            stacklevel=2,
        )

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Evaluates the given params on an environment and returns relevent metrics.
        """

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _: Any) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, n_vmapped_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))

            metrics = timesteps.extras["episode_metrics"] | timesteps.extras["env_metrics"]

            # find the first instance of done to get the metrics at that timestep, we don't
            # care about subsequent steps because we only the results from the first episode
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, jnp.arange(n_vmapped_envs)], metrics)

            return key, metrics

        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)  # flatten metrics
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        metrics = jax.pmap(eval_fn)(params, key, init_act_state)
        metrics = jax.block_until_ready(metrics)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn


def make_ff_eval_act_fn(actor_apply_fn: ActorApply, config: DictConfig) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    feed forward mava actor network."""

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        pi = actor_apply_fn(params, timestep.observation)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action, {}

    return eval_act_fn


def get_nx_eval_fn(
    env: NavixWrapper,
    act_fn: Callable,
    config: DictConfig,
    absolute_metric: bool = False,
) -> Callable:
    """Creates an evaluation function for Navix environments."""
    total_eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )
    num_eval_envs = config.arch.num_envs
    eval_episodes_per_env = max(1, int(math.ceil(total_eval_episodes / num_eval_envs)))
    max_episode_length = config.env.time_limit

    def eval_fn(params: FrozenDict, key: PRNGKey, init_actor_state: Dict) -> Metrics:
        def _env_step(carry: Tuple, _: Any) -> Tuple[Tuple, NavixTimestep]:
            ts, step_key, actor_state = carry
            step_key, action_key = jax.random.split(step_key)
            action, actor_state = act_fn(params, ts, action_key, actor_state)
            new_ts = env.step(ts, action)
            return (new_ts, step_key, actor_state), new_ts

        def _episode(carry: Tuple, _: Any) -> Tuple[Tuple, Metrics]:
            _, episode_key, actor_state = carry
            episode_key, reset_key, step_key = jax.random.split(episode_key, 3)
            reset_keys = jax.random.split(reset_key, num_eval_envs)
            ts = env.reset(reset_keys)

            step_state = (ts, step_key, actor_state)
            _, timesteps = jax.lax.scan(_env_step, step_state, None, length=max_episode_length + 1)

            done_steps = timesteps.is_done()
            done_idx = jnp.argmax(done_steps, axis=0)
            episode_completed = jnp.any(done_steps, axis=0)
            done_idx = jnp.where(episode_completed, done_idx, max_episode_length)

            rewards = timesteps.reward
            episode_return = jnp.sum(
                rewards * (jnp.arange(max_episode_length + 1)[:, None] <= done_idx[None, :]),
                axis=0,
            )
            episode_length = done_idx + 1

            goal_reached = rewards > 0
            won_episode = jnp.any(
                goal_reached & (jnp.arange(max_episode_length + 1)[:, None] <= done_idx[None, :]),
                axis=0,
            )

            final_positions = env.get_player_position(timesteps)
            goals = timesteps.goal_pos
            distances = jnp.sum(jnp.abs(final_positions - goals), axis=-1)
            env_indices = jnp.arange(num_eval_envs)
            final_distance = jnp.where(won_episode, 0.0, distances[done_idx, env_indices])

            metrics = {
                "episode_return": episode_return,
                "episode_length": episode_length,
                "won_episode": won_episode.astype(jnp.float32),
                "avg_distance_to_goal": final_distance,
            }
            return (None, episode_key, init_actor_state), metrics

        initial_carry = (None, key, init_actor_state)
        _, metrics = jax.lax.scan(_episode, initial_carry, None, length=eval_episodes_per_env)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)
        metrics["win_rate"] = jnp.mean(metrics["won_episode"]) * 100
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_actor_state: Dict) -> Metrics:
        start_time = time.time()
        metrics = jax.pmap(eval_fn)(params, key, init_actor_state)
        metrics = jax.block_until_ready(metrics)
        end_time = time.time()
        metrics["win_rate"] = float(jnp.mean(metrics["won_episode"]) * 100)
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn


def make_nx_eval_act_fn(compute_q_values: Callable, config: DictConfig) -> Callable:
    """Creates an act function for Navix evaluation."""
    temperature = config.system.get("eval_temperature", 0.1)
    greedy = config.arch.get("evaluation_greedy", False)

    def eval_act_fn(
        params: FrozenDict,
        timestep: NavixTimestep,
        key: PRNGKey,
        actor_state: Dict,
    ) -> Tuple[Array, Dict]:
        logits = compute_q_values(params, timestep.observation, timestep.ultimate_goal)
        if greedy:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = jax.random.categorical(key, logits / temperature, axis=-1)
        return action, {}

    return eval_act_fn
def get_icrl_eval_fn(
    env: MarlEnv, act_fn: EvalActFn, config: DictConfig, absolute_metric: bool
) -> EvalFn:
    """Creates an ICRL-specific evaluation function with custom win rate handling.

    This function extends the standard evaluation with ICRL-specific win rate
    metric processing that handles various dimension configurations.

    Args:
    ----
        env: an environment that conforms to the mava environment spec.
        act_fn: a function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        config: the system config.
        absolute_metric: whether or not this evaluator calculates the absolute_metric.
                This determines how many evaluation episodes it does.
    """
    n_devices = jax.device_count()
    eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )
    n_vmapped_envs = get_num_eval_envs(config, absolute_metric)
    n_parallel_envs = n_vmapped_envs * n_devices
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by `num_envs` * "
            f"`num_devices` ({n_parallel_envs} * {n_devices}). Some extra evaluations will be "
            f"executed. New number of evaluation episodes = {episode_loops * n_parallel_envs}",
            stacklevel=2,
        )

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Evaluates the given params on an environment and returns relevent metrics.

        Metrics are collected by the `RecordEpisodeMetrics` wrapper: episode return and length,
        also win rate for environments that support it.

        Returns: Dict[str, Array] - dictionary of metric name to metric values for each episode.
        """

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _: Any) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, n_vmapped_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))


            metrics = timesteps.extras["episode_metrics"] | timesteps.extras["env_metrics"]

            # find the first instance of done to get the metrics at that timestep, we don't
            # care about subsequent steps because we only the results from the first episode
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, jnp.arange(n_vmapped_envs)], metrics)

            return key, metrics

        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)  # flatten metrics
        
        # ICRL-specific: If won_episode has agent dimension, reduce to per-episode
        if "won_episode" in metrics:
            won_episode = metrics["won_episode"]
            # If per-agent (multi-dimensional after flatten), take any agent winning
            if won_episode.ndim > 1:
                won_episode = jnp.any(won_episode, axis=-1)
            success = won_episode.astype(jnp.float32)
            metrics["won_episode"] = success
            metrics["eval/episode_success_any"] = jnp.mean(success)

        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        metrics = jax.pmap(eval_fn)(params, key, init_act_state)
        metrics = jax.block_until_ready(metrics)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn


def make_icrl_ff_eval_act_fn(actor_apply_fn: ActorApply, config: DictConfig) -> EvalActFn:
    """Makes an ICRL-specific act function for evaluation.

    This function handles ICRL's discrete/continuous action spaces with action masking.

    Args:
    ----
        actor_apply_fn: ICRL actor network that returns (means, log_stds)
        config: system config with ICRL-specific settings
    """

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        observation = timestep.observation
        obs_view = observation.agents_view
        achieved_goal = observation.achieved_goal
        ultimate_goal = observation.ultimate_goal
        full_obs = jnp.concatenate([obs_view, achieved_goal, ultimate_goal], axis=-1)
        action_mask = getattr(observation, "action_mask", None)

        # ICRL actor returns (means, log_stds)
        means, log_stds = actor_apply_fn(params, full_obs)

        if config.system.discrete_actions:
            # Discrete action space: use logits directly
            if action_mask is not None:
                mask = action_mask.astype(means.dtype)
                means = means - ((1.0 - mask) * 1e10)
            action = jnp.argmax(means, axis=-1)
        else:
            # Continuous action space: apply tanh and then argmax
            if action_mask is not None:
                mask = action_mask.astype(means.dtype)
                means = means - ((1.0 - mask) * 1e10)
            action_continuous = nn.tanh(means)
            action = jnp.argmax(action_continuous, axis=-1)

        return action, {}

    return eval_act_fn


def get_nx_eval_fn(
    env: NavixWrapper,
    act_fn: Callable,
    config: DictConfig,
    absolute_metric: bool = False,
) -> Callable:
    """Creates an evaluation function for Navix environments."""
    total_eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )
    num_eval_envs = config.arch.num_envs
    eval_episodes_per_env = max(1, int(math.ceil(total_eval_episodes / num_eval_envs)))
    max_episode_length = config.env.time_limit

    def eval_fn(params: FrozenDict, key: PRNGKey, init_actor_state: Dict) -> Metrics:
        def _env_step(carry: Tuple, _: Any) -> Tuple[Tuple, NavixTimestep]:
            ts, step_key, actor_state = carry
            step_key, action_key = jax.random.split(step_key)
            action, actor_state = act_fn(params, ts, action_key, actor_state)
            new_ts = env.step(ts, action)
            return (new_ts, step_key, actor_state), new_ts

        def _episode(carry: Tuple, _: Any) -> Tuple[Tuple, Metrics]:
            _, episode_key, actor_state = carry
            episode_key, reset_key, step_key = jax.random.split(episode_key, 3)
            reset_keys = jax.random.split(reset_key, num_eval_envs)
            ts = env.reset(reset_keys)

            step_state = (ts, step_key, actor_state)
            _, timesteps = jax.lax.scan(_env_step, step_state, None, length=max_episode_length + 1)

            done_steps = timesteps.is_done()
            done_idx = jnp.argmax(done_steps, axis=0)
            episode_completed = jnp.any(done_steps, axis=0)
            done_idx = jnp.where(episode_completed, done_idx, max_episode_length)

            rewards = timesteps.reward
            episode_return = jnp.sum(
                rewards * (jnp.arange(max_episode_length + 1)[:, None] <= done_idx[None, :]),
                axis=0,
            )
            episode_length = done_idx + 1

            goal_reached = rewards > 0
            won_episode = jnp.any(
                goal_reached & (jnp.arange(max_episode_length + 1)[:, None] <= done_idx[None, :]),
                axis=0,
            )

            final_positions = env.get_player_position(timesteps)
            goals = timesteps.goal_pos
            distances = jnp.sum(jnp.abs(final_positions - goals), axis=-1)
            env_indices = jnp.arange(num_eval_envs)
            final_distance = jnp.where(won_episode, 0.0, distances[done_idx, env_indices])

            metrics = {
                "episode_return": episode_return,
                "episode_length": episode_length,
                "won_episode": won_episode.astype(jnp.float32),
                "avg_distance_to_goal": final_distance,
            }
            return (None, episode_key, init_actor_state), metrics

        initial_carry = (None, key, init_actor_state)
        _, metrics = jax.lax.scan(_episode, initial_carry, None, length=eval_episodes_per_env)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)
        metrics["win_rate"] = jnp.mean(metrics["won_episode"]) * 100
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_actor_state: Dict) -> Metrics:
        start_time = time.time()
        metrics = jax.pmap(eval_fn)(params, key, init_actor_state)
        metrics = jax.block_until_ready(metrics)
        end_time = time.time()
        metrics["win_rate"] = float(jnp.mean(metrics["won_episode"]) * 100)
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn


def make_nx_eval_act_fn(compute_q_values: Callable, config: DictConfig) -> Callable:
    """Creates an act function for Navix evaluation."""
    temperature = config.system.get("eval_temperature", 0.1)
    greedy = config.arch.get("evaluation_greedy", False)

    def eval_act_fn(
        params: FrozenDict,
        timestep: NavixTimestep,
        key: PRNGKey,
        actor_state: Dict,
    ) -> Tuple[Array, Dict]:
        logits = compute_q_values(params, timestep.observation, timestep.ultimate_goal)
        if greedy:
            action = jnp.argmax(logits, axis=-1)
        else:
            action = jax.random.categorical(key, logits / temperature, axis=-1)
        return action, {}

    return eval_act_fn
