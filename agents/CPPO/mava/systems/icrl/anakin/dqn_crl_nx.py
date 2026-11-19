
import copy
import os
import pickle
import time
from typing import Any, Dict, Tuple

import chex
import flax
import flax.linen as nn  # noqa: F401  — kept for parity with sibling files
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

from mava.evaluator import get_nx_eval_fn
from hydra.utils import instantiate
from mava.networks import GoalEncoder
from mava.networks.icrl import sa_ConvEncoder
from mava.systems.icrl.utils import (
    compute_contrastive_metrics,
    compute_logits,
    contrastive_loss_fn,
    energy_fn,
)
from mava.types import ExperimentOutput
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.icrl_buffer import TrajectoryUniformSamplingQueue
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.navix_wrapper import (
    NavixTimestep,
    make_navix_env,
)


# HER relabel for grid observations (per-trajectory)


def relabel_fn_nx(gamma, transition, sample_key):
    """Hindsight relabeling on a single (T, ...) trajectory with grid obs.

    """
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    seeds = transition.extras["state_extras"]["seed"]
    single_trajectories = jnp.concatenate([seeds[:, jnp.newaxis].T] * seq_len, axis=0)
    seed_mask = jnp.equal(single_trajectories, single_trajectories.T)
    probs = probs * seed_mask + jnp.eye(seq_len) * 1e-5

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    relabeled_ultimate_goal = jnp.take(transition.achieved_goal, goal_index, axis=0)

    extras = {
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"]),
            "seed": jnp.squeeze(transition.extras["state_extras"]["seed"]),
        },
        "real_ultimate_goal": transition.ultimate_goal,
    }

    return transition._replace(
        ultimate_goal=relabeled_ultimate_goal,
        extras=extras,
    )


# Types


class Params(NamedTuple):
    sa_encoder: Any
    goal_encoder: Any


class OptStates(NamedTuple):
    critic: optax.OptState


class LearnerState(NamedTuple):
    params: Params
    opt_states: OptStates
    buffer_state: Any
    key: chex.PRNGKey
    env_state: NavixTimestep
    t: chex.Array  # env-step counter, used by the linear epsilon schedule


class Transition(NamedTuple):
    observation: chex.Array    # (H, W, 3)
    achieved_goal: chex.Array  # (goal_dim,)
    ultimate_goal: chex.Array  # (goal_dim,)
    action: chex.Array         # (action_size,) — one-hot
    reward: chex.Array         # ()
    discount: chex.Array       # ()
    avail_actions: chex.Array  # (action_size,) — all-ones on Navix
    extras: Dict


# Learner


def get_learner_fn(env, buffer, sa_encoder, g_encoder, critic_update_fn, config):
    num_envs = config.arch.num_envs
    rollout_length = config.system.rollout_length
    batch_size = config.system.batch_size
    action_size = config.system.action_size
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    energy_fn_name = config.system.get("energy_fn", "norm")
    energy_eps = config.system.get("energy_eps", 1e-6)
    contrastive_loss_name = config.system.get("contrastive_loss_fn", "fwd_infonce")

    eps_start = float(config.system.get("eps_start", 1.0))
    eps_min = float(config.system.get("eps_min", 0.05))
    eps_decay = float(config.system.get("eps_decay", 50000))

    def _epsilon(t):
        return jnp.maximum(eps_min, eps_start - t * (eps_start - eps_min) / eps_decay)

    def _compute_q_per_action(sa_params, g_repr, obs, achieved_goal):
        """Q(s, a) for every a — matches ff_ppo_crl_nx._compute_q_values."""
        B = obs.shape[0]

        def q_for_action(action_idx):
            a_onehot = jax.nn.one_hot(jnp.full(B, action_idx), action_size)
            sa_repr = sa_encoder.apply(sa_params, obs, achieved_goal, a_onehot)
            return energy_fn(energy_fn_name, sa_repr, g_repr, eps=energy_eps)  # (B,)

        q = jax.vmap(q_for_action)(jnp.arange(action_size))  # (A, B)
        return q.T  # (B, A)

    def epsilon_greedy_step(params, ts: NavixTimestep, key, t):
        obs = ts.observation                          # (N, H, W, 3)
        ultimate = ts.ultimate_goal                   # (N, goal_dim)
        achieved = env.get_achieved_goal(ts)          # (N, goal_dim)

        g_repr = g_encoder.apply(params.goal_encoder, ultimate)
        q_values = _compute_q_per_action(params.sa_encoder, g_repr, obs, achieved)  # (N, A)

        explore_key, rand_key = jax.random.split(key)
        greedy = jnp.argmax(q_values, axis=-1)
        rand = jax.random.randint(rand_key, shape=greedy.shape, minval=0, maxval=action_size)
        eps = _epsilon(t)
        explore = jax.random.uniform(explore_key, shape=greedy.shape) < eps
        actions = jnp.where(explore, rand, greedy)

        new_ts = env.step(ts, actions)

        actions_onehot = jax.nn.one_hot(actions, action_size)

        transition = Transition(
            observation=obs,
            achieved_goal=achieved,
            ultimate_goal=ultimate,
            action=actions_onehot,
            reward=new_ts.reward,
            discount=1.0 - new_ts.is_done().astype(jnp.float32),
            avail_actions=jnp.ones((num_envs, action_size)),
            extras={
                "state_extras": {
                    "truncation": new_ts.is_truncation().astype(jnp.float32),
                    "seed": new_ts.seed,
                }
            },
        )
        return new_ts, transition

    def collect_experience(params, ts, key, t):
        def f(carry, _):
            ts, k, t = carry
            k, sub = jax.random.split(k)
            new_ts, tr = epsilon_greedy_step(params, ts, sub, t)
            # Linear-eps schedule counts every parallel env step.
            return (new_ts, k, t + num_envs), tr

        (final_ts, _, final_t), trajs = jax.lax.scan(f, (ts, key, t), None, length=rollout_length)
        return final_ts, trajs, final_t

    def _env_step_only(learner_state: LearnerState, _: Any):
        """Collect rollout_length steps, append to buffer (used by prefill + train)."""
        params, opt_states, buffer_state, key, env_state, t = learner_state
        key, exp_key = jax.random.split(key)
        env_state, traj_batch, new_t = collect_experience(params, env_state, exp_key, t)
        buffer_state = buffer.insert(buffer_state, traj_batch)
        rollout_metrics = {
            "rollout_reward_mean": traj_batch.reward.mean(),
            "rollout_done_rate": (1.0 - traj_batch.discount).mean(),
            "rollout_win_rate": (traj_batch.reward > 0).mean() * 100.0,
            "epsilon": _epsilon(t),
        }
        return LearnerState(params, opt_states, buffer_state, key, env_state, new_t), rollout_metrics

    def _update_step(learner_state: LearnerState, _: Any):
        #  Rollout + buffer 
        learner_state, act_metrics = _env_step_only(learner_state, None)
        params, opt_states, buffer_state, key, env_state, t = learner_state

        #  Sample replay 
        key, sample_key, perm_key, mb_key = jax.random.split(key, 4)
        buffer_state, transitions = buffer.sample(buffer_state)

        #  HER relabel per trajectory 
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        transitions = jax.vmap(relabel_fn_nx, in_axes=(None, 0, 0))(
            config.system.gamma, transitions, batch_keys
        )
        # (num_envs, T, ...) → (num_envs * T, ...)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        # Permute, truncate to multiple of batch_size, reshape into minibatches.
        n = transitions.observation.shape[0]
        permutation = jax.random.permutation(perm_key, n)
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        n_keep = (n // batch_size) * batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:n_keep], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )

        #  Per-minibatch contrastive critic update 
        def _update_minibatch(carry, batch):
            params, opt_states, k = carry

            def critic_loss_fn(critic_params):
                sa_p = critic_params["sa_encoder"]
                g_p = critic_params["goal_encoder"]
                # batch.action is the one-hot action that was actually taken.
                sa_repr = sa_encoder.apply(sa_p, batch.observation, batch.achieved_goal, batch.action)
                g_repr = g_encoder.apply(g_p, batch.ultimate_goal)
                logits = compute_logits(energy_fn_name, sa_repr, g_repr, eps=energy_eps)
                loss = contrastive_loss_fn(contrastive_loss_name, logits, logsumexp_penalty_coeff)
                m = compute_contrastive_metrics(logits)
                # Diagnostic: Q(s, taken_a, g) on this minibatch's diagonal.
                q_taken = energy_fn(energy_fn_name, sa_repr, g_repr, eps=energy_eps)
                return loss, {
                    "critic_loss": loss,
                    "logits_pos": m["logits_pos"],
                    "logits_neg": m["logits_neg"],
                    "categorical_accuracy": m["categorical_accuracy"],
                    "logsumexp": m["logsumexp"],
                    "q_taken_mean": q_taken.mean(),
                }

            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            (_, info), grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_params)
            grads, info = lax.pmean((grads, info), axis_name="device")
            updates, new_c_opt = critic_update_fn(grads, opt_states.critic)
            new_sa = optax.apply_updates(params.sa_encoder, updates["sa_encoder"])
            new_g = optax.apply_updates(params.goal_encoder, updates["goal_encoder"])

            new_params = Params(sa_encoder=new_sa, goal_encoder=new_g)
            new_opts = OptStates(critic=new_c_opt)
            return (new_params, new_opts, k), info

        (params, opt_states, _), train_info = jax.lax.scan(
            _update_minibatch, (params, opt_states, mb_key), transitions
        )

        new_learner_state = LearnerState(
            params=params,
            opt_states=opt_states,
            buffer_state=buffer_state,
            key=key,
            env_state=env_state,
            t=t,
        )
        return new_learner_state, (act_metrics, train_info)

    def learner_fn(learner_state: LearnerState):
        learner_state, (act_metrics, train_metrics) = jax.lax.scan(
            _update_step, learner_state, None, length=config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=act_metrics,
            train_metrics=train_metrics,
        )

    def prefill_fn(learner_state: LearnerState):
        explore_steps = config.system.get("explore_steps", 1000)

        def cond(ls):
            return ls.buffer_state.size < explore_steps

        def body(ls):
            ls, _ = _env_step_only(ls, None)
            return ls

        return jax.lax.while_loop(cond, body, learner_state)

    return learner_fn, prefill_fn


# Eval act-fn (greedy = argmax of contrastive Q)


def make_eval_act_fn(sa_encoder, g_encoder, eval_env, action_size, config):
    energy_fn_name = config.system.get("energy_fn", "norm")
    energy_eps = config.system.get("energy_eps", 1e-6)

    def act_fn(params, ts: NavixTimestep, key, actor_state):
        # `params` arrives as a dict {"sa_encoder": ..., "goal_encoder": ...}
        # built by run_experiment from the (already device-replicated) tuple.
        achieved = eval_env.get_achieved_goal(ts)
        g_repr = g_encoder.apply(params["goal_encoder"], ts.ultimate_goal)
        B = ts.observation.shape[0]

        def q_for_action(a_idx):
            a_onehot = jax.nn.one_hot(jnp.full(B, a_idx), action_size)
            sa_repr = sa_encoder.apply(params["sa_encoder"], ts.observation, achieved, a_onehot)
            return energy_fn(energy_fn_name, sa_repr, g_repr, eps=energy_eps)

        q = jax.vmap(q_for_action)(jnp.arange(action_size))  # (A, B)
        action = jnp.argmax(q.T, axis=-1)
        return action, {}

    return act_fn


# Setup


def learner_setup(config: DictConfig):
    n_devices = len(jax.devices())

    terminate_on_goal_train = config.env.get("terminate_on_goal", False)
    terminate_on_goal_eval = config.env.get("terminate_on_goal_eval", True)
    max_steps = config.env.get("time_limit", 100)
    obs_type = config.env.get("obs_type", "full")
    fov_radius = config.env.get("fov_radius", 3)

    env = make_navix_env(
        env_name=config.env.env_name,
        terminate_on_goal=terminate_on_goal_train,
        max_steps=max_steps,
        obs_type=obs_type,
        fov_radius=fov_radius,
    )
    eval_env = make_navix_env(
        env_name=config.env.env_name,
        terminate_on_goal=terminate_on_goal_eval,
        max_steps=max_steps,
        obs_type=obs_type,
        fov_radius=fov_radius,
    )

    action_size = int(env.action_size)
    obs_shape = env.observation_shape
    goal_dim = env.goal_dim
    config.system.action_size = action_size
    config.system.obs_size = int(np.prod(obs_shape))

    key = jax.random.PRNGKey(config.system.seed)
    key, env_key, sa_key, g_key, buffer_key = jax.random.split(key, 5)

    num_envs = config.arch.num_envs
    env_keys = jax.random.split(env_key, n_devices * num_envs).reshape(n_devices, num_envs, -1)
    env_states = jax.vmap(env.reset)(env_keys)
    print(f"Goal coordinates from environment: {env_states.goal_pos[0, 0]}")

    # Networks
    sa_encoder = sa_ConvEncoder(torso=instantiate(config.network.sa_encoder_network.torso), output_dim=config.system.rep_size)
    g_encoder = GoalEncoder(torso=instantiate(config.network.goal_encoder_network.torso), output_dim=config.system.rep_size)

    dummy_obs = jnp.ones([1, *obs_shape])
    dummy_action = jnp.ones([1, action_size])
    dummy_goal = jnp.ones([1, goal_dim])
    dummy_achieved = jnp.ones([1, goal_dim])

    sa_params = sa_encoder.init(sa_key, dummy_obs, dummy_achieved, dummy_action)
    g_params = g_encoder.init(g_key, dummy_goal)
    params = Params(sa_encoder=sa_params, goal_encoder=g_params)

    # Optimizer (single optimizer for both encoders)
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    critic_opt = optax.chain(grad_clip, optax.adam(config.system.q_lr))
    critic_opt_state = critic_opt.init({"sa_encoder": sa_params, "goal_encoder": g_params})
    opt_states = OptStates(critic=critic_opt_state)

    # Replay buffer
    dummy_transition = Transition(
        observation=jnp.zeros(obs_shape, dtype=jnp.float32),
        achieved_goal=jnp.zeros((goal_dim,), dtype=jnp.float32),
        ultimate_goal=jnp.zeros((goal_dim,), dtype=jnp.float32),
        action=jnp.zeros((action_size,), dtype=jnp.float32),
        reward=jnp.zeros((), dtype=jnp.float32),
        discount=jnp.zeros((), dtype=jnp.float32),
        avail_actions=jnp.ones((action_size,), dtype=jnp.float32),
        extras={
            "state_extras": {
                "truncation": jnp.zeros((), dtype=jnp.float32),
                "seed": jnp.zeros((), dtype=jnp.int32),
            }
        },
    )

    def jit_wrap(b):
        b.insert_internal = jax.jit(b.insert_internal)
        b.sample_internal = jax.jit(b.sample_internal)
        return b

    buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.system.buffer_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=config.system.batch_size,
            num_envs=num_envs,
            episode_length=max_steps,
        )
    )
    buffer_state = jax.jit(buffer.init)(buffer_key)

    learn, prefill = get_learner_fn(env, buffer, sa_encoder, g_encoder, critic_opt.update, config)
    learn = jax.pmap(learn, axis_name="device")
    prefill = jax.pmap(prefill, axis_name="device")

    # Replicate per-device state
    params = flax.jax_utils.replicate(params, devices=jax.devices())
    opt_states = flax.jax_utils.replicate(opt_states, devices=jax.devices())
    buffer_state = flax.jax_utils.replicate(buffer_state, devices=jax.devices())
    key, sk = jax.random.split(key)
    step_keys = jax.random.split(sk, n_devices)
    t0 = jnp.zeros((n_devices,), dtype=jnp.int32)

    learner_state = LearnerState(
        params=params,
        opt_states=opt_states,
        buffer_state=buffer_state,
        key=step_keys,
        env_state=env_states,
        t=t0,
    )

    eval_act_fn = make_eval_act_fn(sa_encoder, g_encoder, eval_env, action_size, config)
    evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)
    abs_metric_evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)

    return learn, prefill, learner_state, evaluator, abs_metric_evaluator, sa_encoder, g_encoder


# Run


def run_experiment(_config: DictConfig) -> float:
    _config.logger.system_name = "dqn_crl_nx"
    config = copy.deepcopy(_config)
    n_devices = len(jax.devices())

    learn, prefill, learner_state, evaluator, abs_metric_evaluator, sa_encoder, g_encoder = (
        learner_setup(config)
    )

    # Warm up the replay buffer.
    learner_state = prefill(learner_state)
    jax.block_until_ready(learner_state)

    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "num_updates must exceed num_evaluation"
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation

    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.arch.num_envs
    )

    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,
        )

    save_dir = f"results/{config.env.env_name}"
    os.makedirs(save_dir, exist_ok=True)

    eval_key = jax.random.PRNGKey(config.system.seed + 1000)
    max_win_rate = -jnp.inf
    best_params = None

    for eval_step in range(config.arch.num_evaluation):
        start = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)
        elapsed = time.time() - start
        t = int(steps_per_rollout * (eval_step + 1))

        train_metrics = jax.tree_util.tree_map(jnp.mean, learner_output.train_metrics)
        train_metrics["steps_per_second"] = steps_per_rollout / elapsed
        act_metrics = jax.tree_util.tree_map(jnp.mean, learner_output.episode_metrics)

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        logger.log(act_metrics, t, eval_step, LogEvent.ACT)
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        eval_key, *keys = jax.random.split(eval_key, n_devices + 1)
        eval_keys = jnp.stack(keys)
        eval_params = {
            "sa_encoder": learner_output.learner_state.params.sa_encoder,
            "goal_encoder": learner_output.learner_state.params.goal_encoder,
        }
        eval_metrics = evaluator(eval_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        win_rate = float(jnp.mean(eval_metrics["win_rate"]))
        if win_rate > max_win_rate:
            max_win_rate = win_rate
            best_params = {
                "sa_encoder": jax.tree_util.tree_map(
                    lambda x: np.array(x[0]), learner_output.learner_state.params.sa_encoder
                ),
                "goal_encoder": jax.tree_util.tree_map(
                    lambda x: np.array(x[0]), learner_output.learner_state.params.goal_encoder
                ),
            }
            with open(f"{save_dir}/best_policy_dqn_crl_nx.pkl", "wb") as f:
                pickle.dump(best_params, f)

        if save_checkpoint:
            checkpointer.save(
                timestep=t,
                unreplicated_learner_state=jax.tree_util.tree_map(
                    lambda x: x[0], learner_output.learner_state
                ),
                episode_return=jnp.mean(eval_metrics.get("episode_return", jnp.zeros(()))),
            )

        learner_state = learner_output.learner_state

    if config.arch.absolute_metric and best_params is not None:
        eval_key, *keys = jax.random.split(eval_key, n_devices + 1)
        eval_keys = jnp.stack(keys)
        best_eval_params = {
            "sa_encoder": flax.jax_utils.replicate(best_params["sa_encoder"]),
            "goal_encoder": flax.jax_utils.replicate(best_params["goal_encoder"]),
        }
        eval_metrics = abs_metric_evaluator(best_eval_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return max_win_rate


@hydra.main(
    config_path="../../../configs/default",
    config_name="dqn_crl_nx.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    try:
        return run_experiment(cfg)
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e!s}")
        import traceback

        traceback.print_exc()
        return float("-inf")


if __name__ == "__main__":
    hydra_entry_point()
