import copy
import time
from typing import Any, Dict, Tuple

import chex
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

from mava.evaluator import get_icrl_eval_fn
from mava.networks import GoalEncoder, ICRLActor, SAEncoder
from mava.systems.icrl.types import ICRLParams, LearnerState, OptStates
from mava.types import ExperimentOutput, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.icrl_buffer import TrajectoryUniformSamplingQueue
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.episode_metrics import get_final_step_metrics


class JaxNavTransition(NamedTuple):
    """Transition for ff_icrl_jaxnav. Carries achieved_goal in extras (not in obs)."""

    observation: chex.Array
    action: chex.Array
    reward: chex.Array
    discount: chex.Array
    avail_actions: chex.Array
    extras: Dict[str, Any]


def _flatten_crl_jaxnav(gamma: float, transition: JaxNavTransition, sample_key: chex.PRNGKey) -> Dict:
    """HER relabeling for one trajectory.

    Samples a future achieved_goal per timestep and returns a flat dict
    ready for minibatch training. Returns seq_len-1 samples (drops last).
    """
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    seed = transition.extras["state_extras"]["seed"]
    single_trajectories = jnp.concatenate(
        [seed[:, jnp.newaxis].T] * seq_len, axis=0
    )
    seed_mask = jnp.equal(single_trajectories, single_trajectories.T)
    probs = probs * seed_mask + jnp.eye(seq_len) * 1e-5

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    relabeled_goal = jnp.take(transition.extras["achieved_goal"], goal_index[:-1], axis=0)

    return {
        "observation": transition.observation[:-1],
        "goal": relabeled_goal,
        "action": transition.action[:-1],
        "reward": transition.reward[:-1],
        "discount": transition.discount[:-1],
        "avail_actions": transition.avail_actions[:-1],
    }


def make_eval_act_fn(actor_apply_fn, env):
    """Deterministic eval act function for JaxNav CRL.

    Actor input: [obs, goal]. Ultimate goal is always zero distance for the
    distance goal_type used in JaxNav paper experiments.
    """
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0
    goal_dim = env.goal_dim

    def eval_act_fn(params, timestep, key, actor_state):
        obs_view = timestep.observation.agents_view
        goal = jnp.zeros(obs_view.shape[:-1] + (goal_dim,), dtype=obs_view.dtype)
        gc_obs = jnp.concatenate([obs_view, goal], axis=-1)
        means, _ = actor_apply_fn(params, gc_obs)
        action = action_bias + action_scale * nn.tanh(means)
        return action, actor_state

    return eval_act_fn


def get_learner_fn(
    env: MarlEnv,
    buffer: TrajectoryUniformSamplingQueue,
    apply_fns: Tuple,
    update_fns: Tuple,
    config: DictConfig,
) -> Any:
    """Get the learner function."""
    sa_encoder_apply, goal_encoder_apply, actor_apply = apply_fns
    actor_update_fn, critic_update_fn, alpha_update_fn = update_fns

    n_agents = env.num_agents
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0

    num_envs_agents = config.arch.num_envs * n_agents

    target_entropy = -env.action_dim * config.system.target_entropy_scale
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    gamma = config.system.gamma

    def _env_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Any]:
        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        def single_step(carry, _):
            key, env_state, last_timestep, buffer_state = carry
            key, noise_key = jax.random.split(key)

            obs = last_timestep.observation.agents_view
            # Access goal through RecordEpisodeMetricsState → JaxNavState
            goal = env_state.env_state.goal
            gc_obs = jnp.concatenate([obs, goal], axis=-1)
            means, log_stds = actor_apply(params.actor, gc_obs)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
            x_ts = means + stds * noise
            action_squashed = nn.tanh(x_ts)
            action = action_bias + action_scale * action_squashed

            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            flat_obs = last_timestep.observation.agents_view.reshape(num_envs_agents, -1)
            flat_action = action.reshape(num_envs_agents, -1)
            flat_reward = timestep.reward.reshape(num_envs_agents)
            flat_discount = timestep.discount.reshape(num_envs_agents)
            flat_avail = jnp.ones((num_envs_agents, env.action_dim))
            flat_achieved_goal = timestep.extras["env_metrics"]["achieved_goal"].reshape(
                num_envs_agents, -1
            )

            trunc_per_env = timestep.extras.get(
                "truncation", jnp.zeros(config.arch.num_envs, dtype=jnp.float32)
            )
            # Access episode_seed through RecordEpisodeMetricsState → JaxNavState
            seed_per_env = env_state.env_state.episode_seed
            trunc = jnp.repeat(trunc_per_env, n_agents)
            seed = jnp.repeat(seed_per_env, n_agents)

            transition = JaxNavTransition(
                observation=flat_obs,
                action=flat_action,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,
                extras={
                    "state_extras": {"truncation": trunc, "seed": seed},
                    "achieved_goal": flat_achieved_goal,
                },
            )

            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return (key, env_state, timestep, buffer_state), (transition, metrics)

        (key, env_state, last_timestep, buffer_state), (traj_batch, episode_metrics) = jax.lax.scan(
            single_step, (key, env_state, last_timestep, buffer_state), None, config.system.rollout_length
        )
        buffer_state = buffer.insert(buffer_state, traj_batch)

        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)
        return learner_state, episode_metrics

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        learner_state, episode_metrics = _env_step(learner_state, None)

        params, opt_states, buffer_state, key, env_state, last_timestep = learner_state

        key, sample_key = jax.random.split(key)
        buffer_state, transitions = buffer.sample(buffer_state)

        # HER relabeling: one key per trajectory in the sampled batch
        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        relabeled = jax.vmap(_flatten_crl_jaxnav, in_axes=(None, 0, 0))(
            gamma, transitions, batch_keys
        )

        # Flatten with Fortran order (interleave envs)
        relabeled = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), relabeled
        )

        # Shuffle and truncate to batch_size boundary
        perm_key, _ = jax.random.split(sample_key)
        perm = jax.random.permutation(perm_key, relabeled["observation"].shape[0])
        relabeled = jax.tree_util.tree_map(lambda x: x[perm], relabeled)

        num_samples = (relabeled["observation"].shape[0] // config.system.batch_size) * config.system.batch_size
        relabeled = jax.tree_util.tree_map(lambda x: x[:num_samples], relabeled)

        batched = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]), relabeled
        )

        def _update_minibatch(carry, batch):
            params, opt_states, key = carry
            key, actor_key = jax.random.split(key)

            obs = batch["observation"]
            action = batch["action"]
            goal = batch["goal"]
            avail_actions = batch["avail_actions"]

            def _critic_loss_fn(critic_params, obs, action, goal):
                sa_repr = sa_encoder_apply(critic_params["sa_encoder"], obs, action)
                g_repr = goal_encoder_apply(critic_params["goal_encoder"], goal)

                logits = -jnp.sqrt(
                    jnp.sum((sa_repr[:, None, :] - g_repr[None, :, :]) ** 2, axis=-1) + 1e-8
                )
                critic_loss = -jnp.mean(jnp.diag(logits) - jax.nn.logsumexp(logits, axis=1))
                logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
                critic_loss += logsumexp_penalty_coeff * jnp.mean(logsumexp**2)

                logits_pos = jnp.diag(logits).mean()
                logits_neg = (logits.sum() - jnp.diag(logits).sum()) / (logits.size - logits.shape[0])
                categorical_accuracy = (logits.argmax(axis=1) == jnp.arange(logits.shape[0])).mean()
                return critic_loss, {
                    "critic_loss": critic_loss,
                    "logits_pos": logits_pos,
                    "logits_neg": logits_neg,
                    "categorical_accuracy": categorical_accuracy,
                }

            def _actor_loss_fn(actor_params, critic_params, obs, goal, alpha, key):
                actor_input = jnp.concatenate([obs, goal], axis=-1)
                means, log_stds = actor_apply(actor_params, actor_input)
                stds = jnp.exp(log_stds)
                noise = jax.random.normal(key, shape=means.shape, dtype=means.dtype)
                x_ts = means + stds * noise
                action_squashed = nn.tanh(x_ts)
                action = action_bias + action_scale * action_squashed

                log_prob = -0.5 * noise**2 - log_stds
                log_prob -= jnp.log(1 - jnp.square(action_squashed) + 1e-6)
                log_prob = log_prob.sum(-1)

                sa_repr = sa_encoder_apply(critic_params["sa_encoder"], obs, action)
                g_repr = goal_encoder_apply(critic_params["goal_encoder"], goal)
                q_value = -jnp.sqrt(jnp.sum((sa_repr - g_repr) ** 2, axis=-1) + 1e-8)

                actor_loss = (alpha * log_prob - q_value).mean()
                return actor_loss, {"actor_loss": actor_loss, "entropy": -log_prob.mean()}

            def _alpha_loss_fn(log_alpha, entropy):
                alpha = jnp.exp(log_alpha)
                alpha_loss = alpha * jax.lax.stop_gradient(entropy - target_entropy)
                return alpha_loss, {"alpha_loss": alpha_loss, "alpha": alpha}

            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (_, critic_info), critic_grads = critic_grad_fn(critic_params, obs, action, goal)
            critic_updates, new_critic_opt_state = critic_update_fn(critic_grads, opt_states.critic)
            new_sa_encoder = optax.apply_updates(params.sa_encoder, critic_updates["sa_encoder"])
            new_goal_encoder = optax.apply_updates(params.goal_encoder, critic_updates["goal_encoder"])

            alpha = jnp.exp(params.log_alpha)
            critic_params_sg = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            (_, actor_info), actor_grads = actor_grad_fn(
                params.actor, critic_params_sg, obs, goal, alpha, actor_key
            )
            actor_updates, new_actor_opt_state = actor_update_fn(actor_grads, opt_states.actor)
            new_actor = optax.apply_updates(params.actor, actor_updates)

            if config.system.icrl.learnable_temperature:
                alpha_grad_fn = jax.value_and_grad(_alpha_loss_fn, has_aux=True)
                (_, alpha_info), alpha_grads = alpha_grad_fn(params.log_alpha, actor_info["entropy"])
                alpha_updates, new_alpha_opt_state = alpha_update_fn(alpha_grads, opt_states.alpha)
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
            else:
                new_log_alpha = params.log_alpha
                new_alpha_opt_state = opt_states.alpha
                alpha_info = {"alpha": alpha, "alpha_loss": 0.0}

            new_params = ICRLParams(
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
                actor=new_actor,
                log_alpha=new_log_alpha,
            )
            new_opt_states = OptStates(
                actor=new_actor_opt_state,
                critic=new_critic_opt_state,
                alpha=new_alpha_opt_state,
            )
            return (new_params, new_opt_states, key), critic_info | actor_info | alpha_info

        (params, opt_states, key), train_metrics = jax.lax.scan(
            _update_minibatch, (params, opt_states, key), batched
        )

        learner_state = LearnerState(params, opt_states, buffer_state, key, env_state, last_timestep)
        return learner_state, (episode_metrics, train_metrics)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput:
        learner_state, (episode_metrics, train_metrics) = jax.lax.scan(
            _update_step, learner_state, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_metrics,
            train_metrics=train_metrics,
        )

    def prefill_fn(learner_state):
        def filled_enough(ls):
            return jnp.min(ls.buffer_state.size) >= config.system.explore_steps

        return jax.lax.while_loop(
            lambda ls: jnp.logical_not(filled_enough(ls)),
            lambda ls: _env_step(ls, None)[0],
            learner_state,
        )

    return learner_fn, prefill_fn


def learner_setup(env: MarlEnv, keys: chex.Array, config: DictConfig) -> Tuple:
    config.system.num_agents = env.num_agents

    key, buffer_key, sa_key, goal_key, actor_key = keys

    sa_encoder_torso = instantiate(config.network.sa_encoder_network.pre_torso)
    goal_encoder_torso = instantiate(config.network.goal_encoder_network.pre_torso)
    actor_torso = instantiate(config.network.actor_network.pre_torso)

    rep_size = config.system.get("rep_size", 64)
    sa_encoder = SAEncoder(torso=sa_encoder_torso, output_dim=rep_size)
    goal_encoder = GoalEncoder(torso=goal_encoder_torso, output_dim=rep_size)
    actor_network = ICRLActor(
        torso=actor_torso,
        action_size=env.action_dim,
        LOG_STD_MIN=config.system.get("log_std_min", -5.0),
        LOG_STD_MAX=config.system.get("log_std_max", 2.0),
    )

    obs_dim = env.observation_spec.agents_view.shape[-1]
    sa_encoder_params = sa_encoder.init(sa_key, jnp.zeros((1, obs_dim)), jnp.zeros((1, env.action_dim)))
    goal_encoder_params = goal_encoder.init(goal_key, jnp.zeros((1, env.goal_dim)))
    actor_params = actor_network.init(actor_key, jnp.zeros((1, obs_dim + env.goal_dim)))
    log_alpha = jnp.log(config.system.get("init_alpha", 0.1))

    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
        actor=actor_params,
        log_alpha=log_alpha,
    )

    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    actor_opt = optax.chain(grad_clip, optax.adam(config.system.policy_lr))
    critic_opt = optax.chain(grad_clip, optax.adam(config.system.q_lr))
    alpha_opt = optax.chain(grad_clip, optax.adam(config.system.alpha_lr))

    critic_params_struct = {"sa_encoder": sa_encoder_params, "goal_encoder": goal_encoder_params}
    opt_states = OptStates(
        actor=actor_opt.init(actor_params),
        critic=critic_opt.init(critic_params_struct),
        alpha=alpha_opt.init(log_alpha),
    )

    num_envs_agents = config.arch.num_envs * env.num_agents
    dummy_transition = JaxNavTransition(
        observation=jnp.zeros((obs_dim,)),
        action=jnp.zeros((env.action_dim,)),
        reward=0.0,
        discount=0.0,
        avail_actions=jnp.ones((env.action_dim,)),
        extras={
            "state_extras": {"truncation": 0.0, "seed": 0.0},
            "achieved_goal": jnp.zeros((env.goal_dim,)),
        },
    )

    def jit_wrap(buf):
        buf.insert_internal = jax.jit(buf.insert_internal)
        buf.sample_internal = jax.jit(buf.sample_internal)
        return buf

    buffer = jit_wrap(
        TrajectoryUniformSamplingQueue(
            max_replay_size=config.system.buffer_size,
            dummy_data_sample=dummy_transition,
            sample_batch_size=config.system.batch_size,
            num_envs=num_envs_agents,
            episode_length=config.env.max_steps,
        )
    )
    buffer_state = jax.jit(buffer.init)(buffer_key)

    apply_fns = (sa_encoder.apply, goal_encoder.apply, actor_network.apply)
    update_fns = (actor_opt.update, critic_opt.update, alpha_opt.update)

    learn, prefill = get_learner_fn(env, buffer, apply_fns, update_fns, config)
    learn = jax.jit(learn)
    prefill = jax.jit(prefill)

    key, *env_keys = jax.random.split(key, config.arch.num_envs + 1)
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0,))(jnp.stack(env_keys))

    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    key, step_key = jax.random.split(key)
    init_learner_state = LearnerState(params, opt_states, buffer_state, step_key, env_states, timesteps)
    return learn, prefill, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    _config.logger.system_name = "ff_icrl_jaxnav"
    config = copy.deepcopy(_config)

    env, eval_env = environments.make(config)

    key, buffer_key, key_e, sa_key, goal_key, actor_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=6
    )

    learn, prefill, actor_network, learner_state = learner_setup(
        env, (key, buffer_key, sa_key, goal_key, actor_key), config
    )

    learner_state = prefill(learner_state)
    jax.block_until_ready(learner_state)

    eval_act_fn = make_eval_act_fn(actor_network.apply, eval_env)
    evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    config = check_total_timesteps(config)
    assert config.system.num_updates >= config.arch.num_evaluation, (
        "Number of updates must be >= number of evaluations."
    )

    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        config.system.num_updates_per_eval * config.system.rollout_length * config.arch.num_envs
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

    max_success = -jnp.inf
    best_params = None
    fixed_eval_key = jax.random.split(key_e, jax.device_count())

    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()
        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        trained_params = learner_output.learner_state.params.actor
        eval_metrics = evaluator(
            jax.device_put_replicated(trained_params, jax.local_devices()), fixed_eval_key, {}
        )
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        success = jnp.mean(eval_metrics["success"])

        if save_checkpoint:
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_output.learner_state,
                episode_return=success,
            )

        if config.arch.absolute_metric and max_success <= success:
            best_params = copy.deepcopy(trained_params)
            max_success = success

        learner_state = learner_output.learner_state

    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    if config.arch.absolute_metric:
        abs_metric_evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_metrics = abs_metric_evaluator(
            jax.device_put_replicated(best_params, jax.local_devices()), fixed_eval_key, {}
        )
        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_icrl_jaxnav.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    try:
        return run_experiment(cfg)
    except Exception as e:
        try:
            override_info = cfg.hydra.job.override_dirname
        except Exception:
            override_info = "unknown"
        print(f"Error executing job with overrides: {override_info}")
        print(f"Exception: {type(e).__name__}: {e!s}")
        import traceback
        traceback.print_exc()
        return float("-inf")


if __name__ == "__main__":
    hydra_entry_point()
