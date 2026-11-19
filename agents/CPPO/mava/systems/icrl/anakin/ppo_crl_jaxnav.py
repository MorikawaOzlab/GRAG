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
from mava.systems.icrl.utils import compute_contrastive_metrics, compute_logits, contrastive_loss_fn, energy_fn
from mava.systems.icrl.types import ICRLParams, OptStates
from mava.types import ExperimentOutput, MarlEnv
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.episode_metrics import get_final_step_metrics


def make_eval_act_fn(actor_apply_fn, env, deterministic=False):
    """Make eval act function."""
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0
    goal_dim = env.goal_dim

    def eval_act_fn(params, timestep, key, actor_state):
        obs_view = timestep.observation.agents_view
        # Ultimate goal is always zero distance .
        goal = jnp.zeros(obs_view.shape[:-1] + (goal_dim,), dtype=obs_view.dtype)
        current_ag = timestep.extras["env_metrics"]["achieved_goal"]
        while current_ag.ndim < obs_view.ndim:
            current_ag = current_ag[..., None]
        gc_obs = jnp.concatenate([obs_view, current_ag, goal], axis=-1)

        means, log_stds = actor_apply_fn(params, gc_obs)
        if deterministic:
            action = action_bias + action_scale * nn.tanh(means)
        else:
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            x_ts = means + stds * noise
            action = action_bias + action_scale * nn.tanh(x_ts)
        return action, actor_state

    return eval_act_fn


class PPOCRLLearnerState(NamedTuple):
    """Learner state for pure PPO-CRL continuous."""

    params: ICRLParams
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: Any
    last_timestep: Any
    update_count: chex.Array  


class PPOTransition(NamedTuple):
    """Transition for PPO-CRL with log probs for importance sampling."""

    observation: chex.Array  
    action: chex.Array  
    x_t: chex.Array  
    log_prob: chex.Array  
    reward: chex.Array
    discount: chex.Array
    avail_actions: chex.Array
    extras: Dict[str, Any]


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple,
    update_fns: Tuple,
    config: DictConfig,
) -> Any:
    """Get the learner function for pure PPO-CRL continuous."""

    sa_encoder_apply, goal_encoder_apply, actor_apply = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    n_agents = env.num_agents
    action_dim = env.action_dim
    action_spec = env.action_spec
    action_scale = (action_spec.maximum - action_spec.minimum) / 2.0
    action_bias = (action_spec.maximum + action_spec.minimum) / 2.0

    num_envs_agents = config.arch.num_envs * n_agents

    # CRL config
    energy_fn_name = config.system.get("energy_fn", "norm")
    contrastive_loss_name = config.system.get("contrastive_loss_fn", "fwd_infonce")
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff

    # PPO config
    clip_eps = config.system.clip_eps
    ent_coef = float(config.system.ent_coef)  # constant scalar
    num_mc_samples = config.system.num_mc_samples
    num_epochs = config.system.get("num_epochs", 4)

    def _env_step(
        learner_state: PPOCRLLearnerState, _: Any
    ) -> Tuple[PPOCRLLearnerState, Tuple[dict, PPOTransition]]:
        """Step the environment for rollout_length steps."""
        params, opt_states, key, env_state, last_timestep, update_count = learner_state

        def single_step(carry, _):
            key, env_state, last_timestep = carry
            key, noise_key = jax.random.split(key)

            obs = last_timestep.observation.agents_view  # [N_env, N_agent, obs_dim]
            current_ag = last_timestep.extras["env_metrics"]["achieved_goal"]
            while current_ag.ndim < obs.ndim:
                current_ag = current_ag[..., None]
            gc_obs = jnp.concatenate([obs, current_ag, env_state.env_state.goal], axis=-1)

            means, log_stds = actor_apply(params.actor, gc_obs)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(noise_key, shape=means.shape, dtype=means.dtype)
            x_ts = means + stds * noise
            action_squashed = nn.tanh(x_ts)
            action = action_bias + action_scale * action_squashed

            log_prob = -0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)
            log_prob = log_prob.sum(-1)  # [N_env, N_agent]

            # Real env goal seen by this rollout step .
            real_goal = env_state.env_state.goal

            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # Flatten across (env, agent) for processing.
            flat_obs = last_timestep.observation.agents_view.reshape(num_envs_agents, -1)
            flat_action = action.reshape(num_envs_agents, -1)
            flat_x_t = x_ts.reshape(num_envs_agents, -1)
            flat_log_prob = log_prob.reshape(num_envs_agents)
            flat_reward = timestep.reward.reshape(num_envs_agents)
            flat_discount = timestep.discount.reshape(num_envs_agents)
            flat_avail = jnp.ones((num_envs_agents, action_dim))
            flat_achieved_goal = timestep.extras["env_metrics"]["achieved_goal"].reshape(
                num_envs_agents, -1
            )
            flat_real_goal = real_goal.reshape(num_envs_agents, -1)

            # Broadcast seed and truncation to all agents for HER masking.
            trunc_per_env = timestep.extras.get(
                "truncation", jnp.zeros(config.arch.num_envs, dtype=jnp.float32)
            )
            seed_per_env = env_state.env_state.episode_seed
            trunc = jnp.repeat(trunc_per_env, n_agents)
            seed = jnp.repeat(seed_per_env, n_agents)

            extras = {
                "state_extras": {"truncation": trunc, "seed": seed},
                "achieved_goal": flat_achieved_goal,
                "real_goal": flat_real_goal,  # for actor + advantage during training
            }

            flat_current_ag = current_ag.reshape(num_envs_agents, -1)
            extras["current_achieved_goal"] = flat_current_ag

            transition = PPOTransition(
                observation=flat_obs,
                action=flat_action,
                x_t=flat_x_t,
                log_prob=flat_log_prob,
                reward=flat_reward,
                discount=flat_discount,
                avail_actions=flat_avail,
                extras=extras,
            )

            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]

            return (key, env_state, timestep), (transition, metrics)

        (key, env_state, last_timestep), (traj_batch, episode_metrics) = jax.lax.scan(
            single_step,
            (key, env_state, last_timestep),
            None,
            config.system.rollout_length,
        )

        learner_state = PPOCRLLearnerState(
            params, opt_states, key, env_state, last_timestep, update_count
        )
        return learner_state, (episode_metrics, traj_batch)

    def _flatten_crl(buffer_config, transition, sample_key):
        """HER goal relabeling per trajectory.

        Saves both the relabeled goal (for the critic) and the original real env
        goal (for the actor + advantage).
        """
        gamma, _ = buffer_config

        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(
            arrangement[:, None] < arrangement[None], dtype=jnp.float32
        )
        discount = gamma ** jnp.array(
            arrangement[None] - arrangement[:, None], dtype=jnp.float32
        )
        probs = is_future_mask * discount

        # Same-episode mask via per-step seed.
        single_trajectories = jnp.concatenate(
            [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len,
            axis=0,
        )
        seed_mask = jnp.equal(single_trajectories, single_trajectories.T)
        probs = probs * seed_mask + jnp.eye(seq_len) * 1e-5

        # Sample a future achieved goal per timestep.
        goal_index = jax.random.categorical(sample_key, jnp.log(probs))
        relabeled_goal = jnp.take(
            transition.extras["achieved_goal"], goal_index[:-1], axis=0
        )

        result = {
            "observation": transition.observation[:-1],
            "goal": relabeled_goal,                              # HER (critic only)
            "real_goal": transition.extras["real_goal"][:-1],    # actor + advantage
            "action": transition.action[:-1],
            "x_t": transition.x_t[:-1],
            "old_log_prob": transition.log_prob[:-1],
            "reward": transition.reward[:-1],
        }

        result["current_achieved_goal"] = transition.extras["current_achieved_goal"][:-1]
        return result

    def _update_step(
        learner_state: PPOCRLLearnerState, _: Any
    ) -> Tuple[PPOCRLLearnerState, Tuple]:
        """Collect rollout, HER-relabel, compute advantages, run PPO epochs."""

        learner_state, (episode_metrics, traj_batch) = _env_step(learner_state, None)

        params, opt_states, key, env_state, last_timestep, update_count = learner_state

        key, sample_key = jax.random.split(key)

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(x, 0, 1), traj_batch
        )

        batch_keys = jax.random.split(sample_key, transitions.observation.shape[0])
        relabeled = jax.vmap(_flatten_crl, in_axes=(None, 0, 0))(
            (config.system.gamma, None), transitions, batch_keys
        )

        relabeled = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), relabeled
        )

        # Truncate to make divisible by batch_size.
        num_samples = (
            relabeled["observation"].shape[0] // config.system.batch_size
        ) * config.system.batch_size
        relabeled = jax.tree_util.tree_map(lambda x: x[:num_samples], relabeled)

        # Compute CRL advantages: A = Q(s, a_taken, g_real) - V_MC(s, g_real)
        key, adv_key = jax.random.split(key)

        def _compute_advantages(carry, batch):
            """Compute Q and V baseline, Conditions on REAL env goal."""
            key = carry
            key, mc_key = jax.random.split(key)

            obs = batch["observation"]
            goal = batch["real_goal"]
            action = batch["action"]
            current_ag = batch["current_achieved_goal"]
            sa_state = jnp.concatenate([obs, current_ag], axis=-1)
            actor_input = jnp.concatenate([obs, current_ag, goal], axis=-1)
    
            # Actor forward (for MC sampling).
            means, log_stds = actor_apply(params.actor, actor_input)
            stds = jnp.exp(log_stds)

            # Q(s, a_taken, g_real).
            sa_repr_taken = sa_encoder_apply(params.sa_encoder, sa_state, action)
            g_repr = goal_encoder_apply(params.goal_encoder, goal)
            q_taken = energy_fn(energy_fn_name, sa_repr_taken, g_repr)

            # MC V(s, g_real).
            mc_noise = jax.random.normal(
                mc_key, shape=(num_mc_samples,) + means.shape, dtype=means.dtype
            )
            mc_x_ts = means[None, :, :] + stds[None, :, :] * mc_noise
            mc_actions_squashed = nn.tanh(mc_x_ts)
            mc_actions = action_bias + action_scale * mc_actions_squashed

            def compute_q_for_sample(mc_action):
                sa_repr = sa_encoder_apply(params.sa_encoder, sa_state, mc_action)
                return energy_fn(energy_fn_name, sa_repr, g_repr)

            mc_q_values = jax.vmap(compute_q_for_sample)(mc_actions)
            v_mc = jnp.mean(mc_q_values, axis=0)

            advantage = q_taken - v_mc
            return key, {"advantage": advantage, "q_taken": q_taken, "v_mc": v_mc}

        adv_batched = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
            relabeled,
        )
        _, adv_results = jax.lax.scan(_compute_advantages, adv_key, adv_batched)
        advantages = adv_results["advantage"].reshape(-1)
        q_taken_all = adv_results["q_taken"].reshape(-1)
        v_mc_all = adv_results["v_mc"].reshape(-1)

        # Normalize advantages.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        relabeled["advantage"] = advantages

        # PPO epochs over the relabeled batch
        def _update_minibatch(carry, batch):
            params, opt_states, key = carry
            key, _ = jax.random.split(key)

            obs = batch["observation"]
            relabeled_g = batch["goal"]      # for critic (CRL contrastive)
            real_g = batch["real_goal"]      # for actor (PPO ratio)
            action = batch["action"]
            x_t_stored = batch["x_t"]
            old_log_prob = batch["old_log_prob"]
            advantage = batch["advantage"]
            current_ag = batch["current_achieved_goal"]
            sa_state = jnp.concatenate([obs, current_ag], axis=-1)

            # Critic update: InfoNCE on (s, a) <-> relabeled goal
            def _critic_loss_fn(critic_params, sa_state, action, goal):
                sa_repr = sa_encoder_apply(critic_params["sa_encoder"], sa_state, action)
                g_repr = goal_encoder_apply(critic_params["goal_encoder"], goal)

                logits = compute_logits(energy_fn_name, sa_repr, g_repr)
                critic_loss = contrastive_loss_fn(
                    contrastive_loss_name, logits, logsumexp_penalty_coeff
                )

                metrics = compute_contrastive_metrics(logits)
                loss_info = {
                    "critic_loss": critic_loss,
                    "logits_pos": metrics["logits_pos"],
                    "logits_neg": metrics["logits_neg"],
                    "categorical_accuracy": metrics["categorical_accuracy"],
                    "logsumexp": metrics["logsumexp"],
                }
                return critic_loss, loss_info

            critic_params = {
                "sa_encoder": params.sa_encoder,
                "goal_encoder": params.goal_encoder,
            }
            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            (_, critic_info), critic_grads = critic_grad_fn(
                critic_params, sa_state, action, relabeled_g
            )

            critic_updates, new_critic_opt_state = critic_update_fn(
                critic_grads, opt_states.critic
            )
            new_sa_encoder = optax.apply_updates(
                params.sa_encoder, critic_updates["sa_encoder"]
            )
            new_goal_encoder = optax.apply_updates(
                params.goal_encoder, critic_updates["goal_encoder"]
            )

            # Actor update: PPO clipped on REAL env goal
            def _actor_loss_fn(actor_params, obs, real_g, x_t_stored, old_log_prob, advantage):
                actor_input = jnp.concatenate([obs, batch["current_achieved_goal"], real_g], axis=-1)
                means, log_stds = actor_apply(actor_params, actor_input)
                stds = jnp.exp(log_stds)

                # Pre-tanh Gaussian log_prob from stored x_t (must mirror rollout form).
                noise = (x_t_stored - means) / (stds + 1e-8)
                log_prob = -0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)
                log_prob = log_prob.sum(-1)

                # PPO clipped loss with pre-computed advantage.
                ratio = jnp.exp(log_prob - old_log_prob)
                ratio_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
                loss_unclipped = -advantage * ratio
                loss_clipped = -advantage * ratio_clipped
                ppo_loss = jnp.mean(jnp.maximum(loss_unclipped, loss_clipped))

                # Constant entropy bonus (Gaussian entropy from log_stds).
                entropy = (log_stds + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1).mean()
                actor_loss = ppo_loss - ent_coef * entropy

                loss_info = {
                    "actor_loss": actor_loss,
                    "ppo_loss": ppo_loss,
                    "entropy": entropy,
                    "ratio_mean": ratio.mean(),
                }
                return actor_loss, loss_info

            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            (_, actor_info), actor_grads = actor_grad_fn(
                params.actor, obs, real_g, x_t_stored, old_log_prob, advantage
            )

            actor_updates, new_actor_opt_state = actor_update_fn(
                actor_grads, opt_states.actor
            )
            new_actor = optax.apply_updates(params.actor, actor_updates)

            new_params = ICRLParams(
                sa_encoder=new_sa_encoder,
                goal_encoder=new_goal_encoder,
                actor=new_actor,
            )
            new_opt_states = OptStates(
                actor=new_actor_opt_state,
                critic=new_critic_opt_state,
            )

            metrics = critic_info | actor_info
            metrics["advantage_mean"] = advantage.mean()
            metrics["q_taken_mean"] = q_taken_all.mean()
            metrics["v_mc_mean"] = v_mc_all.mean()
            metrics["ent_coef"] = jnp.asarray(ent_coef)
            return (new_params, new_opt_states, key), metrics

        def _train_epoch(carry, _):
            params, opt_states, key, flat_data = carry

            key, perm_key = jax.random.split(key)
            perm = jax.random.permutation(perm_key, flat_data["observation"].shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_data)

            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, config.system.batch_size) + x.shape[1:]),
                shuffled,
            )

            (params, opt_states, key), epoch_metrics = jax.lax.scan(
                _update_minibatch, (params, opt_states, key), batched
            )

            return (params, opt_states, key, flat_data), epoch_metrics

        (params, opt_states, key, _), train_metrics = jax.lax.scan(
            _train_epoch, (params, opt_states, key, relabeled), None, num_epochs
        )

        learner_state = PPOCRLLearnerState(
            params, opt_states, key, env_state, last_timestep, update_count + 1
        )
        return learner_state, (episode_metrics, train_metrics)

    def learner_fn(learner_state: PPOCRLLearnerState) -> ExperimentOutput:
        """Run num_updates_per_eval update steps."""
        learner_state, (episode_metrics, train_metrics) = jax.lax.scan(
            _update_step, learner_state, None, config.system.num_updates_per_eval
        )

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_metrics,
            train_metrics=train_metrics,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple:
    """Initialize learner_fn, networks, optimizers, and states."""
    config.system.num_agents = env.num_agents

    key, sa_key, goal_key, actor_key = jax.random.split(keys[0], 4)

    # Networks (encoders + actor).
    sa_encoder_torso = instantiate(config.network.sa_encoder_network.pre_torso)
    goal_encoder_torso = instantiate(config.network.goal_encoder_network.pre_torso)
    actor_torso = instantiate(config.network.actor_network.pre_torso)

    rep_size = config.system.get("rep_size", 64)
    sa_encoder = SAEncoder(torso=sa_encoder_torso, output_dim=rep_size)
    goal_encoder = GoalEncoder(torso=goal_encoder_torso, output_dim=rep_size)
    log_std_min = config.system.get("log_std_min", -5.0)
    log_std_max = config.system.get("log_std_max", 2.0)
    actor_network = ICRLActor(
        torso=actor_torso, action_size=env.action_dim,
        LOG_STD_MIN=log_std_min, LOG_STD_MAX=log_std_max,
    )

    # Dummy inputs for init.
    obs_dim = env.observation_spec.agents_view.shape[-1]
    goal_dim = env.goal_dim

    init_sa_state = jnp.zeros((1, obs_dim + goal_dim))
    init_actor_input = jnp.zeros((1, obs_dim + goal_dim + goal_dim))

    init_action = jnp.zeros((1, env.action_dim))
    init_goal = jnp.zeros((1, goal_dim))

    sa_encoder_params = sa_encoder.init(sa_key, init_sa_state, init_action)
    goal_encoder_params = goal_encoder.init(goal_key, init_goal)
    actor_params = actor_network.init(actor_key, init_actor_input)

    params = ICRLParams(
        sa_encoder=sa_encoder_params,
        goal_encoder=goal_encoder_params,
        actor=actor_params,
    )

    # Optimizers (Adam with global grad clip).
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    actor_lr = config.system.actor_lr
    critic_lr = config.system.q_lr
    actor_opt = optax.chain(grad_clip, optax.adam(actor_lr))
    critic_opt = optax.chain(grad_clip, optax.adam(critic_lr))

    critic_params_struct = {
        "sa_encoder": sa_encoder_params,
        "goal_encoder": goal_encoder_params,
    }
    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params_struct)
    opt_states = OptStates(actor=actor_opt_state, critic=critic_opt_state)

    apply_fns = (sa_encoder.apply, goal_encoder.apply, actor_network.apply)
    update_fns = (actor_opt.update, critic_opt.update)

    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.jit(learn)

    # Initial env states.
    key, *env_keys = jax.random.split(key, config.arch.num_envs + 1)
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0,))(jnp.stack(env_keys))

    # Optional checkpoint load.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    key, step_key = jax.random.split(key)
    init_learner_state = PPOCRLLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_key,
        env_state=env_states,
        last_timestep=timesteps,
        update_count=jnp.array(0, dtype=jnp.int32),
    )

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Run pure PPO-CRL continuous experiment."""
    _config.logger.system_name = "ppo_crl_jaxnav"
    config = copy.deepcopy(_config)

    env, eval_env = environments.make(config)

    key, key_e, sa_key, goal_key, actor_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=5
    )

    learn, actor_network, learner_state = learner_setup(
        env, (key, sa_key, goal_key, actor_key), config
    )

    jax.block_until_ready(learner_state)

    # eval
    eval_act_fn = make_eval_act_fn(actor_network.apply, eval_env)
    evaluator = get_icrl_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    config = check_total_timesteps(config)
    assert config.system.num_updates >= config.arch.num_evaluation, (
        "Number of updates must be >= number of evaluations."
    )

    config.system.num_updates_per_eval = (
        config.system.num_updates // config.arch.num_evaluation
    )
    steps_per_rollout = (
        config.system.num_updates_per_eval
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

    max_success = -jnp.inf
    best_params = None

    fixed_eval_key = jax.random.split(key_e, jax.device_count())

    for eval_step in range(config.arch.num_evaluation):
        start_time = time.time()

        learner_output = learn(learner_state)
        jax.block_until_ready(learner_output)

        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))

        episode_metrics, ep_completed = get_final_step_metrics(
            learner_output.episode_metrics
        )
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        trained_params = learner_output.learner_state.params.actor

        eval_metrics = evaluator(jax.device_put_replicated(trained_params, jax.local_devices()), fixed_eval_key, {})
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
        abs_metric_evaluator = get_icrl_eval_fn(
            eval_env, eval_act_fn, config, absolute_metric=True
        )
        eval_metrics = abs_metric_evaluator(jax.device_put_replicated(best_params, jax.local_devices()), fixed_eval_key, {})
        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="ppo_crl_jaxnav.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)

    try:
        eval_performance = run_experiment(cfg)
        return eval_performance
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
