
import copy
import os
import pickle
import time
from typing import Any, Dict, Tuple

import chex
import flax
import flax.linen as nn
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import variance_scaling
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

from mava.evaluator import get_nx_eval_fn
from mava.networks.icrl import sa_ConvEncoder, small_G_encoder
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
    actor: Any
    sa_encoder: Any
    goal_encoder: Any
    log_alpha: chex.Array


class OptStates(NamedTuple):
    actor: optax.OptState
    critic: optax.OptState
    alpha: optax.OptState


class LearnerState(NamedTuple):
    params: Params
    opt_states: OptStates
    buffer_state: Any
    key: chex.PRNGKey
    env_state: NavixTimestep
    t: chex.Array


class Transition(NamedTuple):
    observation: chex.Array    # (H, W, 3)
    achieved_goal: chex.Array  # (goal_dim,)
    ultimate_goal: chex.Array  # (goal_dim,)
    action: chex.Array         # (action_size,) — continuous (post-tanh)
    reward: chex.Array         # ()
    discount: chex.Array       # ()
    avail_actions: chex.Array  # (action_size,)
    extras: Dict


# Goal-conditioned CNN actor with (mean, log_std) output heads


class ConvActorGaussian(nn.Module):
    """SAC-style actor for grid observations.

    Input: grid_obs (..., H, W, 3), achieved_goal (..., goal_dim),
           ultimate_goal (..., goal_dim).
    Output: (mean, log_std) each of shape (..., action_size).

    """

    num_actions: int
    mlp_hidden: int = 256
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        grid_obs: chex.Array,
        achieved_goal: chex.Array,
        ultimate_goal: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        lead = grid_obs.shape[:-3]
        h, w, c = grid_obs.shape[-3:]
        x = grid_obs.reshape(-1, h, w, c)

        x = nn.Conv(16, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(32, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(64, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = x.reshape(x.shape[0], -1)

        ag = achieved_goal.reshape(-1, achieved_goal.shape[-1])
        g = ultimate_goal.reshape(-1, ultimate_goal.shape[-1])
        x = jnp.concatenate([x, ag, g], axis=-1)

        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.swish(x)

        mean = nn.Dense(self.num_actions, kernel_init=init, bias_init=bias_init)(x)
        log_std = nn.Dense(self.num_actions, kernel_init=init, bias_init=bias_init)(x)
        log_std = nn.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)

        mean = mean.reshape(*lead, self.num_actions)
        log_std = log_std.reshape(*lead, self.num_actions)
        return mean, log_std


def _sample_action_and_log_prob(means, log_stds, avail_mask, key):
    stds = jnp.exp(log_stds)
    masked_means = means - (1.0 - avail_mask) * 1e10
    noise = jax.random.gumbel(key, shape=means.shape, dtype=means.dtype)
    x_ts = masked_means + stds * noise
    action = nn.tanh(x_ts)

    x_std = (x_ts - masked_means) / stds
    log_prob = -(x_std + jnp.exp(-x_std)) - log_stds
    log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
    log_prob = jnp.where(avail_mask == 0, 0.0, log_prob)
    log_prob = log_prob.sum(-1)
    return action, log_prob


# Learner


def get_learner_fn(
    env,
    buffer: TrajectoryUniformSamplingQueue,
    actor,
    sa_encoder,
    g_encoder,
    actor_update_fn,
    critic_update_fn,
    alpha_update_fn,
    config: DictConfig,
):
    num_envs = config.arch.num_envs
    rollout_length = config.system.rollout_length
    batch_size = config.system.batch_size
    action_size = config.system.action_size
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    energy_fn_name = config.system.get("energy_fn", "norm")
    energy_eps = config.system.get("energy_eps", 1e-6)
    contrastive_loss_name = config.system.get("contrastive_loss_fn", "fwd_infonce")
    target_entropy = -jnp.log(action_size) * config.system.target_entropy_scale
    autotune = config.system.get("autotune", True)
    alpha_min = config.system.get("alpha_min", 0.0)

    def actor_step(params, ts: NavixTimestep, key):
        obs = ts.observation                          # (N, H, W, 3)
        ultimate = ts.ultimate_goal                   # (N, goal_dim)
        achieved = env.get_achieved_goal(ts)          # (N, goal_dim)

        means, log_stds = actor.apply(params.actor, obs, achieved, ultimate)
        avail_mask = jnp.ones_like(means)
        action_continuous, _ = _sample_action_and_log_prob(means, log_stds, avail_mask, key)
        action_discrete = jnp.argmax(action_continuous, axis=-1)

        new_ts = env.step(ts, action_discrete)

        transition = Transition(
            observation=obs,
            achieved_goal=achieved,
            ultimate_goal=ultimate,
            action=action_continuous,
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

    def collect_experience(params, ts, key):
        def f(carry, _):
            ts, k = carry
            k, sub = jax.random.split(k)
            new_ts, tr = actor_step(params, ts, sub)
            return (new_ts, k), tr

        (final_ts, _), trajs = jax.lax.scan(f, (ts, key), None, length=rollout_length)
        return final_ts, trajs

    def _env_step_only(learner_state: LearnerState, _: Any):
        """Collect rollout_length steps and append to the buffer (used by prefill)."""
        params, opt_states, buffer_state, key, env_state, t = learner_state
        key, exp_key = jax.random.split(key)
        env_state, traj_batch = collect_experience(params, env_state, exp_key)
        buffer_state = buffer.insert(buffer_state, traj_batch)
        rollout_metrics = {
            "rollout_reward_mean": traj_batch.reward.mean(),
            "rollout_done_rate": (1.0 - traj_batch.discount).mean(),
            "rollout_win_rate": (traj_batch.reward > 0).mean() * 100.0,
        }
        new_t = t + num_envs * rollout_length
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
        # (num_envs, T, ...) → (num_envs * T, ...). Fortran order matches ff_icrl.
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )

        # Permute, truncate to a multiple of batch_size, reshape into minibatches.
        n = transitions.observation.shape[0]
        permutation = jax.random.permutation(perm_key, n)
        transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
        n_keep = (n // batch_size) * batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:n_keep], transitions)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
            transitions,
        )

        #  Per-minibatch SAC update 
        def _update_minibatch(carry, batch):
            params, opt_states, k = carry
            k, a_key = jax.random.split(k)

            obs = batch.observation
            achieved = batch.achieved_goal
            ultimate = batch.ultimate_goal
            action = batch.action
            avail = batch.avail_actions

            def critic_loss_fn(critic_params):
                sa_p = critic_params["sa_encoder"]
                g_p = critic_params["goal_encoder"]
                sa_repr = sa_encoder.apply(sa_p, obs, achieved, action)
                g_repr = g_encoder.apply(g_p, ultimate)
                logits = compute_logits(energy_fn_name, sa_repr, g_repr, eps=energy_eps)
                loss = contrastive_loss_fn(contrastive_loss_name, logits, logsumexp_penalty_coeff)
                m = compute_contrastive_metrics(logits)
                return loss, {
                    "critic_loss": loss,
                    "logits_pos": m["logits_pos"],
                    "logits_neg": m["logits_neg"],
                    "categorical_accuracy": m["categorical_accuracy"],
                    "logsumexp": m["logsumexp"],
                }

            def actor_loss_fn(actor_params, sa_p, g_p, alpha, key):
                means, log_stds = actor.apply(actor_params, obs, achieved, ultimate)
                act_resampled, log_prob = _sample_action_and_log_prob(
                    means, log_stds, avail, key
                )
                sa_repr = sa_encoder.apply(sa_p, obs, achieved, act_resampled)
                g_repr = g_encoder.apply(g_p, ultimate)
                q_value = energy_fn(energy_fn_name, sa_repr, g_repr, eps=energy_eps)
                actor_loss = (alpha * log_prob - q_value).mean()
                entropy = -log_prob.mean()
                return actor_loss, {
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                    "q_value_mean": q_value.mean(),
                }

            # Critic update
            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            (_, c_info), c_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_params)
            c_grads, c_info = lax.pmean((c_grads, c_info), axis_name="device")
            c_updates, new_c_opt = critic_update_fn(c_grads, opt_states.critic)
            new_sa = optax.apply_updates(params.sa_encoder, c_updates["sa_encoder"])
            new_g = optax.apply_updates(params.goal_encoder, c_updates["goal_encoder"])

            alpha = jnp.exp(params.log_alpha)
            (_, a_info), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                params.actor, new_sa, new_g, alpha, a_key
            )
            a_grads, a_info = lax.pmean((a_grads, a_info), axis_name="device")
            a_updates, new_a_opt = actor_update_fn(a_grads, opt_states.actor)
            new_actor = optax.apply_updates(params.actor, a_updates)

            # Alpha (temperature) update
            if autotune:
                def alpha_loss_fn(log_alpha):
                    a = jnp.exp(log_alpha)
                    loss = a * jax.lax.stop_gradient(a_info["entropy"] - target_entropy)
                    return loss, {"alpha_loss": loss, "alpha": a}

                (_, alpha_info), alpha_grads = jax.value_and_grad(alpha_loss_fn, has_aux=True)(
                    params.log_alpha
                )
                alpha_grads, alpha_info = lax.pmean((alpha_grads, alpha_info), axis_name="device")
                alpha_updates, new_alpha_opt = alpha_update_fn(alpha_grads, opt_states.alpha)
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)
                if alpha_min > 0:
                    new_log_alpha = jnp.maximum(new_log_alpha, jnp.log(alpha_min))
            else:
                new_log_alpha = params.log_alpha
                new_alpha_opt = opt_states.alpha
                alpha_info = {"alpha": alpha, "alpha_loss": jnp.zeros(())}

            new_params = Params(
                actor=new_actor,
                sa_encoder=new_sa,
                goal_encoder=new_g,
                log_alpha=new_log_alpha,
            )
            new_opts = OptStates(actor=new_a_opt, critic=new_c_opt, alpha=new_alpha_opt)
            return (new_params, new_opts, k), {**c_info, **a_info, **alpha_info}

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
        """Run env steps until the buffer holds at least `explore_steps` transitions."""
        explore_steps = config.system.get("explore_steps", 0)

        def cond(ls):
            return ls.buffer_state.size < explore_steps

        def body(ls):
            ls, _ = _env_step_only(ls, None)
            return ls

        return jax.lax.while_loop(cond, body, learner_state)

    return learner_fn, prefill_fn


# Eval act-fn 

def make_eval_act_fn(actor, eval_env, config):
    greedy = config.arch.evaluation_greedy

    def act_fn(params, ts: NavixTimestep, key, actor_state):
        achieved = eval_env.get_achieved_goal(ts)
        means, log_stds = actor.apply(params, ts.observation, achieved, ts.ultimate_goal)
        if greedy:
            action = jnp.argmax(means, axis=-1)
        else:
            avail = jnp.ones_like(means)
            cont, _ = _sample_action_and_log_prob(means, log_stds, avail, key)
            action = jnp.argmax(cont, axis=-1)
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
    key, env_key, sa_key, g_key, actor_key, buffer_key = jax.random.split(key, 6)

    num_envs = config.arch.num_envs
    env_keys = jax.random.split(env_key, n_devices * num_envs).reshape(n_devices, num_envs, -1)
    env_states = jax.vmap(env.reset)(env_keys)
    print(f"Goal coordinates from environment: {env_states.goal_pos[0, 0]}")

    # Networks
    sa_encoder = sa_ConvEncoder(rep_size=config.system.rep_size)
    g_encoder = small_G_encoder(rep_size=config.system.rep_size)
    actor = ConvActorGaussian(
        num_actions=action_size,
        mlp_hidden=config.system.get("actor_mlp_hidden", 256),
        log_std_min=config.system.get("log_std_min", -5.0),
        log_std_max=config.system.get("log_std_max", 2.0),
    )

    dummy_obs = jnp.ones([1, *obs_shape])
    dummy_action = jnp.ones([1, action_size])
    dummy_goal = jnp.ones([1, goal_dim])
    dummy_achieved = jnp.ones([1, goal_dim])

    sa_params = sa_encoder.init(sa_key, dummy_obs, dummy_achieved, dummy_action)
    g_params = g_encoder.init(g_key, dummy_goal)
    actor_params = actor.init(actor_key, dummy_obs, dummy_achieved, dummy_goal)
    log_alpha = jnp.log(jnp.array(config.system.get("init_alpha", 0.1), dtype=jnp.float32))

    params = Params(actor=actor_params, sa_encoder=sa_params, goal_encoder=g_params, log_alpha=log_alpha)

    # Optimizers
    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    actor_opt = optax.chain(grad_clip, optax.adam(config.system.policy_lr))
    critic_opt = optax.chain(grad_clip, optax.adam(config.system.q_lr))
    alpha_opt = optax.chain(grad_clip, optax.adam(config.system.alpha_lr))

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init({"sa_encoder": sa_params, "goal_encoder": g_params})
    alpha_opt_state = alpha_opt.init(log_alpha)
    opt_states = OptStates(actor=actor_opt_state, critic=critic_opt_state, alpha=alpha_opt_state)

    # Replay buffer (per device)
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

    learn, prefill = get_learner_fn(
        env, buffer, actor, sa_encoder, g_encoder,
        actor_opt.update, critic_opt.update, alpha_opt.update, config,
    )
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

    eval_act_fn = make_eval_act_fn(actor, eval_env, config)
    evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)
    abs_metric_evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)

    return learn, prefill, learner_state, evaluator, abs_metric_evaluator, actor


# Run


def run_experiment(_config: DictConfig) -> float:
    _config.logger.system_name = "ff_icrl_nx"
    config = copy.deepcopy(_config)
    n_devices = len(jax.devices())

    learn, prefill, learner_state, evaluator, abs_metric_evaluator, actor = learner_setup(config)

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
        eval_metrics = evaluator(learner_output.learner_state.params.actor, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        win_rate = float(jnp.mean(eval_metrics["win_rate"]))
        if win_rate > max_win_rate:
            max_win_rate = win_rate
            best_params = jax.tree_util.tree_map(
                lambda x: np.array(x[0]), learner_output.learner_state.params.actor
            )
            with open(f"{save_dir}/best_policy_ff_icrl_nx.pkl", "wb") as f:
                pickle.dump({"actor": best_params}, f)
            print(f"New best policy saved - Win rate: {win_rate:.2f}%")

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
        best_params_rep = flax.jax_utils.replicate(best_params)
        eval_metrics = abs_metric_evaluator(best_params_rep, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return max_win_rate


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_icrl_nx.yaml",
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
