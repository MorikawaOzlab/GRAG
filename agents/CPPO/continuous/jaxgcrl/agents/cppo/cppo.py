
import functools
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass as flax_dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import TerminateOnSuccessWrapper, TrajectoryIdWrapper
from jaxgcrl.utils.evaluator import ActorEvaluator

from .losses import (
    compute_contrastive_metrics,
    compute_logits,
    contrastive_loss_fn as contrastive_loss_fn_,
    energy_fn as energy_fn_,
)
from .networks import Actor, GoalEncoder, SAEncoder

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]


@flax_dataclass
class TrainingState:
    env_steps: jnp.ndarray
    update_count: jnp.ndarray
    actor_state: TrainState
    sa_encoder_state: TrainState
    g_encoder_state: TrainState


class Transition(NamedTuple):
    observation: jnp.ndarray  # full obs [state | target_goal] from env
    action: jnp.ndarray  # post-tanh action in [-1, 1]
    x_t: jnp.ndarray  # pre-tanh value (for stable log_prob recomputation)
    log_prob: jnp.ndarray  # Gaussian log_prob in pre-tanh space (sum over action dims)
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: Dict[str, Any]


@dataclass
class CPPO:
    """PPO actor + CRL contrastive critic."""

    # Optimization
    actor_lr: float = 3e-4
    q_lr: float = 3e-4
    max_grad_norm: float = 1.0
    lr_linear_decay: bool = False
    lr_end: float = 1e-7

    # Rollout / batching
    unroll_length: int = 128
    num_epochs: int = 8
    batch_size: int = 256

    # PPO
    discounting: float = 0.9999
    clip_eps: float = 0.2
    ent_coef: float = 1e-4

    # MC V(s,g) estimate
    num_mc_samples: int = 32

    # CRL
    logsumexp_penalty_coeff: float = 0.1
    repr_dim: int = 64
    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # Network
    h_dim: int = 512
    n_hidden: int = 4
    use_layer_norm: bool = True
    use_relu: bool = False
    skip_connections: int = 4

    log_std_min: float = -5.0
    log_std_max: float = 2.0

    # Goal handling
    terminate_on_success: bool = True

    def check_config(self, config):
        assert config.num_envs * self.unroll_length % self.batch_size == 0, (
            "num_envs * unroll_length must be divisible by batch_size"
        )

    def train_fn(
        self,
        config,
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[Callable] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        # Environment wrappers
        unwrapped_env = train_env
        if self.terminate_on_success:
            train_env = TerminateOnSuccessWrapper(train_env)
            eval_env = TerminateOnSuccessWrapper(eval_env)
        train_env_w = TrajectoryIdWrapper(train_env)
        train_env_w = envs.training.wrap(
            train_env_w,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        eval_env_w = TrajectoryIdWrapper(eval_env)
        eval_env_w = envs.training.wrap(
            eval_env_w,
            episode_length=config.episode_length,
            action_repeat=config.action_repeat,
        )

        action_size = int(unwrapped_env.action_size)
        state_size = int(unwrapped_env.state_dim)
        goal_indices = tuple(int(i) for i in unwrapped_env.goal_indices)
        goal_size = len(goal_indices)
        obs_size = state_size + goal_size
        assert obs_size == int(unwrapped_env.observation_size), (
            f"obs_size {obs_size} != env.observation_size {unwrapped_env.observation_size}"
        )

        # SA input size:
        # [state, achieved_goal]
        sa_state_size = state_size + goal_size

        # Actor input size.
        # [state, achieved_goal, goal]
        actor_input_size = state_size + goal_size + goal_size

        # Schedules
        env_steps_per_update = config.num_envs * self.unroll_length
        num_updates_total = int(np.ceil(config.total_env_steps / env_steps_per_update))
        num_evals_after_init = min(max(config.num_evals - 1, 1), num_updates_total)
        num_updates_per_epoch = max(1, int(np.ceil(num_updates_total / num_evals_after_init)))

        if self.lr_linear_decay:
            samples_per_update = config.num_envs * self.unroll_length
            mb_per_update = samples_per_update // self.batch_size
            grad_steps = num_updates_total * self.num_epochs * mb_per_update
            actor_lr_sched = optax.linear_schedule(self.actor_lr, self.lr_end, grad_steps)
            critic_lr_sched = optax.linear_schedule(self.q_lr, self.lr_end, grad_steps)
        else:
            actor_lr_sched = self.actor_lr
            critic_lr_sched = self.q_lr

        logging.info(
            "CPPO: total_updates=%d  per_eval=%d  env_steps_per_update=%d",
            num_updates_total,
            num_updates_per_epoch,
            env_steps_per_update,
        )

        # Networks
        hidden_sizes = tuple([self.h_dim] * self.n_hidden)
        actor = Actor(
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
            use_layer_norm=self.use_layer_norm,
            use_relu=self.use_relu,
            skip_connections=self.skip_connections,
        )
        sa_encoder = SAEncoder(
            hidden_sizes=hidden_sizes,
            output_dim=self.repr_dim,
            use_layer_norm=self.use_layer_norm,
            use_relu=self.use_relu,
            skip_connections=self.skip_connections,
        )
        g_encoder = GoalEncoder(
            hidden_sizes=hidden_sizes,
            output_dim=self.repr_dim,
            use_layer_norm=self.use_layer_norm,
            use_relu=self.use_relu,
            skip_connections=self.skip_connections,
        )

        key = jax.random.PRNGKey(config.seed)
        key, env_key, eval_key, actor_key, sa_key, g_key, train_key = jax.random.split(key, 7)

        actor_params = actor.init(actor_key, jnp.zeros((1, actor_input_size)))
        sa_params = sa_encoder.init(sa_key, jnp.zeros((1, sa_state_size)), jnp.zeros((1, action_size)))
        g_params = g_encoder.init(g_key, jnp.zeros((1, goal_size)))

        actor_opt = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(actor_lr_sched))
        sa_opt = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(critic_lr_sched))
        g_opt = optax.chain(optax.clip_by_global_norm(self.max_grad_norm), optax.adam(critic_lr_sched))

        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=actor_opt)
        sa_state = TrainState.create(apply_fn=sa_encoder.apply, params=sa_params, tx=sa_opt)
        g_state = TrainState.create(apply_fn=g_encoder.apply, params=g_params, tx=g_opt)

        training_state = TrainingState(
            env_steps=jnp.zeros((), dtype=jnp.int64),
            update_count=jnp.zeros((), dtype=jnp.int32),
            actor_state=actor_state,
            sa_encoder_state=sa_state,
            g_encoder_state=g_state,
        )

        # Helpers
        loss_name = self.contrastive_loss_fn
        ene_name = self.energy_fn
        lse_pen = self.logsumexp_penalty_coeff
        clip_eps = self.clip_eps
        gamma = self.discounting
        K = self.num_mc_samples
        ent_coef = self.ent_coef

        def split_obs(obs):
            target_goal = obs[..., -goal_size:]
            achieved_goal = obs[..., jnp.array(goal_indices)]
            return obs, target_goal, achieved_goal

        def build_actor_input(obs_full, achieved_goal, target_goal):
            state = obs_full[..., :state_size]
            return jnp.concatenate([state, achieved_goal, target_goal], axis=-1)

        def build_sa_state(obs_full, achieved_goal):
            base = obs_full[..., :state_size]
            return jnp.concatenate([base, achieved_goal], axis=-1)

        def actor_sample(actor_params, obs, key):
            """Sample action from policy. Returns (action_tanh, x_t, log_prob)."""
            state, target, ach = split_obs(obs)
            inp = build_actor_input(state, ach, target)
            means, log_stds = actor.apply(actor_params, inp)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(key, shape=means.shape, dtype=means.dtype)
            x_t = means + stds * noise
            action = nn.tanh(x_t)
            log_prob = (-0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
            return action, x_t, log_prob

        # Rollout
        def actor_step_for_rollout(carry, _):
            env_state, key, actor_params = carry
            key, sample_key = jax.random.split(key)
            action, x_t, log_prob = actor_sample(actor_params, env_state.obs, sample_key)
            n_state = train_env_w.step(env_state, action)
            extras = {
                "state_extras": {
                    "truncation": n_state.info["truncation"],
                    "traj_id": n_state.info["traj_id"],
                }
            }
            transition = Transition(
                observation=env_state.obs,
                action=action,
                x_t=x_t,
                log_prob=log_prob,
                reward=n_state.reward,
                discount=1.0 - n_state.done,
                extras=extras,
            )
            return (n_state, key, actor_params), transition

        def collect_rollout(env_state, key, actor_params):
            (env_state, key, _), traj = jax.lax.scan(
                actor_step_for_rollout,
                (env_state, key, actor_params),
                None,
                length=self.unroll_length,
            )
            return env_state, key, traj

        # Hindsight relabeling
        # Operates per-env trajectory of length T = unroll_length.
        def her_relabel(transition, sample_key):
            T = transition.observation.shape[0]
            arr = jnp.arange(T)
            future_mask = jnp.array(arr[:, None] < arr[None], dtype=jnp.float32)
            disc = gamma ** jnp.array(arr[None] - arr[:, None], dtype=jnp.float32)
            probs = future_mask * disc
            traj_id = transition.extras["state_extras"]["traj_id"]
            same_traj = jnp.equal(traj_id[:, None], traj_id[None, :]).astype(jnp.float32)
            probs = probs * same_traj + jnp.eye(T) * 1e-5
            goal_idx = jax.random.categorical(sample_key, jnp.log(probs))
            future_obs = jnp.take(transition.observation, goal_idx[:-1], axis=0)
            relabeled_goal = future_obs[:, jnp.array(goal_indices)]  # (T-1, goal_size)

            obs = transition.observation[:-1]  # (T-1, obs_size)
            achieved = obs[:, jnp.array(goal_indices)]

            return {
                "obs": obs,
                "achieved_goal": achieved,
                "target_goal": obs[:, -goal_size:],  # original env target — used by actor/advantage
                "relabeled_goal": relabeled_goal,  # HER future-achieved — used by critic only
                "action": transition.action[:-1],
                "x_t": transition.x_t[:-1],
                "old_log_prob": transition.log_prob[:-1],
                "reward": transition.reward[:-1],
                "discount": transition.discount[:-1],
            }

        # Critic loss
        def critic_loss_fn(sa_params, g_params, sa_state_arr, action, goal):
            sa_repr = sa_encoder.apply(sa_params, sa_state_arr, action)
            g_repr = g_encoder.apply(g_params, goal)
            logits = compute_logits(ene_name, sa_repr, g_repr)
            loss = contrastive_loss_fn_(loss_name, logits, lse_pen)
            metrics = compute_contrastive_metrics(logits)
            return loss, {"critic_loss": loss, **metrics}

        # Actor loss
        def actor_loss_fn(actor_params, batch, advantage, ent_coef):
            obs_full = batch["obs"]
            ach = batch["achieved_goal"]
            # Actor uses the REAL env goal — matches the rollout actor input,
            goal = batch["target_goal"]
            x_t = batch["x_t"]
            old_log_prob = batch["old_log_prob"]

            inp = build_actor_input(obs_full, ach, goal)
            means, log_stds = actor.apply(actor_params, inp)
            stds = jnp.exp(log_stds)
            noise = (x_t - means) / (stds + 1e-8)
            log_prob = (-0.5 * noise**2 - log_stds - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)

            ratio = jnp.exp(log_prob - old_log_prob)
            ratio_clipped = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
            ppo_loss = jnp.mean(jnp.maximum(-advantage * ratio, -advantage * ratio_clipped))

            entropy = (log_stds + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum(-1).mean()
            actor_loss = ppo_loss - ent_coef * entropy

            return actor_loss, {
                "actor_loss": actor_loss,
                "ppo_loss": ppo_loss,
                "entropy": entropy,
                "ratio_mean": ratio.mean(),
                "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio + 1e-8)),
            }

        # Advantage (no-grad)
        def compute_advantage(actor_params, sa_params, g_params, batch, key):
            obs_full = batch["obs"]
            ach = batch["achieved_goal"]
            # Advantage uses REAL env goal — Q(s,a,g_real) vs V(s,g_real) under
            # actor(·|s, g_real). Matches rollout/eval conditioning. 
            goal = batch["target_goal"]
            action = batch["action"]
            sa_in = build_sa_state(obs_full, ach)

            actor_in = build_actor_input(obs_full, ach, goal)
            means, log_stds = actor.apply(actor_params, actor_in)
            stds = jnp.exp(log_stds)

            sa_repr_taken = sa_encoder.apply(sa_params, sa_in, action)
            g_repr = g_encoder.apply(g_params, goal)
            q_taken = energy_fn_(ene_name, sa_repr_taken, g_repr)

            mc_noise = jax.random.normal(key, shape=(K,) + means.shape, dtype=means.dtype)
            mc_xt = means[None] + stds[None] * mc_noise
            mc_actions = nn.tanh(mc_xt)

            def q_for_sample(a_):
                sa_r = sa_encoder.apply(sa_params, sa_in, a_)
                return energy_fn_(ene_name, sa_r, g_repr)

            mc_q = jax.vmap(q_for_sample)(mc_actions)  # (K, B)
            v_mc = mc_q.mean(axis=0)
            adv = q_taken - v_mc
            return adv, q_taken, v_mc

        # Single update step
        @jax.jit
        def update_step(training_state, env_state, key):
            key, rollout_key, her_key, adv_key, train_key, perm_key = jax.random.split(key, 6)

            actor_p = training_state.actor_state.params
            sa_p = training_state.sa_encoder_state.params
            g_p = training_state.g_encoder_state.params

            # Rollout
            env_state, _, traj = collect_rollout(env_state, rollout_key, actor_p)

            # Trajectory shape: (T, num_envs, ...). Swap to (num_envs, T, ...) for per-env HER.
            traj_per_env = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj)

            her_keys = jax.random.split(her_key, traj_per_env.observation.shape[0])
            relabeled = jax.vmap(her_relabel)(traj_per_env, her_keys)
            # shape: (num_envs, T-1, ...)

            # Flatten env+time into batch dimension (Fortran order to interleave envs).
            relabeled = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), relabeled
            )

            B_total = relabeled["obs"].shape[0]
            B_use = (B_total // self.batch_size) * self.batch_size
            relabeled = jax.tree_util.tree_map(lambda x: x[:B_use], relabeled)

            # Compute advantages over full data in batch_size chunks.
            adv_batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]), relabeled
            )

            def adv_scan(carry_key, batch):
                k, sub_k = jax.random.split(carry_key)
                adv, q, v = compute_advantage(actor_p, sa_p, g_p, batch, sub_k)
                return k, {"advantage": adv, "q_taken": q, "v_mc": v}

            _, adv_results = jax.lax.scan(adv_scan, adv_key, adv_batched)
            advantages = adv_results["advantage"].reshape(-1)
            q_taken_all = adv_results["q_taken"].reshape(-1)
            v_mc_all = adv_results["v_mc"].reshape(-1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            relabeled["advantage"] = advantages
            # Train epochs
            def minibatch_step(carry, batch):
                a_state, sa_state_, g_state_ = carry

                # Critic
                def c_loss_fn(c_params, batch):
                    sa_in = build_sa_state(batch["obs"], batch["achieved_goal"])
                    return critic_loss_fn(c_params["sa"], c_params["g"], sa_in, batch["action"], batch["relabeled_goal"])

                c_params = {"sa": sa_state_.params, "g": g_state_.params}
                (_, c_info), c_grads = jax.value_and_grad(c_loss_fn, has_aux=True)(c_params, batch)
                sa_state_ = sa_state_.apply_gradients(grads=c_grads["sa"])
                g_state_ = g_state_.apply_gradients(grads=c_grads["g"])

                # Actor
                (_, a_info), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(
                    a_state.params, batch, batch["advantage"], ent_coef
                )
                a_state = a_state.apply_gradients(grads=a_grads)

                metrics = {**c_info, **a_info, "ent_coef": ent_coef}
                return (a_state, sa_state_, g_state_), metrics

            def train_epoch(carry, _):
                a_state, sa_state_, g_state_, key, flat_data = carry
                key, perm_k = jax.random.split(key)
                perm = jax.random.permutation(perm_k, flat_data["obs"].shape[0])
                shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat_data)
                batched = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (-1, self.batch_size) + x.shape[1:]), shuffled
                )
                (a_state, sa_state_, g_state_), m = jax.lax.scan(
                    minibatch_step, (a_state, sa_state_, g_state_), batched
                )
                return (a_state, sa_state_, g_state_, key, flat_data), m

            (a_state, sa_state_, g_state_, _, _), epoch_metrics = jax.lax.scan(
                train_epoch,
                (training_state.actor_state, training_state.sa_encoder_state, training_state.g_encoder_state,
                 train_key, relabeled),
                None,
                self.num_epochs,
            )

            new_state = TrainingState(
                env_steps=training_state.env_steps + env_steps_per_update,
                update_count=training_state.update_count + 1,
                actor_state=a_state,
                sa_encoder_state=sa_state_,
                g_encoder_state=g_state_,
            )

            metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
            metrics["advantage_mean"] = advantages.mean()
            metrics["advantage_std"] = advantages.std()
            metrics["q_taken_mean"] = q_taken_all.mean()
            metrics["v_mc_mean"] = v_mc_all.mean()
            metrics["rollout_reward_mean"] = traj.reward.mean()
            return new_state, env_state, metrics

        # Eval
        def eval_actor_step(ts, env, env_state, extra_fields):
            means, log_stds = actor.apply(ts.actor_state.params, _build_actor_input_full(env_state.obs))
            stds = jnp.exp(log_stds)
            # Stochastic eval matches paper.
            # Use a key derived from env_state to keep determinism per call.
            noise = jax.random.normal(jax.random.PRNGKey(0), shape=means.shape, dtype=means.dtype)
            x_t = means + stds * noise
            action = nn.tanh(x_t)
            n_state = env.step(env_state, action)
            return n_state, Transition(
                observation=env_state.obs,
                action=action,
                x_t=x_t,
                log_prob=jnp.zeros(action.shape[:-1]),
                reward=n_state.reward,
                discount=1.0 - n_state.done,
                extras={"state_extras": {x: n_state.info[x] for x in extra_fields}},
            )

        def _build_actor_input_full(obs):
            state, target, ach = split_obs(obs)
            return build_actor_input(state, ach, target)

        evaluator = ActorEvaluator(
            actor_step=eval_actor_step,
            eval_env=eval_env_w,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_key,
        )

        # Main training loop
        env_state = jax.jit(train_env_w.reset)(jax.random.split(env_key, config.num_envs))

        def make_policy(params):
            def policy(obs, rng):
                state, target, ach = split_obs(obs)
                inp = build_actor_input(state, ach, target)
                means, log_stds = actor.apply(params, inp)
                stds = jnp.exp(log_stds)
                noise = jax.random.normal(rng, shape=means.shape, dtype=means.dtype)
                return nn.tanh(means + stds * noise)
            return policy

        # Initial eval
        eval_metrics = evaluator.run_evaluation(training_state, {})
        progress_fn(
            int(training_state.env_steps),
            eval_metrics,
            make_policy,
            training_state.actor_state.params,
            unwrapped_env,
            do_render=False,
        )

        loop_key = train_key
        for eval_idx in range(num_evals_after_init):
            t0 = time.time()
            for _ in range(num_updates_per_epoch):
                loop_key, sub_key = jax.random.split(loop_key)
                training_state, env_state, train_metrics = update_step(training_state, env_state, sub_key)
            jax.block_until_ready(training_state.env_steps)
            sps = (num_updates_per_epoch * env_steps_per_update) / (time.time() - t0)

            train_metrics = jax.tree_util.tree_map(lambda x: float(np.asarray(x)), train_metrics)
            train_metrics = {f"training/{k}": v for k, v in train_metrics.items()}
            train_metrics["training/sps"] = sps

            eval_metrics = evaluator.run_evaluation(training_state, train_metrics)
            progress_fn(
            int(training_state.env_steps),
            eval_metrics,
            make_policy,
            training_state.actor_state.params,
            unwrapped_env,
            do_render=False,
        )

        params = (
            training_state.actor_state.params,
            training_state.sa_encoder_state.params,
            training_state.g_encoder_state.params,
        )
        logging.info("CPPO total steps: %d", int(training_state.env_steps))

        # Fresh post-training evaluation with more envs for absolute metric.
        abs_metrics = None
        absolute_num_eval_envs = getattr(config, "absolute_num_eval_envs", 0)
        if absolute_num_eval_envs > 0:
            logging.info("Running absolute eval with %d envs...", absolute_num_eval_envs)
            abs_evaluator = ActorEvaluator(
                actor_step=eval_actor_step,
                eval_env=eval_env_w,
                num_eval_envs=absolute_num_eval_envs,
                episode_length=config.episode_length,
                key=jax.random.PRNGKey(int(config.seed) + 9999),
            )
            abs_metrics = abs_evaluator.run_evaluation(training_state, {})

        return make_policy, params, eval_metrics, abs_metrics
