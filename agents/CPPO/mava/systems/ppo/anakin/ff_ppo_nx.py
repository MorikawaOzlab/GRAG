
import copy
import os
import time
from typing import Any

import chex
import distrax
import flax
import flax.linen as nn
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import orthogonal, variance_scaling
from omegaconf import DictConfig, OmegaConf
from typing_extensions import NamedTuple

from mava.evaluator import get_nx_eval_fn
from mava.types import ExperimentOutput
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.navix_wrapper import (
    NavixTimestep,
    make_navix_env,
)



class Params(NamedTuple):
    actor: Any
    critic: Any


class OptStates(NamedTuple):
    actor: optax.OptState
    critic: optax.OptState


class LearnerState(NamedTuple):
    params: Params
    opt_states: OptStates
    key: chex.PRNGKey
    env_state: NavixTimestep
    t: chex.Array


class Transition(NamedTuple):
    obs: chex.Array       # (N, H, W, 3)
    action: chex.Array    # (N,)
    value: chex.Array     # (N,)
    reward: chex.Array    # (N,)
    done: chex.Array      # (N,)  1.0 if terminal else 0.0
    log_prob: chex.Array  # (N,)
    t: chex.Array         # (N,)  episode timestep for gamma**t discount


class ConvActor(nn.Module):
    num_actions: int
    mlp_hidden: int = 256

    @nn.compact
    def __call__(self, obs: chex.Array) -> distrax.Categorical:
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        lead = obs.shape[:-3]
        h, w, c = obs.shape[-3:]
        x = obs.reshape(-1, h, w, c).astype(jnp.float32)

        x = nn.Conv(16, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(32, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(x)
        return distrax.Categorical(logits=logits.reshape(*lead, self.num_actions))


class ConvCritic(nn.Module):
    mlp_hidden: int = 256

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        lead = obs.shape[:-3]
        h, w, c = obs.shape[-3:]
        x = obs.reshape(-1, h, w, c).astype(jnp.float32)

        x = nn.Conv(16, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(32, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = nn.Conv(64, (2, 2), kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = x.reshape(x.shape[0], -1)

        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.relu(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0))(x)
        return value.reshape(*lead)



def _gae(traj: Transition, last_val: chex.Array, gamma: float, lam: float):
    def _step(carry, tr):
        gae, next_val = carry
        discount = (gamma ** tr.t) * (1.0 - tr.done)
        delta = tr.reward + discount * next_val - tr.value
        gae = delta + lam * discount * gae
        return (gae, tr.value), gae

    _, advs = jax.lax.scan(
        _step,
        (jnp.zeros_like(last_val), last_val),
        traj,
        reverse=True,
        unroll=16,
    )
    return advs, advs + traj.value


# Learner


def get_learner_fn(env, actor, critic, actor_update_fn, critic_update_fn, config):
    rollout_length = config.system.rollout_length
    num_minibatches = config.system.num_minibatches
    clip_eps = config.system.clip_eps
    ent_coef = config.system.ent_coef
    vf_coef = config.system.vf_coef
    gamma = config.system.gamma
    gae_lambda = config.system.gae_lambda
    ppo_epochs = config.system.ppo_epochs

    def _actor_step(params, ts: NavixTimestep, key):
        obs = ts.observation
        pi = actor.apply(params.actor, obs)
        key, sk = jax.random.split(key)
        action = pi.sample(seed=sk)
        log_prob = pi.log_prob(action)
        value = critic.apply(params.critic, obs)
        new_ts = env.step(ts, action)
        done = new_ts.is_done().astype(jnp.float32)
        transition = Transition(
            obs=obs,
            action=action,
            value=value,
            reward=new_ts.reward,
            done=done,
            log_prob=log_prob,
            t=ts.t,
        )
        return new_ts, transition, key

    def _collect(params, ts, key):
        def f(carry, _):
            ts, k = carry
            new_ts, tr, k = _actor_step(params, ts, k)
            return (new_ts, k), tr

        (final_ts, key), traj = jax.lax.scan(f, (ts, key), None, length=rollout_length)
        return final_ts, traj, key

    def _update_step(learner_state: LearnerState, _: Any):
        key = learner_state.key
        rollout_key, shuffle_key, new_key = jax.random.split(key, 3)

        env_state, traj, _ = _collect(learner_state.params, learner_state.env_state, rollout_key)

        last_val = critic.apply(learner_state.params.critic, env_state.observation)
        advantages, targets = _gae(traj, last_val, gamma, gae_lambda)

        batch_size = rollout_length * config.arch.num_envs
        flat = jax.tree_util.tree_map(lambda x: x.reshape(batch_size, *x.shape[2:]), traj)
        flat_adv = advantages.reshape(batch_size)
        flat_tgt = targets.reshape(batch_size)

        def _epoch(carry, _):
            params, opt_states, k = carry
            k, perm_key = jax.random.split(k)
            perm = jax.random.permutation(perm_key, batch_size)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat)
            adv = flat_adv[perm]
            tgt = flat_tgt[perm]

            mb_size = batch_size // num_minibatches
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape(num_minibatches, mb_size, *x.shape[1:]), shuffled
            )
            mb_adv = adv.reshape(num_minibatches, mb_size)
            mb_tgt = tgt.reshape(num_minibatches, mb_size)

            def _mb(carry, batch):
                params, opt_states, k = carry
                mb, mb_adv, mb_tgt = batch

                def _actor_loss(actor_params):
                    pi = actor.apply(actor_params, mb.obs)
                    new_lp = pi.log_prob(mb.action)
                    ratio = jnp.exp(new_lp - mb.log_prob)
                    norm_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                    l1 = ratio * norm_adv
                    l2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * norm_adv
                    actor_loss = -jnp.mean(jnp.minimum(l1, l2))
                    entropy = pi.entropy().mean()
                    return actor_loss - ent_coef * entropy, (actor_loss, entropy)

                def _critic_loss(critic_params):
                    v = critic.apply(critic_params, mb.obs)
                    v_clipped = mb.value + jnp.clip(v - mb.value, -clip_eps, clip_eps)
                    vf_loss = jnp.square(v - mb_tgt)
                    vf_loss_clipped = jnp.square(v_clipped - mb_tgt)
                    value_loss = 0.5 * jnp.maximum(vf_loss, vf_loss_clipped).mean()
                    return vf_coef * value_loss, value_loss

                (al, (actor_loss, entropy)), a_grads = jax.value_and_grad(
                    _actor_loss, has_aux=True
                )(params.actor)
                a_grads, al = lax.pmean((a_grads, al), axis_name="device")
                a_upd, new_a_opt = actor_update_fn(a_grads, opt_states.actor)
                new_actor = optax.apply_updates(params.actor, a_upd)

                (cl, value_loss), c_grads = jax.value_and_grad(
                    _critic_loss, has_aux=True
                )(params.critic)
                c_grads, cl = lax.pmean((c_grads, cl), axis_name="device")
                c_upd, new_c_opt = critic_update_fn(c_grads, opt_states.critic)
                new_critic = optax.apply_updates(params.critic, c_upd)

                loss_info = {
                    "total_loss": al + cl,
                    "actor_loss": actor_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                }
                new_params = Params(actor=new_actor, critic=new_critic)
                new_opts = OptStates(actor=new_a_opt, critic=new_c_opt)
                return (new_params, new_opts, k), loss_info

            (params, opt_states, k), info = jax.lax.scan(
                _mb, (params, opt_states, k), (minibatches, mb_adv, mb_tgt)
            )
            return (params, opt_states, k), info

        (params, opt_states, _), train_info = jax.lax.scan(
            _epoch,
            (learner_state.params, learner_state.opt_states, shuffle_key),
            None,
            length=ppo_epochs,
        )

        act_metrics = {
            "rollout_reward_mean": traj.reward.mean(),
            "rollout_win_rate": (traj.reward > 0).mean() * 100.0,
            "rollout_done_rate": traj.done.mean(),
        }

        new_learner_state = LearnerState(
            params=params,
            opt_states=opt_states,
            key=new_key,
            env_state=env_state,
            t=learner_state.t + 1,
        )
        return new_learner_state, (act_metrics, train_info)

    def learner_fn(learner_state: LearnerState):
        learner_state, (act_metrics, train_metrics) = jax.lax.scan(
            _update_step, learner_state, None,
            length=config.system.num_updates_per_eval,
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=act_metrics,
            train_metrics=train_metrics,
        )

    return learner_fn


# Eval


def make_eval_act_fn(actor, config):
    greedy = config.arch.get("evaluation_greedy", True)

    def act_fn(params, ts: NavixTimestep, key, actor_state):
        pi = actor.apply(params, ts.observation)
        action = jnp.argmax(pi.logits, axis=-1) if greedy else pi.sample(seed=key)
        return action, {}

    return act_fn


# Setup


def learner_setup(config: DictConfig):
    n_devices = len(jax.devices())

    obs_type = config.env.get("obs_type", "full")
    fov_radius = config.env.get("fov_radius", 3)
    max_steps = config.env.get("time_limit", 100)

    env = make_navix_env(
        env_name=config.env.env_name,
        terminate_on_goal=config.env.get("terminate_on_goal", False),
        max_steps=max_steps,
        obs_type=obs_type,
        fov_radius=fov_radius,
    )
    eval_env = make_navix_env(
        env_name=config.env.env_name,
        terminate_on_goal=config.env.get("terminate_on_goal_eval", True),
        max_steps=max_steps,
        obs_type=obs_type,
        fov_radius=fov_radius,
    )

    action_size = int(env.action_size)
    obs_shape = env.observation_shape
    config.system.action_size = action_size

    key = jax.random.PRNGKey(config.system.seed)
    key, env_key, actor_key, critic_key = jax.random.split(key, 4)

    num_envs = config.arch.num_envs
    env_keys = jax.random.split(env_key, n_devices * num_envs).reshape(n_devices, num_envs, -1)
    env_states = jax.vmap(env.reset)(env_keys)
    print(f"Goal coordinates from environment: {env_states.goal_pos[0, 0]}")

    mlp_hidden = config.system.get("mlp_hidden", 256)
    actor = ConvActor(num_actions=action_size, mlp_hidden=mlp_hidden)
    critic = ConvCritic(mlp_hidden=mlp_hidden)

    dummy_obs = jnp.ones([1, *obs_shape])
    actor_params = actor.init(actor_key, dummy_obs)
    critic_params = critic.init(critic_key, dummy_obs)
    params = Params(actor=actor_params, critic=critic_params)

    def _lr_schedule(base_lr):
        decay = config.system.get("lr_decay_type", "linear")
        end = config.system.get("lr_end", 0.0)
        total_ts = config.system.get("total_timesteps", None)
        if total_ts is not None:
            n_updates = total_ts // (n_devices * num_envs * config.system.rollout_length)
        else:
            n_updates = config.system.num_updates
        steps = max(1, n_updates * config.system.ppo_epochs * config.system.num_minibatches)
        if decay == "linear":
            return optax.linear_schedule(base_lr, end, steps)
        if decay == "cosine":
            return optax.cosine_decay_schedule(base_lr, steps, alpha=end / base_lr)
        return base_lr

    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    actor_opt = optax.chain(grad_clip, optax.adam(_lr_schedule(config.system.actor_lr), eps=1e-5))
    critic_opt = optax.chain(grad_clip, optax.adam(_lr_schedule(config.system.critic_lr), eps=1e-5))

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init(critic_params)
    opt_states = OptStates(actor=actor_opt_state, critic=critic_opt_state)

    learn = get_learner_fn(env, actor, critic, actor_opt.update, critic_opt.update, config)
    learn = jax.pmap(learn, axis_name="device")

    params = flax.jax_utils.replicate(params)
    opt_states = flax.jax_utils.replicate(opt_states)
    step_keys = jax.random.split(jax.random.PRNGKey(config.system.seed), n_devices)
    t0 = jnp.zeros((n_devices,), dtype=jnp.int32)

    learner_state = LearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        t=t0,
    )

    eval_act_fn = make_eval_act_fn(actor, config)
    evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)
    abs_metric_evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)

    return learn, learner_state, evaluator, abs_metric_evaluator, actor


# Run


def run_experiment(_config: DictConfig) -> float:
    _config.logger.system_name = "ff_ppo_nx"
    config = copy.deepcopy(_config)
    n_devices = len(jax.devices())

    learn, learner_state, evaluator, abs_metric_evaluator, actor = learner_setup(config)
    jax.block_until_ready(learner_state)

    config = check_total_timesteps(config)
    assert config.system.num_updates > config.arch.num_evaluation, \
        "num_updates must exceed num_evaluation"
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
            with open(f"{save_dir}/best_policy_ff_ppo_nx.pkl", "wb") as f:
                import pickle
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

    # Measure absolute metric using the best params found during training.
    if config.arch.absolute_metric:
        eval_key, *keys = jax.random.split(eval_key, n_devices + 1)
        eval_keys = jnp.stack(keys)
        best_params_rep = flax.jax_utils.replicate(best_params)
        eval_metrics = abs_metric_evaluator(best_params_rep, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    logger.stop()
    return max_win_rate


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_ppo_nx.yaml",
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
