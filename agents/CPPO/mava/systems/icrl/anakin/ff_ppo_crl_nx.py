
import copy
import os
import pickle
import time
from typing import Any, Dict, Tuple

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

from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.navix_wrapper import make_navix_env, NavixTimestep


def relabel_fn(buffer_config, transition, sample_key):
    """HER-style goal relabeling for Navix environments with grid observations."""
    gamma = buffer_config

    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # Ensure goals from same episode
    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["seed"][:, jnp.newaxis].T] * seq_len, axis=0
    )

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5

    # Sample future goal indices and get goals from state_extras
    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    # Goals are stored in state_extras as (seq_len, 2) coordinates
    future_goals = jnp.take(transition.extras["state_extras"]["goal"], goal_index[:-1], axis=0)

    # Observations are grid format (seq_len, H, W, 3) - no squeeze needed
    obs = transition.observation[:-1]

    extras = {
        "state_extras": {
            "truncation": transition.extras["state_extras"]["truncation"][:-1],
            "seed": transition.extras["state_extras"]["seed"][:-1],
        },
        "goal": future_goals,
    }

    return transition._replace(
        observation=obs,  # Grid observations (seq_len-1, H, W, 3)
        action=transition.action[:-1],
        reward=transition.reward[:-1],
        extras=extras,
    )


# Types


class Params(NamedTuple):
    actor: Any
    sa_encoder: Any
    goal_encoder: Any


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
    observation: chex.Array    # (N, H, W, 3)
    achieved_goal: chex.Array  # (N, 4)
    ultimate_goal: chex.Array  # (N, 4)
    action: chex.Array         # (N,) int32
    reward: chex.Array         # (N,)
    discount: chex.Array       # (N,)
    log_prob: chex.Array       # (N,)
    avail_actions: chex.Array  # (N, A)
    extras: Dict


# Goal-conditioned CNN actor


class ConvActor(nn.Module):
    """Goal-conditioned feed-forward actor for grid observations.

    """

    num_actions: int
    mlp_hidden: int = 256

    @nn.compact
    def __call__(self, grid_obs: chex.Array, achieved_goal: chex.Array, goal: chex.Array) -> distrax.Categorical:
        init = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        # Flatten leading batch dims so nn.Conv sees (B, H, W, C)
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

        # Concat achieved_goal (absolute agent position) + ultimate goal
        ag = achieved_goal.reshape(-1, achieved_goal.shape[-1])
        g = goal.reshape(-1, goal.shape[-1])
        x = jnp.concatenate([x, ag, g], axis=-1)

        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Dense(self.mlp_hidden, kernel_init=init, bias_init=bias_init)(x)
        x = nn.swish(x)
        logits = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01))(x)

        logits = logits.reshape(*lead, self.num_actions)
        return distrax.Categorical(logits=logits)


# Learner


def get_learner_fn(env, actor, sa_encoder, g_encoder, actor_update_fn, critic_update_fn, config):
    num_envs = config.arch.num_envs
    rollout_length = config.system.rollout_length
    batch_size = config.system.batch_size
    clip_eps = config.system.clip_eps
    ent_coef = config.system.ent_coef
    logsumexp_penalty_coeff = config.system.logsumexp_penalty_coeff
    energy_fn_name = config.system.get("energy_fn", "l2")
    contrastive_loss_name = config.system.get("contrastive_loss_fn", "fwd_infonce")
    action_size = config.system.action_size

    def _compute_q_values(sa_params, g_repr, obs, achieved_goal):
        """Q-values per action for a batch of grid observations.

        Shapes:
            obs           : (B, H, W, 3)
            achieved_goal : (B, goal_dim)
            g_repr        : (B, rep_size)
            return        : (B, A)
        """
        B = obs.shape[0]

        def q_for_action(action_idx):
            a_onehot = jax.nn.one_hot(jnp.full(B, action_idx), action_size)
            sa_repr = sa_encoder.apply(sa_params, obs, achieved_goal, a_onehot)
            return energy_fn(energy_fn_name, sa_repr, g_repr)  # (B,)

        q = jax.vmap(q_for_action)(jnp.arange(action_size))  # (A, B)
        return q.T  # (B, A)

    def actor_step(params, ts: NavixTimestep, key):
        obs = ts.observation                         # (N, H, W, 3)
        ultimate = ts.ultimate_goal                  # (N, goal_dim) commanded goal
        achieved = env.get_achieved_goal(ts)         # (N, goal_dim) current position

        key, act_key = jax.random.split(key)
        pi = actor.apply(params.actor, obs, achieved, ultimate)
        actions = pi.sample(seed=act_key)                # (N,)
        log_prob = pi.log_prob(actions)                  # (N,)

        new_ts = env.step(ts, actions)

        state_extras = {
            "seed": new_ts.seed,
            "truncation": new_ts.is_truncation(),
            "goal": env.get_achieved_goal(new_ts),       # (N, 4)
        }

        transition = Transition(
            observation=obs,
            achieved_goal=env.get_achieved_goal(ts),     # (N, 4)
            ultimate_goal=ultimate,                      # (N, 4)
            action=actions,
            reward=new_ts.reward,
            discount=1.0 - new_ts.is_done().astype(jnp.float32),
            log_prob=log_prob,
            avail_actions=jnp.ones((num_envs, action_size)),
            extras={"state_extras": state_extras},
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

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        key = learner_state.key
        exp_key, relabel_key, shuffle_key, train_key, new_key = jax.random.split(key, 5)

        #  Rollout 
        env_state, traj_batch = collect_experience(learner_state.params, learner_state.env_state, exp_key)

        # Transpose (T, N, ...) -> (N, T, ...) for per-env relabel
        transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), traj_batch)

        # Crude rollout metrics to log under ACT (no episode boundaries needed)
        act_metrics = {
            "rollout_reward_mean": traj_batch.reward.mean(),
            "rollout_done_rate": (1.0 - traj_batch.discount).mean(),
            "rollout_win_rate": (traj_batch.reward > 0).mean() * 100.0,
        }

        #  HER relabel 
        batch_keys = jax.random.split(relabel_key, transitions.observation.shape[0])
        transitions = jax.vmap(relabel_fn, in_axes=(None, 0, 0))(
            config.system.gamma, transitions, batch_keys
        )
        # Flatten env x (T-1) with Fortran order (matches ff_ppo_crl / pqn_crl_nx)
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        num_samples = (transitions.observation.shape[0] // batch_size) * batch_size
        transitions = jax.tree_util.tree_map(lambda x: x[:num_samples], transitions)

        #  CRL advantages 
        params = learner_state.params

        def compute_advantages(transitions, params):
            ultimate = transitions.ultimate_goal   # (B, goal_dim)
            achieved = transitions.achieved_goal   # (B, goal_dim) absolute agent position
            g_repr = g_encoder.apply(params.goal_encoder, ultimate)

            q_values = _compute_q_values(params.sa_encoder, g_repr, transitions.observation, achieved)

            pi = actor.apply(params.actor, transitions.observation, achieved, ultimate)
            probs = jax.nn.softmax(pi.logits)                        # (B, A)
            v = jnp.sum(probs * q_values, axis=-1)                    # (B,)
            q_taken = q_values[jnp.arange(q_values.shape[0]), transitions.action]
            adv = q_taken - v
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            return jax.lax.stop_gradient(adv)

        advantages = compute_advantages(transitions, params)
        transitions = transitions._replace(
            extras={**transitions.extras, "advantages": advantages}
        )

        #  PPO epochs 
        def _minibatch_update(carry, mb):
            params, opt_states, k = carry
            k, a_key, c_key = jax.random.split(k, 3)

            def actor_loss_fn(actor_params, batch):
                pi = actor.apply(actor_params, batch.observation, batch.achieved_goal, batch.ultimate_goal)
                new_log_prob = pi.log_prob(batch.action)
                ratio = jnp.exp(new_log_prob - batch.log_prob)
                adv = batch.extras["advantages"]
                l1 = ratio * adv
                l2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                actor_loss = -jnp.mean(jnp.minimum(l1, l2))
                entropy = pi.entropy().mean()
                total = actor_loss - ent_coef * entropy
                return total, {"actor_loss": actor_loss, "entropy": entropy}

            def critic_loss_fn(critic_params, batch):
                sa_p = critic_params["sa_encoder"]
                g_p = critic_params["goal_encoder"]

                goal = batch.extras["goal"]
                action_onehot = jax.nn.one_hot(batch.action, action_size)
                sa_repr = sa_encoder.apply(sa_p, batch.observation, batch.achieved_goal, action_onehot)
                g_repr = g_encoder.apply(g_p, goal)
                logits = compute_logits(energy_fn_name, sa_repr, g_repr)
                loss = contrastive_loss_fn(contrastive_loss_name, logits, logsumexp_penalty_coeff)
                metrics = compute_contrastive_metrics(logits)
                return loss, {
                    "critic_loss": loss,
                    "logits_pos": metrics["logits_pos"],
                    "logits_neg": metrics["logits_neg"],
                    "categorical_accuracy": metrics["categorical_accuracy"],
                    "logsumexp": metrics["logsumexp"],
                }

            (_, a_info), a_grads = jax.value_and_grad(actor_loss_fn, has_aux=True)(params.actor, mb)
            a_grads, a_info = lax.pmean((a_grads, a_info), axis_name="device")
            a_updates, new_a_opt = actor_update_fn(a_grads, opt_states.actor)
            new_actor_params = optax.apply_updates(params.actor, a_updates)

            critic_params = {"sa_encoder": params.sa_encoder, "goal_encoder": params.goal_encoder}
            (_, c_info), c_grads = jax.value_and_grad(critic_loss_fn, has_aux=True)(critic_params, mb)
            c_grads, c_info = lax.pmean((c_grads, c_info), axis_name="device")
            c_updates, new_c_opt = critic_update_fn(c_grads, opt_states.critic)
            new_sa = optax.apply_updates(params.sa_encoder, c_updates["sa_encoder"])
            new_g = optax.apply_updates(params.goal_encoder, c_updates["goal_encoder"])

            new_params = Params(actor=new_actor_params, sa_encoder=new_sa, goal_encoder=new_g)
            new_opts = OptStates(actor=new_a_opt, critic=new_c_opt)
            return (new_params, new_opts, k), {**a_info, **c_info}

        def _epoch(carry, _):
            params, opt_states, k, flat = carry
            k, perm_key = jax.random.split(k)
            perm = jax.random.permutation(perm_key, flat.observation.shape[0])
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], flat)
            batched = jax.tree_util.tree_map(
                lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
                shuffled,
            )
            (params, opt_states, k), info = jax.lax.scan(
                _minibatch_update, (params, opt_states, k), batched
            )
            return (params, opt_states, k, flat), info

        (params, opt_states, _, _), train_info = jax.lax.scan(
            _epoch,
            (params, learner_state.opt_states, train_key, transitions),
            None,
            length=config.system.ppo_epochs,
        )

        new_learner_state = LearnerState(
            params=params,
            opt_states=opt_states,
            key=new_key,
            env_state=env_state,
            t=learner_state.t + num_envs * rollout_length,
        )
        return new_learner_state, (act_metrics, train_info)

    def learner_fn(learner_state: LearnerState):
        def body(state, _):
            return _update_step(state, None)

        learner_state, (act_metrics, train_metrics) = jax.lax.scan(
            body, learner_state, None, length=config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=act_metrics,
            train_metrics=train_metrics,
        )

    return learner_fn


# Eval act-fn 


def make_eval_act_fn(actor, eval_env, config):
    def act_fn(params, ts: NavixTimestep, key, actor_state):
        achieved = eval_env.get_achieved_goal(ts)
        pi = actor.apply(params, ts.observation, achieved, ts.ultimate_goal)

        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
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
    obs_shape = env.observation_shape  # (H, W, 3) full or (2R+1, 2R+1, 3) partial
    config.system.action_size = action_size
    config.system.obs_size = int(np.prod(obs_shape))

    key = jax.random.PRNGKey(config.system.seed)
    key, env_key, sa_key, g_key, actor_key = jax.random.split(key, 5)

    # Env states per device
    num_envs = config.arch.num_envs
    env_keys = jax.random.split(env_key, n_devices * num_envs).reshape(n_devices, num_envs, -1)
    env_states = jax.vmap(env.reset)(env_keys)
    print(f"Goal coordinates from environment: {env_states.goal_pos[0, 0]}")

    # Networks
    sa_encoder = sa_ConvEncoder(torso=instantiate(config.network.sa_encoder_network.torso), output_dim=config.system.rep_size)
    g_encoder = GoalEncoder(torso=instantiate(config.network.goal_encoder_network.torso), output_dim=config.system.rep_size)
    actor = ConvActor(num_actions=action_size, mlp_hidden=config.system.get("actor_mlp_hidden", 256))

    dummy_obs = jnp.ones([1, *obs_shape])
    dummy_action = jnp.ones([1, action_size])
    dummy_goal = jnp.ones([1, env.goal_dim])
    dummy_achieved = jnp.ones([1, env.goal_dim])

    sa_params = sa_encoder.init(sa_key, dummy_obs, dummy_achieved, dummy_action)
    g_params = g_encoder.init(g_key, dummy_goal)
    actor_params = actor.init(actor_key, dummy_obs, dummy_achieved, dummy_goal)

    params = Params(actor=actor_params, sa_encoder=sa_params, goal_encoder=g_params)

    # LR schedules
    def _schedule(base_lr):
        decay = config.system.get("lr_decay_type", "none")
        end = config.system.get("lr_end", 1e-5)
        total_timesteps = config.system.get("total_timesteps", None)
        if total_timesteps is not None:
            updates = total_timesteps // (n_devices * num_envs * config.system.rollout_length)
        else:
            updates = config.system.num_updates
        # approximate grad-step count — minibatches/epoch computed at train time
        steps = max(1, int(updates * config.system.ppo_epochs * max(
            1, (num_envs * (config.system.rollout_length - 1)) // config.system.batch_size
        )))
        if decay == "linear":
            return optax.linear_schedule(base_lr, end, steps)
        if decay == "cosine":
            return optax.cosine_decay_schedule(base_lr, steps, alpha=end / base_lr)
        return base_lr

    grad_clip = optax.clip_by_global_norm(config.system.max_grad_norm)
    actor_opt = optax.chain(grad_clip, optax.adam(_schedule(config.system.actor_lr), eps=1e-5))
    critic_opt = optax.chain(grad_clip, optax.adam(_schedule(config.system.q_lr)))

    actor_opt_state = actor_opt.init(actor_params)
    critic_opt_state = critic_opt.init({"sa_encoder": sa_params, "goal_encoder": g_params})
    opt_states = OptStates(actor=actor_opt_state, critic=critic_opt_state)

    learn = get_learner_fn(env, actor, sa_encoder, g_encoder, actor_opt.update, critic_opt.update, config)
    learn = jax.pmap(learn, axis_name="device")

    params = flax.jax_utils.replicate(params, devices=jax.devices())
    opt_states = flax.jax_utils.replicate(opt_states, devices=jax.devices())
    key, sk = jax.random.split(key)
    step_keys = jax.random.split(sk, n_devices)
    t0 = jnp.zeros((n_devices,), dtype=jnp.int32)

    learner_state = LearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        t=t0,
    )

    eval_act_fn = make_eval_act_fn(actor, eval_env, config)
    evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)
    abs_metric_evaluator = get_nx_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)

    return learn, learner_state, evaluator, abs_metric_evaluator, actor


def run_experiment(_config: DictConfig) -> float:
    _config.logger.system_name = "ff_ppo_crl_nx"
    config = copy.deepcopy(_config)
    n_devices = len(jax.devices())

    learn, learner_state, evaluator, abs_metric_evaluator, actor = learner_setup(config)
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

    for eval_step in range(config.arch.num_evaluation):
        start = time.time()
        new_learner_state, act_metrics, train_metrics = None, None, None
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
        eval_metrics = evaluator(learner_state.params.actor, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        win_rate = float(jnp.mean(eval_metrics["win_rate"]))
        if win_rate > max_win_rate:
            max_win_rate = win_rate
            best_params = jax.tree_util.tree_map(
                lambda x: np.array(x[0]), learner_output.learner_state.params.actor
            )
            with open(f"{save_dir}/best_policy_ff_ppo_crl_nx.pkl", "wb") as f:
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
    config_name="ff_ppo_crl_nx.yaml",
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
