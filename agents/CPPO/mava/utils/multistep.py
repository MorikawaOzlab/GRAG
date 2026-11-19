
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp

from mava.systems.ppo.types import PPOTransition, RNNPPOTransition


def calculate_gae(
    traj_batch: Union[PPOTransition, RNNPPOTransition],
    last_val: chex.Array,
    last_done: chex.Array,
    gamma: float,
    gae_lambda: float,
    unroll: int = 16,
) -> Tuple[chex.Array, chex.Array]:
    """Computes truncated generalized advantage estimates.

    The advantages are computed in a backwards fashion according to the equation:
    Âₜ = δₜ + (γλ) * δₜ₊₁ + ... + ... + (γλ)ᵏ⁻ᵗ⁺¹ * δₖ₋₁
    where δₜ = rₜ₊₁ + γₜ₊₁ * v(sₜ₊₁) - v(sₜ).
    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347

    Args:
        traj_batch (B, T, N, ...): a batch of trajectories.
        last_val  (B, N): value of the final timestep.
        last_done (B, N): whether the last timestep was a terminated or truncated.
        gamma (float): discount factor.
        gae_lambda (float): GAE mixing parameter.
        unroll (int): how much XLA should unroll the scan used to calculate GAE.

    Returns Tuple[(B, T, N), (B, T, N)]: advantages and target values.
    """

    def _get_advantages(
        carry: Tuple[chex.Array, chex.Array, chex.Array], transition: RNNPPOTransition
    ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
        gae, next_value, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward

        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, done), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, last_done),
        traj_batch,
        reverse=True,
        unroll=unroll,
    )
    return advantages, advantages + traj_batch.value
