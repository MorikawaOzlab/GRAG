from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import variance_scaling
from mava.networks.torsos import MLPTorso


class SAEncoder(nn.Module):
    """State-Action Encoder for ICRL - encodes (s,a) pairs using a configurable torso."""

    torso: nn.Module
    output_dim: int = 64

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = jnp.concatenate([s, a], axis=-1)
        x = self.torso(x)
        x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class GoalEncoder(nn.Module):
    """Goal Encoder for ICRL - encodes goals using a configurable torso."""

    torso: nn.Module
    output_dim: int = 64

    @nn.compact
    def __call__(self, g: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = self.torso(g)
        x = nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


class ICRLActor(nn.Module):
    """Actor network for ICRL - outputs continuous action distribution using a configurable torso."""

    torso: nn.Module
    action_size: int
    LOG_STD_MAX: float = 2.0
    LOG_STD_MIN: float = -5.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = self.torso(x)

        mean = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)

        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


class sa_ConvEncoder(nn.Module):
    """State-Action ConvEncoder for grid observations.

    Takes (grid_obs, achieved_goal, action_onehot). The achieved_goal (absolute
    agent position) is concatenated with CNN features so the encoder can predict
    future absolute positions from egocentric observations.
    """
    torso: nn.Module
    output_dim: int = 64

    @nn.compact
    def __call__(self, grid_obs: jnp.ndarray, achieved_goal: jnp.ndarray, action_onehot: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        x = nn.Conv(16, kernel_size=(2, 2), kernel_init=lecun_uniform, bias_init=bias_init)(grid_obs)
        x = nn.swish(x)
        x = nn.Conv(32, kernel_size=(2, 2), kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = nn.swish(x)
        x = nn.Conv(64, kernel_size=(2, 2), kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = nn.swish(x)

        x = x.reshape((x.shape[0], -1))
        x = jnp.concatenate([x, achieved_goal, action_onehot], axis=-1)

        x = self.torso(x)
        return nn.Dense(self.output_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
