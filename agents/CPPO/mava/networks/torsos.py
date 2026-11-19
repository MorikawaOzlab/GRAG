
from typing import Callable, Dict, Sequence

import chex
import jax
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal, variance_scaling


def _parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
        "swish": nn.swish,
    }
    return activation_fns[activation_fn_name]


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    use_layer_norm_scale: bool = False
    activate_final: bool = True

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        x = observation
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size, kernel_init=lecun_uniform)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=self.use_layer_norm_scale)(x)

            should_activate = (i < len(self.layer_sizes) - 1) or self.activate_final
            x = self.activation_fn(x) if should_activate else x

        return x
