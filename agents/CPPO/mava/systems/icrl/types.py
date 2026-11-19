
from typing import Any, Dict, Tuple

import flax.linen as nn
import optax
from chex import Array, PRNGKey
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from typing_extensions import NamedTuple, TypeAlias

from mava.types import Observation, State

# Type aliases
Metrics: TypeAlias = Dict[str, Array]
Networks: TypeAlias = Tuple[nn.Module, nn.Module, nn.Module]  # (actor, sa_encoder, goal_encoder)
Optimisers: TypeAlias = Tuple[
    optax.GradientTransformation,  # actor optimizer
    optax.GradientTransformation,  # critic (both encoders) optimizer
    optax.GradientTransformation,  # alpha optimizer
]


class ICRLParams(NamedTuple):
    """Parameters for ICRL networks."""

    actor: FrozenVariableDict = None
    sa_encoder: FrozenVariableDict = None
    goal_encoder: FrozenVariableDict = None
    log_alpha: Array = None


class OptStates(NamedTuple):
    """Optimizer states for ICRL."""

    actor: optax.OptState = None
    critic: optax.OptState = None
    alpha: optax.OptState = None


class Transition(NamedTuple):
    """Transition for ICRL with separated goal fields."""

    observation: Array  # [batch, base_obs_dim] — pure env obs (flat)
    achieved_goal: Array  # [batch, goal_dim] — current metric value
    ultimate_goal: Array  # [batch, goal_dim] — commanded goal (0 at eval, relabeled in hindsight)
    action: Array
    reward: Array
    discount: Array
    avail_actions: Array  # Available actions mask for discrete action spaces
    extras: Dict[str, Any]
    log_prob: Array = None

BufferState: TypeAlias = TrajectoryBufferState[Transition]


class LearnerState(NamedTuple):
    """Complete learner state for ICRL (restructured version)."""

    params: ICRLParams
    opt_states: OptStates
    buffer_state: BufferState
    key: PRNGKey
    env_state: State
    last_timestep: Any  # TimeStep from environment


class PQNLearnerState(NamedTuple):
    """Learner state for PQN-CRL (no buffer)."""

    params: ICRLParams
    opt_states: OptStates
    key: PRNGKey
    env_state: State
    last_timestep: Any  # TimeStep from environment
    t: Array = None  # Timestep counter for epsilon decay


class RNNLearnerState(NamedTuple):
    """State of the `Learner` for recurrent architectures."""

    params: ICRLParams
    opt_states: OptStates
    key: PRNGKey
    env_state: State
    timestep: Any  # TimeStep from environment
    dones: Array
    hstates: Any
    t: Array = None  # Timestep counter for epsilon decay