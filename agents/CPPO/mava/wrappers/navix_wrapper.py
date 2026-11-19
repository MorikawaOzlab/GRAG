
from typing import Tuple
from functools import cached_property

import jax
import jax.numpy as jnp
import navix as nx
import navix.observations as navix_obs
from jax import Array
from flax import struct
from navix.environments.environment import Timestep, Environment, StepType
from navix.states import State


def _get_goal_position(state: State) -> Array:
    """Extract goal position from Navix state."""
    if "goal" in state.entities:
        return state.entities["goal"].position[0]
    raise ValueError("Goal entity not found in state")


def _get_door_position(state: State) -> Array:
    """Extract door position from Navix state (assumes a door entity exists)."""
    return state.entities["door"].position[0]


def _get_virtual_key_position(state: State) -> Array:

    key_pos = state.entities["key"].position[0]
    player_pos = state.entities["player"].position[0]
    door_pos = state.entities["door"].position[0]
    door_open = state.entities["door"].open[0]
    pocket = state.entities["player"].pocket[0]
    key_id = state.entities["key"].id[0]
    carrying = pocket == key_id

    vkey = jnp.where(door_open, door_pos,
                     jnp.where(carrying, player_pos, key_pos))
    return vkey


def termination_never(prev_state: State, action: Array, state: State) -> Array:
    """Never terminate — only truncate at max_steps."""
    return jnp.asarray(False, dtype=jnp.bool_)


@struct.dataclass
class NavixTimestep:

    timestep: Timestep
    seed: Array
    goal_pos: Array
    ultimate_goal: Array
    key: Array

    @property
    def t(self) -> Array:
        return self.timestep.t

    @property
    def observation(self) -> Array:
        return self.timestep.observation

    @property
    def action(self) -> Array:
        return self.timestep.action

    @property
    def reward(self) -> Array:
        return self.timestep.reward

    @property
    def step_type(self) -> Array:
        return self.timestep.step_type

    @property
    def state(self) -> State:
        return self.timestep.state

    @property
    def info(self):
        return self.timestep.info

    def is_done(self) -> Array:
        """Episode ended (goal reached OR truncated)."""
        return self.timestep.step_type > StepType.TRANSITION

    def is_truncation(self) -> Array:
        """Episode truncated (max steps reached without goal)."""
        return self.timestep.step_type == StepType.TRUNCATION

    def is_termination(self) -> Array:
        """Episode terminated (goal reached)."""
        return self.timestep.step_type == StepType.TERMINATION


class NavixWrapper:
    """ Navix wrapper for CRL training ."""

    def __init__(
        self,
        env_name: str,
        terminate_on_goal: bool = True,
        max_steps: int = None,
        obs_type: str = "full",
        fov_radius: int = 3,
    ):

        self.env_name = env_name
        self.terminate_on_goal = terminate_on_goal
        self.obs_type = obs_type

        # Build kwargs for env creation
        make_kwargs = {}
        if max_steps is not None:
            make_kwargs["max_steps"] = max_steps
        if not terminate_on_goal:
            make_kwargs["termination_fn"] = termination_never
        if obs_type == "partial":
            navix_obs.set_radius(fov_radius)
            make_kwargs["observation_fn"] = navix_obs.symbolic_first_person

        # Create the underlying environment
        self._env: Environment = nx.make(env_name, **make_kwargs)

        # Cache static properties
        self._height = self._env.height
        self._width = self._env.width
        self._max_steps = self._env.max_steps
        self._action_size = self._env.action_space.n

        probe_ts = self._env.reset(jax.random.PRNGKey(0))
        probe_entities = probe_ts.state.entities
        self._has_key_door = ("key" in probe_entities) and ("door" in probe_entities)
        self._obs_shape: Tuple[int, ...] = tuple(probe_ts.observation.shape)
        # On DoorKey: 1D scalar progress goal. Else: 2D player position goal.
        self._goal_dim = 1 if self._has_key_door else 2
        # Max manhattan distance on the grid, used to normalize progress.
        self._max_manhattan = float(self._height + self._width)

        # Closures over `env` / has_key_door to avoid `self` in JIT
        env = self._env
        has_key_door = self._has_key_door
        max_manhattan = self._max_manhattan

        def _build_ultimate_goal(state: State) -> Array:
            if has_key_door:
                # Scalar 1.0: key delivered to door AND player at goal.
                return jnp.ones((1,), dtype=jnp.float32)
            return _get_goal_position(state).astype(jnp.float32)

        def reset_single_fn(key: Array) -> NavixTimestep:
            timestep = env.reset(key)
            seed = jax.random.randint(key, (), 0, 2**31 - 1)
            goal_pos = _get_goal_position(timestep.state)
            ultimate_goal = _build_ultimate_goal(timestep.state)
            return NavixTimestep(
                timestep=timestep,
                seed=seed,
                goal_pos=goal_pos,
                ultimate_goal=ultimate_goal,
                key=key,
            )

        def step_single_fn(navix_ts: NavixTimestep, action: Array) -> NavixTimestep:
            old_ts = navix_ts.timestep
            old_t = old_ts.t
            action_scalar = jnp.squeeze(action)

            # Use env.step which includes Navix's built-in random autoreset
            new_ts = env.step(old_ts, action_scalar)

            # Detect autoreset: t went from > 0 back to 0
            autoreset = jnp.logical_and(new_ts.t == 0, old_t > 0)
            new_seed = jnp.where(autoreset, navix_ts.seed + 1, navix_ts.seed)
            goal_pos = _get_goal_position(new_ts.state)
            ultimate_goal = _build_ultimate_goal(new_ts.state)

            return NavixTimestep(
                timestep=new_ts,
                seed=new_seed,
                goal_pos=goal_pos,
                ultimate_goal=ultimate_goal,
                key=navix_ts.key,
            )

        # Stash and create vmapped/jitted versions
        self._reset_single_fn = reset_single_fn
        self._step_single_fn = step_single_fn
        self._reset_single = jax.jit(reset_single_fn)
        self._step_single = jax.jit(step_single_fn)
        self.reset = jax.jit(jax.vmap(reset_single_fn))
        self.step = jax.jit(jax.vmap(step_single_fn))

    @cached_property
    def observation_shape(self) -> Tuple[int, ...]:
        """Observation shape: (H, W, 3) for full obs, (2R+1, 2R+1, 3) for partial."""
        return self._obs_shape

    @cached_property
    def action_size(self) -> int:
        return self._action_size

    @cached_property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def goal_dim(self) -> int:
        """CRL goal dimensionality. 5 on DoorKey (4 positions + 1 progress), 2 otherwise."""
        return self._goal_dim

    @property
    def max_manhattan(self) -> float:
        return self._max_manhattan

    @property
    def has_key_door(self) -> bool:
        return self._has_key_door

    def get_player_position(self, navix_ts: NavixTimestep) -> Array:
        """Get current player position from a (vmapped) timestep.
        """
        return navix_ts.state.entities["player"].position[:, 0]

    def get_goal_position(self, navix_ts: NavixTimestep) -> Array:
        return navix_ts.goal_pos

    def get_virtual_key_position(self, navix_ts: NavixTimestep) -> Array:
        """Virtual key position per the CRL spec. Shape `(num_envs, 2)`.
        """
        state = navix_ts.state
        key_pos = state.entities["key"].position[:, 0]
        player_pos = state.entities["player"].position[:, 0]
        door_pos = state.entities["door"].position[:, 0]
        door_open = state.entities["door"].open[:, 0]
        pocket = state.entities["player"].pocket[:, 0]
        key_id = state.entities["key"].id[:, 0]
        carrying = pocket == key_id

        return jnp.where(
            door_open[:, None], door_pos,
            jnp.where(carrying[:, None], player_pos, key_pos),
        )

    def get_achieved_goal(self, navix_ts: NavixTimestep) -> Array:
        """CRL achieved_goal.
        """
        player_pos = self.get_player_position(navix_ts).astype(jnp.float32)
        if not self._has_key_door:
            return player_pos

        vkey_pos = self.get_virtual_key_position(navix_ts).astype(jnp.float32)
        door_pos = navix_ts.state.entities["door"].position[:, 0].astype(jnp.float32)
        goal_pos = navix_ts.state.entities["goal"].position[:, 0].astype(jnp.float32)

        key_dist = jnp.sum(jnp.abs(vkey_pos - door_pos), axis=-1)
        player_dist = jnp.sum(jnp.abs(player_pos - goal_pos), axis=-1)
        key_prog = 1.0 - key_dist / self._max_manhattan
        player_prog = 1.0 - player_dist / self._max_manhattan
        progress = 0.5 * key_prog + 0.5 * player_prog  # (num_envs,)

        return progress[:, None]

    def get_ultimate_goal(self, navix_ts: NavixTimestep) -> Array:
        """CRL ultimate_goal, 4D on DoorKey, 2D elsewhere."""
        return navix_ts.ultimate_goal


def make_navix_env(
    env_name: str = "Navix-Empty-8x8-v0",
    terminate_on_goal: bool = True,
    max_steps: int = None,
    obs_type: str = "full",
    fov_radius: int = 3,
) -> NavixWrapper:
    """Create a simple Navix environment wrapper.

    Args:
        env_name: Any registered Navix environment name.
        terminate_on_goal: Whether to terminate on goal reach.
        max_steps: Maximum episode length (None → Navix default).
        obs_type: "full" for global grid (H, W, 3); "partial" for egocentric
            first-person view (2*fov_radius+1, 2*fov_radius+1, 3).
        fov_radius: Agent field-of-view half-width (only used when obs_type="partial").

    Returns:
        A `NavixWrapper` instance with vmapped + jitted reset/step.
    """
    return NavixWrapper(env_name, terminate_on_goal, max_steps, obs_type, fov_radius)
