
from typing import Tuple

import chex
import jax
from jumanji.env import State
from jumanji.types import TimeStep
from jumanji.wrappers import Observation, Wrapper

from mava.types import MarlEnv, ObservableMarlEnv


class AutoResetWrapper(Wrapper):
    """Automatically resets environments that are done. Once the terminal state is reached,
    the state, observation, and step_type are reset. The observation and step_type of the
    terminal TimeStep is reset to the reset observation and StepType.LAST, respectively.
    The reward, discount, and extras retrieved from the transition to the terminal state.
    """

    OBS_IN_EXTRAS_KEY = "real_next_obs"

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv, deterministic: bool = False):
        super().__init__(env)
        self._env: MarlEnv

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim
        self.deterministic = deterministic
    def _obs_in_extras(self, state: State, timestep: TimeStep[Observation]) -> Tuple[State, TimeStep[Observation]]:
        """Place the observation in timestep.extras[real_next_obs]."""
        extras = timestep.extras
        extras[AutoResetWrapper.OBS_IN_EXTRAS_KEY] = timestep.observation
        return state, timestep.replace(extras=extras)

    def _auto_reset(self, state: State, timestep: TimeStep[Observation]) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Save and increment episode_seed for ICRL buffer relabeling
        old_episode_seed = state.episode_seed if hasattr(state, "episode_seed") else 0

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)  # type: ignore
        if hasattr(state, "reset_key") and self.deterministic:
            key = state.reset_key
        state, reset_timestep = self._env.reset(key)

        # Restore episode seed counter (reset() returns 0, but we need to preserve it)
        if hasattr(state, "episode_seed"):
            new_episode_seed = old_episode_seed + 1
            state = state.replace(episode_seed=new_episode_seed)

        # Place original observation in extras.
        state, timestep = self._obs_in_extras(state, timestep)

        # Replace observation with reset observation.
        timestep = timestep.replace(observation=reset_timestep.observation)  # type: ignore

        return state, timestep

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        return self._obs_in_extras(*super().reset(key))

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Step the environment, with automatic resetting if the episode terminates."""
        state, timestep = self._env.step(state, action)

        # Overwrite the state and timestep appropriately if the episode terminates.
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            self._obs_in_extras,
            state,
            timestep,
        )

        return state, timestep
