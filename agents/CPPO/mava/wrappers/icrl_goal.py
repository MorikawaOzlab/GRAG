
from typing import Tuple, Dict, Any, Union
from functools import cached_property

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import TimeStep
from jumanji.environments.routing.connector.types import State as ConnectorState

from dataclasses import replace as dataclass_replace
from jumanji.types import StepType
from mava.wrappers.jumanji import VectorConnectorWrapper
from mava.wrappers.jaxmarl import SmaxWrapper, MPEWrapper, JaxMarlState
from mava.types import Observation, ObservationGlobalState

@chex.dataclass
class ICRLVectorConnectorState:
    """State wrapper for ICRL VectorConnector.
    """
    state: ConnectorState
    episode_seed: int = 0 
    reset_key: chex.PRNGKey = 0
    win_steps_remaining: int = 0  

    @property
    def key(self) -> chex.PRNGKey:
        """Forward key access to underlying state for AutoResetWrapper compatibility."""
        return self.state.key

    def replace(self, **kwargs) -> "ICRLVectorConnectorState":
        return dataclass_replace(self, **kwargs)


class ICRLVectorConnectorWrapper(VectorConnectorWrapper):
    """ICRL wrapper for VectorConnector that adds goal fields to Observation.

    Goals are stored as separate achieved_goal/ultimate_goal fields on the Observation,
    """

    def __init__(
        self,
        env,
        add_global_state: bool = False,
        aggregate_rewards: bool = True,
        goal_type: str = "distance_per_agent",
        goal_dim: int = 1,
        win_repeat_steps: int = 0,
    ):
        super().__init__(env, add_global_state, aggregate_rewards)

        self.goal_type = goal_type
        self._goal_dim = goal_dim
        self._win_repeat = win_repeat_steps

        # Select goal extraction function
        _goal_registry = {
            "total_distance": self._goal_total_distance,           
            "distance_per_agent": self._goal_distance_per_agent,   
            "ratio_connected": self._goal_ratio_connected,         
            "position_coords": self._goal_position_coords,         
        }
        if goal_type not in _goal_registry:
            raise ValueError(
                f"Unknown goal_type: {goal_type}. "
                f"Available: {list(_goal_registry.keys())}"
            )
        self._extract_goal = _goal_registry[goal_type]

    def reset(self, key: chex.PRNGKey) -> Tuple[ICRLVectorConnectorState, TimeStep]:
        """Reset with episode_seed tracking."""
        state, timestep = self._env.reset(key)
        # episode_seed = jax.random.randint(key, (), 0, 2**31 - 1, dtype=jnp.int32)
        episode_seed = jnp.int32(0)
        timestep = self.modify_timestep(timestep, episode_seed=episode_seed)

        icrl_state = ICRLVectorConnectorState(
            state=state,
            episode_seed=episode_seed,
            reset_key=key,
            win_steps_remaining=jnp.int32(0),
        )

        # Set goal fields on observation
        timestep = self._set_goal_fields(icrl_state, timestep)

        if self.add_global_state:
            from mava.types import ObservationGlobalState
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return icrl_state, timestep.replace(observation=observation)

        return icrl_state, timestep

    def step(self, icrl_state: ICRLVectorConnectorState, action: chex.Array) -> Tuple[ICRLVectorConnectorState, TimeStep]:
        prev_remaining = icrl_state.win_steps_remaining
        in_repeat = prev_remaining > 0

        new_conn_state, raw_timestep = self._env.step(icrl_state.state, action)

        raw_won = raw_timestep.extras.get("ratio_connections", 0.0) == 1.0
        raw_last = raw_timestep.step_type == StepType.LAST
        won_now = raw_won & raw_last & (~in_repeat) & (self._win_repeat > 0)

        new_remaining = jnp.where(
            won_now, self._win_repeat,
            jnp.where(in_repeat, prev_remaining - 1, 0)
        )

        final_conn_state = jax.tree.map(
            lambda old, new: jnp.where(in_repeat, old, new),
            icrl_state.state, new_conn_state
        )

        timestep = self.modify_timestep(
            raw_timestep,
            episode_seed=icrl_state.episode_seed,
        )

        new_icrl_state = ICRLVectorConnectorState(
            state=final_conn_state,
            episode_seed=icrl_state.episode_seed,
            reset_key=icrl_state.reset_key,
            win_steps_remaining=new_remaining,
        )

        timestep = self._set_goal_fields(new_icrl_state, timestep)

        should_be_mid = (won_now | in_repeat) & (new_remaining > 0)
        # LAST when countdown reaches 0
        should_be_last = in_repeat & (new_remaining == 0)

        step_type = timestep.step_type
        step_type = jnp.where(should_be_mid, StepType.MID, step_type)
        step_type = jnp.where(should_be_last, StepType.LAST, step_type)

        # Zero reward during repeat steps 
        reward = jnp.where(in_repeat, jnp.zeros_like(timestep.reward), timestep.reward)

        # Discount: 1.0 during MID repeat, 0.0 on final LAST
        discount = timestep.discount
        discount = jnp.where(should_be_mid, jnp.ones_like(discount), discount)
        discount = jnp.where(should_be_last, jnp.zeros_like(discount), discount)

        timestep = timestep.replace(
            step_type=step_type,
            reward=reward,
            discount=discount,
        )

        if self.add_global_state:
            from mava.types import ObservationGlobalState
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return new_icrl_state, timestep.replace(observation=observation)

        return new_icrl_state, timestep

    def modify_timestep(self, timestep: TimeStep, episode_seed: int = 0):
        """Modify the timestep for the Connector environment - adds ICRL extras."""
        timestep_base = super().modify_timestep(timestep)

        obs_data = {
            "agents_view": timestep_base.observation.agents_view,
            "action_mask": timestep_base.observation.action_mask,
            "step_count": jnp.repeat(timestep.observation.step_count, self.num_agents),
        }

        won_episode = timestep.extras.get("ratio_connections", 0.0) == 1.0
        is_last = timestep.step_type == StepType.LAST
        truncation = jnp.where(is_last & ~won_episode, 1.0, 0.0).astype(jnp.float32)

        new_extras = {
            "seed": jnp.asarray(episode_seed, dtype=jnp.int32),
            "truncation": truncation,
        }
        metrics: Dict[str, Any] = {**timestep_base.extras, **new_extras}

        return timestep_base.replace(observation=Observation(**obs_data), extras=metrics)

    def _set_goal_fields(
        self,
        icrl_state: ICRLVectorConnectorState,
        timestep: TimeStep
    ) -> TimeStep:
        """Extract goal metric and set achieved_goal/ultimate_goal on observation."""
        achieved_goal = self._extract_goal(icrl_state)  # [num_agents, goal_dim]
        ultimate_goal = self._ultimate_goal(icrl_state, achieved_goal)

        new_observation = timestep.observation._replace(
            achieved_goal=achieved_goal,
            ultimate_goal=ultimate_goal,
        )

        return timestep.replace(observation=new_observation)

    def _ultimate_goal(
        self,
        icrl_state: ICRLVectorConnectorState,
        achieved_goal: chex.Array,
    ) -> chex.Array:
        """Return the ultimate goal for the current goal_type.
        """
        if self.goal_type == "position_coords":
            return icrl_state.state.agents.target.astype(jnp.float32)
        return jnp.ones_like(achieved_goal)


    def _goal_total_distance(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Mean team progress (1 - mean normalized Manhattan distance).

        Goes from ~0 to 1.0 as agents approach their targets.
        Shared across all agents (same value tiled).
        """
        state = icrl_state.state

        positions = state.agents.position
        targets = state.agents.target

        distance = jnp.sum(jnp.abs(positions - targets), axis=-1)
        mean_distance = jnp.mean(distance)

        grid_size = state.grid.shape[0]
        normalized_distance = mean_distance / (2 * grid_size)
        current_progress = 1.0 - normalized_distance

        return jnp.full((self.num_agents, 1), current_progress)

    def _goal_distance_per_agent(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Per-agent progress toward target (1 - normalized distance).

        Goes from ~0 to 1.0 as agent approaches its target.
        Matches the old working CRL signal: achieved=progress, ultimate=1.0.
        """
        state = icrl_state.state

        positions = state.agents.position
        targets = state.agents.target

        distance = jnp.sum(jnp.abs(positions - targets), axis=-1)

        grid_size = state.grid.shape[0]
        normalized_distances = distance / (2 * grid_size)
        current_progress = 1.0 - normalized_distances

        return current_progress[:, None]  # [num_agents, 1]

    def _goal_ratio_connected(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Ratio of agents connected (team metric).

        Goes from 0 to 1.0 as agents connect to their targets.
        Shared across all agents (same value tiled).
        """
        state = icrl_state.state

        ratio_connected = jnp.mean(state.agents.connected.astype(jnp.float32))

        return jnp.full((self.num_agents, 1), ratio_connected)

    def _goal_position_coords(
        self,
        icrl_state: ICRLVectorConnectorState,
    ) -> chex.Array:
        """Current (x, y) position of each agent. goal_dim=2.

        During hindsight, future positions become the goal.
        At eval, ultimate_goal = zeros (origin), though for position_coords
        the interpretation is different from distance-based goals.
        """
        state = icrl_state.state
        positions = state.agents.position.astype(jnp.float32)
        return positions  # [num_agents, 2]

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """Observation spec — agents_view is unchanged (no goal concatenated)."""
        return super().observation_spec




@chex.dataclass
class ICRLSmaxState:
    """State wrapper for ICRL SMAX
    """
    state: JaxMarlState
    episode_seed: int = 0
    reset_key: chex.PRNGKey = 0
    win_steps_remaining: int = 0
    frozen_agents_view: chex.Array = None
    frozen_action_mask: chex.Array = None

    @property
    def key(self) -> chex.PRNGKey:
        """Forward key access to underlying state for AutoResetWrapper compatibility."""
        return self.state.key

    def replace(self, **kwargs) -> "ICRLSmaxState":
        return dataclass_replace(self, **kwargs)


class ICRLSmaxWrapper(SmaxWrapper):
    """ICRL wrapper for SMAX that adds goal fields to Observation.
    Goal types:
    - "damage_mean": scalar mean damage (goal_dim=1). 1 - mean(normalized enemy health).
    - "damage_per_enemy": per-enemy damage vector (goal_dim=num_enemies).
      Each element is 1 - normalized_health for that enemy.
    """

    def __init__(
        self,
        env,
        has_global_state: bool = False,
        goal_dim: int = 1,
        goal_type: str = "damage_mean",
        win_repeat_steps: int = 0,
    ):
        super().__init__(env, has_global_state)

        self._goal_dim = goal_dim
        self._goal_type = goal_type
        self._win_repeat = win_repeat_steps

    def reset(self, key: chex.PRNGKey) -> Tuple[ICRLSmaxState, TimeStep]:
        """Reset with episode_seed tracking."""
        state, timestep = super().reset(key)
        episode_seed = jnp.int32(0)

        icrl_state = ICRLSmaxState(
            state=state,
            episode_seed=episode_seed,
            reset_key=key,
            win_steps_remaining=jnp.int32(0),
            frozen_agents_view=timestep.observation.agents_view,
            frozen_action_mask=timestep.observation.action_mask,
        )

        timestep = self._add_icrl_fields(icrl_state, timestep)
        return icrl_state, timestep

    def step(
        self, icrl_state: ICRLSmaxState, action: chex.Array
    ) -> Tuple[ICRLSmaxState, TimeStep]:
        """Step the environment with ICRL goal tracking and optional win_repeat.

        """
        prev_remaining = icrl_state.win_steps_remaining
        in_repeat = prev_remaining > 0

        # Always step the real env
        state, timestep = super().step(icrl_state.state, action)

        # Detect win from the raw timestep
        won_episode = timestep.extras["env_metrics"]["won_episode"]
        is_last = timestep.step_type == StepType.LAST
        won_now = won_episode & is_last & (~in_repeat) & (self._win_repeat > 0)

        new_remaining = jnp.where(
            won_now, self._win_repeat,
            jnp.where(in_repeat, prev_remaining - 1, 0),
        )

        should_override = won_now | in_repeat
        agents_view = jax.tree.map(
            lambda frozen, real: jnp.where(should_override, frozen, real),
            icrl_state.frozen_agents_view, timestep.observation.agents_view,
        )
        action_mask = jax.tree.map(
            lambda frozen, real: jnp.where(should_override, frozen, real),
            icrl_state.frozen_action_mask, timestep.observation.action_mask,
        )

        achieved_goal_normal = self._extract_enemy_health_goal_from_obs(
            timestep.observation.agents_view, state
        )
        success_value = self._success_goal_value()
        achieved_goal = jnp.where(should_override, success_value, achieved_goal_normal)
        ultimate_goal = success_value

        # Build observation
        new_obs = timestep.observation._replace(
            agents_view=agents_view,
            action_mask=action_mask,
            achieved_goal=achieved_goal,
            ultimate_goal=ultimate_goal,
        )

        # Compute truncation 
        truncation = jnp.where(is_last & ~won_episode & ~in_repeat, 1.0, 0.0).astype(jnp.float32)

        #  Win repeat step_type overrides 
        step_type = timestep.step_type
        should_be_mid = (won_now | in_repeat) & (new_remaining > 0)
        should_be_last = in_repeat & (new_remaining == 0)
        step_type = jnp.where(should_be_mid, StepType.MID, step_type)
        step_type = jnp.where(should_be_last, StepType.LAST, step_type)

        # Zero reward during repeat steps
        reward = jnp.where(should_override, jnp.zeros_like(timestep.reward), timestep.reward)

        # Discount: 1.0 during MID repeat, 0.0 on final LAST
        discount = timestep.discount
        discount = jnp.where(should_be_mid, jnp.ones_like(discount), discount)
        discount = jnp.where(should_be_last, jnp.zeros_like(discount), discount)

        # Fix won_episode during win_repeat: any episode in repeat is by definition a win
        env_metrics = timestep.extras.get("env_metrics", {})
        if "won_episode" in env_metrics:
            env_metrics = {**env_metrics, "won_episode": won_episode | won_now | in_repeat}

        new_extras = {
            **timestep.extras,
            "env_metrics": env_metrics,
            "seed": jnp.asarray(icrl_state.episode_seed, dtype=jnp.int32),
            "truncation": truncation,
        }

        timestep = timestep.replace(
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_obs,
            extras=new_extras,
        )

        # Update frozen view: save current obs for potential future win freezing
        # During repeat, keep the frozen view unchanged
        new_frozen_view = jnp.where(
            should_override, icrl_state.frozen_agents_view, timestep.observation.agents_view
        )
        new_frozen_mask = jnp.where(
            should_override, icrl_state.frozen_action_mask, timestep.observation.action_mask
        )

        new_icrl_state = ICRLSmaxState(
            state=state,
            episode_seed=icrl_state.episode_seed,
            reset_key=icrl_state.reset_key,
            win_steps_remaining=new_remaining,
            frozen_agents_view=new_frozen_view,
            frozen_action_mask=new_frozen_mask,
        )

        return new_icrl_state, timestep

    def _add_icrl_fields(
        self, icrl_state: ICRLSmaxState, timestep: TimeStep
    ) -> TimeStep:
        """Add achieved_goal, ultimate_goal, seed, and truncation to timestep."""
        achieved_goal = self._extract_enemy_health_goal_from_obs(
            timestep.observation.agents_view, icrl_state.state
        )
        ultimate_goal = self._success_goal_value()

        new_obs = timestep.observation._replace(
            achieved_goal=achieved_goal,
            ultimate_goal=ultimate_goal,
        )

        won_episode = timestep.extras["env_metrics"]["won_episode"]
        is_last = timestep.step_type == StepType.LAST
        truncation = jnp.where(is_last & ~won_episode, 1.0, 0.0).astype(jnp.float32)

        new_extras = {
            **timestep.extras,
            "seed": jnp.asarray(icrl_state.episode_seed, dtype=jnp.int32),
            "truncation": truncation,
        }

        return timestep.replace(observation=new_obs, extras=new_extras)

    def _per_agent_visible_enemy_health(self, agents_view: chex.Array, smax_state) -> chex.Array:
        """Per-agent visibility-masked normalized enemy health matrix.

        Returns [num_agents, num_enemies] with values in [0, 1].
        """
        num_allies = self._env.num_allies
        num_enemies = self._env.num_enemies
        num_feat = len(self._env.unit_features)

        enemy_health = smax_state.unit_health[num_allies : num_allies + num_enemies]
        enemy_types = smax_state.unit_types[num_allies : num_allies + num_enemies]
        max_health = self._env.unit_type_health[enemy_types]
        real_health = enemy_health / (max_health + 1e-8)  # [num_enemies]

        enemy_start = (num_allies - 1) * num_feat
        enemy_health_indices = enemy_start + jnp.arange(num_enemies) * num_feat
        obs_health = agents_view[:, enemy_health_indices]  # [num_agents, num_enemies]

        health_diff_sq = (obs_health - real_health[None, :]) ** 2
        visible = health_diff_sq < 0.01

        enemy_health_per_agent = jnp.where(visible, obs_health, 1.0)

        # Dead agents observe an all-zero vector; fall back to real health for them.
        agent_dead = jnp.all(agents_view == 0, axis=-1, keepdims=True)
        enemy_health_per_agent = jnp.where(
            agent_dead, real_health[None, :], enemy_health_per_agent
        )

        return enemy_health_per_agent

    def _compute_per_agent_visible_health(self, agents_view: chex.Array, smax_state) -> chex.Array:
        """Sum of per-agent visible enemy health. Shape [num_agents, 1], values in [0, num_enemies]."""
        return jnp.sum(
            self._per_agent_visible_enemy_health(agents_view, smax_state),
            axis=-1, keepdims=True,
        )

    def _get_smax_state(self, jaxmarl_state: JaxMarlState):
        """Get the raw SMAX State from JaxMarlState, handling HeuristicEnemySMAX nesting."""
        s = jaxmarl_state.state
        # HeuristicEnemySMAX wraps State in another State with .state and .enemy_policy_state
        if hasattr(s, 'unit_health'):
            return s
        if hasattr(s, 'state') and hasattr(s.state, 'unit_health'):
            return s.state
        raise AttributeError(f"Cannot find unit_health in state nesting: {type(s)}")

    def _extract_enemy_health_goal_from_obs(
        self, agents_view: chex.Array, state: JaxMarlState
    ) -> chex.Array:
        """Per-agent achieved-goal vector. Shape [num_agents, goal_dim].

        Both reset and step use this (via per-agent visibility), so phi is
        consistent across the trajectory.
        - damage_per_enemy → 1 - per-agent visible enemy health, [N, num_enemies]
        - visible_alive_ratio → fraction of enemies still alive from this agent's POV, [N, 1]
        - damage_mean (and any unrecognised type) → sum of per-agent visible health, [N, 1]
        """
        smax_state = self._get_smax_state(state)
        enemy_health_per_agent = self._per_agent_visible_enemy_health(agents_view, smax_state)
        if self._goal_type == "damage_per_enemy":
            return 1.0 - enemy_health_per_agent
        if self._goal_type == "visible_alive_ratio":
            alive = (enemy_health_per_agent > 0.01).astype(jnp.float32)
            return jnp.mean(alive, axis=-1, keepdims=True)

        return jnp.sum(enemy_health_per_agent, axis=-1, keepdims=True)

    def _success_goal_value(self) -> chex.Array:
        """Goal-vector value at success. Also used as ultimate_goal target.

        Convention is per goal_type:
        - damage_per_enemy stores damage proportion → success = ones
        - damage_mean / visible_alive_ratio store remaining health → success = zeros
        """
        if self._goal_type in ("damage_per_enemy"):
            return jnp.ones((self.num_agents, self._goal_dim))
        return jnp.zeros((self.num_agents, self._goal_dim))

