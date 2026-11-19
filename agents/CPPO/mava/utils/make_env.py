
from typing import Tuple, Type, TypeAlias

import chex
import jax.numpy as jnp
import jaxmarl
import jumanji
from jaxmarl.environments.smax import map_name_to_scenario
from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator as ConnectorRandomGenerator,
)
from omegaconf import DictConfig

from mava.types import MarlEnv
from mava.wrappers import (
    AgentIDWrapper,
    AutoResetWrapper,
    JaxNavWrapper,
    RecordEpisodeMetrics,
    SmaxWrapper,
    VectorConnectorWrapper,
    ICRLVectorConnectorWrapper,
)
from mava.wrappers.icrl_goal import ICRLSmaxWrapper

from jumanji.environments.routing.connector.reward import (
    DenseRewardFn,
    RewardFn,
    SparseRewardFn,
    SharedDenseRewardFn,
    SharedSparseRewardFn,
)
from jumanji.environments.routing.connector.types import State as ConnectorState


class CompleteSparseRewardFn(RewardFn):
    """Reward all agents equally, but only when ALL agents connect simultaneously."""
    def __call__(self, state: ConnectorState, action: chex.Array, next_state: ConnectorState) -> float:
        all_connected = jnp.all(next_state.agents.connected) & ~jnp.all(state.agents.connected)
        num_agents = state.agents.id.shape[0]
        return all_connected.repeat(num_agents) * 1.0


_reward_fn_registry = {
    "dense": DenseRewardFn,
    "sparse": SparseRewardFn,
    "shared_dense": SharedDenseRewardFn,
    "shared_sparse": SharedSparseRewardFn,
    "complete_sparse": CompleteSparseRewardFn,
}

registry_type: TypeAlias = dict[str, dict[str, Type]]

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry: registry_type = {
    "VectorConnector": {
        "generator": ConnectorRandomGenerator,
        "wrapper": VectorConnectorWrapper,
        "icrl_wrapper": ICRLVectorConnectorWrapper,
    },
}

# Registry mapping environment names directly to the corresponding wrapper classes.
_jaxmarl_registry: registry_type = {
    "Smax": {"wrapper": SmaxWrapper, "icrl_wrapper": ICRLSmaxWrapper},
    "JaxNav": {"wrapper": JaxNavWrapper},
}


def add_extra_wrappers(
    train_env: MarlEnv, eval_env: MarlEnv, config: DictConfig, registry: registry_type
) -> Tuple[MarlEnv, MarlEnv]:
    """Wrappers that access and modify observations (like AgentIDWrapper) must come before
    GraphWrapper to avoid special casing observation handling for both regular and graph
    observations. For example, AgentIDWrapper adds agent IDs to observations, which should happen
    before converting observation to GraphObservation."""

    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    train_env = AutoResetWrapper(train_env)
    train_env = RecordEpisodeMetrics(train_env)

    eval_env = AutoResetWrapper(eval_env)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_jumanji_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """

    # Config generator and select the wrapper.
    generator_cls = _jumanji_registry[config.env.env_name]["generator"]
    eval_generator = generator_cls(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[config.env.env_name]["wrapper"]

    train_generator = generator_cls(**config.env.scenario.task_config)

    reward_fn = _reward_fn_registry[config.env.reward_fn]()
    # Create envs.
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    train_env = jumanji.make(config.env.scenario.name, generator=train_generator, reward_fn=reward_fn, **env_config)
    eval_env = jumanji.make(config.env.scenario.name, generator=eval_generator, reward_fn=reward_fn, **env_config)

    if config.system.get("use_icrl", False):
        # Get ICRL wrapper and parameters
        icrl_wrapper = _jumanji_registry[config.env.env_name]["icrl_wrapper"]

        # First wrap with standard wrapper to get observation spec
        temp_env = wrapper(train_env, add_global_state=add_global_state)
        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]

        # ICRL parameters — goal_dim is determined by goal_type
        goal_type = config.system.icrl.get("goal_type", "distance_per_agent")
        _goal_dim_lookup = {
            "total_distance": 1,
            "distance_per_agent": 1,
            "ratio_connected": 1,
            "position_coords": 2,
        }
        goal_dim = _goal_dim_lookup.get(goal_type, config.system.icrl.get("goal_dim", 1))
        # Set config for ICRL — no more goal_start_idx/goal_end_idx
        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

        # Win repeat: only for training (gives contrastive loss more goal=0 examples)
        win_repeat_steps = config.system.get("win_repeat_steps", 0)

        # Wrap with ICRL wrapper (goals stored as separate Observation fields)
        train_env = icrl_wrapper(
            train_env,
            add_global_state=add_global_state,
            aggregate_rewards=True,
            goal_type=goal_type,
            goal_dim=goal_dim,
            win_repeat_steps=win_repeat_steps,
        )
        eval_env = icrl_wrapper(
            eval_env,
            add_global_state=add_global_state,
            aggregate_rewards=True,
            goal_type=goal_type,
            goal_dim=goal_dim,
            win_repeat_steps=0,  # Never repeat in eval — clean episode boundaries
        )
    else:
        # Standard wrapper
        train_env = wrapper(train_env, add_global_state=add_global_state)
        eval_env = wrapper(eval_env, add_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config, _jumanji_registry)
    return train_env, eval_env


def make_jaxmarl_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a JAXMARL environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A JAXMARL environment.

    """
    kwargs = dict(config.env.kwargs)
    if "smax" in config.env.env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    use_icrl = config.system.get("use_icrl", False)
    env_name = config.env.env_name

    if use_icrl and "icrl_wrapper" in _jaxmarl_registry.get(env_name, {}):
        # ICRL mode: use ICRL wrapper with goal extraction
        icrl_wrapper_cls = _jaxmarl_registry[env_name]["icrl_wrapper"]
        standard_wrapper_cls = _jaxmarl_registry[env_name]["wrapper"]

        # ICRL parameters
        goal_type = config.system.icrl.get("goal_type", "damage_mean")

        # Get base obs dim from a temp standard wrapper
        temp_env = standard_wrapper_cls(
            jaxmarl.make(config.env.scenario.name, **kwargs), add_global_state
        )

        if goal_type == "damage_per_enemy":
            goal_dim = temp_env._env.num_enemies
        else:
            goal_dim = config.system.icrl.get("goal_dim", 1)

        base_obs_dim = temp_env.observation_spec.agents_view.shape[-1]

        # Account for agent IDs that AgentIDWrapper will prepend
        will_add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)
        if will_add_agent_id:
            base_obs_dim += temp_env.num_agents

        config.system.icrl.base_obs_dim = base_obs_dim
        config.system.icrl.goal_dim = goal_dim

        # Win repeat: only for training (gives contrastive loss more goal=0 examples)
        win_repeat_steps = config.system.get("win_repeat_steps", 0)

        # Create ICRL-wrapped envs
        train_env: MarlEnv = icrl_wrapper_cls(
            jaxmarl.make(config.env.scenario.name, **kwargs),
            has_global_state=add_global_state,
            goal_dim=goal_dim,
            goal_type=goal_type,
            win_repeat_steps=win_repeat_steps,
        )
        eval_env: MarlEnv = icrl_wrapper_cls(
            jaxmarl.make(config.env.scenario.name, **kwargs),
            has_global_state=add_global_state,
            goal_dim=goal_dim,
            goal_type=goal_type,
            win_repeat_steps=0,  # Never repeat in eval — clean episode boundaries
        )
    else:
        wrapper = _jaxmarl_registry[env_name]["wrapper"]

        if env_name == "JaxNav":
            reward_type = kwargs.pop("reward_type", "full")
            goal_type = config.env.get("goal_type", "distance")
            train_env: MarlEnv = wrapper(
                jaxmarl.make(config.env.scenario.name, **kwargs),
                has_global_state=add_global_state,
                goal_type=goal_type,
                reward_type=reward_type,
            )
            eval_env: MarlEnv = wrapper(
                jaxmarl.make(config.env.scenario.name, **kwargs),
                has_global_state=add_global_state,
                goal_type=goal_type,
                reward_type=reward_type,
            )
        else:
            train_env: MarlEnv = wrapper(
                jaxmarl.make(config.env.scenario.name, **kwargs),
                add_global_state,
            )
            eval_env: MarlEnv = wrapper(
                jaxmarl.make(config.env.scenario.name, **kwargs),
                add_global_state,
            )

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config, _jaxmarl_registry)

    return train_env, eval_env


def make(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create environments for training and evaluation.

    Args:
    ----
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    env_name = config.env.env_name

    if env_name in _jumanji_registry:
        return make_jumanji_env(config, add_global_state)
    elif env_name in _jaxmarl_registry:
        return make_jaxmarl_env(config, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
