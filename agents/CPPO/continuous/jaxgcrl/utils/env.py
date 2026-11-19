
from jaxgcrl.envs.ant import Ant
from jaxgcrl.envs.ant_maze import AntMaze
from jaxgcrl.envs.reacher import Reacher

legal_envs = (
    "ant",
    "ant_random_start",
    "reacher",
    "ant_u_maze",
    "ant_big_maze",
    "ant_hardest_maze",
)


def create_env(env_name: str, backend: str = None, dense_reward: bool = False, **kwargs) -> object:
    """Instantiate a JaxGCRL env by name.

    Args:
        env_name: env identifier (must be in `legal_envs`).
        backend: brax backend (`"mjx"`, `"spring"`, `"positional"`, `"generalized"`).
                 None lets each env pick its own default.
        dense_reward: use the dense (velocity-to-target + healthy) reward instead
                      of the sparse goal-reaching reward.

    Returns:
        A brax `PipelineEnv` subclass with `state_dim`, `goal_indices`,
        `observation_size`, and `action_size` attributes.
    """
    if env_name == "reacher":
        env = Reacher(backend=backend or "generalized", dense_reward=dense_reward)
    elif env_name == "ant":
        env = Ant(backend=backend or "spring", dense_reward=dense_reward)
    elif env_name == "ant_random_start":
        env = Ant(backend=backend or "spring", randomize_start=True, dense_reward=dense_reward)
    elif "maze" in env_name:
        # ant_u_maze -> "u_maze", ant_big_maze -> "big_maze", ant_hardest_maze -> "hardest_maze"
        env = AntMaze(backend=backend or "spring", maze_layout_name=env_name[4:], dense_reward=dense_reward)
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    return env
