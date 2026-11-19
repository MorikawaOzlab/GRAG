"""Run-level configuration dataclass, constructed from the Hydra config."""

from typing import Literal, Optional

from flax.struct import dataclass

from .env import legal_envs


@dataclass
class RunConfig:

    env: Literal[legal_envs]

    total_env_steps: int = 50_000_000
    episode_length: int = 1001
    eval_env: Optional[Literal[legal_envs]] = None
    num_envs: int = 256
    num_eval_envs: int = 256
    action_repeat: int = 1
    num_evals: int = 200
    seed: int = 0
    backend: Optional[Literal["mjx", "spring", "positional", "generalized"]] = None

    # wandb / experiment metadata (consumed by the hydra entry, not the agents)
    exp_name: str = "run"
    log_wandb: bool = True
    wandb_project_name: str = "jaxgcrl"
    wandb_group: str = "."
    wandb_mode: Literal["online", "offline"] = "online"

    absolute_num_eval_envs: int = 0

    visualization_interval: int = 5
    vis_length: int = 1000
    checkpoint_logdir: Optional[str] = None
    max_devices_per_host: int = 1
    cuda: bool = True
