
import jax
from colorama import Fore, Style
from omegaconf import DictConfig


def check_total_timesteps(config: DictConfig) -> DictConfig:
    """Check if total_timesteps is set, if not, set it based on the other parameters"""

    if config.arch.architecture_name == "anakin":
        n_devices = len(jax.devices())
        update_batch_size = config.system.get("update_batch_size", 1)
    else:
        n_devices = 1  # We only use a single device's output when updating.
        update_batch_size = config.system.get("update_batch_size", 1)

    if config.system.total_timesteps is None:
        config.system.num_updates = int(config.system.num_updates)
        config.system.total_timesteps = int(
            n_devices
            * config.system.num_updates
            * config.system.rollout_length
            * update_batch_size
            * config.arch.num_envs
        )
    else:
        config.system.total_timesteps = int(config.system.total_timesteps)
        config.system.num_updates = int(
            config.system.total_timesteps
            // config.system.rollout_length
            // update_batch_size
            // config.arch.num_envs
            // n_devices
        )
        print(
            f"{Fore.RED}{Style.BRIGHT} Changing the number of updates "
            + f"to {config.system.num_updates}: If you want to train"
            + " for a specific number of updates, please set total_timesteps to None!"
            + f"{Style.RESET_ALL}"
        )
    return config
