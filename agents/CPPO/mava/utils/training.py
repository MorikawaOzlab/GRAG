
from typing import Callable, Union

from omegaconf import DictConfig

import optax


def make_learning_rate_schedule(init_lr: float, config: DictConfig) -> Callable:
    """Makes a very simple linear learning rate scheduler.

    Args:
    ----
        init_lr: initial learning rate.
        config: system configuration.

    Note:
    ----
        We use a simple linear learning rate scheduler based on the suggestions from a blog on PPO
        implementation details which can be viewed at http://tinyurl.com/mr3chs4p
        This function can be extended to have more complex learning rate schedules by adding any
        relevant arguments to the system config and then parsing them accordingly here.

    """

    def linear_scedule(count: int) -> float:
        frac: float = (
            1.0 - (count // (config.system.ppo_epochs * config.system.num_minibatches)) / config.system.num_updates
        )
        return init_lr * frac

    return linear_scedule


def make_learning_rate(init_lr: float, config: DictConfig) -> Union[float, Callable]:
    """Retuns a constant learning rate or a learning rate schedule.

    Args:
    ----
        init_lr: initial learning rate.
        config: system configuration.

    Returns:
    -------
        A learning rate schedule or fixed learning rate.

    """

    if config.system.get("lr_decay_type", "none") == "linear":
        return optax.linear_schedule(
            init_value=init_lr,
            end_value=config.system.lr_end,
            transition_steps=config.system.total_grad_steps,
        )
    elif config.system.get("lr_decay_type", "none") == "cosine":
        return optax.cosine_decay_schedule(
            init_value=init_lr,
            decay_steps=config.system.total_grad_steps,
            alpha=config.system.lr_end / init_lr,
        )
    elif config.system.get("decay_learning_rates", False):
        return make_learning_rate_schedule(init_lr, config)
    else:
        return init_lr
