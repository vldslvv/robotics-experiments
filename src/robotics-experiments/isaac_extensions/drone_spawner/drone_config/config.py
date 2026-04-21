from experiment_config import DroneConfig
from experiment_config import get_config as _get_experiment_config


def get_config() -> DroneConfig:
    """Return the drone section of the shared experiment config.

    Returns:
        The :class:`DroneConfig` resolved from
        :func:`experiment_config.get_config`.
    """
    return _get_experiment_config().drone


__all__ = ["DroneConfig", "get_config"]
