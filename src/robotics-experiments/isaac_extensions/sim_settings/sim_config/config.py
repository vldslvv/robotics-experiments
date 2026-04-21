from experiment_config import SimSettings
from experiment_config import get_config as _get_experiment_config


def get_config() -> SimSettings:
    """Return the simulation-settings section of the shared experiment config.

    Returns:
        The :class:`SimSettings` resolved from
        :func:`experiment_config.get_config`.
    """
    return _get_experiment_config().sim


__all__ = ["SimSettings", "get_config"]
