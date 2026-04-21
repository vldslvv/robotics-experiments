from experiment_config import BoundingBox, LightSwitchConfig
from experiment_config import get_config as _get_experiment_config


def get_config() -> LightSwitchConfig:
    """Return the light-switch section of the shared experiment config.

    Returns:
        The :class:`LightSwitchConfig` resolved from
        :func:`experiment_config.get_config`.
    """
    return _get_experiment_config().switches


__all__ = ["BoundingBox", "LightSwitchConfig", "get_config"]
