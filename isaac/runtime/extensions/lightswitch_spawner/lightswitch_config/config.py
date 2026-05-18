from robotics_experiments.config import BoundingBox, LightSwitchConfig
from robotics_experiments.config import get_config as _get_experiment_config

__all__ = ["BoundingBox", "LightSwitchConfig", "get_config"]


def get_config() -> LightSwitchConfig:
    return _get_experiment_config().switches
