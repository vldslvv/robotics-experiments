from robotics_experiments.config import SimSettings
from robotics_experiments.config import get_config as _get_experiment_config

__all__ = ["SimSettings", "get_config"]


def get_config() -> SimSettings:
    return _get_experiment_config().sim
