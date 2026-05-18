from robotics_experiments.config import DroneConfig
from robotics_experiments.config import get_config as _get_experiment_config


def get_config() -> DroneConfig:
    return _get_experiment_config().drone
