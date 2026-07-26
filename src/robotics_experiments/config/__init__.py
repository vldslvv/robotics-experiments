"""Public configuration contract for robotics experiments.

The package root is the only supported import surface outside this package.
Internal modules own Hydra composition, schema registration, and enrichment
details; callers consume the stable config model types and loader entrypoints
exported here.
"""

from robotics_experiments.config._environment import ExperimentConfig, VehicleConfig
from robotics_experiments.config._px4 import Px4BridgeConfig
from robotics_experiments.config._rl import (
    IsaacLabVehicleConfig,
    PressActionConfig,
    PressEvaluationConfig,
    PressResetConfig,
    PressRewardConfig,
    PressTerminationConfig,
    RlPolicyActorConfig,
    RlPolicyConfig,
    RlPolicyCriticConfig,
)
from robotics_experiments.config._scene import BoundingBox, DroneSpawnConfig, LightSwitchConfig
from robotics_experiments.config._simulation import SimSettings

__all__ = [
    "BoundingBox",
    "DroneSpawnConfig",
    "ExperimentConfig",
    "IsaacLabVehicleConfig",
    "LightSwitchConfig",
    "PressActionConfig",
    "PressEvaluationConfig",
    "PressResetConfig",
    "PressRewardConfig",
    "PressTerminationConfig",
    "Px4BridgeConfig",
    "RlPolicyActorConfig",
    "RlPolicyConfig",
    "RlPolicyCriticConfig",
    "SimSettings",
    "VehicleConfig",
    "load_config",
]


def __getattr__(name: str):
    if name == "load_config":
        from robotics_experiments.config.loading import load_config

        return load_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
