"""Public configuration contract for robotics experiments.

The package root is the only supported import surface outside this package.
Internal modules own Hydra composition, schema registration, and enrichment
details; callers consume the stable config model types and loader entrypoints
exported here.
"""

from robotics_experiments.config._commanding import RoutineStepConfig
from robotics_experiments.config._environment import ExperimentConfig, VehicleConfig
from robotics_experiments.config._px4 import Px4BridgeConfig
from robotics_experiments.config._rl import (
    HoverTaskConfig,
    IsaacLabVehicleConfig,
    PressApproachEvaluationConfig,
    PressApproachGoalConfig,
    PressApproachRewardConfig,
    PressApproachTerminationConfig,
    PressResetConfig,
    RlActionConfig,
    RlPolicyActorConfig,
    RlPolicyConfig,
    RlPolicyCriticConfig,
    RlTrainingConfig,
)
from robotics_experiments.config._scene import BoundingBox, DroneSpawnConfig, LightSwitchConfig
from robotics_experiments.config._simulation import SimSettings
from robotics_experiments.config.loading import load_config

__all__ = [
    "BoundingBox",
    "DroneSpawnConfig",
    "ExperimentConfig",
    "HoverTaskConfig",
    "IsaacLabVehicleConfig",
    "LightSwitchConfig",
    "PressApproachEvaluationConfig",
    "PressApproachGoalConfig",
    "PressApproachRewardConfig",
    "PressApproachTerminationConfig",
    "PressResetConfig",
    "Px4BridgeConfig",
    "RlActionConfig",
    "RlPolicyActorConfig",
    "RlPolicyConfig",
    "RlPolicyCriticConfig",
    "RlTrainingConfig",
    "RoutineStepConfig",
    "SimSettings",
    "VehicleConfig",
    "load_config",
]
