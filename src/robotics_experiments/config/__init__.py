"""Public configuration contract for robotics experiments.

The package root is the only supported import surface outside this package.
Internal modules own Hydra composition, schema registration, and enrichment
details; callers consume the stable config model types and loader entrypoints
exported here.
"""

from robotics_experiments.config._commanding import RoutineStepConfig
from robotics_experiments.config._curriculum import CurriculumConfig, CurriculumStageConfig
from robotics_experiments.config._environment import ExperimentConfig, VehicleConfig
from robotics_experiments.config._px4 import Px4BridgeConfig
from robotics_experiments.config._rl import (
    HoverTaskConfig,
    IsaacLabVehicleConfig,
    PressApproachAcceptanceConfig,
    PressApproachEvaluationConfig,
    PressApproachGoalConfig,
    PressApproachRewardConfig,
    PressApproachTerminationConfig,
    PressApproachVelocityReferenceConfig,
    PressHandoffConfig,
    PressResetConfig,
    PressResetSourceConfig,
    PressTaskConfig,
    RlActionConfig,
    RlInputsConfig,
    RlPlateauCriteriaConfig,
    RlPolicyActorConfig,
    RlPolicyConfig,
    RlPolicyCriticConfig,
    RlTrainingConfig,
    RlTrainingPhaseConfig,
)
from robotics_experiments.config._scene import BoundingBox, DroneSpawnConfig, LightSwitchConfig
from robotics_experiments.config._simulation import SimSettings
from robotics_experiments.config.loading import load_config, load_curriculum_config

__all__ = [
    "CurriculumConfig",
    "CurriculumStageConfig",
    "load_curriculum_config",
    "BoundingBox",
    "DroneSpawnConfig",
    "ExperimentConfig",
    "HoverTaskConfig",
    "IsaacLabVehicleConfig",
    "LightSwitchConfig",
    "PressApproachAcceptanceConfig",
    "PressApproachEvaluationConfig",
    "PressApproachGoalConfig",
    "PressApproachRewardConfig",
    "PressApproachTerminationConfig",
    "PressApproachVelocityReferenceConfig",
    "PressTaskConfig",
    "PressResetSourceConfig",
    "PressHandoffConfig",
    "PressResetConfig",
    "Px4BridgeConfig",
    "RlActionConfig",
    "RlInputsConfig",
    "RlPolicyActorConfig",
    "RlPolicyConfig",
    "RlPolicyCriticConfig",
    "RlPlateauCriteriaConfig",
    "RlTrainingPhaseConfig",
    "RlTrainingConfig",
    "RoutineStepConfig",
    "SimSettings",
    "VehicleConfig",
    "load_config",
]
