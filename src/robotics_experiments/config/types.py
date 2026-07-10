"""Compatibility aggregator for the private domain-oriented config schemas.

The supported public boundary remains ``robotics_experiments.config``.
Internal config modules may import this aggregator when they need the complete
schema graph; individual schema ownership lives in the private modules.
"""

from robotics_experiments.config._commanding import DriverConfig, RoutineConfig
from robotics_experiments.config._environment import EnvConfig
from robotics_experiments.config._px4 import (
    LockstepConfig,
    Px4BridgeConfig,
    Px4SitlLaunchConfig,
    ResolvedVehicle,
    VehicleConfig,
)
from robotics_experiments.config._rl import (
    PressActionConfig,
    PressHandoffConfig,
    PressHoverTaskConfig,
    PressTaskConfig,
    RlConfig,
    RlTrainingConfig,
)
from robotics_experiments.config._scene import (
    BoundingBox,
    DroneConfig,
    LightSwitchConfig,
    LightSwitchStatePublisherConfig,
    RosQosConfig,
    RosVehicleConfig,
)
from robotics_experiments.config._simulation import SimSettings

__all__ = [
    "BoundingBox",
    "DriverConfig",
    "DroneConfig",
    "EnvConfig",
    "LightSwitchConfig",
    "LightSwitchStatePublisherConfig",
    "LockstepConfig",
    "PressActionConfig",
    "PressHandoffConfig",
    "PressHoverTaskConfig",
    "PressTaskConfig",
    "Px4BridgeConfig",
    "Px4SitlLaunchConfig",
    "ResolvedVehicle",
    "RlConfig",
    "RlTrainingConfig",
    "RosQosConfig",
    "RosVehicleConfig",
    "RoutineConfig",
    "SimSettings",
    "VehicleConfig",
]
