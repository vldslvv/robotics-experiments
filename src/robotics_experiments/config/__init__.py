"""Public configuration contract for robotics experiments.

The package root is the only supported import surface outside this package.
Internal modules own Hydra composition, schema registration, and enrichment
details; callers consume the stable config model types and loader entrypoints
exported here.
"""

from robotics_experiments.config.types import (
    BoundingBox,
    DriverConfig,
    DroneSpawnConfig,
    ExperimentConfig,
    LightSwitchConfig,
    LightSwitchStatePublisherConfig,
    LockstepConfig,
    PressActionConfig,
    PressHandoffConfig,
    PressHoverTaskConfig,
    PressTaskConfig,
    Px4BridgeConfig,
    Px4SitlLaunchConfig,
    RlConfig,
    RlTrainingConfig,
    RosQosConfig,
    RosVehicleConfig,
    RoutineConfig,
    RoutineStepConfig,
    SimSettings,
    VehicleConfig,
)

__all__ = [
    "BoundingBox",
    "DriverConfig",
    "DroneSpawnConfig",
    "ExperimentConfig",
    "LightSwitchConfig",
    "LightSwitchStatePublisherConfig",
    "LockstepConfig",
    "PressActionConfig",
    "PressHandoffConfig",
    "PressHoverTaskConfig",
    "PressTaskConfig",
    "Px4BridgeConfig",
    "Px4SitlLaunchConfig",
    "RlConfig",
    "RlTrainingConfig",
    "RosQosConfig",
    "RosVehicleConfig",
    "RoutineConfig",
    "RoutineStepConfig",
    "SimSettings",
    "VehicleConfig",
    "get_config",
    "get_enabled_px4_launch_configs",
    "load_config",
    "reset_config_cache",
]


def __getattr__(name: str):
    if name in {
        "get_config",
        "get_enabled_px4_launch_configs",
        "load_config",
        "reset_config_cache",
    }:
        from robotics_experiments.config.loading import (
            get_config,
            get_enabled_px4_launch_configs,
            load_config,
            reset_config_cache,
        )

        return {
            "get_config": get_config,
            "get_enabled_px4_launch_configs": get_enabled_px4_launch_configs,
            "load_config": load_config,
            "reset_config_cache": reset_config_cache,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
