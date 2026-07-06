"""Public configuration contract for robotics experiments.

The package root is the only supported import surface outside this package.
Internal modules own Hydra composition, schema registration, and enrichment
details; callers consume the stable config model types and loader entrypoints
exported here.
"""

from robotics_experiments.config.types import (
    BoundingBox,
    DriverConfig,
    DroneConfig,
    EnvConfig,
    LightSwitchConfig,
    LightSwitchStatePublisherConfig,
    LockstepConfig,
    PressActionConfig,
    PressHandoffConfig,
    PressHoverTaskConfig,
    PressTaskConfig,
    Px4BridgeConfig,
    Px4SitlLaunchConfig,
    ResolvedVehicle,
    RlConfig,
    RlTrainingConfig,
    RosQosConfig,
    RosVehicleConfig,
    RoutineConfig,
    SimSettings,
    VehicleConfig,
)

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
    "get_config",
    "get_enabled_px4_launch_configs",
    "reset_config_cache",
    "resolve_drone_bridge_config",
]


def __getattr__(name: str):
    if name in {
        "get_config",
        "get_enabled_px4_launch_configs",
        "reset_config_cache",
        "resolve_drone_bridge_config",
    }:
        from robotics_experiments.config.loading import (
            get_config,
            get_enabled_px4_launch_configs,
            reset_config_cache,
            resolve_drone_bridge_config,
        )

        return {
            "get_config": get_config,
            "get_enabled_px4_launch_configs": get_enabled_px4_launch_configs,
            "reset_config_cache": reset_config_cache,
            "resolve_drone_bridge_config": resolve_drone_bridge_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
