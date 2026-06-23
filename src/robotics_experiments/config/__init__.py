from .types import (
    BoundingBox,
    DroneConfig,
    EnvConfig,
    IsaacLabConfig,
    LightSwitchConfig,
    Px4BridgeConfig,
    ResolvedVehicle,
    SimSettings,
    VehicleConfig,
)

__all__ = [
    "BoundingBox",
    "DroneConfig",
    "EnvConfig",
    "IsaacLabConfig",
    "LightSwitchConfig",
    "Px4BridgeConfig",
    "ResolvedVehicle",
    "SimSettings",
    "VehicleConfig",
    "get_enabled_px4_vehicle_count",
    "get_enabled_px4_instance_indices",
    "get_config",
    "reset_config",
]


def __getattr__(name: str):
    if name in {
        "get_config",
        "get_enabled_px4_vehicle_count",
        "get_enabled_px4_instance_indices",
        "reset_config",
    }:
        from .load import (
            get_config,
            get_enabled_px4_instance_indices,
            get_enabled_px4_vehicle_count,
            reset_config,
        )

        return {
            "get_config": get_config,
            "get_enabled_px4_vehicle_count": get_enabled_px4_vehicle_count,
            "get_enabled_px4_instance_indices": get_enabled_px4_instance_indices,
            "reset_config": reset_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
