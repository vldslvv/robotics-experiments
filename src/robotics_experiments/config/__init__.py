from ._types import Px4SitlLaunchConfig, ResolvedVehicle

__all__ = [
    "ResolvedVehicle",
    "Px4SitlLaunchConfig",
    "get_enabled_px4_instance_indices",
    "get_enabled_px4_launch_configs",
    "get_config",
    "resolve_drone_bridge_config",
]


def __getattr__(name: str):
    if name in {
        "get_config",
        "get_enabled_px4_instance_indices",
        "get_enabled_px4_launch_configs",
        "resolve_drone_bridge_config",
    }:
        from ._load import (
            get_config,
            get_enabled_px4_instance_indices,
            get_enabled_px4_launch_configs,
            resolve_drone_bridge_config,
        )

        return {
            "get_config": get_config,
            "get_enabled_px4_instance_indices": get_enabled_px4_instance_indices,
            "get_enabled_px4_launch_configs": get_enabled_px4_launch_configs,
            "resolve_drone_bridge_config": resolve_drone_bridge_config,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
