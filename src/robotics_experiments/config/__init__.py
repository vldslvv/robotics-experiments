from ._types import ResolvedVehicle

__all__ = [
    "ResolvedVehicle",
    "get_enabled_px4_instance_indices",
    "get_config",
]


def __getattr__(name: str):
    if name in {
        "get_config",
        "get_enabled_px4_instance_indices",
    }:
        from ._load import (
            get_config,
            get_enabled_px4_instance_indices,
        )

        return {
            "get_config": get_config,
            "get_enabled_px4_instance_indices": get_enabled_px4_instance_indices,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
