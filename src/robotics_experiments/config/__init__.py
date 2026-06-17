from .types import (
    BoundingBox,
    DroneConfig,
    EnvConfig,
    IsaacLabConfig,
    LightSwitchConfig,
    Px4BridgeConfig,
    SimSettings,
)

__all__ = [
    "BoundingBox",
    "DroneConfig",
    "EnvConfig",
    "IsaacLabConfig",
    "LightSwitchConfig",
    "Px4BridgeConfig",
    "SimSettings",
    "get_config",
    "reset_config",
]


def __getattr__(name: str):
    if name in {"get_config", "reset_config"}:
        from .load import get_config, reset_config

        return {"get_config": get_config, "reset_config": reset_config}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
