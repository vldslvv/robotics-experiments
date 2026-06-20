from hydra.core.config_store import ConfigStore

from .types import (
    BoundingBox,
    DroneConfig,
    EnvConfig,
    IsaacLabConfig,
    LightSwitchConfig,
    LightSwitchStatePublisherConfig,
    Px4BridgeConfig,
    RosQosConfig,
    SimSettings,
    VehicleConfig,
)

__all__ = [
    "SimSettings",
    "DroneConfig",
    "BoundingBox",
    "RosQosConfig",
    "LightSwitchStatePublisherConfig",
    "LightSwitchConfig",
    "Px4BridgeConfig",
    "VehicleConfig",
    "EnvConfig",
    "IsaacLabConfig",
    "register_schemas",
]


def register_schemas() -> None:
    """Register the structured-config schemas with Hydra's ``ConfigStore``.

    Idempotent: safe to call on every config composition. The group schemas
    let YAML files under ``conf/{sim,drone,switches,px4}/`` be validated
    against their dataclass shape.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_env", node=EnvConfig)
    cs.store(group="sim", name="base_sim", node=SimSettings)
    cs.store(group="drone", name="base_drone", node=DroneConfig)
    cs.store(group="switches", name="base_switches", node=LightSwitchConfig)
    cs.store(group="px4", name="base_px4", node=Px4BridgeConfig)
    cs.store(group="isaac_lab", name="base_isaac_lab", node=IsaacLabConfig)
