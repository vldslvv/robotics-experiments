from hydra.core.config_store import ConfigStore

from .types import (
    BoundingBox,
    DroneConfig,
    EnvConfig,
    IntegrationTestConfig,
    IsaacLabConfig,
    LightSwitchConfig,
    LightSwitchStatePublisherConfig,
    Px4BridgeConfig,
    RosQosConfig,
    RoutineConfig,
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
    "RoutineConfig",
    "IntegrationTestConfig",
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
    # Group schemas are stored under the experiment/ subdir where their YAML
    # building blocks live, so each group file's ``defaults: - base_<group>``
    # resolves within its own group; the top-level entry configs at configs/
    # select them via ``experiment/<group>@<group>``.
    cs.store(group="experiment/sim", name="base_sim", node=SimSettings)
    cs.store(group="experiment/drone", name="base_drone", node=DroneConfig)
    cs.store(group="experiment/switches", name="base_switches", node=LightSwitchConfig)
    cs.store(group="experiment/px4", name="base_px4", node=Px4BridgeConfig)
    cs.store(group="experiment/isaac_lab", name="base_isaac_lab", node=IsaacLabConfig)
    cs.store(group="experiment/routine", name="base_routine", node=RoutineConfig)
    cs.store(
        group="experiment/integration_test",
        name="base_integration_test",
        node=IntegrationTestConfig,
    )
