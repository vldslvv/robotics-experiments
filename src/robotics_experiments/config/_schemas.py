from hydra.core.config_store import ConfigStore

from ._types import (
    DroneConfig,
    EnvConfig,
    LightSwitchConfig,
    PressConfig,
    Px4BridgeConfig,
    RoutineConfig,
    SimSettings,
)

__all__: list[str] = []


def _register_schemas() -> None:
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
    cs.store(group="experiment/press", name="base_press", node=PressConfig)
    cs.store(group="experiment/routine", name="base_routine", node=RoutineConfig)
