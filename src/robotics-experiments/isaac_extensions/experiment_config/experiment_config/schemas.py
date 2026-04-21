from dataclasses import dataclass, field
from typing import List, Tuple

from hydra.core.config_store import ConfigStore


@dataclass
class SimSettings:
    """Simulation settings applied by the sim_settings extension.

    Attributes:
        physics_hz: PhysX steps per second.
        rendering_hz: Kit main-loop target render frequency in Hz.
        solver_type: PhysX solver identifier (``"TGS"`` or ``"PGS"``).
        gravity: World gravity vector in m/s^2 as ``(gx, gy, gz)``.
    """

    physics_hz: float = 500.0
    rendering_hz: float = 60.0
    solver_type: str = "TGS"
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class DroneConfig:
    """Configuration for the drone spawner extension.

    Attributes:
        usd_path: Absolute path to the drone USD asset referenced on spawn.
        prim_name: Child name used under the stage default prim for the drone.
        spawn_volume_prim_name: Child name used for the guide cube that
            visualizes the spawn volume.
        spawn_volume_center: World-space center of the spawn volume.
        spawn_volume_half_extents: Per-axis half-extents of the spawn volume.
    """

    usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/hexacopter_s550.usda"
    prim_name: str = "DroneSpawned"
    spawn_volume_prim_name: str = "DroneSpawnVolume"
    spawn_volume_center: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    spawn_volume_half_extents: Tuple[float, float, float] = (2.0, 2.0, 1.0)


@dataclass
class BoundingBox:
    """Axis-aligned world-space region, authored as corner tuples.

    Used instead of ``pxr.Gf.Range3d`` so the value is serializable by
    OmegaConf / Hydra. Consumers convert to ``Gf.Range3d`` at the call site.

    Attributes:
        min: Lower corner ``(x, y, z)`` in world units.
        max: Upper corner ``(x, y, z)`` in world units.
    """

    min: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    max: Tuple[float, float, float] = (1.0, 1.0, 1.0)


def _default_bounds() -> List[BoundingBox]:
    """Return the default list of spawn volumes for switches.

    Returns:
        A single axis-aligned region along the warehouse wall at y=17.99 m.
    """
    return [BoundingBox(min=(-3.0, 17.99, 0.5), max=(8.0, 17.99, 2.7))]


@dataclass
class LightSwitchConfig:
    """Configuration for the light-switch spawner extension.

    Attributes:
        switch_usd_path: Absolute path to the switch USD asset.
        lamp_usd_path: Absolute path to the lamp USD asset.
        num_switches: Number of switch/lamp pairs to spawn.
        press_threshold: Joint position above which the button is treated as pressed.
        min_spawn_dist: Minimum pairwise distance between spawned switches, in meters.
        max_spawn_retries: Retry budget per position before accepting overlap.
        lamp_on_intensity: ``SphereLight`` intensity applied when the lamp is on.
        spawn_bounds: Candidate axis-aligned regions used to sample positions.
    """

    switch_usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/LightSwitch.usd"
    lamp_usd_path: str = "/isaac-sim/proj/assets/LightSwitchWarehouse/Lamp.usd"
    num_switches: int = 5
    press_threshold: float = 0.01
    min_spawn_dist: float = 0.5
    max_spawn_retries: int = 100
    lamp_on_intensity: float = 5000.0
    spawn_bounds: List[BoundingBox] = field(default_factory=_default_bounds)


@dataclass
class EnvConfig:
    """Composite experiment config consumed by the Isaac Sim extensions.

    Attributes:
        sim: Simulation-level settings (physics, rendering, gravity).
        drone: Drone spawner configuration.
        switches: Light-switch spawner configuration.
    """

    sim: SimSettings = field(default_factory=SimSettings)
    drone: DroneConfig = field(default_factory=DroneConfig)
    switches: LightSwitchConfig = field(default_factory=LightSwitchConfig)


def register_schemas() -> None:
    """Register the structured-config schemas with Hydra's ``ConfigStore``.

    Idempotent: safe to call on every config composition. The group schemas
    let YAML files under ``conf/{sim,drone,switches}/`` be validated against
    their dataclass shape.
    """
    cs = ConfigStore.instance()
    cs.store(name="base_env", node=EnvConfig)
    cs.store(group="sim", name="base_sim", node=SimSettings)
    cs.store(group="drone", name="base_drone", node=DroneConfig)
    cs.store(group="switches", name="base_switches", node=LightSwitchConfig)
