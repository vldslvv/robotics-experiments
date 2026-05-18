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
class Px4BridgeConfig:
    """Configuration for the PX4 SITL bridge.

    Attributes:
        enabled: When false, the drone-spawner host does not construct or
            tick a bridge. Default false so existing scenes are unchanged
            until PX4 is wired up end-to-end.
        articulation_path: USD path of the drone root articulation. When
            empty (default), the host derives it from the drone-spawner
            config (``<default_prim>/<drone.prim_name>``).
        rotor_link_paths: USD paths of rotor links **in PX4 motor-index
            order** (FR-CW, RL-CW, FL-CCW, RR-CCW for Generic Quad X). The
            host prefixes them with the articulation path if they start
            with a ``/``, otherwise treats them as relative.
        num_rotors: Number of rotors driven by the bridge.
        max_rotor_force_n: Force in Newtons produced at command 1.0 by the
            default linear thrust model.
        connection_type: "tcpin" (bridge listens) or "tcpout".
        connection_ip: Address used by ``connection_type``.
        connection_baseport: TCP port for the simulator MAVLink API.
        vehicle_id: Added to ``connection_baseport`` when more than one
            vehicle shares a host.
        px4_autolaunch: When true, the bridge spawns a PX4 SITL process via
            ``PX4LaunchTool`` on start. Leave false when running PX4 in
            Docker.
        px4_dir: Path to the PX4-Autopilot tree, only used when autolaunch.
        px4_vehicle_model: ``PX4_SIM_MODEL`` env value passed to autolaunch.
        lockstep_timeout_s: Hard deadline on receiving an actuator message
            each tick. On timeout the port returns zeros and Isaac keeps
            ticking - required so a stalled flight stack does not freeze
            the simulator.
        gps_send_rate_hz: Rate at which synthesized ``HIL_GPS`` is sent.
        home_lat_deg, home_lon_deg, home_alt_m: Home position used when
            synthesizing GPS from the local ENU pose. Defaults match PX4's
            built-in default (Zurich).
    """

    enabled: bool = False
    articulation_path: str = ""
    rotor_link_paths: List[str] = field(default_factory=list)
    num_rotors: int = 4
    max_rotor_force_n: float = 8.0
    connection_type: str = "tcpin"
    connection_ip: str = "localhost"
    connection_baseport: int = 4560
    vehicle_id: int = 0
    px4_autolaunch: bool = False
    px4_dir: str = ""
    px4_vehicle_model: str = "gazebo-classic_iris"
    lockstep_timeout_s: float = 0.1
    gps_send_rate_hz: float = 5.0
    home_lat_deg: float = 47.3977
    home_lon_deg: float = 8.5456
    home_alt_m: float = 488.0


@dataclass
class EnvConfig:
    """Composite experiment config consumed by the Isaac Sim extensions.

    Attributes:
        sim: Simulation-level settings (physics, rendering, gravity).
        drone: Drone spawner configuration.
        switches: Light-switch spawner configuration.
        px4: PX4 SITL bridge configuration.
    """

    sim: SimSettings = field(default_factory=SimSettings)
    drone: DroneConfig = field(default_factory=DroneConfig)
    switches: LightSwitchConfig = field(default_factory=LightSwitchConfig)
    px4: Px4BridgeConfig = field(default_factory=Px4BridgeConfig)


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
