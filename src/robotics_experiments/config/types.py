from dataclasses import dataclass
from typing import List, Tuple

__all__ = [
    "SimSettings",
    "DroneConfig",
    "BoundingBox",
    "RosQosConfig",
    "LightSwitchStatePublisherConfig",
    "LightSwitchConfig",
    "Px4BridgeConfig",
    "EnvConfig",
]


@dataclass
class SimSettings:
    """Simulation settings applied by the sim_settings extension.

    Attributes:
        physics_hz: PhysX steps per second.
        rendering_hz: Kit main-loop target render frequency in Hz.
        solver_type: PhysX solver identifier (``"TGS"`` or ``"PGS"``).
        gravity: World gravity vector in m/s^2 as ``(gx, gy, gz)``.
    """

    physics_hz: float
    rendering_hz: float
    solver_type: str
    gravity: Tuple[float, float, float]


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
        spawn_stationary: When true, spawn at sampled X/Y and yaw but compute
            Z from the referenced drone bounds so the vehicle starts resting
            near the ground instead of falling through PX4 startup.
        stationary_ground_z: World Z coordinate of the ground plane used for
            stationary placement.
        stationary_ground_clearance_m: Clearance added above the computed
            lower bound to avoid initial interpenetration.
        stationary_spawn_z_fallback: Root Z used if composed bounds cannot be
            computed for the referenced drone.
    """

    usd_path: str
    prim_name: str
    spawn_volume_prim_name: str
    spawn_volume_center: Tuple[float, float, float]
    spawn_volume_half_extents: Tuple[float, float, float]
    spawn_stationary: bool
    stationary_ground_z: float
    stationary_ground_clearance_m: float
    stationary_spawn_z_fallback: float


@dataclass
class BoundingBox:
    """Axis-aligned world-space region, authored as corner tuples.

    Used instead of ``pxr.Gf.Range3d`` so the value is serializable by
    OmegaConf / Hydra. Consumers convert to ``Gf.Range3d`` at the call site.

    Attributes:
        min: Lower corner ``(x, y, z)`` in world units.
        max: Upper corner ``(x, y, z)`` in world units.
    """

    min: Tuple[float, float, float]
    max: Tuple[float, float, float]


@dataclass
class RosQosConfig:
    """ROS 2 QoS settings for a publisher or subscription.

    Attributes:
        depth: Queue depth for ``KEEP_LAST`` history.
        durability: Durability policy name, such as ``"transient_local"``.
        reliability: Reliability policy name, such as ``"reliable"``.
        history: History policy name, such as ``"keep_last"``.
    """

    depth: int
    durability: str
    reliability: str
    history: str


@dataclass
class LightSwitchStatePublisherConfig:
    """ROS publisher configuration for light-switch state events.

    Attributes:
        topic: ROS topic used for switch toggle events.
        qos: ROS 2 QoS settings for the publisher.
    """

    topic: str
    qos: RosQosConfig


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
        state_publisher: ROS publisher settings for switch toggle events.
    """

    switch_usd_path: str
    lamp_usd_path: str
    num_switches: int
    press_threshold: float
    min_spawn_dist: float
    max_spawn_retries: int
    lamp_on_intensity: float
    spawn_bounds: List[BoundingBox]
    state_publisher: LightSwitchStatePublisherConfig


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
        max_rotor_force_n: Legacy staging value for the linear thrust model.
            The default PX4 path uses Pegasus' quadratic rotor model instead.
        input_offset, input_scaling, zero_position_armed: Pegasus MAVLink
            actuator scaling from PX4 controls to rotor angular velocity
            references.
        rotor_constant, rolling_moment_coefficient, min_rotor_velocity,
            max_rotor_velocity: Pegasus quadratic thrust curve parameters.
            Rotor count and spin direction come from authored USD rotor
            metadata on the spawned drone asset.
        linear_drag: Pegasus body-frame linear drag coefficients.
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
        first_heartbeat_timeout_s: Hard deadline on receiving PX4's first
            heartbeat after the port starts. On timeout, the bridge marks
            the port disconnected instead of waiting indefinitely.
        max_bootstrap_actuator_lag_s: Upper bound on the lag of PX4's first
            HIL_ACTUATOR_CONTROLS packet relative to current sim time. A
            host-load-dependent PX4 SITL startup can emit its first actuator
            packet several seconds behind; accepting that as the steady-state
            baseline makes the bridge apply multi-second-stale motor commands
            and destabilizes takeoff. When the first lag exceeds this bound the
            bridge disconnects (fail closed) with a distinct "actuator bootstrap
            lag too large" reason, catching the pathological startup early. Set
            to zero or a negative value to disable the guard.
        heartbeat_rate_hz: Wall-clock rate for bridge-originated MAVLink
            heartbeat messages.
        startup_settle_time_s: Sim-time delay before opening the PX4
            simulator MAVLink endpoint. The drone is allowed to settle in
            Isaac physics first so PX4 does not initialize from spawn-drop
            transients.
        gps_enabled: When true, update and send Pegasus ``HIL_GPS`` from the
            local ENU pose. Keep true for the default PX4 SITL estimator path.
        gps_send_rate_hz: Rate at which ``HIL_GPS`` is sent.
        sensor_noise_enabled: When true, Pegasus sensor noise/bias models are
            active. Integration and real simulator runs should keep this true;
            precision tests may disable it through Hydra or direct port config.
        home_lat_deg, home_lon_deg, home_alt_m: Home position used when
            synthesizing GPS from the local ENU pose. Defaults match PX4's
            built-in default (Zurich).
        debug_log_path: Path template for persistent bridge diagnostics. Empty
            disables file logging. Supports ``{date}`` (``YYYY-MM-DD``),
            ``{time}`` (``HH_MM_SS``), and ``{timestamp}``
            (``YYYY-MM-DD/HH_MM_SS``) so bridge logs can be paired with PX4
            ULog files without appending across runs.
        debug_log_level: Python logging level name used by the bridge file log.
        sensor_debug_log_interval: Emit one detailed HIL_SENSOR payload log
            every N sensor sends after the initial startup samples.
        max_tilt_angle_deg: Maximum allowed tilt angle in degrees between the
            drone's body-Z axis and world-Z. When exceeded, the bridge zeros
            all rotor forces to prevent inverted thrust after a crash. Set to
            zero or a negative value to disable.
    """

    enabled: bool
    articulation_path: str
    max_rotor_force_n: float
    input_offset: List[float]
    input_scaling: List[float]
    zero_position_armed: List[float]
    rotor_constant: List[float]
    rolling_moment_coefficient: List[float]
    min_rotor_velocity: List[float]
    max_rotor_velocity: List[float]
    linear_drag: List[float]
    connection_type: str
    connection_ip: str
    connection_baseport: int
    vehicle_id: int
    px4_autolaunch: bool
    px4_dir: str
    px4_vehicle_model: str
    lockstep_timeout_s: float
    first_heartbeat_timeout_s: float
    max_bootstrap_actuator_lag_s: float
    heartbeat_rate_hz: float
    startup_settle_time_s: float
    gps_enabled: bool
    gps_send_rate_hz: float
    sensor_noise_enabled: bool
    home_lat_deg: float
    home_lon_deg: float
    home_alt_m: float
    debug_log_path: str
    debug_log_level: str
    sensor_debug_log_interval: int
    max_tilt_angle_deg: float


@dataclass
class EnvConfig:
    """Composite experiment config consumed by the Isaac Sim extensions.

    Attributes:
        sim: Simulation-level settings (physics, rendering, gravity).
        drone: Drone spawner configuration.
        switches: Light-switch spawner configuration.
        px4: PX4 SITL bridge configuration.
    """

    sim: SimSettings
    drone: DroneConfig
    switches: LightSwitchConfig
    px4: Px4BridgeConfig
