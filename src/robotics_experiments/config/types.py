from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

__all__ = [
    "SimSettings",
    "DroneConfig",
    "BoundingBox",
    "RosQosConfig",
    "LightSwitchStatePublisherConfig",
    "LightSwitchConfig",
    "LockstepConfig",
    "Px4BridgeConfig",
    "VehicleConfig",
    "ResolvedVehicle",
    "IsaacLabConfig",
    "RoutineConfig",
    "IntegrationTestConfig",
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
class LockstepConfig:
    """Tunables for the PX4 lockstep exchange (the ``px4.lockstep`` group).

    Maps onto the library :class:`robotics_experiments.px4_bridge.LockstepSettings`
    at construction time, with ``physics_rate_hz`` injected from ``sim.physics_hz``
    (the single source of truth for the physics/sensor cadence, so it is not a
    standalone knob here).

    Attributes:
        lockstep_timeout_s: Hard deadline on receiving an actuator message each
            steady-state tick. On timeout the port returns zeros and Isaac keeps
            ticking - required so a stalled flight stack does not freeze the
            simulator.
        first_heartbeat_timeout_s: Hard deadline on receiving PX4's first
            heartbeat after the port starts. On timeout, the bridge marks the
            port disconnected instead of waiting indefinitely.
        bootstrap_prime_ticks: Number of priming HIL_SENSOR sends allowed before
            the bridge blocks for PX4's first HIL_ACTUATOR_CONTROLS. During PX4
            boot the controller needs a few sensor packets (~15) before it emits
            actuators; without a cap, Isaac free-runs and floods PX4, building a
            startup backlog that PX4 delivers with stale timestamps and that may
            never drain, so the bridge flies on stale motor commands. Capping the
            free-run bounds the backlog to ~this many ticks, keeping steady-state
            actuator lag at ~1 tick. Must exceed PX4's boot sensor need or the
            first-actuator wait times out and the bridge disconnects.
        first_actuator_timeout_s: Hard deadline on receiving PX4's first
            HIL_ACTUATOR_CONTROLS once the bridge starts blocking for it. PX4
            boots its controllers only after Isaac connects and that boot competes
            with Isaac for the host CPU, so under load it can take well over the
            nominal couple of seconds. This wait is a strict per-tick lockstep
            block, so a generous deadline patiently waits out a slow boot without
            building a startup backlog; only a dead PX4 trips it. Much larger than
            the per-tick lockstep_timeout_s, which is used only after the first
            actuator arrives.
        max_sustained_actuator_lag_s: Safety tripwire on steady-state actuator
            lag (current sim time minus the newest applied actuator timestamp).
            This is a regression net, not a happy-path control: strict blocking
            lockstep already keeps the lag at ~1 tick, prime-then-block prevents
            the startup backlog, and a PX4 stall is caught by lockstep_timeout_s.
            It exists to catch the one remaining failure mode -- a regression that
            reintroduces a stale-but-still-flowing actuator backlog (PX4 keeps
            answering, but with persistently old timestamps), which would otherwise
            silently fly stale motor commands. On exceed the bridge fails closed
            with a distinct "actuator lag exceeded sustained limit" reason instead
            of surfacing as a confusing attitude divergence. Set above the brief
            post-bootstrap drain transient and below the lag that destabilizes
            flight.
        heartbeat_rate_hz: Wall-clock rate for bridge-originated MAVLink heartbeat
            messages.
    """

    lockstep_timeout_s: float
    first_heartbeat_timeout_s: float
    bootstrap_prime_ticks: int
    first_actuator_timeout_s: float
    max_sustained_actuator_lag_s: float
    heartbeat_rate_hz: float


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
        connection_baseport: Base TCP port for the simulator MAVLink API. The
            resolved per-vehicle port is ``connection_baseport + vehicle.id``
            (see :class:`ResolvedVehicle.mavlink_port`); this base is shared by
            every vehicle and the per-vehicle offset is derived from ``id``.
        gcs_baseport: Base UDP port for the per-drone GCS MAVLink link. The
            resolved per-vehicle port is ``gcs_baseport + vehicle.id`` (see
            :class:`ResolvedVehicle.gcs_port`), parallel to
            ``connection_baseport``. Used by control/test code (the standalone
            takeoff scenario's GCS endpoint); the bridge itself does not use it.
        lockstep: Lockstep exchange tunables (timeouts, bootstrap priming,
            actuator-lag tripwire, heartbeat rate). See :class:`LockstepConfig`.
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
            ``{time}`` (``HH_MM_SS``), ``{timestamp}`` (``YYYY-MM-DD/HH_MM_SS``),
            and ``{vehicle}`` (the per-vehicle ``id``) so bridge logs can
            be paired with PX4 ULog files without appending across runs and so
            multiple vehicles never overwrite each other's logs.
        debug_log_level: Python logging level name used by the bridge file log.
        sensor_debug_log_interval: Emit one detailed HIL_SENSOR payload log
            every N sensor sends after the initial startup samples.
        max_tilt_angle_deg: Maximum allowed tilt angle in degrees between the
            drone's body-Z axis and world-Z. When exceeded, the bridge zeros
            all rotor forces to prevent inverted thrust after a crash. Set to
            zero or a negative value to disable.
        rotor_spin_visual_scale: Cosmetic-only factor that scales the real
            per-rotor angular velocity into a non-aliasing on-screen rotor-spin
            rate. Visual only - it never affects forces or the PX4 lockstep.
    """

    enabled: bool
    articulation_path: str
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
    gcs_baseport: int
    lockstep: LockstepConfig
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
    rotor_spin_visual_scale: float


@dataclass
class VehicleConfig:
    """One drone instance: its asset/spawn and its PX4 bridge config.

    The pure authored config object. ``drone`` and ``px4`` are fully resolved
    (the global ``EnvConfig`` defaults merged with the per-vehicle override).
    ``spawn_xy`` is the world X/Y; Z is computed from asset bounds at spawn; yaw
    is 0. ``id`` is the sole authored identity: the MAVLink port, PX4 ``-i``
    index, and per-vehicle log token are all derived from it (see
    :class:`ResolvedVehicle`), never hand-set.

    Attributes:
        id: Stable, unique vehicle identifier within the scene; the single
            source of every derived per-vehicle runtime value.
        spawn_xy: World ``(x, y)`` spawn position; Z comes from asset bounds.
        drone: Fully resolved per-vehicle drone spawner configuration.
        px4: Fully resolved per-vehicle PX4 bridge configuration.
    """

    id: int
    spawn_xy: Tuple[float, float]
    drone: DroneConfig
    px4: Px4BridgeConfig


@dataclass(frozen=True)
class ResolvedVehicle:
    """One fully resolved vehicle: pure config plus derived runtime fields.

    Produced by the single enrichment step ``config.load.resolve_fleet`` from a
    :class:`VehicleConfig`. Every consumer (Isaac orchestration, the bridge, and
    the host-side PX4 launch) reads the derived fields here instead of
    recomputing ``connection_baseport + id`` ad hoc. ``id`` is the sole identity
    and drives every derived value.

    Attributes:
        id: Stable, unique vehicle identifier (the sole authored identity).
        spawn_xy: World ``(x, y)`` spawn position; Z comes from asset bounds.
        drone: Fully resolved per-vehicle drone spawner configuration.
        px4: Fully resolved per-vehicle PX4 bridge configuration.
        mavlink_port: Simulator MAVLink TCP port for this vehicle, derived as
            ``px4.connection_baseport + id``.
        px4_instance_index: PX4 ``-i`` instance index, equal to ``id`` (so the
            PX4-side port shift matches ``mavlink_port`` by construction).
        gcs_port: Per-drone GCS MAVLink link UDP port, derived as
            ``px4.gcs_baseport + id``. Used by control/test code (the
            standalone takeoff scenario's GCS endpoint); the bridge itself does
            not use it. Folding it here keeps ``ResolvedVehicle`` the single home
            for every port the project derives from ``id``.
        debug_log_path: ``px4.debug_log_path`` with the ``{vehicle}`` token
            rendered to ``id``; ``{date}`` / ``{time}`` tokens stay for runtime.
        prim_name: ``drone.prim_name`` passthrough; the full USD ``prim_path``
            stays runtime-derived from the live stage default prim in
            ``spawn_drone`` and flows to the bridge as ``articulation_path``.
    """

    id: int
    spawn_xy: Tuple[float, float]
    drone: DroneConfig
    px4: Px4BridgeConfig
    mavlink_port: int
    px4_instance_index: int
    gcs_port: int
    debug_log_path: str
    prim_name: str


@dataclass
class IsaacLabConfig:
    """Configuration for Isaac Lab workflow skeletons.

    Attributes:
        drone_hover_episode_length_s: Episode timeout for the DirectRLEnv
            drone-hover scaffold.
        drone_hover_num_envs: Number of parallel envs used by the scaffold
            unless overridden by Isaac Lab tooling.
        drone_hover_env_spacing: Distance between cloned env origins in meters.
        drone_hover_action_scale: Scale applied to normalized placeholder
            actions before future dynamics consume them.
    """

    drone_hover_episode_length_s: float
    drone_hover_num_envs: int
    drone_hover_env_spacing: float
    drone_hover_action_scale: float


@dataclass
class IntegrationTestConfig:
    """Tunables for the standalone PX4 takeoff/hover integration test.

    Single source of truth for the integration test's command-flow timeouts,
    takeoff target, and the flight-envelope gates the test asserts (previously
    module-level constants in the standalone test). This is test-harness config:
    it is composed only by the ``integration_test_*`` experiments and is ``None``
    in the default runtime experiment (see :attr:`EnvConfig.integration_test`).
    The same test drives one drone (single-vehicle baseline) or N drones
    (multi-vehicle takeoff); each gate is evaluated per drone against its own
    spawn position. The scene is not here -- it is a top-level per-experiment
    field (:attr:`EnvConfig.scene`); the GCS link baseport lives in the ``px4``
    group (:attr:`Px4BridgeConfig.gcs_baseport`).

    Attributes:
        gcs_heartbeat_period_s: Minimum wall-clock spacing between GCS heartbeats.
        ground_settle_s: Sim-time the drone settles on the ground before the
            command flow starts.
        heartbeat_timeout_s: Wall-clock deadline for each handshake wait (bridge
            heartbeat, GCS heartbeat, takeoff telemetry).
        prearm_settle_s: Sim-time the PX4 estimator is allowed to settle before
            arming.
        command_ack_timeout_s: Wall-clock deadline for a MAVLink COMMAND_ACK.
        takeoff_altitude_m: Climb commanded in MAV_CMD_NAV_TAKEOFF.
        min_takeoff_climb_m: Minimum climb (m) that counts as airborne before the
            hover envelope is evaluated.
        takeoff_timeout_s: Sim-time budget to reach ``min_takeoff_climb_m``.
        hover_settle_s: Sim-time after reaching altitude before hover sampling.
        hover_observation_s: Sim-time the hover envelope is sampled.
        hover_max_altitude_range_m: Max allowed spread of hover climb height.
        hover_max_abs_vertical_speed_m_s: Max allowed hover vertical speed.
        hover_max_horizontal_range_m: Max allowed hover horizontal bbox diagonal.
        hover_max_abs_horizontal_speed_m_s: Max allowed hover horizontal speed.
        max_tilt_deg: Continuous tilt gate during takeoff.
        max_lateral_drift_m: Continuous lateral-drift gate during takeoff.
        max_vertical_speed_m_s: Continuous upward-speed divergence gate.
        max_downward_speed_m_s: Continuous downward-speed divergence gate
            (negative).
        max_angular_speed_rad_s: Continuous angular-speed divergence gate.
    """

    gcs_heartbeat_period_s: float
    ground_settle_s: float
    heartbeat_timeout_s: float
    prearm_settle_s: float
    command_ack_timeout_s: float
    takeoff_altitude_m: float
    min_takeoff_climb_m: float
    takeoff_timeout_s: float
    hover_settle_s: float
    hover_observation_s: float
    hover_max_altitude_range_m: float
    hover_max_abs_vertical_speed_m_s: float
    hover_max_horizontal_range_m: float
    hover_max_abs_horizontal_speed_m_s: float
    max_tilt_deg: float
    max_lateral_drift_m: float
    max_vertical_speed_m_s: float
    max_downward_speed_m_s: float
    max_angular_speed_rad_s: float


@dataclass
class RoutineConfig:
    """High-level flight routine executed by the live ``drone_spawner`` extension.

    The single source of truth for whether and what the live runtime flies. It is
    composed into every experiment (always present on :class:`EnvConfig`, unlike
    the optional ``integration_test`` harness group), and the rule is simply:
    a **non-empty** ``steps`` list makes the extension build a control driver that
    arms/takes-off the fleet on Play, while an **empty** ``steps`` list means
    purely manual operation (no driver is constructed). There is no separate
    enable flag -- ``steps`` is the switch.

    This is the *control* layer's config, parallel to the bridge's ``px4`` group:
    the GCS link baseport lives in the ``px4`` group
    (:attr:`Px4BridgeConfig.gcs_baseport`); the per-vehicle ``gcs_port`` the driver
    talks is derived from it (see :class:`ResolvedVehicle`).

    Attributes:
        steps: Ordered step descriptors authored in YAML, each a mapping with a
            ``kind`` (``arm`` or ``takeoff``; ``takeoff`` carries ``climb_m``).
            Empty means manual (no routine driver). Typed ``List[Any]`` so Hydra's
            structured-config validation accepts the partial per-step mappings
            during ``compose()`` (the same escape hatch as ``vehicles``); the
            driver maps them to a pure ``VehicleRoutine`` via
            ``commanding.build_vehicle_routine``.
        gcs_heartbeat_period_s: Minimum wall-clock spacing between GCS heartbeats.
        command_ack_timeout_s: Sim-time budget for each routine step's COMMAND_ACK.
        bridge_heartbeat_timeout_s: Wall-clock deadline to wait for every bridge
            to connect before the routine starts.
        handshake_timeout_s: Wall-clock deadline to learn the PX4 target system on
            every GCS link.
        prearm_settle_s: Sim-time the PX4 estimator is allowed to settle before
            arming.
        takeoff_target_timeout_s: Wall-clock deadline to receive each drone's
            global-position telemetry (the takeoff target) before commanding.
    """

    steps: List[Any]
    gcs_heartbeat_period_s: float
    command_ack_timeout_s: float
    bridge_heartbeat_timeout_s: float
    handshake_timeout_s: float
    prearm_settle_s: float
    takeoff_target_timeout_s: float


@dataclass
class EnvConfig:
    """Composite experiment config consumed by the Isaac Sim extensions.

    Attributes:
        sim: Simulation-level settings (physics, rendering, gravity).
        drone: Drone spawner configuration.
        switches: Light-switch spawner configuration.
        px4: PX4 SITL bridge configuration.
        isaac_lab: Isaac Lab workflow configuration.
        scene: USD scene to load for this experiment. ``@``-prefixed paths are
            repository-relative. Every experiment (test or not) sets a scene;
            the standalone runner uses it when not given an explicit ``--scene``.
        vehicles: Multi-vehicle scaffolding. Each entry merges its ``drone`` /
            ``px4`` partial over the global ``drone`` / ``px4`` defaults above.
            The list is the only spawn path (one vehicle = a one-element list;
            an empty list is a valid empty fleet).

            Typing note: the annotation is intentionally ``List[Any]`` so Hydra's
            structured-config validation accepts the *partial* YAML entries
            during ``compose()``. The loader replaces the raw dicts with a
            fully enriched, validated ``list[ResolvedVehicle]`` (see
            ``config.load.resolve_fleet``), which is the public/runtime type.
        routine: High-level flight routine for the live ``drone_spawner``
            extension (see :class:`RoutineConfig`). Always present; an empty
            ``routine.steps`` means manual operation, a non-empty list makes the
            runtime arm/take-off the fleet on Play. This is the runtime
            counterpart of the test-only ``integration_test`` group.
        integration_test: Standalone integration-test harness tunables
            (command-flow timeouts, takeoff target, flight-envelope gates).
            ``None`` for the default runtime experiment; populated only by the
            ``integration_test_*`` experiments that compose the
            ``integration_test`` group. The ``None`` value is the genuinely
            distinct state "this experiment carries no test harness", not a
            routine optional.
    """

    sim: SimSettings
    drone: DroneConfig
    switches: LightSwitchConfig
    px4: Px4BridgeConfig
    isaac_lab: IsaacLabConfig
    scene: str
    vehicles: List[Any]
    routine: RoutineConfig
    integration_test: Optional[IntegrationTestConfig] = None
