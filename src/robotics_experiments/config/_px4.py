from dataclasses import dataclass, field
from typing import List, Tuple

from robotics_experiments.config._commanding import RoutineConfig
from robotics_experiments.config._immutability import _FreezableConfig
from robotics_experiments.config._scene import DroneConfig, RosVehicleConfig

__all__ = [
    "LockstepConfig",
    "Px4BridgeConfig",
    "Px4SitlLaunchConfig",
    "ResolvedVehicle",
    "VehicleConfig",
]


@dataclass
class LockstepConfig(_FreezableConfig):
    """Tunables for the PX4 lockstep exchange (the ``px4.lockstep`` group).

    Maps onto the library
    :class:`robotics_experiments.px4_bridge.LockstepSettings` at construction
    time, with ``physics_rate_hz`` injected from ``sim.physics_hz`` (the single source
    of truth for the physics/sensor cadence, so it is not a standalone knob here).

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
class Px4BridgeConfig(_FreezableConfig):
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
        max_thrust_per_rotor_g: Authored, vendor-citable per-rotor static max
            thrust in grams-force (e.g. the motor+prop datasheet figure). The
            single source for the thrust scale; ``rotor_constant`` is derived
            from it, not authored.
        rolling_moment_coefficient, min_rotor_velocity, max_rotor_velocity:
            Pegasus quadratic thrust/torque curve parameters. Rotor count and
            spin direction come from authored USD rotor metadata on the spawned
            drone asset.
        rotor_constant: Pegasus quadratic thrust coefficient per rotor
            (``T_i = rotor_constant_i * omega_i^2``). NOT authored -- it is
            computed during config package loading (and for the global default
            in :func:`robotics_experiments.config.get_config`) from
            ``max_thrust_per_rotor_g``, ``max_rotor_velocity``, and standard
            gravity, so a datasheet thrust change is a single-line edit.
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
            rate. The source velocity is the resolved rotor omega after actuator
            scaling and min/max rotor clipping, so KV/battery effects should be
            represented by ``max_rotor_velocity`` unless a future config
            generator computes that value. Visual only - it never affects forces
            or the PX4 lockstep.
    """

    enabled: bool
    articulation_path: str
    input_offset: List[float]
    input_scaling: List[float]
    zero_position_armed: List[float]
    max_thrust_per_rotor_g: float
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

    # Computed (not authored): derived from max_thrust_per_rotor_g +
    # max_rotor_velocity + standard gravity in the config loader. Kept last so the
    # authored fields above stay required. See the class docstring.
    rotor_constant: List[float] = field(default_factory=list)


@dataclass
class VehicleConfig(_FreezableConfig):
    """One drone instance: its asset/spawn and its PX4 bridge config.

    The pure authored config object. ``drone`` and ``px4`` are fully resolved
    (the global ``EnvConfig`` defaults merged with the per-vehicle override).
    ``spawn_xy`` is the world X/Y; Z is computed from asset bounds at spawn; yaw
    is 0. ``id`` is the PX4 numeric identity: the MAVLink port, PX4 ``-i``
    index, and per-vehicle log token are all derived from it (see
    :class:`ResolvedVehicle`), never hand-set. ``ros.namespace`` is the explicit
    operator-facing ROS identity.

    Attributes:
        id: Stable, unique numeric vehicle identifier within the scene; the
            source of the derived PX4 runtime values.
        spawn_xy: World ``(x, y)`` spawn position; Z comes from asset bounds.
        drone: Fully resolved per-vehicle drone spawner configuration.
        px4: Fully resolved per-vehicle PX4 bridge configuration.
        ros: Per-vehicle ROS identity.
        routine: Fully resolved per-vehicle flight routine (the global
            ``routine`` with this vehicle's partial merged over it). A non-empty
            ``routine.steps`` makes this drone fly; empty means manual.
    """

    id: int
    spawn_xy: Tuple[float, float]
    drone: DroneConfig
    px4: Px4BridgeConfig
    ros: RosVehicleConfig
    routine: "RoutineConfig"


@dataclass(frozen=True)
class ResolvedVehicle:
    """One fully resolved vehicle: pure config plus derived runtime fields.

    Produced by the config package's single enrichment step from a
    :class:`VehicleConfig`. Every consumer (Isaac orchestration, the bridge, and
    the host-side PX4 launch) reads the derived fields here instead of
    recomputing ``connection_baseport + id`` ad hoc. ``id`` drives the PX4
    numeric runtime values; ``ros_namespace`` is the ROS graph identity.

    Attributes:
        id: Stable, unique numeric vehicle identifier.
        spawn_xy: World ``(x, y)`` spawn position; Z comes from asset bounds.
        drone: Fully resolved per-vehicle drone spawner configuration.
        px4: Fully resolved per-vehicle PX4 bridge configuration.
        routine: Fully resolved per-vehicle flight routine; a non-empty
            ``routine.steps`` makes this drone fly its routine, empty = manual.
        mavlink_port: Simulator MAVLink TCP port for this vehicle, derived as
            ``px4.connection_baseport + id``.
        px4_instance_index: PX4 ``-i`` instance index, equal to ``id`` (so the
            PX4-side port shift matches ``mavlink_port`` by construction).
        gcs_port: Per-drone GCS MAVLink link UDP port, derived as
            ``px4.gcs_baseport + id``. Used by control/test code (the
            standalone takeoff scenario's GCS endpoint); the bridge itself does
            not use it. Folding it here keeps ``ResolvedVehicle`` the single home
            for every port the project derives from ``id``.
        ros_namespace: Authored absolute ROS namespace for per-drone topics,
            nodes, services, actions, and parameters.
        ros_frame_prefix: Relative frame prefix derived from
            ``ros_namespace`` for future per-drone TF/message frame IDs. Nested
            namespaces use underscores here (``/fleet_a/x500_01`` ->
            ``fleet_a_x500_01``).
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
    routine: "RoutineConfig"
    mavlink_port: int
    px4_instance_index: int
    gcs_port: int
    ros_namespace: str
    ros_frame_prefix: str
    debug_log_path: str
    prim_name: str


@dataclass(frozen=True)
class Px4SitlLaunchConfig:
    """PX4 SITL launch airframe derived from a resolved vehicle's config.

    Produced by the config loader from a :class:`ResolvedVehicle`. The drone USD
    asset selects the PX4 simulator airframe so each asset launches the PX4
    control allocation generated from the same ``configs/airframes`` source as
    the USD. Threaded by the host-side launch scripts into
    each PX4 container as ``PX4_INSTANCE`` / ``PX4_SIM_MODEL`` /
    ``PX4_SYS_AUTOSTART``.

    Attributes:
        instance_index: PX4 ``-i`` instance index (the resolved
            ``px4_instance_index``); the per-container port offset.
        sim_model: ``PX4_SIM_MODEL`` selecting the external-MAVLink SITL
            airframe (``none_quad_x`` for the generic Quad X asset,
            ``none_hexa_x`` for the Hex X asset); both still start
            ``simulator_mavlink``.
        sys_autostart: ``PX4_SYS_AUTOSTART`` canonical PX4 copter airframe id
            (``4001`` for generic Quad X, ``6001`` for Hex X).
    """

    instance_index: int
    sim_model: str
    sys_autostart: int
