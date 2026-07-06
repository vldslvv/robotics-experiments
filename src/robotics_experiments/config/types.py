from dataclasses import dataclass, field
from typing import Any, List, Tuple

__all__ = [
    "SimSettings",
    "DroneConfig",
    "BoundingBox",
    "RosQosConfig",
    "RosVehicleConfig",
    "LightSwitchStatePublisherConfig",
    "LightSwitchConfig",
    "LockstepConfig",
    "Px4BridgeConfig",
    "VehicleConfig",
    "ResolvedVehicle",
    "Px4SitlLaunchConfig",
    "PressActionConfig",
    "PressHoverTaskConfig",
    "PressHandoffConfig",
    "PressTaskConfig",
    "RlTrainingConfig",
    "RlConfig",
    "RoutineConfig",
    "DriverConfig",
    "EnvConfig",
]


@dataclass
class SimSettings:
    """Simulation settings applied by the fleet orchestrator's ``apply_settings``.

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
class RosVehicleConfig:
    """Per-vehicle ROS graph identity.

    Attributes:
        namespace: Absolute ROS namespace for this vehicle, such as
            ``"/x500_01"``. Authored per vehicle so real fleet names can stay
            stable independently of the numeric runtime ``id`` used by PX4.
    """

    namespace: str


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
        press_confirm_steps: Consecutive steps the button must stay pressed before
            a toggle is confirmed (the light-state latch / confirmation window).
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
    press_confirm_steps: int
    min_spawn_dist: float
    max_spawn_retries: int
    lamp_on_intensity: float
    spawn_bounds: List[BoundingBox]
    state_publisher: LightSwitchStatePublisherConfig


@dataclass
class LockstepConfig:
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
class VehicleConfig:
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


@dataclass
class PressActionConfig:
    """RL action-level (``kind``) selection plus its controller tunables.

    The ``kind`` picks the in-Lab actuation model (see
    :mod:`robotics_experiments.rl`): ``velocity`` (cascaded PD:
    body-frame velocity + yaw rate) or ``ctbr`` (collective thrust + body-rate
    setpoint). The gains/limits below are a superset covering both; the selected
    ``kind`` reads the subset it needs, so switching is a one-line ``kind`` edit.

    Attributes:
        kind: Action level selecting the actuation model (``velocity`` /
            ``ctbr``; other levels are reserved and raise if selected).
        action_dim: Length of the normalized policy action vector (4 for both
            ``velocity`` = ``[vx, vy, vz, yaw_rate]`` and ``ctbr`` =
            ``[thrust, wx, wy, wz]``).
        max_body_velocity_mps: Body-frame linear-velocity command scale (m/s) a
            normalized ``velocity`` action of magnitude 1 maps to.
        max_yaw_rate_radps: Yaw-rate command scale (rad/s) for a normalized
            ``velocity`` action.
        max_body_rate_radps: Body-rate command scale (rad/s) for a normalized
            ``ctbr`` action.
        max_tilt_cmd_deg: Saturation on the commanded tilt of the desired thrust
            axis in the ``velocity`` cascade, degrees.
        hover_thrust_fraction: Collective thrust at hover as a fraction of the
            summed per-rotor static max thrust; centers the ``ctbr`` collective
            command and the ``velocity`` thrust feed-forward.
        controller_kp_vel: Per-axis P gain on world-frame velocity error
            (``velocity`` cascade), length 3.
        controller_kd_vel: Per-axis damping gain on world-frame velocity
            (``velocity`` cascade), length 3.
        controller_kp_att: Per-axis P gain mapping geometric attitude error to a
            body-rate setpoint (``velocity`` cascade), length 3.
        controller_kp_rate: Per-axis P gain mapping body-rate error to desired
            body torque (both models), length 3.
    """

    kind: str
    action_dim: int
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_body_rate_radps: float
    max_tilt_cmd_deg: float
    hover_thrust_fraction: float
    controller_kp_vel: List[float]
    controller_kd_vel: List[float]
    controller_kp_att: List[float]
    controller_kp_rate: List[float]


@dataclass
class PressHoverTaskConfig:
    """The hover task definition: reward weights, termination, and reset ranges.

    The task half of the ``task definition -> policy`` contract for the hover
    task (`Isaac-Drone-Hover-Direct-v0`): authoring these values IS defining the
    task; the env module carries no task-tuning literals.

    Attributes:
        reward_pos_weight: Penalty weight on the position error norm, per meter.
        reward_linvel_weight: Penalty weight on linear speed, per m/s.
        reward_angvel_weight: Penalty weight on angular speed, per rad/s.
        reward_alive: Constant per-step alive bonus.
        crash_tilt_cos: Terminate when the body-Z world-Z alignment (cosine)
            drops below this (0.0 = more than 90 deg tilt).
        min_altitude_m: Terminate below this altitude, meters.
        max_position_error_m: Terminate beyond this distance from the hover
            target, meters.
        reset_xy_half_m: Half-extent of the horizontal start-pose randomization
            box around the hover target, meters.
        reset_z_half_m: Half-extent of the vertical start-pose randomization box
            around the hover target, meters.
    """

    reward_pos_weight: float
    reward_linvel_weight: float
    reward_angvel_weight: float
    reward_alive: float
    crash_tilt_cos: float
    min_altitude_m: float
    max_position_error_m: float
    reset_xy_half_m: float
    reset_z_half_m: float


@dataclass
class PressHandoffConfig:
    """Thresholds for the nav -> contact handoff gate (``rl.press.handoff_ready``).

    All checks are over the task-relative :class:`PressObservation` (contact
    frame); the gate decides whether the contact behavior may start.

    Attributes:
        pose_confidence_min: Minimum pose confidence, [0, 1].
        pose_age_max_s: Maximum pose-estimate age, seconds.
        switch_range_max_m: Maximum wall-normal range to the switch, meters.
        lateral_align_max_m: Maximum lateral (in-plate) alignment error, meters.
        wall_normal_speed_max_mps: Maximum speed along the press axis, m/s.
        lateral_speed_max_mps: Maximum in-plate speed, m/s.
        angular_speed_max_radps: Maximum body angular speed, rad/s.
        estimator_health_min: Minimum aggregate estimator health, [0, 1].
    """

    pose_confidence_min: float
    pose_age_max_s: float
    switch_range_max_m: float
    lateral_align_max_m: float
    wall_normal_speed_max_mps: float
    lateral_speed_max_mps: float
    angular_speed_max_radps: float
    estimator_health_min: float


@dataclass
class RlTrainingConfig:
    """PPO training hyperparameters + task selection for ``make train-rl``.

    Consumed by the ``rl.training`` adapter (rsl_rl today); no hyperparameter
    lives in Python.

    Attributes:
        task_id: Gymnasium task id the entrypoint trains
            (``Isaac-Drone-Hover-Direct-v0`` for the F2b pipeline; F3 points it
            at the press task).
        policy_name: Required artifact namespace under ``.build/training``;
            for example ``hover`` writes ``.build/training/hover/latest``.
        max_iterations: Training iterations for a full run.
        num_steps_per_env: On-policy rollout length per env per iteration.
        save_interval: Checkpoint save interval, iterations.
        learning_rate: PPO learning rate (adaptive schedule).
        gamma: Discount factor.
        lam: GAE lambda.
        entropy_coef: Entropy bonus coefficient.
        clip_param: PPO clip parameter.
        value_loss_coef: Value-loss coefficient.
        num_learning_epochs: PPO epochs per iteration.
        num_mini_batches: Minibatches per epoch.
        desired_kl: Adaptive-schedule KL target.
        max_grad_norm: Gradient clipping norm.
        init_noise_std: Initial action noise std (normalized action units).
        actor_hidden_dims: Actor MLP hidden layer sizes.
        critic_hidden_dims: Critic MLP hidden layer sizes.
        activation: MLP activation name (e.g. ``elu``).
        seed: Training seed (env reset + torch).
    """

    task_id: str
    policy_name: str
    max_iterations: int
    num_steps_per_env: int
    save_interval: int
    learning_rate: float
    gamma: float
    lam: float
    entropy_coef: float
    clip_param: float
    value_loss_coef: float
    num_learning_epochs: int
    num_mini_batches: int
    desired_kl: float
    max_grad_norm: float
    init_noise_std: float
    actor_hidden_dims: List[int]
    critic_hidden_dims: List[int]
    activation: str
    seed: int


@dataclass
class PressTaskConfig:
    """Press-task-specific RL config.

    Attributes:
        hover_task: The hover task definition (reward/termination/reset).
        handoff: Thresholds for the nav -> contact handoff gate.
    """

    hover_task: PressHoverTaskConfig
    handoff: PressHandoffConfig


@dataclass
class RlConfig:
    """Shared Isaac Lab RL scale + the pluggable action level.

    Replaces the retired ``isaac_lab`` group as the single source of truth for
    the DirectRLEnv Lab-scale tunables. ``decimation`` is not authored here: the
    env derives it as ``round(sim.physics_hz / policy_rate_hz)`` so the physics
    cadence stays single-sourced on ``sim.physics_hz``.

    Attributes:
        action: The RL action-level selection and controller tunables.
        training: PPO hyperparameters + task selection for training runs.
        press: Press-task-specific config.
        num_envs: Number of parallel cloned envs.
        env_spacing: Distance between cloned env origins, meters.
        episode_length_s: Episode timeout, seconds.
        policy_rate_hz: RL policy decision rate; the physics decimation is
            ``round(sim.physics_hz / policy_rate_hz)``.
    """

    action: PressActionConfig
    training: RlTrainingConfig
    press: PressTaskConfig
    num_envs: int
    env_spacing: float
    episode_length_s: float
    policy_rate_hz: float


@dataclass
class RoutineConfig:
    """High-level flight routine flown in every workflow over the GCS link.

    The single source of truth for whether and what is flown, and how a good
    flight is verified. It is composed into every experiment (always present on
    :class:`EnvConfig`), and the rule is simply: a **non-empty** ``steps`` list
    makes a control driver arm/take-off/verify the fleet, while an **empty**
    ``steps`` list means purely manual operation (no driver is constructed).
    There is no separate enable flag -- ``steps`` is the switch.

    The *same* routine runs in the live extension (``make run-all``), a regular
    standalone run, and the strict standalone integration test; only the driver's
    reaction to a flight-envelope breach differs (the strict integration test
    raises, the forgiving live/regular run logs and stops). The verification is
    the routine's own job: a takeoff step completes on a real climb and a hover
    step on a sustained stable hover, and the ``max_*`` / ``hover_max_*`` gates
    below are wired by the command driver's private routine builder onto the
    flight steps as continuous invariants whose breach fails the routine.

    This is the *control* layer's config, parallel to the bridge's ``px4`` group:
    the GCS link baseport lives in the ``px4`` group
    (:attr:`Px4BridgeConfig.gcs_baseport`); the per-vehicle ``gcs_port`` the driver
    talks is derived from it (see :class:`ResolvedVehicle`).

    Attributes:
        steps: Ordered step descriptors authored in YAML, each a mapping with a
            ``kind`` (``arm``; ``takeoff`` carrying ``climb_m`` + ``min_climb_m``;
            ``hover`` carrying ``hold_s``; ``forward`` carrying ``distance_m`` +
            ``timeout_s``; ``goto`` carrying absolute world-ENU ``x`` + ``y`` +
            ``z`` + ``timeout_s`` and optional ``position_tol_m`` /
            ``heading_tol_deg``; ``land`` carrying ``timeout_s``). Empty means manual
            (no routine driver). Typed ``List[Any]`` so Hydra's
            structured-config validation accepts the partial per-step mappings
            during ``compose()`` (the same escape hatch as ``vehicles``); the
            driver maps them to a pure routine object through private
            command-layer implementation.
        gcs_heartbeat_period_s: Minimum wall-clock spacing between GCS heartbeats.
        command_ack_timeout_s: Sim-time budget for each command step's COMMAND_ACK.
        bridge_heartbeat_timeout_s: Wall-clock deadline to wait for every bridge
            to connect before the routine starts.
        handshake_timeout_s: Wall-clock deadline to learn the PX4 target system on
            every GCS link.
        prearm_settle_s: Sim-time the PX4 estimator is allowed to settle before
            arming.
        takeoff_target_timeout_s: Wall-clock deadline to receive each drone's
            global-position telemetry (the takeoff target) before commanding.
        ground_settle_s: Sim-time the drone settles on the ground before the
            command flow starts (used by the standalone driver).
        takeoff_timeout_s: Sim-time budget for the takeoff step to climb to its
            ``min_climb_m``.
        hover_settle_s: Sim-time a hover step is allowed to stabilize before its
            ``hold_s`` window must be sustained (the hover step timeout is
            ``hover_settle_s + hold_s``).
        max_tilt_deg: Continuous tilt gate on the flight steps.
        max_lateral_drift_m: Continuous lateral-drift gate. Takeoff/hover treat
            it as radial horizontal excursion from the active step baseline;
            forward treats it as cross-track drift so commanded +X/east motion
            is allowed.
        max_vertical_speed_m_s: Continuous upward-speed divergence gate.
        max_downward_speed_m_s: Continuous downward-speed divergence gate
            (negative).
        max_angular_speed_rad_s: Continuous angular-speed divergence gate.
        hover_max_abs_vertical_speed_m_s: Hover-completion vertical-speed bound.
        hover_max_abs_horizontal_speed_m_s: Hover-completion horizontal-speed bound.
        goto_position_tol_m: Default radius within which a ``goto`` waypoint counts
            as reached (a per-step ``position_tol_m`` overrides it).
        goto_heading_tol_deg: Default heading tolerance for a ``goto`` to count as
            aligned with the leg direction (a per-step ``heading_tol_deg``
            overrides it).
        goto_cross_track_m: Continuous gate on a ``goto`` leg: maximum perpendicular
            deviation from the straight active step baseline to waypoint path.
        goto_offboard_prime_ticks: Number of OFFBOARD setpoints streamed before the
            mode is engaged (PX4 rejects OFFBOARD until a setpoint stream flows).
    """

    steps: List[Any]
    gcs_heartbeat_period_s: float
    command_ack_timeout_s: float
    bridge_heartbeat_timeout_s: float
    handshake_timeout_s: float
    prearm_settle_s: float
    takeoff_target_timeout_s: float
    ground_settle_s: float
    takeoff_timeout_s: float
    hover_settle_s: float
    max_tilt_deg: float
    max_lateral_drift_m: float
    max_vertical_speed_m_s: float
    max_downward_speed_m_s: float
    max_angular_speed_rad_s: float
    hover_max_abs_vertical_speed_m_s: float
    hover_max_abs_horizontal_speed_m_s: float
    goto_position_tol_m: float
    goto_heading_tol_deg: float
    goto_cross_track_m: float
    goto_offboard_prime_ticks: int


@dataclass
class DriverConfig:
    """Routine-execution driver heuristics, separate from the routine itself.

    The ``routine`` group says *what* is flown and how a good flight is
    verified; this group tunes *how the driver executes it*: the geometric
    epsilon of the ``goto`` heading logic and the driver's wall-clock backstops
    (last-resort deadlines that only fire when the simulation itself stalls, on
    top of the sim-time budgets in the routine). Global, not per vehicle.

    Attributes:
        goto_min_heading_dist_m: Below this horizontal baseline-to-target
            distance a ``goto`` leg has no meaningful heading (a near-vertical
            move): the streamed setpoint holds heading and the
            heading-alignment completion passes trivially.
        prearm_backstop_floor_s: Minimum wall-clock backstop for the
            PREARM_SETTLE phase (which is budgeted in sim time).
        prearm_backstop_multiplier: PREARM_SETTLE wall-clock backstop as a
            multiple of ``routine.prearm_settle_s``; the backstop is
            ``max(floor, prearm_settle_s * multiplier)``.
        drive_budget_multiplier: DRIVE-phase wall-clock backstop as a multiple
            of the routine's summed sim-time step budgets (plus handshake/ack
            slack).
    """

    goto_min_heading_dist_m: float
    prearm_backstop_floor_s: float
    prearm_backstop_multiplier: float
    drive_budget_multiplier: float


@dataclass
class EnvConfig:
    """Composite experiment config consumed by the Isaac Sim extensions.

    Attributes:
        sim: Simulation-level settings (physics, rendering, gravity).
        drone: Drone spawner configuration.
        switches: Light-switch spawner configuration.
        px4: PX4 SITL bridge configuration.
        rl: Shared RL training/scale/action config plus nested task config.
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
            fully enriched, validated ``list[ResolvedVehicle]``, which is the
            public/runtime type.
        routine: High-level flight routine plus its verification gates (see
            :class:`RoutineConfig`). Always present; an empty ``routine.steps``
            means manual operation, a non-empty list makes every workflow
            arm/take-off/verify the fleet. The integration test is just this
            routine driven in strict mode.
        driver: Routine-execution driver heuristics (see :class:`DriverConfig`):
            the ``goto`` heading epsilon and the driver's wall-clock backstops.
            Global (not merged per vehicle).
    """

    sim: SimSettings
    drone: DroneConfig
    switches: LightSwitchConfig
    px4: Px4BridgeConfig
    rl: RlConfig
    scene: str
    vehicles: List[Any]
    routine: RoutineConfig
    driver: DriverConfig
