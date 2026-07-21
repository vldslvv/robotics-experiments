from dataclasses import dataclass
from typing import List, Literal, NotRequired, Required, TypedDict

from robotics_experiments.config._immutability import _FreezableConfig

__all__ = ["DriverConfig", "RoutineConfig", "RoutineStepConfig"]


class RoutineStepConfig(TypedDict, total=False):
    """Typed YAML descriptor for one high-level routine step."""

    kind: Required[Literal["arm", "takeoff", "hover", "forward", "goto", "policy", "land"]]
    climb_m: NotRequired[float]
    min_climb_m: NotRequired[float]
    hold_s: NotRequired[float]
    distance_m: NotRequired[float]
    timeout_s: NotRequired[float]
    x: NotRequired[float]
    y: NotRequired[float]
    z: NotRequired[float]
    position_tol_m: NotRequired[float]
    heading_tol_deg: NotRequired[float]
    artifact: NotRequired[str]
    name: NotRequired[str]
    target: NotRequired[List[float]]


@dataclass
class RoutineConfig(_FreezableConfig):
    """High-level flight routine flown in every workflow over the GCS link.

    The single source of truth for what is flown and how a good flight is
    verified. It is composed beneath every :class:`VehicleConfig`. Every
    bridge-enabled vehicle receives a control driver; an empty ``steps`` list
    completes without sending commands and leaves manual control available.

    The *same* routine runs in the live extension (``make run-all``), a regular
    standalone run, and the strict standalone integration test; only the driver's
    reaction to a flight-envelope breach differs (the strict integration test
    raises, the forgiving live/regular run logs and stops). The verification is
    the routine's own job: a takeoff step completes on a real climb and a hover
    step on a sustained stable hover, and the ``max_*`` / ``hover_max_*`` gates
    below are wired by the command driver's private routine builder onto the
    flight steps as continuous invariants whose breach fails the routine.

    This is the control layer's config, parallel to the vehicle's ``bridge`` group:
    the GCS link baseport lives in that bridge config
    (:attr:`Px4BridgeConfig.gcs_baseport`); the per-vehicle ``gcs_port`` the driver
    talks is derived from it (see :class:`VehicleConfig`).

    Attributes:
        steps: Ordered step descriptors authored in YAML, each a mapping with a
            ``kind`` (``arm``; ``takeoff`` carrying ``climb_m`` + ``min_climb_m``;
            ``hover`` carrying ``hold_s``; ``forward`` carrying ``distance_m`` +
            ``timeout_s``; ``goto`` carrying absolute world-ENU ``x`` + ``y`` +
            ``z`` + ``timeout_s`` and optional ``position_tol_m`` /
            ``heading_tol_deg``; ``land`` carrying ``timeout_s``). Empty sends no
            flight commands. The resolved graph exposes read-only typed mappings.
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

    steps: List[RoutineStepConfig]
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
class DriverConfig(_FreezableConfig):
    """Routine-execution driver heuristics, separate from the routine itself.

    The ``routine`` group says *what* is flown and how a good flight is
    verified; this group tunes *how the driver executes it*: the geometric
    epsilon of the ``goto`` heading logic and the driver's wall-clock backstops
    (last-resort deadlines that only fire when the simulation itself stalls, on
    top of the sim-time budgets in the routine). Global, not per vehicle.

    Attributes:
        control_rate_hz: Fleet routine-driver rate derived from physics ticks.
        offboard_setpoint_rate_hz: OFFBOARD setpoint transmission rate.
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

    control_rate_hz: float
    offboard_setpoint_rate_hz: float
    goto_min_heading_dist_m: float
    prearm_backstop_floor_s: float
    prearm_backstop_multiplier: float
    drive_budget_multiplier: float
