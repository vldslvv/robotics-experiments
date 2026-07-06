"""The pure command plan and its stepper - the command-layer contract centerpiece.

A :class:`VehicleRoutine` is an ordered tuple of :class:`RoutineStep`s; each step
carries one optional :class:`MavlinkCommand`, a pure completion predicate over a
:class:`RoutineObservation` (and the active step's captured baseline position),
an optional sustained-hold window, and continuous invariant gates. :func:`decide`
is the pure transform at the heart of the layer - the command-layer analogue of
the thrust model's ``commands_to_dynamics``: given the routine, the current
:class:`RoutineState`, and an observation, it returns the next state and a
:class:`RoutineDecision` (``SEND`` / ``WAIT`` / ``ADVANCE`` / ``DONE`` /
``FAIL``). No sockets, no pymavlink, no Isaac - it is unit-testable with plain
dataclasses, and the side-effecting :class:`VehicleCommander` (see
``_commander.py``) is the only thing that turns a ``SEND`` decision into MAVLink.

The same routine verifies a takeoff in every workflow: a step's completion
predicate can require a physical state (climbed at least N metres, hovering
stable for N seconds), and its ``invariants`` are continuous flight-envelope
gates (tilt / drift / vertical+angular speed) whose breach fails the routine with
``failure_kind == FailureKind.ENVELOPE``. A strict driver (the standalone
integration test) raises on that breach; a forgiving driver (the live extension,
a regular standalone run) logs it and stops the routine. Command rejections and
step timeouts fail with ``FailureKind.COMMAND`` instead.

Timeouts and hold windows are measured in sim time (``observation.sim_time_us``),
not wall clock, so the stepper stays pure; the driver pairs this with its own
wall-clock guard to catch a fully stalled simulation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from robotics_experiments.commanding.commands import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    AttitudeRateSetpoint,
    MavlinkCommand,
    PositionSetpoint,
    VelocitySetpoint,
    arm,
    land,
    reposition,
    set_offboard_mode,
    takeoff,
)
from robotics_experiments.commanding.observation import (
    MAV_RESULT_TEMPORARILY_REJECTED,
    RoutineObservation,
)
from robotics_experiments.px4_bridge.frames import (
    rotate_vector_wxyz,
    synth_gps_from_enu,
    tilt_angle_deg,
)

if TYPE_CHECKING:
    from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = ["FailureKind"]

_EARTH_RADIUS_M = 6_378_137.0

# Library-neutral fallback for the ``goto`` heading epsilon: below this
# horizontal active-step-baseline-to-target distance a leg has no meaningful
# heading (a near-vertical move), so the setpoint holds heading (yaw NaN) and
# the heading-alignment predicate passes trivially. The project value lives in
# the Hydra ``driver`` group (``driver.goto_min_heading_dist_m``).
_DEFAULT_MIN_HEADING_DIST_M = 0.25

# A completion predicate reads the observation plus the active step's captured
# baseline position (ENU, set when each step begins); baseline is ``None`` only
# before a step has begun. An invariant gate returns a breach message when the
# continuous flight envelope is violated, or ``None`` when it holds.
_StepPredicate = Callable[[RoutineObservation, Optional[np.ndarray]], bool]
_Invariant = Callable[[RoutineObservation, Optional[np.ndarray]], Optional[str]]

# Any OFFBOARD setpoint payload a STREAM decision can carry; the commander
# dispatches on the concrete type to the matching port send method.
StreamSetpoint = Union[PositionSetpoint, VelocitySetpoint, AttitudeRateSetpoint]

# The injected per-tick pure function of a ``policy`` step: maps the live
# vehicle state and the step's world-ENU target to the OFFBOARD setpoint
# payload to stream (composed at the runtime layer from the observation
# packing, the loaded policy, and the deploy action mapping -- see
# ``rl.press.policy.build_policy_step_fn``).
PolicyStepFn = Callable[["VehicleState", np.ndarray], StreamSetpoint]


class DecisionKind(Enum):
    """The kind of a :class:`RoutineDecision`."""

    SEND = auto()
    STREAM = auto()
    WAIT = auto()
    ADVANCE = auto()
    DONE = auto()
    FAIL = auto()


class FailureKind(Enum):
    """Why a routine failed, so a driver can react in kind.

    ``COMMAND`` is a command rejection or a step timeout (the routine could not
    make progress); ``ENVELOPE`` is a flight-envelope invariant breach (the
    vehicle flew out of bounds); ``FAILSAFE`` is a PX4 failsafe system status on
    the GCS link; ``TIMEOUT`` is a driver phase exceeding its wall-clock
    backstop. The pure stepper emits only ``COMMAND`` / ``ENVELOPE``; the driver
    adds ``FAILSAFE`` / ``TIMEOUT`` for its own terminal conditions. The
    standalone strict driver maps ``ENVELOPE`` to its
    ``ExpectedInstabilityDetected``, ``FAILSAFE`` to ``Px4FailsafeDetected``,
    and the rest to a plain assertion.
    """

    COMMAND = auto()
    ENVELOPE = auto()
    FAILSAFE = auto()
    TIMEOUT = auto()


@dataclass(frozen=True)
class RoutineDecision:
    """The pure output of :func:`decide` for one tick.

    Exactly one ``kind``; ``command`` is set only for ``SEND``; ``setpoint`` only
    for ``STREAM``; ``reason`` and ``failure_kind`` only for ``FAIL``. Build
    instances through the classmethods.
    """

    kind: DecisionKind
    command: Optional[MavlinkCommand] = None
    setpoint: Optional[StreamSetpoint] = None
    reason: Optional[str] = None
    failure_kind: Optional[FailureKind] = None

    @classmethod
    def send(cls, command: MavlinkCommand) -> "RoutineDecision":
        return cls(DecisionKind.SEND, command=command)

    @classmethod
    def stream(cls, setpoint: StreamSetpoint) -> "RoutineDecision":
        return cls(DecisionKind.STREAM, setpoint=setpoint)

    @classmethod
    def wait(cls) -> "RoutineDecision":
        return cls(DecisionKind.WAIT)

    @classmethod
    def advance(cls) -> "RoutineDecision":
        return cls(DecisionKind.ADVANCE)

    @classmethod
    def done(cls) -> "RoutineDecision":
        return cls(DecisionKind.DONE)

    @classmethod
    def fail(
        cls, reason: str, *, failure_kind: FailureKind = FailureKind.COMMAND
    ) -> "RoutineDecision":
        return cls(DecisionKind.FAIL, reason=reason, failure_kind=failure_kind)


@dataclass(frozen=True)
class RoutineStep:
    """One step of a routine: a command, its completion predicate, and gates.

    Attributes:
        command: The :class:`MavlinkCommand` sent when the step begins, or
            ``None`` for a verification-only step (e.g. ``hover``) that only
            waits on its completion predicate and invariants. If a command-less
            step follows an OFFBOARD streaming step, the previous setpoint keeps
            streaming so PX4 does not trip offboard-loss failsafe while the
            verification step is active.
        is_complete: Pure predicate over a :class:`RoutineObservation` and the
            active step's captured baseline position; the step advances when it
            returns ``True`` (and, if ``hold_s`` is set, has held continuously).
        label: Short human-readable name used in logs and failure messages.
        timeout_s: Sim-time budget for the step; on exceed the routine fails with
            ``FailureKind.COMMAND``.
        retry_on_temporarily_rejected: When ``True``, a
            ``MAV_RESULT_TEMPORARILY_REJECTED`` ack for this command re-sends it
            instead of failing (PX4 returns this transiently before it is ready
            to arm). When ``False``, any non-accepted ack fails the routine.
        hold_s: Sim-time the completion predicate must hold *continuously* before
            the step advances (0 = instantaneous, the default). This expresses
            "hover stable for N seconds" as a step completion condition.
        invariants: Continuous flight-envelope gates evaluated every tick the
            step is active; the first breach fails the routine with
            ``FailureKind.ENVELOPE``.
        streaming: When set, this is a streamed OFFBOARD step (a ``goto``): the
            stepper primes setpoints, sends ``command`` once to engage offboard,
            then streams the spec's setpoint every tick until completion. When
            ``None`` (every other kind) the step uses the send-once-then-verify
            flow.
    """

    command: Optional[MavlinkCommand]
    is_complete: _StepPredicate
    label: str
    timeout_s: float
    retry_on_temporarily_rejected: bool = False
    hold_s: float = 0.0
    invariants: tuple[_Invariant, ...] = ()
    streaming: Optional["_StreamingSpec"] = None


@dataclass(frozen=True)
class _StreamingSpec:
    """The streamed-setpoint target for an OFFBOARD ``goto`` step.

    Attributes:
        target_enu: The absolute world-ENU waypoint ``(east, north, up)`` in
            metres, used to compute the per-tick heading and (with the captured
            baseline) the cross-track gate.
        latitude_deg, longitude_deg, altitude_m: The same waypoint precomputed
            (by :func:`build_vehicle_routine` via ``synth_gps_from_enu``) as the
            geodetic target the port streams, kept consistent by construction with
            the bridge's synthesized GPS.
        prime_ticks: How many setpoints to stream before engaging OFFBOARD (PX4
            rejects the mode until a setpoint stream is already flowing).
        min_heading_dist_m: Below this horizontal baseline-to-target distance
            the leg has no meaningful heading and the streamed setpoint holds
            the current heading (yaw ``NaN``).
        setpoint_fn: When set (a ``policy`` step), the per-tick setpoint source:
            a pure function of the observation and the step baseline returning
            the payload to stream, replacing the fixed geodetic target (whose
            fields are then unused, ``NaN``). The OFFBOARD prime/engage/stream
            lifecycle is identical either way.
    """

    target_enu: tuple[float, float, float]
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    prime_ticks: int
    min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M
    setpoint_fn: Optional[Callable[[RoutineObservation, Optional[np.ndarray]], StreamSetpoint]] = (
        None
    )


@dataclass(frozen=True)
class VehicleRoutine:
    """An ordered, immutable plan of :class:`RoutineStep`s for one vehicle."""

    steps: tuple[RoutineStep, ...]


@dataclass(frozen=True)
class RoutineState:
    """Immutable progress marker threaded through :func:`decide`.

    Attributes:
        cursor: Index of the active step; ``>= len(steps)`` means the routine is
            done.
        sent: Whether the active step has begun (its command emitted, or - for a
            command-less step - simply entered) and is now awaiting completion.
        baseline_position_enu: The vehicle position captured when the active
            step began, for climb/displacement predicates and gates that need a
            reference. ``None`` until a step begins.
        step_started_us: Sim time the active step began, used to enforce the step
            timeout. ``None`` before the step begins.
        hold_since_us: Sim time the active step's completion predicate first
            became (and has stayed) ``True``; reset to ``None`` whenever it goes
            ``False`` again. Used to enforce ``RoutineStep.hold_s``.
        setpoints_streamed: Number of OFFBOARD setpoints streamed for the active
            streaming step so far (the priming counter); 0 for a non-streaming
            step.
        offboard_requested: Whether the one-shot OFFBOARD-mode command has been
            sent for the active streaming step.
        offboard_accepted: Whether PX4 accepted the OFFBOARD-mode command for
            the active streaming step. Policy-step hold windows start only
            after this is true, so a policy cannot complete before PX4 enters
            the mode that is supposed to consume its stream.
        offboard_hold_setpoint: Last OFFBOARD setpoint that must keep streaming
            across command-less verification steps after a ``goto``. Cleared
            when a later explicit command begins.
    """

    cursor: int = 0
    sent: bool = False
    baseline_position_enu: Optional[np.ndarray] = None
    step_started_us: Optional[int] = None
    hold_since_us: Optional[int] = None
    setpoints_streamed: int = 0
    offboard_requested: bool = False
    offboard_accepted: bool = False
    offboard_hold_setpoint: Optional[StreamSetpoint] = None


def _advanced(state: RoutineState) -> RoutineState:
    """The state after the active step completes: cursor on the next step, its
    per-step progress (baseline, timer, hold, streaming counters) cleared."""
    return replace(
        state,
        cursor=state.cursor + 1,
        sent=False,
        baseline_position_enu=None,
        step_started_us=None,
        hold_since_us=None,
        setpoints_streamed=0,
        offboard_requested=False,
        offboard_accepted=False,
    )


def decide(
    routine: VehicleRoutine,
    state: RoutineState,
    observation: RoutineObservation,
) -> tuple[RoutineState, RoutineDecision]:
    """Pure routine stepper: map (routine, state, observation) to (state, decision).

    Returns the next :class:`RoutineState` together with the
    :class:`RoutineDecision` for this tick. The decision is purely a function of
    the inputs; the only state mutation is the returned new state. Semantics:

    - cursor past the last step -> ``DONE``.
    - step not yet begun -> ``SEND`` (or ``WAIT``/``STREAM`` for a command-less
      step), capturing that step's baseline and starting the timer.
    - an active step's invariant breached -> ``FAIL`` (``ENVELOPE``).
    - completion predicate true and held for ``hold_s`` -> ``ADVANCE``.
    - a temporarily-rejected ack arrived and the step opts into retry -> ``SEND``
      again (re-send, same step).
    - a non-accepted ack arrived (and no retry) -> ``FAIL`` (``COMMAND``).
    - past ``timeout_s`` of sim time -> ``FAIL`` (``COMMAND``).
    - otherwise -> ``WAIT``.
    """
    steps = routine.steps
    if state.cursor >= len(steps):
        return state, RoutineDecision.done()

    step = steps[state.cursor]

    if not state.sent:
        baseline = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        new_state = replace(
            state,
            sent=True,
            baseline_position_enu=baseline,
            step_started_us=int(observation.sim_time_us),
            hold_since_us=None,
            setpoints_streamed=0,
            offboard_requested=False,
            offboard_accepted=False,
        )
        if step.streaming is not None:
            # Streaming step: begin priming the OFFBOARD setpoint stream
            # immediately; the mode command is sent later (see below).
            setpoint = _setpoint_for(step.streaming, baseline, observation)
            return replace(
                new_state,
                setpoints_streamed=1,
                offboard_hold_setpoint=setpoint,
            ), RoutineDecision.stream(setpoint)
        if step.command is None and state.offboard_hold_setpoint is not None:
            # A command-less verifier after a goto must keep feeding PX4's
            # OFFBOARD stream until it advances or an explicit command changes
            # mode. Without this PX4 declares offboard signal loss.
            return new_state, RoutineDecision.stream(state.offboard_hold_setpoint)
        if step.command is None:
            return new_state, RoutineDecision.wait()
        return replace(new_state, offboard_hold_setpoint=None), RoutineDecision.send(step.command)

    # Continuous flight-envelope gates: a breach fails the routine as an
    # envelope event, distinct from a command rejection or a timeout.
    for invariant in step.invariants:
        breach = invariant(observation, state.baseline_position_enu)
        if breach is not None:
            return state, RoutineDecision.fail(
                f"step {step.label!r} flight envelope breached: {breach}",
                failure_kind=FailureKind.ENVELOPE,
            )

    completion_enabled = (
        not (step.streaming is not None and step.streaming.setpoint_fn is not None)
        or state.offboard_accepted
    )
    if completion_enabled and step.is_complete(observation, state.baseline_position_enu):
        if step.hold_s <= 0.0:
            return _advanced(state), RoutineDecision.advance()
        hold_since_us = state.hold_since_us
        if hold_since_us is None:
            hold_since_us = int(observation.sim_time_us)
        if int(observation.sim_time_us) - hold_since_us >= step.hold_s * 1_000_000:
            return _advanced(state), RoutineDecision.advance()
        # Still accumulating the hold window; remember when it started.
        state = replace(state, hold_since_us=hold_since_us)
    elif state.hold_since_us is not None:
        # The completion condition broke before the hold window elapsed; restart it.
        state = replace(state, hold_since_us=None)

    # Streaming (OFFBOARD goto) lifecycle: prime the setpoint stream, engage
    # OFFBOARD once, then keep streaming until completion or timeout. Handled
    # before the generic ack/timeout tail so the one-shot mode command is not
    # treated as a rejected step command.
    if step.streaming is not None:
        spec = step.streaming
        step_started_us = state.step_started_us
        assert step_started_us is not None  # set on the entry tick, before sent=True
        elapsed_us = int(observation.sim_time_us) - step_started_us
        if elapsed_us > step.timeout_s * 1_000_000:
            return state, RoutineDecision.fail(
                f"step {step.label!r} timed out after {step.timeout_s:.1f}s "
                "(did not reach the waypoint)",
                failure_kind=FailureKind.COMMAND,
            )

        if state.offboard_requested:
            ack = observation.last_ack
            if (
                ack is not None
                and step.command is not None
                and ack.command_id == step.command.command_id
            ):
                setpoint = _setpoint_for(spec, state.baseline_position_enu, observation)
                if ack.accepted:
                    hold_since_us = state.hold_since_us
                    if spec.setpoint_fn is not None and hold_since_us is None:
                        hold_since_us = int(observation.sim_time_us)
                    return replace(
                        state,
                        offboard_hold_setpoint=setpoint,
                        offboard_accepted=True,
                        hold_since_us=hold_since_us,
                    ), RoutineDecision.stream(setpoint)
                if ack.result == MAV_RESULT_TEMPORARILY_REJECTED:
                    # Keep the setpoint stream alive on this tick, then retry
                    # the one-shot OFFBOARD command on a later tick inside the
                    # same step timeout budget.
                    return replace(
                        state,
                        offboard_requested=False,
                        offboard_accepted=False,
                        hold_since_us=None,
                        offboard_hold_setpoint=setpoint,
                    ), RoutineDecision.stream(setpoint)
                return state, RoutineDecision.fail(
                    f"step {step.label!r} rejected: command={step.command.command_id} "
                    f"result={ack.result}",
                    failure_kind=FailureKind.COMMAND,
                )

        if state.setpoints_streamed < spec.prime_ticks:
            setpoint = _setpoint_for(spec, state.baseline_position_enu, observation)
            return replace(
                state,
                setpoints_streamed=state.setpoints_streamed + 1,
                offboard_hold_setpoint=setpoint,
            ), RoutineDecision.stream(setpoint)
        if not state.offboard_requested:
            assert step.command is not None  # a streaming step always carries the mode command
            return replace(state, offboard_requested=True), RoutineDecision.send(step.command)
        setpoint = _setpoint_for(spec, state.baseline_position_enu, observation)
        return replace(state, offboard_hold_setpoint=setpoint), RoutineDecision.stream(setpoint)

    ack = observation.last_ack
    if (
        ack is not None
        and step.command is not None
        and ack.command_id == step.command.command_id
        and not ack.accepted
    ):
        if step.retry_on_temporarily_rejected and ack.result == MAV_RESULT_TEMPORARILY_REJECTED:
            # Re-send the same command; PX4 returned a transient rejection. Keep
            # the step timer running so a vehicle that never becomes armable
            # still fails on the step timeout below.
            return state, RoutineDecision.send(step.command)
        return state, RoutineDecision.fail(
            f"step {step.label!r} rejected: command={step.command.command_id} result={ack.result}",
            failure_kind=FailureKind.COMMAND,
        )

    step_started_us = state.step_started_us
    assert step_started_us is not None  # set on the entry tick, before sent=True
    elapsed_us = int(observation.sim_time_us) - step_started_us
    if elapsed_us > step.timeout_s * 1_000_000:
        return state, RoutineDecision.fail(
            f"step {step.label!r} timed out after {step.timeout_s:.1f}s "
            f"(sim elapsed {elapsed_us * 1e-6:.1f}s)",
            failure_kind=FailureKind.COMMAND,
        )

    if step.command is None and state.offboard_hold_setpoint is not None:
        return state, RoutineDecision.stream(state.offboard_hold_setpoint)

    return state, RoutineDecision.wait()


# ---- Pure completion predicates over (RoutineObservation, baseline) ---------


def _always(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
    """Completion predicate that holds every tick: the step is a pure
    ``hold_s`` wait whose real gate is its continuous invariants (``policy``)."""
    return True


def armed() -> _StepPredicate:
    """Predicate: PX4 currently commands nonzero rotor references (is armed)."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        return bool(observation.bridge_status.is_armed)

    return predicate


def ack_accepted(command_id: int) -> _StepPredicate:
    """Predicate: the last ack this tick accepted ``command_id``."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        ack = observation.last_ack
        return ack is not None and ack.command_id == command_id and ack.accepted

    return predicate


def climbed_at_least(min_climb_m: float) -> _StepPredicate:
    """Predicate: the vehicle climbed at least ``min_climb_m`` above the baseline.

    The baseline is the takeoff step's captured start position (its ground
    altitude), so this is a real airborne check, not just a command ack.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return False
        altitude_m = float(observation.vehicle_state.position_enu[2])
        return altitude_m - float(baseline[2]) >= min_climb_m

    return predicate


def hovering_stable(
    *, max_abs_vertical_speed_m_s: float, max_abs_horizontal_speed_m_s: float
) -> _StepPredicate:
    """Predicate: the vehicle's translational speeds are within hover bounds.

    Combined with a step ``hold_s`` this expresses "held a stable hover for N
    seconds". Reached only after the takeoff step has already verified the climb,
    so a grounded (also slow) vehicle never gets here.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        velocity = np.asarray(observation.vehicle_state.linear_velocity_enu, dtype=np.float64)
        vertical = abs(float(velocity[2]))
        horizontal = float(np.hypot(float(velocity[0]), float(velocity[1])))
        return vertical <= max_abs_vertical_speed_m_s and horizontal <= max_abs_horizontal_speed_m_s

    return predicate


def moved_forward_at_least(distance_m: float) -> _StepPredicate:
    """Predicate: the vehicle moved at least ``distance_m`` along world +X/east."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return False
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        return float(position[0] - baseline[0]) >= distance_m

    return predicate


def disarmed() -> _StepPredicate:
    """Predicate: PX4 no longer commands armed rotor references."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        return not bool(observation.bridge_status.is_armed)

    return predicate


def reached_world_position(x_m: float, y_m: float, z_m: float, tol_m: float) -> _StepPredicate:
    """Predicate: the vehicle is within ``tol_m`` (3D) of absolute world ENU ``(x, y, z)``.

    The first *absolute* world-frame completion predicate: it compares the
    observed world position directly to the target, not a baseline-relative
    displacement.
    """
    target = np.array([x_m, y_m, z_m], dtype=np.float64)

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        return float(np.linalg.norm(position - target)) <= tol_m

    return predicate


def heading_aligned(
    x_m: float,
    y_m: float,
    tol_deg: float,
    min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M,
) -> _StepPredicate:
    """Predicate: the body faces the active-step-baseline-to-target heading.

    Trivially satisfied for a near-vertical leg (horizontal
    active-step-baseline-to-target distance below ``min_heading_dist_m``),
    where heading is undefined.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return True
        de = x_m - float(baseline[0])
        dn = y_m - float(baseline[1])
        if math.hypot(de, dn) < min_heading_dist_m:
            return True
        target_heading = math.atan2(dn, de)  # ENU, CCW from +east
        forward_enu = rotate_vector_wxyz(
            observation.vehicle_state.orientation_flu_enu_wxyz,
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        observed = math.atan2(float(forward_enu[1]), float(forward_enu[0]))
        diff = abs((observed - target_heading + math.pi) % (2.0 * math.pi) - math.pi)
        return math.degrees(diff) <= tol_deg

    return predicate


def _setpoint_for(
    spec: "_StreamingSpec",
    baseline: Optional[np.ndarray],
    observation: RoutineObservation,
) -> StreamSetpoint:
    """Build the streamed setpoint for this tick.

    A ``setpoint_fn`` spec (policy step) delegates to the injected pure
    function. Otherwise (a ``goto``): the fixed geodetic target plus a live
    heading -- yaw faces the active-step-baseline-to-target direction (constant
    per leg); a near-vertical leg (or no baseline yet) holds heading (``NaN``).
    """
    if spec.setpoint_fn is not None:
        return spec.setpoint_fn(observation, baseline)
    east, north, _up = spec.target_enu
    yaw_rad = math.nan
    if baseline is not None:
        de = east - float(baseline[0])
        dn = north - float(baseline[1])
        if math.hypot(de, dn) >= spec.min_heading_dist_m:
            # NED yaw: 0 = north, positive clockwise toward east.
            yaw_rad = math.atan2(de, dn)
    return PositionSetpoint(
        latitude_deg=spec.latitude_deg,
        longitude_deg=spec.longitude_deg,
        altitude_m=spec.altitude_m,
        yaw_rad=yaw_rad,
    )


# ---- Continuous flight-envelope invariant gates ----------------------------


def tilt_below(max_tilt_deg: float) -> _Invariant:
    """Gate: body tilt from upright stays at or below ``max_tilt_deg``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        tilt = float(tilt_angle_deg(observation.vehicle_state.orientation_flu_enu_wxyz))
        if tilt > max_tilt_deg:
            return f"tilt {tilt:.1f}deg exceeds limit {max_tilt_deg:.1f}deg"
        return None

    return gate


def _lateral_drift_below(max_lateral_drift_m: float) -> _Invariant:
    """Gate: horizontal distance from the baseline stays within ``max_lateral_drift_m``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        if baseline is None:
            return None
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        drift = float(np.hypot(float(position[0] - baseline[0]), float(position[1] - baseline[1])))
        if drift > max_lateral_drift_m:
            return f"lateral drift {drift:.2f}m exceeds limit {max_lateral_drift_m:.2f}m"
        return None

    return gate


def cross_track_drift_below(max_cross_track_m: float) -> _Invariant:
    """Gate: world Y/north drift from the baseline stays within ``max_cross_track_m``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        if baseline is None:
            return None
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        drift = abs(float(position[1] - baseline[1]))
        if drift > max_cross_track_m:
            return f"cross-track drift {drift:.2f}m exceeds limit {max_cross_track_m:.2f}m"
        return None

    return gate


def cross_track_to_segment_below(target_x_m: float, target_y_m: float, max_m: float) -> _Invariant:
    """Gate: horizontal distance from the active step's path stays within ``max_m``.

    The direction-agnostic generalization of :func:`cross_track_drift_below`: it
    measures perpendicular deviation from the straight planned path (baseline to
    the absolute target), so along-track progress in any heading is free while a
    sideways excursion off the path trips the envelope.
    """
    target = np.array([target_x_m, target_y_m], dtype=np.float64)

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        if baseline is None:
            return None
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        a = np.array([float(baseline[0]), float(baseline[1])], dtype=np.float64)
        p = position[:2]
        ab = target - a
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-6:
            cross = float(np.linalg.norm(p - a))
        else:
            cross = abs(float(np.cross(ab, p - a))) / ab_len
        if cross > max_m:
            return f"cross-track {cross:.2f}m exceeds limit {max_m:.2f}m"
        return None

    return gate


def _vertical_speed_within(max_up_m_s: float, max_down_m_s: float) -> _Invariant:
    """Gate: vertical speed stays within (``max_down_m_s``, ``max_up_m_s``).

    ``max_up_m_s`` is the positive upward limit; ``max_down_m_s`` is the negative
    downward limit (e.g. ``-4.0``).
    """

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        vertical = float(observation.vehicle_state.linear_velocity_enu[2])
        if vertical > max_up_m_s:
            return f"vertical speed {vertical:.2f}m/s exceeds up limit {max_up_m_s:.2f}m/s"
        if vertical < max_down_m_s:
            return f"vertical speed {vertical:.2f}m/s exceeds down limit {max_down_m_s:.2f}m/s"
        return None

    return gate


def _angular_speed_below(max_angular_speed_rad_s: float) -> _Invariant:
    """Gate: body angular-speed magnitude stays within ``max_angular_speed_rad_s``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        angular = np.asarray(observation.vehicle_state.angular_velocity_flu, dtype=np.float64)
        speed = float(np.linalg.norm(angular))
        if speed > max_angular_speed_rad_s:
            return (
                f"angular speed {speed:.2f}rad/s exceeds limit "
                f"{max_angular_speed_rad_s:.2f}rad/s"
            )
        return None

    return gate


# ---- Config-driven routine assembly ----------------------------------------


@dataclass(frozen=True)
class EnvelopeGates:
    """Flight-envelope thresholds wired onto a routine's flight steps.

    The continuous gates (``max_*``) become per-step :class:`_Invariant`s on the
    takeoff and hover steps; the two ``hover_*`` thresholds parameterize the
    hover step's :func:`hovering_stable` completion predicate. Built by a driver
    from the ``routine`` config and handed to :func:`build_vehicle_routine`.
    """

    max_tilt_deg: float
    max_lateral_drift_m: float
    max_vertical_speed_m_s: float
    max_downward_speed_m_s: float
    max_angular_speed_rad_s: float
    hover_max_abs_vertical_speed_m_s: float
    hover_max_abs_horizontal_speed_m_s: float


def _invariants(gates: EnvelopeGates, lateral: _Invariant) -> tuple[_Invariant, ...]:
    """The four continuous flight-envelope gates, parameterized by the lateral gate.

    Tilt, vertical, and angular gates are shared by every flight step; the lateral
    gate differs per step (baseline drift for takeoff/hover, cross-track for
    forward, cross-track-to-path for goto).
    """
    return (
        tilt_below(gates.max_tilt_deg),
        lateral,
        _vertical_speed_within(gates.max_vertical_speed_m_s, gates.max_downward_speed_m_s),
        _angular_speed_below(gates.max_angular_speed_rad_s),
    )


def _offset_lat_lon(
    latitude_deg: float, longitude_deg: float, *, east_m: float, north_m: float
) -> tuple[float, float]:
    lat_rad = math.radians(float(latitude_deg))
    dlat_deg = (float(north_m) / _EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon_deg = (float(east_m) / (_EARTH_RADIUS_M * math.cos(lat_rad))) * (180.0 / math.pi)
    return float(latitude_deg) + dlat_deg, float(longitude_deg) + dlon_deg


def build_vehicle_routine(
    steps: Sequence[Mapping[str, Any]],
    *,
    target_latitude_deg: float,
    target_longitude_deg: float,
    target_altitude_m: float,
    command_ack_timeout_s: float,
    gates: EnvelopeGates,
    takeoff_timeout_s: float,
    hover_settle_s: float,
    home_lat_deg: float,
    home_lon_deg: float,
    home_alt_m: float,
    goto_position_tol_m: float,
    goto_heading_tol_deg: float,
    goto_cross_track_m: float,
    goto_offboard_prime_ticks: int,
    goto_min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M,
    policy_fn_factory: Optional[Callable[[Mapping[str, Any]], PolicyStepFn]] = None,
) -> VehicleRoutine:
    """Build a :class:`VehicleRoutine` from authored step descriptors and a target.

    The single pure mapping from the ``routine.steps`` config (a list of
    ``{kind: ...}`` mappings) to the immutable routine the commander executes,
    shared by every driver (the live extension and the standalone scenario).
    ``target_*`` is the per-drone takeoff target: the drone's current reported
    geographic position and altitude, to which a ``takeoff`` step adds its
    ``climb_m``. ``gates`` are the flight-envelope thresholds wired onto the
    takeoff and hover steps so the routine verifies the flight in every workflow.

    Step kinds:
    - ``arm``: completes on the arm ``COMMAND_ACK``; retries on a transient PX4
      ``TEMPORARILY_REJECTED``.
    - ``takeoff`` (requires ``climb_m`` and ``min_climb_m``): commands takeoff to
      ``target_altitude_m + climb_m`` and completes once the vehicle has actually
      climbed ``min_climb_m`` above its baseline; ``timeout_s`` is
      ``takeoff_timeout_s``; flight gates apply.
    - ``hover`` (requires ``hold_s``): verifies a stable hover for ``hold_s``
      seconds; if it follows a ``goto`` while PX4 is still in OFFBOARD, the
      previous setpoint keeps streaming until the hover advances; ``timeout_s``
      is ``hover_settle_s + hold_s``; flight gates apply.
    - ``forward`` (requires ``distance_m`` and ``timeout_s``): repositions
      ``distance_m`` along world +X/east and completes once that displacement is
      observed from the active step baseline; flight gates apply.
    - ``goto`` (requires ``x``, ``y``, ``z`` and ``timeout_s``; optional
      ``position_tol_m`` / ``heading_tol_deg``): streams OFFBOARD position
      setpoints toward the absolute world-ENU waypoint (converted to a geodetic
      target via ``synth_gps_from_enu`` against ``home_*``, frame-consistent with
      the bridge's synthesized GPS), yawing toward the leg direction; completes
      once within ``goto_position_tol_m`` of the waypoint and heading-aligned. A
      cross-track-to-path gate plus tilt / vertical / angular gates apply.
    - ``policy`` (requires ``artifact``, ``hold_s`` and ``timeout_s``; optional
      ``target`` as world-ENU ``[x, y, z]``): streams OFFBOARD setpoints
      produced by a trained policy each tick -- the injected
      ``policy_fn_factory(descriptor)`` loads the artifact and returns the pure
      :data:`PolicyStepFn` (raising ``ValueError`` for a missing or mismatched
      artifact, so a bad config fails at build time). Reuses the ``goto``
      OFFBOARD prime/engage/stream lifecycle. Without an authored ``target``
      the policy holds the position captured when the step began. Completes
      after PX4 accepts OFFBOARD and then ``hold_s`` of continuous in-envelope
      flight (the flight gates are the invariants, so a bad policy fails as
      ``ENVELOPE``). A routine should follow a policy step with an
      explicit-command step (e.g. ``land``): a command-less verifier would keep
      re-streaming the last policy payload.
    - ``land`` (requires ``timeout_s``): commands a landing at the current
      position and completes once PX4 disarms.

    An empty ``steps`` yields an empty routine (the manual case). Pure: no Isaac,
    no pymavlink, no sockets.

    Raises:
        ValueError: on an unknown ``kind``, a required descriptor field missing,
            or a ``policy`` step without an injected ``policy_fn_factory``
            (invalid configuration must not pass silently).
    """
    flight_invariants = _invariants(gates, _lateral_drift_below(gates.max_lateral_drift_m))
    forward_invariants = _invariants(gates, cross_track_drift_below(gates.max_lateral_drift_m))
    built: list[RoutineStep] = []
    commanded_altitude_m = float(target_altitude_m)
    for index, descriptor in enumerate(steps):
        kind = descriptor.get("kind")
        if kind == "arm":
            built.append(
                RoutineStep(
                    command=arm(),
                    is_complete=ack_accepted(MAV_CMD_COMPONENT_ARM_DISARM),
                    label="arm",
                    timeout_s=command_ack_timeout_s,
                    retry_on_temporarily_rejected=True,
                )
            )
        elif kind == "takeoff":
            if "climb_m" not in descriptor:
                raise ValueError(f"routine step {index} (takeoff) is missing required 'climb_m'")
            if "min_climb_m" not in descriptor:
                raise ValueError(
                    f"routine step {index} (takeoff) is missing required 'min_climb_m'"
                )
            absolute_altitude_m = float(target_altitude_m) + float(descriptor["climb_m"])
            commanded_altitude_m = absolute_altitude_m
            built.append(
                RoutineStep(
                    command=takeoff(target_latitude_deg, target_longitude_deg, absolute_altitude_m),
                    is_complete=climbed_at_least(float(descriptor["min_climb_m"])),
                    label="takeoff",
                    timeout_s=takeoff_timeout_s,
                    retry_on_temporarily_rejected=True,
                    invariants=flight_invariants,
                )
            )
        elif kind == "hover":
            if "hold_s" not in descriptor:
                raise ValueError(f"routine step {index} (hover) is missing required 'hold_s'")
            hold_s = float(descriptor["hold_s"])
            built.append(
                RoutineStep(
                    command=None,
                    is_complete=hovering_stable(
                        max_abs_vertical_speed_m_s=gates.hover_max_abs_vertical_speed_m_s,
                        max_abs_horizontal_speed_m_s=gates.hover_max_abs_horizontal_speed_m_s,
                    ),
                    label="hover",
                    timeout_s=float(hover_settle_s) + hold_s,
                    hold_s=hold_s,
                    invariants=flight_invariants,
                )
            )
        elif kind == "forward":
            if "distance_m" not in descriptor:
                raise ValueError(f"routine step {index} (forward) is missing required 'distance_m'")
            if "timeout_s" not in descriptor:
                raise ValueError(f"routine step {index} (forward) is missing required 'timeout_s'")
            distance_m = float(descriptor["distance_m"])
            target_latitude_deg, target_longitude_deg = _offset_lat_lon(
                target_latitude_deg,
                target_longitude_deg,
                east_m=distance_m,
                north_m=0.0,
            )
            built.append(
                RoutineStep(
                    command=reposition(
                        target_latitude_deg,
                        target_longitude_deg,
                        commanded_altitude_m,
                    ),
                    is_complete=moved_forward_at_least(distance_m),
                    label="forward",
                    timeout_s=float(descriptor["timeout_s"]),
                    retry_on_temporarily_rejected=True,
                    invariants=forward_invariants,
                )
            )
        elif kind == "goto":
            for field in ("x", "y", "z", "timeout_s"):
                if field not in descriptor:
                    raise ValueError(f"routine step {index} (goto) is missing required {field!r}")
            x_m = float(descriptor["x"])
            y_m = float(descriptor["y"])
            z_m = float(descriptor["z"])
            position_tol_m = float(descriptor.get("position_tol_m", goto_position_tol_m))
            heading_tol_deg = float(descriptor.get("heading_tol_deg", goto_heading_tol_deg))
            gps = synth_gps_from_enu(
                np.array([x_m, y_m, z_m], dtype=np.float64),
                np.zeros(3, dtype=np.float64),
                home_lat_deg,
                home_lon_deg,
                home_alt_m,
            )
            spec = _StreamingSpec(
                target_enu=(x_m, y_m, z_m),
                latitude_deg=float(gps["latitude"]),
                longitude_deg=float(gps["longitude"]),
                altitude_m=float(gps["altitude"]),
                prime_ticks=int(goto_offboard_prime_ticks),
                min_heading_dist_m=float(goto_min_heading_dist_m),
            )
            reached = reached_world_position(x_m, y_m, z_m, position_tol_m)
            aligned = heading_aligned(x_m, y_m, heading_tol_deg, goto_min_heading_dist_m)

            def _reached_and_aligned(
                observation: RoutineObservation,
                baseline: Optional[np.ndarray],
                reached: _StepPredicate = reached,
                aligned: _StepPredicate = aligned,
            ) -> bool:
                return reached(observation, baseline) and aligned(observation, baseline)

            built.append(
                RoutineStep(
                    command=set_offboard_mode(),
                    is_complete=_reached_and_aligned,
                    label=f"goto[{x_m:.1f},{y_m:.1f},{z_m:.1f}]",
                    timeout_s=float(descriptor["timeout_s"]),
                    invariants=_invariants(
                        gates, cross_track_to_segment_below(x_m, y_m, goto_cross_track_m)
                    ),
                    streaming=spec,
                )
            )
        elif kind == "policy":
            for field in ("artifact", "hold_s", "timeout_s"):
                if field not in descriptor:
                    raise ValueError(f"routine step {index} (policy) is missing required {field!r}")
            if policy_fn_factory is None:
                raise ValueError(
                    f"routine step {index} (policy) requires an injected policy_fn_factory "
                    "(the runtime composes it from rl.press.policy)"
                )
            hold_s = float(descriptor["hold_s"])
            target = descriptor.get("target")
            if target is None:
                target_enu = None
            else:
                try:
                    target_values = [float(v) for v in target]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"routine step {index} (policy) target must be a 3-item numeric sequence"
                    ) from exc
                if len(target_values) != 3:
                    raise ValueError(
                        f"routine step {index} (policy) target must be a 3-item numeric sequence"
                    )
                target_enu = np.asarray(target_values, dtype=np.float64)
            policy_fn = policy_fn_factory(descriptor)

            def _policy_setpoint(
                observation: RoutineObservation,
                baseline: Optional[np.ndarray],
                policy_fn: PolicyStepFn = policy_fn,
                target_enu: Optional[np.ndarray] = target_enu,
            ) -> StreamSetpoint:
                goal = target_enu if target_enu is not None else baseline
                assert goal is not None  # baseline is captured on the entry tick
                return policy_fn(observation.vehicle_state, goal)

            spec = _StreamingSpec(
                target_enu=(math.nan, math.nan, math.nan),
                latitude_deg=math.nan,
                longitude_deg=math.nan,
                altitude_m=math.nan,
                prime_ticks=int(goto_offboard_prime_ticks),
                setpoint_fn=_policy_setpoint,
            )
            built.append(
                RoutineStep(
                    command=set_offboard_mode(),
                    is_complete=_always,
                    label=(f"policy[{descriptor['name']}]" if "name" in descriptor else "policy"),
                    timeout_s=float(descriptor["timeout_s"]),
                    hold_s=hold_s,
                    invariants=flight_invariants,
                    streaming=spec,
                )
            )
        elif kind == "land":
            if "timeout_s" not in descriptor:
                raise ValueError(f"routine step {index} (land) is missing required 'timeout_s'")
            built.append(
                RoutineStep(
                    command=land(),
                    is_complete=disarmed(),
                    label="land",
                    timeout_s=float(descriptor["timeout_s"]),
                    retry_on_temporarily_rejected=True,
                )
            )
        else:
            raise ValueError(f"routine step {index} has unknown kind {kind!r}")
    return VehicleRoutine(steps=tuple(built))
