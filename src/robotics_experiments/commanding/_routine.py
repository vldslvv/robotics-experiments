"""The pure command plan and its stepper - the command-layer contract centerpiece.

A :class:`_VehicleRoutine` is an ordered tuple of :class:`_RoutineStep`s; each step
carries one optional :class:`_MavlinkCommand`, a pure completion predicate over a
:class:`RoutineObservation` (and the routine's captured baseline position), an
optional sustained-hold window, and continuous invariant gates. :func:`_decide` is
the pure transform at the heart of the layer - the command-layer analogue of the
thrust model's ``commands_to_dynamics``: given the routine, the current
:class:`_RoutineState`, and an observation, it returns the next state and a
:class:`_RoutineDecision` (``SEND`` / ``WAIT`` / ``ADVANCE`` / ``DONE`` /
``FAIL``). No sockets, no pymavlink, no Isaac - it is unit-testable with plain
dataclasses, and the side-effecting :class:`_VehicleCommander` (see
``_commander.py``) is the only thing that turns a ``SEND`` decision into MAVLink.

The same routine verifies a _takeoff in every workflow: a step's completion
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
from typing import Any, Optional

import numpy as np

from robotics_experiments.px4_bridge.frames import tilt_angle_deg

from ._commands import (
    _MAV_CMD_COMPONENT_ARM_DISARM,
    _arm,
    _land,
    _MavlinkCommand,
    _reposition,
    _takeoff,
)
from ._observation import _MAV_RESULT_TEMPORARILY_REJECTED, RoutineObservation

__all__ = [
    "_DecisionKind",
    "_EnvelopeGates",
    "FailureKind",
    "_Invariant",
    "_RoutineDecision",
    "_RoutineState",
    "_RoutineStep",
    "_StepPredicate",
    "_VehicleRoutine",
    "_ack_accepted",
    "_angular_speed_below",
    "_armed",
    "_build_vehicle_routine",
    "_climbed_at_least",
    "_cross_track_drift_below",
    "_disarmed",
    "_hovering_stable",
    "_lateral_drift_below",
    "_moved_forward_at_least",
    "_tilt_below",
    "_vertical_speed_within",
]

_EARTH_RADIUS_M = 6_378_137.0

# A completion predicate reads the observation plus the routine's captured
# baseline position (ENU, set on the first SEND); baseline is ``None`` only
# before any step has been sent. An invariant gate returns a breach message when
# the continuous flight envelope is violated, or ``None`` when it holds.
_StepPredicate = Callable[[RoutineObservation, Optional[np.ndarray]], bool]
_Invariant = Callable[[RoutineObservation, Optional[np.ndarray]], Optional[str]]


class _DecisionKind(Enum):
    """The kind of a :class:`_RoutineDecision`."""

    SEND = auto()
    WAIT = auto()
    ADVANCE = auto()
    DONE = auto()
    FAIL = auto()


class FailureKind(Enum):
    """Why a routine failed, so a driver can react in kind.

    ``COMMAND`` is a command rejection or a step timeout (the routine could not
    make progress); ``ENVELOPE`` is a flight-envelope invariant breach (the
    vehicle flew out of bounds). The standalone strict driver maps ``ENVELOPE``
    to its ``ExpectedInstabilityDetected`` and ``COMMAND`` to a plain assertion.
    """

    COMMAND = auto()
    ENVELOPE = auto()


@dataclass(frozen=True)
class _RoutineDecision:
    """The pure output of :func:`_decide` for one tick.

    Exactly one ``kind``; ``command`` is set only for ``SEND``; ``reason`` and
    ``failure_kind`` only for ``FAIL``. Build instances through the classmethods.
    """

    kind: _DecisionKind
    command: Optional[_MavlinkCommand] = None
    reason: Optional[str] = None
    failure_kind: Optional[FailureKind] = None

    @classmethod
    def send(cls, command: _MavlinkCommand) -> "_RoutineDecision":
        return cls(_DecisionKind.SEND, command=command)

    @classmethod
    def wait(cls) -> "_RoutineDecision":
        return cls(_DecisionKind.WAIT)

    @classmethod
    def advance(cls) -> "_RoutineDecision":
        return cls(_DecisionKind.ADVANCE)

    @classmethod
    def done(cls) -> "_RoutineDecision":
        return cls(_DecisionKind.DONE)

    @classmethod
    def fail(
        cls, reason: str, *, failure_kind: FailureKind = FailureKind.COMMAND
    ) -> "_RoutineDecision":
        return cls(_DecisionKind.FAIL, reason=reason, failure_kind=failure_kind)


@dataclass(frozen=True)
class _RoutineStep:
    """One step of a routine: a command, its completion predicate, and gates.

    Attributes:
        command: The :class:`_MavlinkCommand` sent when the step begins, or
            ``None`` for a verification-only step (e.g. ``hover``) that sends
            nothing and only waits on its completion predicate and invariants.
        is_complete: Pure predicate over a :class:`RoutineObservation` and the
            routine's captured baseline position; the step advances when it
            returns ``True`` (and, if ``hold_s`` is set, has held continuously).
        label: Short human-readable name used in logs and failure messages.
        timeout_s: Sim-time budget for the step; on exceed the routine fails with
            ``FailureKind.COMMAND``.
        retry_on_temporarily_rejected: When ``True``, a
            ``_MAV_RESULT_TEMPORARILY_REJECTED`` ack for this command re-sends it
            instead of failing (PX4 returns this transiently before it is ready
            to _arm). When ``False``, any non-accepted ack fails the routine.
        hold_s: Sim-time the completion predicate must hold *continuously* before
            the step advances (0 = instantaneous, the default). This expresses
            "hover stable for N seconds" as a step completion condition.
        invariants: Continuous flight-envelope gates evaluated every tick the
            step is active; the first breach fails the routine with
            ``FailureKind.ENVELOPE``.
    """

    command: Optional[_MavlinkCommand]
    is_complete: _StepPredicate
    label: str
    timeout_s: float
    retry_on_temporarily_rejected: bool = False
    hold_s: float = 0.0
    invariants: tuple[_Invariant, ...] = ()


@dataclass(frozen=True)
class _VehicleRoutine:
    """An ordered, immutable plan of :class:`_RoutineStep`s for one vehicle."""

    steps: tuple[_RoutineStep, ...]


@dataclass(frozen=True)
class _RoutineState:
    """Immutable progress marker threaded through :func:`_decide`.

    Attributes:
        cursor: Index of the active step; ``>= len(steps)`` means the routine is
            done.
        sent: Whether the active step has begun (its command emitted, or - for a
            command-less step - simply entered) and is now awaiting completion.
        baseline_position_enu: The vehicle position captured when the routine
            began its first step, for climb/displacement predicates and gates
            that need a reference. ``None`` until the first step begins.
        step_started_us: Sim time the active step began, used to enforce the step
            timeout. ``None`` before the step begins.
        hold_since_us: Sim time the active step's completion predicate first
            became (and has stayed) ``True``; reset to ``None`` whenever it goes
            ``False`` again. Used to enforce ``_RoutineStep.hold_s``.
    """

    cursor: int = 0
    sent: bool = False
    baseline_position_enu: Optional[np.ndarray] = None
    step_started_us: Optional[int] = None
    hold_since_us: Optional[int] = None

    @classmethod
    def initial(cls) -> "_RoutineState":
        return cls()


def _decide(
    routine: _VehicleRoutine,
    state: _RoutineState,
    observation: RoutineObservation,
) -> tuple[_RoutineState, _RoutineDecision]:
    """Pure routine stepper: map (routine, state, observation) to (state, decision).

    Returns the next :class:`_RoutineState` together with the
    :class:`_RoutineDecision` for this tick. The decision is purely a function of
    the inputs; the only state mutation is the returned new state. Semantics:

    - cursor past the last step -> ``DONE``.
    - step not yet begun -> ``SEND`` (or ``WAIT`` for a command-less step),
      capturing the baseline and starting the timer.
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
        return state, _RoutineDecision.done()

    step = steps[state.cursor]

    if not state.sent:
        baseline = state.baseline_position_enu
        if baseline is None:
            baseline = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        new_state = replace(
            state,
            sent=True,
            baseline_position_enu=baseline,
            step_started_us=int(observation.sim_time_us),
            hold_since_us=None,
        )
        if step.command is None:
            return new_state, _RoutineDecision.wait()
        return new_state, _RoutineDecision.send(step.command)

    # Continuous flight-envelope gates: a breach fails the routine as an
    # envelope event, distinct from a command rejection or a timeout.
    for invariant in step.invariants:
        breach = invariant(observation, state.baseline_position_enu)
        if breach is not None:
            return state, _RoutineDecision.fail(
                f"step {step.label!r} flight envelope breached: {breach}",
                failure_kind=FailureKind.ENVELOPE,
            )

    if step.is_complete(observation, state.baseline_position_enu):
        if step.hold_s <= 0.0:
            new_state = replace(
                state, cursor=state.cursor + 1, sent=False, step_started_us=None, hold_since_us=None
            )
            return new_state, _RoutineDecision.advance()
        hold_since_us = state.hold_since_us
        if hold_since_us is None:
            hold_since_us = int(observation.sim_time_us)
        if int(observation.sim_time_us) - hold_since_us >= step.hold_s * 1_000_000:
            new_state = replace(
                state, cursor=state.cursor + 1, sent=False, step_started_us=None, hold_since_us=None
            )
            return new_state, _RoutineDecision.advance()
        # Still accumulating the hold window; remember when it started.
        state = replace(state, hold_since_us=hold_since_us)
    elif state.hold_since_us is not None:
        # The completion condition broke before the hold window elapsed; restart it.
        state = replace(state, hold_since_us=None)

    ack = observation.last_ack
    if (
        ack is not None
        and step.command is not None
        and ack.command_id == step.command.command_id
        and not ack.accepted
    ):
        if step.retry_on_temporarily_rejected and ack.result == _MAV_RESULT_TEMPORARILY_REJECTED:
            # Re-send the same command; PX4 returned a transient rejection. Keep
            # the step timer running so a vehicle that never becomes armable
            # still fails on the step timeout below.
            return state, _RoutineDecision.send(step.command)
        return state, _RoutineDecision.fail(
            f"step {step.label!r} rejected: command={step.command.command_id} result={ack.result}",
            failure_kind=FailureKind.COMMAND,
        )

    step_started_us = state.step_started_us
    assert step_started_us is not None  # set on the entry tick, before sent=True
    elapsed_us = int(observation.sim_time_us) - step_started_us
    if elapsed_us > step.timeout_s * 1_000_000:
        return state, _RoutineDecision.fail(
            f"step {step.label!r} timed out after {step.timeout_s:.1f}s "
            f"(sim elapsed {elapsed_us * 1e-6:.1f}s)",
            failure_kind=FailureKind.COMMAND,
        )

    return state, _RoutineDecision.wait()


# ---- Pure completion predicates over (RoutineObservation, baseline) ---------


def _armed() -> _StepPredicate:
    """Predicate: PX4 currently commands nonzero rotor references (is _armed)."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        return bool(observation.bridge_status.is_armed)

    return predicate


def _ack_accepted(command_id: int) -> _StepPredicate:
    """Predicate: the last ack this tick accepted ``command_id``."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        ack = observation.last_ack
        return ack is not None and ack.command_id == command_id and ack.accepted

    return predicate


def _climbed_at_least(min_climb_m: float) -> _StepPredicate:
    """Predicate: the vehicle climbed at least ``min_climb_m`` above the baseline.

    The baseline is the routine's captured start position (its ground altitude),
    so this is a real airborne check, not just a command ack.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return False
        altitude_m = float(observation.vehicle_state.position_enu[2])
        return altitude_m - float(baseline[2]) >= min_climb_m

    return predicate


def _hovering_stable(
    *, max_abs_vertical_speed_m_s: float, max_abs_horizontal_speed_m_s: float
) -> _StepPredicate:
    """Predicate: the vehicle's translational speeds are within hover bounds.

    Combined with a step ``hold_s`` this expresses "held a stable hover for N
    seconds". Reached only after the _takeoff step has already verified the climb,
    so a grounded (also slow) vehicle never gets here.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        velocity = np.asarray(observation.vehicle_state.linear_velocity_enu, dtype=np.float64)
        vertical = abs(float(velocity[2]))
        horizontal = float(np.hypot(float(velocity[0]), float(velocity[1])))
        return vertical <= max_abs_vertical_speed_m_s and horizontal <= max_abs_horizontal_speed_m_s

    return predicate


def _moved_forward_at_least(distance_m: float) -> _StepPredicate:
    """Predicate: the vehicle moved at least ``distance_m`` along world +X/east."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return False
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        return float(position[0] - baseline[0]) >= distance_m

    return predicate


def _disarmed() -> _StepPredicate:
    """Predicate: PX4 no longer commands _armed rotor references."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        return not bool(observation.bridge_status.is_armed)

    return predicate


# ---- Continuous flight-envelope invariant gates ----------------------------


def _tilt_below(max_tilt_deg: float) -> _Invariant:
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


def _cross_track_drift_below(max_cross_track_m: float) -> _Invariant:
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
class _EnvelopeGates:
    """Flight-envelope thresholds wired onto a routine's flight steps.

    The continuous gates (``max_*``) become per-step :class:`_Invariant`s on the
    _takeoff and hover steps; the two ``hover_*`` thresholds parameterize the
    hover step's :func:`_hovering_stable` completion predicate. Built by a driver
    from the ``routine`` config and handed to :func:`_build_vehicle_routine`.
    """

    max_tilt_deg: float
    max_lateral_drift_m: float
    max_vertical_speed_m_s: float
    max_downward_speed_m_s: float
    max_angular_speed_rad_s: float
    hover_max_abs_vertical_speed_m_s: float
    hover_max_abs_horizontal_speed_m_s: float


def _flight_invariants(gates: _EnvelopeGates) -> tuple[_Invariant, ...]:
    return (
        _tilt_below(gates.max_tilt_deg),
        _lateral_drift_below(gates.max_lateral_drift_m),
        _vertical_speed_within(gates.max_vertical_speed_m_s, gates.max_downward_speed_m_s),
        _angular_speed_below(gates.max_angular_speed_rad_s),
    )


def _forward_invariants(gates: _EnvelopeGates) -> tuple[_Invariant, ...]:
    return (
        _tilt_below(gates.max_tilt_deg),
        _cross_track_drift_below(gates.max_lateral_drift_m),
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


def _build_vehicle_routine(
    steps: Sequence[Mapping[str, Any]],
    *,
    target_latitude_deg: float,
    target_longitude_deg: float,
    target_altitude_m: float,
    command_ack_timeout_s: float,
    gates: _EnvelopeGates,
    takeoff_timeout_s: float,
    hover_settle_s: float,
) -> _VehicleRoutine:
    """Build a :class:`_VehicleRoutine` from authored step descriptors and a target.

    The single pure mapping from the ``routine.steps`` config (a list of
    ``{kind: ...}`` mappings) to the immutable routine the commander executes,
    shared by every driver (the live extension and the standalone scenario).
    ``target_*`` is the per-drone _takeoff target: the drone's current reported
    geographic position and altitude, to which a ``_takeoff`` step adds its
    ``climb_m``. ``gates`` are the flight-envelope thresholds wired onto the
    _takeoff and hover steps so the routine verifies the flight in every workflow.

    Step kinds:
    - ``_arm``: completes on the _arm ``COMMAND_ACK``; retries on a transient PX4
      ``TEMPORARILY_REJECTED``.
    - ``_takeoff`` (requires ``climb_m`` and ``min_climb_m``): commands _takeoff to
      ``target_altitude_m + climb_m`` and completes once the vehicle has actually
      climbed ``min_climb_m`` above its baseline; ``timeout_s`` is
      ``takeoff_timeout_s``; flight gates apply.
    - ``hover`` (requires ``hold_s``): sends nothing and completes once a stable
      hover is held for ``hold_s`` seconds; ``timeout_s`` is
      ``hover_settle_s + hold_s``; flight gates apply.
    - ``forward`` (requires ``distance_m`` and ``timeout_s``): repositions
      ``distance_m`` along world +X/east and completes once that displacement is
      observed from the routine baseline; flight gates apply.
    - ``_land`` (requires ``timeout_s``): commands a landing at the current
      position and completes once PX4 disarms.

    An empty ``steps`` yields an empty routine (the manual case). Pure: no Isaac,
    no pymavlink, no sockets.

    Raises:
        ValueError: on an unknown ``kind`` or a required descriptor field missing
            (invalid configuration must not pass silently).
    """
    flight_invariants = _flight_invariants(gates)
    forward_invariants = _forward_invariants(gates)
    built: list[_RoutineStep] = []
    commanded_altitude_m = float(target_altitude_m)
    for index, descriptor in enumerate(steps):
        kind = descriptor.get("kind")
        if kind == "arm":
            built.append(
                _RoutineStep(
                    command=_arm(),
                    is_complete=_ack_accepted(_MAV_CMD_COMPONENT_ARM_DISARM),
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
                _RoutineStep(
                    command=_takeoff(
                        target_latitude_deg, target_longitude_deg, absolute_altitude_m
                    ),
                    is_complete=_climbed_at_least(float(descriptor["min_climb_m"])),
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
                _RoutineStep(
                    command=None,
                    is_complete=_hovering_stable(
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
                _RoutineStep(
                    command=_reposition(
                        target_latitude_deg,
                        target_longitude_deg,
                        commanded_altitude_m,
                    ),
                    is_complete=_moved_forward_at_least(distance_m),
                    label="forward",
                    timeout_s=float(descriptor["timeout_s"]),
                    retry_on_temporarily_rejected=True,
                    invariants=forward_invariants,
                )
            )
        elif kind == "land":
            if "timeout_s" not in descriptor:
                raise ValueError(f"routine step {index} (land) is missing required 'timeout_s'")
            built.append(
                _RoutineStep(
                    command=_land(),
                    is_complete=_disarmed(),
                    label="land",
                    timeout_s=float(descriptor["timeout_s"]),
                    retry_on_temporarily_rejected=True,
                )
            )
        else:
            raise ValueError(f"routine step {index} has unknown kind {kind!r}")
    return _VehicleRoutine(steps=tuple(built))
