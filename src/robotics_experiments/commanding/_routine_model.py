from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from robotics_experiments.commanding.commands import (
    AttitudeRateSetpoint,
    MavlinkCommand,
    PositionSetpoint,
    VelocitySetpoint,
)
from robotics_experiments.commanding.observation import RoutineObservation

if TYPE_CHECKING:
    from robotics_experiments.px4_bridge import VehicleState

__all__ = [
    "DecisionKind",
    "FailureKind",
    "PolicyStepFn",
    "RoutineDecision",
    "RoutineState",
    "RoutineStep",
    "StreamSetpoint",
    "VehicleRoutine",
]


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
# ``robotics_experiments.rl.press.build_policy_step_fn``).
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
