from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional, Union

import numpy as np

from robotics_experiments._cadence import _CadenceState
from robotics_experiments.commanding.commands import (
    AttitudeRateSetpoint,
    MavlinkCommand,
    PositionSetpoint,
    VelocitySetpoint,
)
from robotics_experiments.commanding.observation import RoutineObservation

__all__ = [
    "DecisionKind",
    "FailureKind",
    "PolicyAdapter",
    "PolicyStepFn",
    "PolicyStepResult",
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

# Any OFFBOARD setpoint payload a decision can carry; the commander
# dispatches on the concrete type to the matching port send method.
StreamSetpoint = Union[PositionSetpoint, VelocitySetpoint, AttitudeRateSetpoint]


@dataclass(frozen=True)
class PolicyStepResult:
    """One task-neutral policy inference result.

    ``progress`` is an opaque frozen value owned by the adapter. Commanding
    stores it but never inspects it. ``complete`` and ``failure_reason`` are
    likewise adapter decisions, allowing hover-duration and contact/latch
    completion to use the same routine kind.
    """

    progress: Any
    setpoint: StreamSetpoint
    complete: bool = False
    failure_reason: Optional[str] = None


PolicyStepFn = Callable[
    [RoutineObservation, Optional[np.ndarray], Any],
    PolicyStepResult,
]


@dataclass(frozen=True)
class PolicyAdapter:
    """Injected policy implementation, cadence, and immutable initial progress."""

    rate_hz: float
    initial_progress: Any
    step: PolicyStepFn


class DecisionKind(Enum):
    """The control-flow outcome of a :class:`RoutineDecision`."""

    CONTINUE = auto()
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

    ``kind`` controls routine progress while the optional ``setpoint`` and
    ``command`` independently describe effects for this cycle. Both may be set;
    the imperative driver transmits the setpoint first. ``reason`` and
    ``failure_kind`` are set only for ``FAIL``.
    """

    kind: DecisionKind
    command: Optional[MavlinkCommand] = None
    setpoint: Optional[StreamSetpoint] = None
    reason: Optional[str] = None
    failure_kind: Optional[FailureKind] = None

    @classmethod
    def continue_(
        cls,
        *,
        command: Optional[MavlinkCommand] = None,
        setpoint: Optional[StreamSetpoint] = None,
    ) -> "RoutineDecision":
        return cls(DecisionKind.CONTINUE, command=command, setpoint=setpoint)

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
            then streams the held setpoint at the configured rate until completion. When
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
            metres, used to compute the streamed heading and (with the captured
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
        policy: When set, the scheduled generic policy adapter. It receives the
            full command observation, optional target/baseline, and its opaque
            frozen progress, then returns a task-neutral result. The OFFBOARD
            prime/engage/stream lifecycle is identical for every adapter.
    """

    target_enu: tuple[float, float, float]
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    prime_ticks: int
    min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M
    policy: Optional[PolicyAdapter] = None


@dataclass(frozen=True)
class VehicleRoutine:
    """An ordered, immutable plan of :class:`RoutineStep`s for one vehicle."""

    steps: tuple[RoutineStep, ...]
    offboard_setpoint_rate_hz: float


@dataclass(frozen=True)
class _OffboardState:
    """Continuous OFFBOARD policy, transmission, payload, and mode progress."""

    policy_cadence: _CadenceState = _CadenceState()
    transmitter_cadence: _CadenceState = _CadenceState()
    transmissions: int = 0
    cached_setpoint: Optional[StreamSetpoint] = None
    requested: bool = False
    accepted: bool = False
    policy_progress: Any = None
    policy_complete: bool = False
    policy_failure_reason: Optional[str] = None


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
        offboard: Continuous OFFBOARD progress: policy and transmitter cadence,
            zero-order-held payload, actual transmission count, and mode request
            / acceptance flags. Compatible streaming and command-less verifier
            steps retain this value so cadence phase remains continuous.
        last_policy_progress: Final opaque adapter progress from the most
            recently completed policy step. Integration/evaluation code may
            inspect its adapter-owned type; commanding never does.
    """

    cursor: int = 0
    sent: bool = False
    baseline_position_enu: Optional[np.ndarray] = None
    step_started_us: Optional[int] = None
    hold_since_us: Optional[int] = None
    offboard: _OffboardState = _OffboardState()
    last_policy_progress: Any = None
