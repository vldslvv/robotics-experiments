from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, auto
from typing import Optional, Protocol, Union

import numpy as np

from robotics_experiments._cadence import _CadenceState
from robotics_experiments.commanding.commands import (
    AttitudeRateSetpoint,
    MavlinkCommand,
    PositionSetpoint,
    VelocitySetpoint,
)
from robotics_experiments.commanding.observation import RoutineObservation
from robotics_experiments.px4_bridge import VehicleState

__all__ = [
    "DecisionKind",
    "FailureKind",
    "PolicyAdapter",
    "PolicyInput",
    "PolicyResult",
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


def _policy_state_is_immutable(*, state: object) -> bool:
    if state is None:
        return True
    if isinstance(state, (bool, bytes, float, frozenset, int, str, tuple)):
        values = state if isinstance(state, (frozenset, tuple)) else ()
        return all(_policy_state_is_immutable(state=value) for value in values)
    if not is_dataclass(state):
        return False
    params = vars(type(state))["__dataclass_params__"]
    return bool(params.frozen) and all(
        _policy_state_is_immutable(state=getattr(state, field.name)) for field in fields(state)
    )


def _require_immutable_policy_state(*, state: object, source: str) -> None:
    if state is None:
        raise ValueError(f"{source} must not be None")
    if not _policy_state_is_immutable(state=state):
        raise TypeError(f"{source} must be an immutable value, got {type(state).__name__}")


@dataclass(frozen=True)
class PolicyInput:
    """The complete task-neutral input for one policy transition."""

    vehicle_state: VehicleState
    sim_time_us: int
    task_observation: object
    reference_position_enu: tuple[float, float, float]


@dataclass(frozen=True)
class PolicyResult:
    """One task-neutral policy transition result."""

    state: object
    setpoint: StreamSetpoint
    complete: bool = False
    failure_reason: Optional[str] = None
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_immutable_policy_state(state=self.state, source="policy successor state")


class _PolicyAdvance(Protocol):
    def __call__(self, *, policy_input: PolicyInput, state: object) -> PolicyResult: ...


class _PolicyFactory(Protocol):
    def __call__(self, *, artifact: str, hold_s: Optional[float]) -> "PolicyAdapter": ...


@dataclass(frozen=True)
class PolicyAdapter:
    """Injected policy transition, cadence, and immutable initial state."""

    policy_rate_hz: float
    initial_state: object
    advance: _PolicyAdvance

    def __post_init__(self) -> None:
        _require_immutable_policy_state(state=self.initial_state, source="policy initial state")


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
    diagnostics: tuple[str, ...] = ()

    @classmethod
    def continue_(
        cls,
        *,
        command: Optional[MavlinkCommand] = None,
        setpoint: Optional[StreamSetpoint] = None,
        diagnostics: tuple[str, ...] = (),
    ) -> "RoutineDecision":
        return cls(
            DecisionKind.CONTINUE,
            command=command,
            setpoint=setpoint,
            diagnostics=diagnostics,
        )

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
            baseline) the cross-track gate. Policy specs retain concrete
            ``NaN`` placeholders so these numeric fields stay non-optional.
        latitude_deg, longitude_deg, altitude_m: The same waypoint precomputed
            (by :func:`build_vehicle_routine` via ``synth_gps_from_enu``) as the
            geodetic target the port streams, kept consistent by construction with
            the bridge's synthesized GPS.
        prime_ticks: How many setpoints to stream before engaging OFFBOARD (PX4
            rejects the mode until a setpoint stream is already flowing).
        min_heading_dist_m: Below this horizontal baseline-to-target distance
            the leg has no meaningful heading and the streamed setpoint holds
            the current heading (yaw ``NaN``).
        policy: When set, the scheduled generic policy adapter. It receives one
            explicit policy input and its opaque frozen state, then returns a
            task-neutral result. The OFFBOARD
            prime/engage/stream lifecycle is identical for every adapter.
        policy_reference_position_enu: An authored policy reference, or an
            all-``NaN`` sentinel. Commanding resolves the sentinel to the
            immutable step-entry baseline before invoking the adapter.
        policy_reference_is_entry: Whether the policy reference must be
            resolved from the step-entry baseline.
    """

    target_enu: tuple[float, float, float]
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    prime_ticks: int
    min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M
    policy: Optional[PolicyAdapter] = None
    policy_reference_position_enu: tuple[float, float, float] = (
        float("nan"),
        float("nan"),
        float("nan"),
    )
    policy_reference_is_entry: bool = False


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
    policy_state: Optional[object] = None
    policy_reference_position_enu: tuple[float, float, float] = (
        float("nan"),
        float("nan"),
        float("nan"),
    )
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
        last_policy_state: Final opaque adapter state from the most
            recently completed policy step. Integration/evaluation code may
            inspect its adapter-owned type; commanding never does.
    """

    cursor: int = 0
    sent: bool = False
    baseline_position_enu: Optional[np.ndarray] = None
    step_started_us: Optional[int] = None
    hold_since_us: Optional[int] = None
    offboard: _OffboardState = _OffboardState()
    last_policy_state: Optional[object] = None
