"""Functional per-drone routine driver over mutable resource seams.

The frozen :class:`DriverContext` carries resources and immutable parameters,
while :class:`RoutineDriverState` carries project-owned progress. Stateless
functions return the next state while MAVLink and environment effects remain at
the imperative boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Optional, Protocol

from robotics_experiments.commanding._routine_builder import build_vehicle_routine
from robotics_experiments.commanding._routine_model import (
    _DEFAULT_MIN_HEADING_DIST_M,
    DecisionKind,
    FailureKind,
    PolicyAdapter,
    RoutineDecision,
    RoutineState,
    VehicleRoutine,
)
from robotics_experiments.commanding._routine_predicates import EnvelopeGates
from robotics_experiments.commanding._routine_stepper import decide
from robotics_experiments.commanding.commands import (
    MAV_CMD_NAV_LAND,
    AttitudeRateSetpoint,
    PositionSetpoint,
    VelocitySetpoint,
    land,
)
from robotics_experiments.commanding.observation import (
    MAV_RESULT_TEMPORARILY_REJECTED,
    CommandAck,
    RoutineObservation,
)
from robotics_experiments.commanding.port import (
    MavlinkCommandPort,
)
from robotics_experiments.px4_bridge import (
    DEFAULT_HOME_ALT_M,
    DEFAULT_HOME_LAT_DEG,
    DEFAULT_HOME_LON_DEG,
)

__all__ = [
    "DriverContext",
    "DriverEnvironment",
    "DriverFailure",
    "DriverParams",
    "DriverPhase",
    "DriverTick",
    "RecoveryOutcome",
    "RecoveryStage",
    "RoutineDriverState",
    "create_routine_driver",
    "routine_driver_is_done",
    "tick_routine_driver",
]


class DriverPhase(Enum):
    """Stages of the per-drone routine flow."""

    WAIT_BRIDGE = auto()
    CONNECT = auto()
    HANDSHAKE = auto()
    PREARM_SETTLE = auto()
    FETCH_TARGET = auto()
    BUILD = auto()
    DRIVE = auto()
    RECOVERING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


class RecoveryStage(Enum):
    """Active stage of an armed vehicle's layered landing recovery."""

    WAIT_LAND_ACK = auto()
    WAIT_LAND_DISARM = auto()
    WAIT_PX4_FAILSAFE = auto()
    WAIT_PX4_FAILSAFE_DISARM = auto()
    WAIT_PX4_FAILSAFE_NO_LINK = auto()


class RecoveryOutcome(Enum):
    """Terminal disposition of a failed driver's recovery attempt."""

    PENDING = auto()
    NOT_REQUIRED = auto()
    LANDED = auto()
    PX4_FAILSAFE_LANDED = auto()
    DISARMED_AFTER_ESCALATION = auto()
    EXHAUSTED = auto()


@dataclass(frozen=True)
class DriverFailure:
    """Original driver failure plus its independent recovery disposition."""

    kind: FailureKind
    message: str
    recovery_outcome: RecoveryOutcome


@dataclass(frozen=True)
class DriverParams:
    """The per-drone routine knobs required by the command driver."""

    steps: Sequence[Mapping[str, Any]]
    gates: EnvelopeGates
    command_ack_timeout_s: float
    failure_recovery_timeout_s: float
    bridge_heartbeat_timeout_s: float
    handshake_timeout_s: float
    prearm_settle_s: float
    takeoff_target_timeout_s: float
    takeoff_timeout_s: float
    hover_settle_s: float
    goto_position_tol_m: float
    goto_heading_tol_deg: float
    goto_cross_track_m: float
    goto_offboard_prime_ticks: int
    offboard_setpoint_rate_hz: float
    home_lat_deg: float = DEFAULT_HOME_LAT_DEG
    home_lon_deg: float = DEFAULT_HOME_LON_DEG
    home_alt_m: float = DEFAULT_HOME_ALT_M
    goto_min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M
    prearm_backstop_floor_s: float = 20.0
    prearm_backstop_multiplier: float = 5.0
    drive_budget_multiplier: float = 4.0
    policy_fn_factory: Optional[Callable[[Mapping[str, Any]], PolicyAdapter]] = None


class DriverEnvironment(Protocol):
    """Mutable resource seam for simulator reads and lifecycle reactions."""

    def is_bridge_connected(self) -> bool: ...

    def read_observation(
        self, *, sim_time_us: int, last_ack: Optional[CommandAck]
    ) -> RoutineObservation: ...

    def log(self, message: str) -> None: ...

    def on_failure(self, kind: FailureKind, message: str) -> None:
        """Notify without raising; recovery must be allowed to finish."""
        ...

    def on_step_advanced(self, step_label: str) -> None: ...


@dataclass(frozen=True)
class DriverContext:
    """Immutable resources and parameters shared across driver transitions."""

    params: DriverParams
    port: MavlinkCommandPort
    env: DriverEnvironment
    label: str = ""


@dataclass(frozen=True)
class DriverTick:
    """One fleet-wide cyclic-executive snapshot."""

    control_time_us: int
    monotonic_s: float


@dataclass(frozen=True)
class RoutineDriverState:
    """Immutable progress for one routine driver."""

    phase: DriverPhase
    deadline_s: float
    settle_target_us: int
    target: Optional[tuple[float, float, float]]
    routine: Optional[VehicleRoutine]
    routine_state: RoutineState
    previous_cursor: int
    failure: Optional[DriverFailure]
    recovery_stage: Optional[RecoveryStage]
    recovery_ack_deadline_s: float


def _deadline_for_phase(
    *,
    context: DriverContext,
    state: RoutineDriverState,
    phase: DriverPhase,
    monotonic_s: float,
) -> float:
    params = context.params
    if phase is DriverPhase.WAIT_BRIDGE:
        return monotonic_s + params.bridge_heartbeat_timeout_s
    if phase is DriverPhase.HANDSHAKE:
        return monotonic_s + params.handshake_timeout_s
    if phase is DriverPhase.PREARM_SETTLE:
        return monotonic_s + max(
            params.prearm_backstop_floor_s,
            params.prearm_settle_s * params.prearm_backstop_multiplier,
        )
    if phase is DriverPhase.FETCH_TARGET:
        return monotonic_s + params.takeoff_target_timeout_s
    if phase is DriverPhase.DRIVE:
        routine_budget_s = (
            sum(step.timeout_s for step in state.routine.steps) if state.routine else 0.0
        )
        return (
            monotonic_s
            + (
                routine_budget_s
                + params.takeoff_timeout_s
                + params.command_ack_timeout_s
                + params.handshake_timeout_s
            )
            * params.drive_budget_multiplier
        )
    return 0.0


def _enter(
    *, context: DriverContext, state: RoutineDriverState, phase: DriverPhase, tick: DriverTick
) -> RoutineDriverState:
    return replace(
        state,
        phase=phase,
        deadline_s=_deadline_for_phase(
            context=context,
            state=state,
            phase=phase,
            monotonic_s=tick.monotonic_s,
        ),
    )


def create_routine_driver(
    *,
    context: DriverContext,
    tick: DriverTick,
) -> RoutineDriverState:
    """Create a fresh frozen driver state without opening its resources."""
    state = RoutineDriverState(
        phase=DriverPhase.WAIT_BRIDGE,
        deadline_s=0.0,
        settle_target_us=0,
        target=None,
        routine=None,
        routine_state=RoutineState(),
        previous_cursor=0,
        failure=None,
        recovery_stage=None,
        recovery_ack_deadline_s=0.0,
    )
    return _enter(
        context=context,
        state=state,
        phase=DriverPhase.WAIT_BRIDGE,
        tick=tick,
    )


def routine_driver_is_done(*, state: RoutineDriverState) -> bool:
    """Return whether the driver reached its terminal phase."""
    return state.phase in (DriverPhase.SUCCEEDED, DriverPhase.FAILED)


def _pump_and_poll(*, context: DriverContext, tick: DriverTick) -> Optional[CommandAck]:
    context.port.pump_heartbeat(now_s=tick.monotonic_s)
    return context.port.poll_feedback()


def _terminal_failure(
    *,
    state: RoutineDriverState,
    outcome: RecoveryOutcome,
) -> RoutineDriverState:
    assert state.failure is not None
    return replace(
        state,
        phase=DriverPhase.FAILED,
        deadline_s=0.0,
        failure=replace(state.failure, recovery_outcome=outcome),
        recovery_stage=None,
        recovery_ack_deadline_s=0.0,
    )


def _start_failure(
    *,
    context: DriverContext,
    state: RoutineDriverState,
    tick: DriverTick,
    kind: FailureKind,
    message: str,
    observation: Optional[RoutineObservation] = None,
) -> RoutineDriverState:
    context.env.on_failure(kind, message)
    failed = replace(
        state,
        failure=DriverFailure(
            kind=kind,
            message=message,
            recovery_outcome=RecoveryOutcome.PENDING,
        ),
    )
    if observation is None or not observation.bridge_status.is_armed:
        return _terminal_failure(state=failed, outcome=RecoveryOutcome.NOT_REQUIRED)

    recovery_deadline_s = tick.monotonic_s + context.params.failure_recovery_timeout_s
    if kind is FailureKind.FAILSAFE or context.port.read_status().failsafe:
        context.env.log(
            f"[{context.label}] failure recovery: PX4 failsafe is active; waiting for disarm"
        )
        return replace(
            failed,
            phase=DriverPhase.RECOVERING,
            deadline_s=recovery_deadline_s,
            recovery_stage=RecoveryStage.WAIT_PX4_FAILSAFE_DISARM,
            recovery_ack_deadline_s=0.0,
        )

    context.env.log(f"[{context.label}] failure recovery: commanding Land at the current position")
    try:
        context.port.send(land())
    except Exception as exc:  # noqa: BLE001 - external link failure must still reach fallback
        context.env.log(
            f"[{context.label}] failure recovery: Land send failed "
            f"({type(exc).__name__}: {exc}); stopping OFFBOARD stream for PX4 fallback"
        )
        return replace(
            failed,
            phase=DriverPhase.RECOVERING,
            deadline_s=recovery_deadline_s,
            recovery_stage=RecoveryStage.WAIT_PX4_FAILSAFE,
            recovery_ack_deadline_s=0.0,
        )
    return replace(
        failed,
        phase=DriverPhase.RECOVERING,
        deadline_s=recovery_deadline_s,
        recovery_stage=RecoveryStage.WAIT_LAND_ACK,
        recovery_ack_deadline_s=tick.monotonic_s + context.params.command_ack_timeout_s,
    )


def _abort_on_failsafe(
    *,
    context: DriverContext,
    state: RoutineDriverState,
    tick: DriverTick,
    phase_description: str,
    observation: Optional[RoutineObservation] = None,
) -> Optional[RoutineDriverState]:
    status = context.port.read_status()
    if not status.failsafe:
        return None
    if observation is None:
        observation = context.env.read_observation(
            sim_time_us=tick.control_time_us,
            last_ack=None,
        )
    return _start_failure(
        context=context,
        state=state,
        tick=tick,
        kind=FailureKind.FAILSAFE,
        message=(
            f"[{context.label}] PX4 entered failsafe {phase_description}: "
            f"system_status={status.failsafe_system_status}"
        ),
        observation=observation,
    )


def _apply_decision(*, port: MavlinkCommandPort, decision: RoutineDecision) -> None:
    """Apply independent effects, transmitting a setpoint before a command."""
    setpoint = decision.setpoint
    if isinstance(setpoint, PositionSetpoint):
        port.send_setpoint(setpoint)
    elif isinstance(setpoint, VelocitySetpoint):
        port.send_velocity_setpoint(setpoint)
    elif isinstance(setpoint, AttitudeRateSetpoint):
        port.send_attitude_target(setpoint)
    elif setpoint is not None:
        raise TypeError(f"routine decision carried unsupported setpoint {setpoint!r}")
    if decision.command is not None:
        port.send(decision.command)


def _tick_drive(
    *, context: DriverContext, state: RoutineDriverState, tick: DriverTick
) -> RoutineDriverState:
    assert state.routine is not None
    last_ack = _pump_and_poll(context=context, tick=tick)
    observation = context.env.read_observation(
        sim_time_us=tick.control_time_us,
        last_ack=last_ack,
    )
    failed = _abort_on_failsafe(
        context=context,
        state=state,
        tick=tick,
        phase_description="during the routine",
        observation=observation,
    )
    if failed is not None:
        return failed
    routine_state, decision = decide(state.routine, state.routine_state, observation)
    _apply_decision(port=context.port, decision=decision)
    if decision.kind is DecisionKind.FAIL:
        assert decision.failure_kind is not None
        return _start_failure(
            context=context,
            state=replace(
                state,
                routine_state=routine_state,
            ),
            tick=tick,
            kind=decision.failure_kind,
            message=f"[{context.label}] routine failed: {decision.reason}",
            observation=observation,
        )
    previous_cursor = state.previous_cursor
    while previous_cursor < routine_state.cursor:
        context.env.on_step_advanced(state.routine.steps[previous_cursor].label)
        previous_cursor += 1
    if routine_state.cursor >= len(state.routine.steps):
        context.env.log(f"[{context.label}] routine complete")
        return replace(
            state,
            phase=DriverPhase.SUCCEEDED,
            deadline_s=0.0,
            routine_state=routine_state,
            previous_cursor=previous_cursor,
        )
    if routine_state is state.routine_state and previous_cursor == state.previous_cursor:
        return state
    return replace(
        state,
        routine_state=routine_state,
        previous_cursor=previous_cursor,
    )


def _tick_recovery(
    *, context: DriverContext, state: RoutineDriverState, tick: DriverTick
) -> RoutineDriverState:
    """Advance explicit Land recovery, then PX4 offboard-loss fallback."""
    assert state.recovery_stage is not None
    stage = state.recovery_stage
    last_ack = None
    if stage is not RecoveryStage.WAIT_PX4_FAILSAFE_NO_LINK:
        try:
            last_ack = _pump_and_poll(context=context, tick=tick)
        except Exception as exc:  # noqa: BLE001 - external link failure must still recover
            context.env.log(
                f"[{context.label}] failure recovery: GCS link unavailable "
                f"({type(exc).__name__}: {exc}); waiting for PX4 fallback without control link"
            )
            state = replace(
                state,
                recovery_stage=RecoveryStage.WAIT_PX4_FAILSAFE_NO_LINK,
                recovery_ack_deadline_s=0.0,
            )
            stage = RecoveryStage.WAIT_PX4_FAILSAFE_NO_LINK
    observation = context.env.read_observation(
        sim_time_us=tick.control_time_us,
        last_ack=last_ack,
    )

    if not observation.bridge_status.is_armed:
        if stage is RecoveryStage.WAIT_PX4_FAILSAFE_DISARM:
            outcome = RecoveryOutcome.PX4_FAILSAFE_LANDED
        elif stage in (
            RecoveryStage.WAIT_PX4_FAILSAFE,
            RecoveryStage.WAIT_PX4_FAILSAFE_NO_LINK,
        ):
            outcome = RecoveryOutcome.DISARMED_AFTER_ESCALATION
        else:
            outcome = RecoveryOutcome.LANDED
        context.env.log(f"[{context.label}] failure recovery complete: vehicle disarmed")
        return _terminal_failure(state=state, outcome=outcome)

    if tick.monotonic_s > state.deadline_s:
        context.env.log(
            f"[{context.label}] failure recovery exhausted after "
            f"{context.params.failure_recovery_timeout_s:.1f}s"
        )
        return _terminal_failure(state=state, outcome=RecoveryOutcome.EXHAUSTED)

    if context.port.read_status().failsafe and stage is not RecoveryStage.WAIT_PX4_FAILSAFE_DISARM:
        context.env.log(
            f"[{context.label}] failure recovery: PX4 failsafe engaged; waiting for disarm"
        )
        return replace(
            state,
            recovery_stage=RecoveryStage.WAIT_PX4_FAILSAFE_DISARM,
            recovery_ack_deadline_s=0.0,
        )

    if stage is RecoveryStage.WAIT_LAND_ACK:
        if last_ack is not None and last_ack.command_id == MAV_CMD_NAV_LAND:
            if last_ack.accepted:
                context.env.log(
                    f"[{context.label}] failure recovery: Land accepted; waiting for disarm"
                )
                return replace(
                    state,
                    recovery_stage=RecoveryStage.WAIT_LAND_DISARM,
                    recovery_ack_deadline_s=0.0,
                )
            if last_ack.result == MAV_RESULT_TEMPORARILY_REJECTED:
                context.env.log(
                    f"[{context.label}] failure recovery: Land temporarily rejected; retrying"
                )
                try:
                    context.port.send(land())
                except Exception as exc:  # noqa: BLE001 - retry failure must still reach fallback
                    context.env.log(
                        f"[{context.label}] failure recovery: Land retry failed "
                        f"({type(exc).__name__}: {exc}); "
                        "stopping OFFBOARD stream for PX4 fallback"
                    )
                    return replace(
                        state,
                        recovery_stage=RecoveryStage.WAIT_PX4_FAILSAFE,
                        recovery_ack_deadline_s=0.0,
                    )
                return replace(
                    state,
                    recovery_ack_deadline_s=(
                        tick.monotonic_s + context.params.command_ack_timeout_s
                    ),
                )
            context.env.log(
                f"[{context.label}] failure recovery: Land rejected "
                f"(result={last_ack.result}); stopping OFFBOARD stream for PX4 fallback"
            )
            return replace(
                state,
                recovery_stage=RecoveryStage.WAIT_PX4_FAILSAFE,
                recovery_ack_deadline_s=0.0,
            )
        if tick.monotonic_s > state.recovery_ack_deadline_s:
            context.env.log(
                f"[{context.label}] failure recovery: no Land acknowledgement; "
                "stopping OFFBOARD stream for PX4 fallback"
            )
            return replace(
                state,
                recovery_stage=RecoveryStage.WAIT_PX4_FAILSAFE,
                recovery_ack_deadline_s=0.0,
            )
    return state


def tick_routine_driver(
    *, context: DriverContext, state: RoutineDriverState, tick: DriverTick
) -> RoutineDriverState:
    """Advance one control frame and return the successor driver state."""
    if state.phase in (DriverPhase.SUCCEEDED, DriverPhase.FAILED):
        _pump_and_poll(context=context, tick=tick)
        return state
    if state.phase is DriverPhase.RECOVERING:
        return _tick_recovery(context=context, state=state, tick=tick)
    if state.deadline_s and tick.monotonic_s > state.deadline_s:
        observation = None
        if state.phase is DriverPhase.DRIVE:
            last_ack = _pump_and_poll(context=context, tick=tick)
            observation = context.env.read_observation(
                sim_time_us=tick.control_time_us,
                last_ack=last_ack,
            )
        return _start_failure(
            context=context,
            state=state,
            tick=tick,
            kind=FailureKind.TIMEOUT,
            message=f"phase {state.phase.name} timed out",
            observation=observation,
        )
    if state.phase is DriverPhase.WAIT_BRIDGE:
        if context.env.is_bridge_connected():
            context.env.log(f"[{context.label}] bridge connected; connecting GCS link")
            return _enter(context=context, state=state, phase=DriverPhase.CONNECT, tick=tick)
        return state
    if state.phase is DriverPhase.CONNECT:
        context.port.connect()
        context.port.pump_heartbeat(now_s=tick.monotonic_s, force=True)
        return _enter(context=context, state=state, phase=DriverPhase.HANDSHAKE, tick=tick)
    if state.phase is DriverPhase.HANDSHAKE:
        _pump_and_poll(context=context, tick=tick)
        if context.port.read_status().is_handshaked:
            context.env.log(f"[{context.label}] GCS handshake complete; settling PX4 before arm")
            settled = replace(
                state,
                settle_target_us=tick.control_time_us
                + int(round(context.params.prearm_settle_s * 1_000_000)),
            )
            return _enter(
                context=context,
                state=settled,
                phase=DriverPhase.PREARM_SETTLE,
                tick=tick,
            )
        return state
    if state.phase is DriverPhase.PREARM_SETTLE:
        _pump_and_poll(context=context, tick=tick)
        failed = _abort_on_failsafe(
            context=context,
            state=state,
            tick=tick,
            phase_description="while settling before arm",
        )
        if failed is not None:
            return failed
        if tick.control_time_us >= state.settle_target_us:
            context.env.log(
                f"[{context.label}] prearm settle complete; waiting for takeoff telemetry"
            )
            return _enter(
                context=context,
                state=state,
                phase=DriverPhase.FETCH_TARGET,
                tick=tick,
            )
        return state
    if state.phase is DriverPhase.FETCH_TARGET:
        _pump_and_poll(context=context, tick=tick)
        failed = _abort_on_failsafe(
            context=context,
            state=state,
            tick=tick,
            phase_description="while waiting for altitude telemetry",
        )
        if failed is not None:
            return failed
        target = context.port.read_status().latest_global_position
        if target is not None:
            return _enter(
                context=context,
                state=replace(state, target=target),
                phase=DriverPhase.BUILD,
                tick=tick,
            )
        return state
    if state.phase is DriverPhase.BUILD:
        assert state.target is not None
        latitude_deg, longitude_deg, altitude_m = state.target
        try:
            routine = build_vehicle_routine(
                steps=context.params.steps,
                target_latitude_deg=latitude_deg,
                target_longitude_deg=longitude_deg,
                target_altitude_m=altitude_m,
                command_ack_timeout_s=context.params.command_ack_timeout_s,
                gates=context.params.gates,
                takeoff_timeout_s=context.params.takeoff_timeout_s,
                hover_settle_s=context.params.hover_settle_s,
                home_lat_deg=context.params.home_lat_deg,
                home_lon_deg=context.params.home_lon_deg,
                home_alt_m=context.params.home_alt_m,
                goto_position_tol_m=context.params.goto_position_tol_m,
                goto_heading_tol_deg=context.params.goto_heading_tol_deg,
                goto_cross_track_m=context.params.goto_cross_track_m,
                goto_offboard_prime_ticks=context.params.goto_offboard_prime_ticks,
                offboard_setpoint_rate_hz=context.params.offboard_setpoint_rate_hz,
                goto_min_heading_dist_m=context.params.goto_min_heading_dist_m,
                policy_fn_factory=context.params.policy_fn_factory,
            )
        except ValueError as exc:
            return _start_failure(
                context=context,
                state=state,
                tick=tick,
                kind=FailureKind.COMMAND,
                message=f"[{context.label}] routine build failed: {exc}",
            )
        context.env.log(f"[{context.label}] driving routine ({len(routine.steps)} step(s))")
        built = replace(
            state,
            routine=routine,
            routine_state=RoutineState(),
            previous_cursor=0,
        )
        return _enter(context=context, state=built, phase=DriverPhase.DRIVE, tick=tick)
    if state.phase is DriverPhase.DRIVE:
        return _tick_drive(context=context, state=state, tick=tick)
    return state
