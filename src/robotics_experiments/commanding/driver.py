"""Functional per-drone routine driver over mutable resource seams.

The frozen :class:`RoutineDriverState` carries project-owned progress. Stateless
functions return its successor while MAVLink and environment effects remain at
the imperative boundary.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Optional, Protocol

from robotics_experiments.commanding._routine_builder import build_vehicle_routine
from robotics_experiments.commanding._routine_model import (
    _DEFAULT_MIN_HEADING_DIST_M,
    DecisionKind,
    FailureKind,
    PolicyStepFn,
    RoutineDecision,
    RoutineState,
    VehicleRoutine,
)
from robotics_experiments.commanding._routine_predicates import EnvelopeGates
from robotics_experiments.commanding._routine_stepper import decide
from robotics_experiments.commanding.commands import (
    AttitudeRateSetpoint,
    PositionSetpoint,
    VelocitySetpoint,
)
from robotics_experiments.commanding.observation import CommandAck, RoutineObservation
from robotics_experiments.commanding.port import (
    MavlinkCommandPort,
    read_command_port_status,
)
from robotics_experiments.px4_bridge import (
    DEFAULT_HOME_ALT_M,
    DEFAULT_HOME_LAT_DEG,
    DEFAULT_HOME_LON_DEG,
)

__all__ = [
    "DriverEnvironment",
    "DriverParams",
    "DriverPhase",
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
    DONE = auto()


@dataclass(frozen=True)
class DriverParams:
    """The per-drone routine knobs required by the command driver."""

    steps: Sequence[Mapping[str, Any]]
    gates: EnvelopeGates
    command_ack_timeout_s: float
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
    home_lat_deg: float = DEFAULT_HOME_LAT_DEG
    home_lon_deg: float = DEFAULT_HOME_LON_DEG
    home_alt_m: float = DEFAULT_HOME_ALT_M
    goto_min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M
    prearm_backstop_floor_s: float = 20.0
    prearm_backstop_multiplier: float = 5.0
    drive_budget_multiplier: float = 4.0
    policy_fn_factory: Optional[Callable[[Mapping[str, Any]], PolicyStepFn]] = None


class DriverEnvironment(Protocol):
    """Mutable resource seam for simulator reads and lifecycle reactions."""

    def is_bridge_connected(self) -> bool: ...

    def sim_time_us(self) -> int: ...

    def read_observation(
        self, *, sim_time_us: int, last_ack: Optional[CommandAck]
    ) -> RoutineObservation: ...

    def log(self, message: str) -> None: ...

    def on_failure(self, kind: FailureKind, message: str) -> None: ...

    def on_step_advanced(self, step_label: str) -> None: ...


@dataclass(frozen=True)
class RoutineDriverState:
    """Immutable progress plus explicit references to the driver's resources."""

    params: DriverParams
    port: MavlinkCommandPort
    env: DriverEnvironment
    label: str
    monotonic: Callable[[], float]
    phase: DriverPhase
    deadline_s: float
    settle_target_us: int
    target: Optional[tuple[float, float, float]]
    routine: Optional[VehicleRoutine]
    routine_state: RoutineState
    previous_cursor: int


def _deadline_for_phase(*, state: RoutineDriverState, phase: DriverPhase) -> float:
    params = state.params
    now = state.monotonic()
    if phase is DriverPhase.WAIT_BRIDGE:
        return now + params.bridge_heartbeat_timeout_s
    if phase is DriverPhase.HANDSHAKE:
        return now + params.handshake_timeout_s
    if phase is DriverPhase.PREARM_SETTLE:
        return now + max(
            params.prearm_backstop_floor_s,
            params.prearm_settle_s * params.prearm_backstop_multiplier,
        )
    if phase is DriverPhase.FETCH_TARGET:
        return now + params.takeoff_target_timeout_s
    if phase is DriverPhase.DRIVE:
        routine_budget_s = (
            sum(step.timeout_s for step in state.routine.steps) if state.routine else 0.0
        )
        return (
            now
            + (
                routine_budget_s
                + params.takeoff_timeout_s
                + params.command_ack_timeout_s
                + params.handshake_timeout_s
            )
            * params.drive_budget_multiplier
        )
    return 0.0


def _enter(*, state: RoutineDriverState, phase: DriverPhase) -> RoutineDriverState:
    entered = replace(state, phase=phase)
    return replace(entered, deadline_s=_deadline_for_phase(state=entered, phase=phase))


def create_routine_driver(
    *,
    params: DriverParams,
    port: MavlinkCommandPort,
    env: DriverEnvironment,
    label: str = "",
    monotonic: Callable[[], float] = time.monotonic,
) -> RoutineDriverState:
    """Create a fresh frozen driver state without opening its resources."""
    state = RoutineDriverState(
        params=params,
        port=port,
        env=env,
        label=label,
        monotonic=monotonic,
        phase=DriverPhase.WAIT_BRIDGE,
        deadline_s=0.0,
        settle_target_us=0,
        target=None,
        routine=None,
        routine_state=RoutineState(),
        previous_cursor=0,
    )
    return _enter(state=state, phase=DriverPhase.WAIT_BRIDGE)


def routine_driver_is_done(*, state: RoutineDriverState) -> bool:
    """Return whether the driver reached its terminal phase."""
    return state.phase is DriverPhase.DONE


def _pump_and_poll(*, state: RoutineDriverState) -> Optional[CommandAck]:
    state.port.pump_heartbeat()
    return state.port.poll_feedback()


def _fail(*, state: RoutineDriverState, kind: FailureKind, message: str) -> RoutineDriverState:
    state.env.on_failure(kind, message)
    return _enter(state=state, phase=DriverPhase.DONE)


def _abort_on_failsafe(
    *, state: RoutineDriverState, phase_description: str
) -> Optional[RoutineDriverState]:
    status = read_command_port_status(port=state.port)
    if not status.failsafe:
        return None
    return _fail(
        state=state,
        kind=FailureKind.FAILSAFE,
        message=(
            f"[{state.label}] PX4 entered failsafe {phase_description}: "
            f"system_status={status.failsafe_system_status}"
        ),
    )


def _apply_decision(*, port: MavlinkCommandPort, decision: RoutineDecision) -> None:
    """Perform the single MAVLink side effect described by a pure decision."""
    if decision.kind is DecisionKind.SEND:
        assert decision.command is not None
        port.send(decision.command)
        return
    if decision.kind is not DecisionKind.STREAM:
        return
    setpoint = decision.setpoint
    if isinstance(setpoint, PositionSetpoint):
        port.send_setpoint(setpoint)
    elif isinstance(setpoint, VelocitySetpoint):
        port.send_velocity_setpoint(setpoint)
    elif isinstance(setpoint, AttitudeRateSetpoint):
        port.send_attitude_target(setpoint)
    else:
        raise TypeError(f"STREAM decision carried unsupported payload {setpoint!r}")


def _tick_drive(*, state: RoutineDriverState) -> RoutineDriverState:
    assert state.routine is not None
    last_ack = _pump_and_poll(state=state)
    failed = _abort_on_failsafe(state=state, phase_description="during the routine")
    if failed is not None:
        return failed
    sim_time_us = state.env.sim_time_us()
    observation = state.env.read_observation(sim_time_us=sim_time_us, last_ack=last_ack)
    routine_state, decision = decide(state.routine, state.routine_state, observation)
    _apply_decision(port=state.port, decision=decision)
    advanced = replace(state, routine_state=routine_state)
    if decision.kind is DecisionKind.FAIL:
        assert decision.failure_kind is not None
        return _fail(
            state=advanced,
            kind=decision.failure_kind,
            message=f"[{state.label}] routine failed: {decision.reason}",
        )
    previous_cursor = state.previous_cursor
    while previous_cursor < routine_state.cursor:
        state.env.on_step_advanced(state.routine.steps[previous_cursor].label)
        previous_cursor += 1
    advanced = replace(advanced, previous_cursor=previous_cursor)
    if routine_state.cursor >= len(state.routine.steps):
        state.env.log(f"[{state.label}] routine complete; holding, manual control available")
        return _enter(state=advanced, phase=DriverPhase.DONE)
    return advanced


def tick_routine_driver(*, state: RoutineDriverState) -> RoutineDriverState:
    """Advance one control frame and return the successor driver state."""
    if state.phase is DriverPhase.DONE:
        _pump_and_poll(state=state)
        return state
    if state.deadline_s and state.monotonic() > state.deadline_s:
        return _fail(
            state=state,
            kind=FailureKind.TIMEOUT,
            message=f"phase {state.phase.name} timed out",
        )
    if state.phase is DriverPhase.WAIT_BRIDGE:
        if state.env.is_bridge_connected():
            state.env.log(f"[{state.label}] bridge connected; connecting GCS link")
            return _enter(state=state, phase=DriverPhase.CONNECT)
        return state
    if state.phase is DriverPhase.CONNECT:
        state.port.connect()
        state.port.pump_heartbeat(force=True)
        return _enter(state=state, phase=DriverPhase.HANDSHAKE)
    if state.phase is DriverPhase.HANDSHAKE:
        _pump_and_poll(state=state)
        if read_command_port_status(port=state.port).is_handshaked:
            state.env.log(f"[{state.label}] GCS handshake complete; settling PX4 before arm")
            settled = replace(
                state,
                settle_target_us=state.env.sim_time_us()
                + int(round(state.params.prearm_settle_s * 1_000_000)),
            )
            return _enter(state=settled, phase=DriverPhase.PREARM_SETTLE)
        return state
    if state.phase is DriverPhase.PREARM_SETTLE:
        _pump_and_poll(state=state)
        failed = _abort_on_failsafe(state=state, phase_description="while settling before arm")
        if failed is not None:
            return failed
        if state.env.sim_time_us() >= state.settle_target_us:
            state.env.log(f"[{state.label}] prearm settle complete; waiting for takeoff telemetry")
            return _enter(state=state, phase=DriverPhase.FETCH_TARGET)
        return state
    if state.phase is DriverPhase.FETCH_TARGET:
        _pump_and_poll(state=state)
        failed = _abort_on_failsafe(
            state=state, phase_description="while waiting for altitude telemetry"
        )
        if failed is not None:
            return failed
        target = read_command_port_status(port=state.port).latest_global_position
        if target:
            return _enter(
                state=replace(state, target=target),
                phase=DriverPhase.BUILD,
            )
        return state
    if state.phase is DriverPhase.BUILD:
        assert state.target is not None
        latitude_deg, longitude_deg, altitude_m = state.target
        try:
            routine = build_vehicle_routine(
                state.params.steps,
                target_latitude_deg=latitude_deg,
                target_longitude_deg=longitude_deg,
                target_altitude_m=altitude_m,
                command_ack_timeout_s=state.params.command_ack_timeout_s,
                gates=state.params.gates,
                takeoff_timeout_s=state.params.takeoff_timeout_s,
                hover_settle_s=state.params.hover_settle_s,
                home_lat_deg=state.params.home_lat_deg,
                home_lon_deg=state.params.home_lon_deg,
                home_alt_m=state.params.home_alt_m,
                goto_position_tol_m=state.params.goto_position_tol_m,
                goto_heading_tol_deg=state.params.goto_heading_tol_deg,
                goto_cross_track_m=state.params.goto_cross_track_m,
                goto_offboard_prime_ticks=state.params.goto_offboard_prime_ticks,
                goto_min_heading_dist_m=state.params.goto_min_heading_dist_m,
                policy_fn_factory=state.params.policy_fn_factory,
            )
        except ValueError as exc:
            return _fail(
                state=state,
                kind=FailureKind.COMMAND,
                message=f"[{state.label}] routine build failed: {exc}",
            )
        state.env.log(f"[{state.label}] driving routine ({len(routine.steps)} step(s))")
        built = replace(
            state,
            routine=routine,
            routine_state=RoutineState(),
            previous_cursor=0,
        )
        return _enter(state=built, phase=DriverPhase.DRIVE)
    if state.phase is DriverPhase.DRIVE:
        return _tick_drive(state=state)
    return state
