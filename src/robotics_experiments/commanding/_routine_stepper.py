import math
from dataclasses import replace
from typing import Optional

import numpy as np

from robotics_experiments._cadence import _advance_cadence, _CadenceState
from robotics_experiments.commanding._routine_model import (
    FailureKind,
    PolicyInput,
    RoutineDecision,
    RoutineState,
    StreamSetpoint,
    VehicleRoutine,
    _OffboardState,
    _StreamingSpec,
)
from robotics_experiments.commanding.commands import MavlinkCommand, PositionSetpoint
from robotics_experiments.commanding.observation import (
    MAV_RESULT_TEMPORARILY_REJECTED,
    RoutineObservation,
)

__all__ = ["decide"]


def _anchored_cadence(*, sim_time_us: int) -> _CadenceState:
    return _CadenceState(previous_time_ns=sim_time_us * 1_000)


def _advanced(*, state: RoutineState, policy_state: Optional[object] = None) -> RoutineState:
    """Advance without interrupting a compatible OFFBOARD stream."""
    return replace(
        state,
        cursor=state.cursor + 1,
        sent=False,
        baseline_position_enu=None,
        step_started_us=None,
        hold_since_us=None,
        offboard=replace(
            state.offboard,
            policy_state=None,
            policy_reference_position_enu=(math.nan, math.nan, math.nan),
            policy_complete=False,
            policy_failure_reason=None,
        ),
        last_policy_state=(state.last_policy_state if policy_state is None else policy_state),
    )


def _policy_input(
    *,
    reference_position_enu: tuple[float, float, float],
    observation: RoutineObservation,
) -> PolicyInput:
    return PolicyInput(
        vehicle_state=observation.vehicle_state,
        sim_time_us=int(observation.sim_time_us),
        task_observation=observation.task_observation,
        reference_position_enu=reference_position_enu,
    )


def _advance_stream_cadences(
    *,
    routine: VehicleRoutine,
    offboard: _OffboardState,
    spec: Optional[_StreamingSpec],
    observation: RoutineObservation,
    requested: bool,
    accepted: bool,
    prime_ticks: int,
    command: Optional[MavlinkCommand],
    retry_deferred: bool,
) -> tuple[
    _OffboardState,
    Optional[StreamSetpoint],
    Optional[MavlinkCommand],
    tuple[str, ...],
]:
    """Calculate one OFFBOARD successor plus independent wire effects."""
    now_ns = int(observation.sim_time_us) * 1_000
    policy_cadence = offboard.policy_cadence
    cached_setpoint = offboard.cached_setpoint
    policy_state = offboard.policy_state
    policy_complete = offboard.policy_complete
    policy_failure_reason = offboard.policy_failure_reason
    diagnostics: tuple[str, ...] = ()
    if spec is not None and spec.policy is not None:
        policy_cadence, policy_due = _advance_cadence(
            state=policy_cadence,
            time_ns=now_ns,
            rate_hz=spec.policy.policy_rate_hz,
        )
        if policy_due:
            result = spec.policy.advance(
                policy_input=_policy_input(
                    reference_position_enu=offboard.policy_reference_position_enu,
                    observation=observation,
                ),
                state=spec.policy.initial_state if policy_state is None else policy_state,
            )
            cached_setpoint = result.setpoint
            policy_state = result.state
            policy_complete = result.complete
            policy_failure_reason = result.failure_reason
            diagnostics = result.diagnostics
    assert cached_setpoint is not None

    transmitter_cadence, transmit_due = _advance_cadence(
        state=offboard.transmitter_cadence,
        time_ns=now_ns,
        rate_hz=routine.offboard_setpoint_rate_hz,
    )
    transmissions = offboard.transmissions + int(transmit_due)
    command_effect = None
    if not retry_deferred and not requested and transmissions >= prime_ticks:
        assert command is not None
        command_effect = command
        requested = True
    successor = _OffboardState(
        policy_cadence=policy_cadence,
        transmitter_cadence=transmitter_cadence,
        transmissions=transmissions,
        cached_setpoint=cached_setpoint,
        requested=requested,
        accepted=accepted,
        policy_state=policy_state,
        policy_reference_position_enu=offboard.policy_reference_position_enu,
        policy_complete=policy_complete,
        policy_failure_reason=policy_failure_reason,
    )
    return successor, cached_setpoint if transmit_due else None, command_effect, diagnostics


def _enter_stream(
    *,
    routine: VehicleRoutine,
    offboard: _OffboardState,
    spec: _StreamingSpec,
    baseline: np.ndarray,
    observation: RoutineObservation,
    command: Optional[MavlinkCommand],
) -> tuple[
    _OffboardState,
    Optional[StreamSetpoint],
    Optional[MavlinkCommand],
    tuple[str, ...],
]:
    """Enter a streaming step without resetting an active OFFBOARD phase."""
    now_us = int(observation.sim_time_us)
    policy_reference_position_enu = offboard.policy_reference_position_enu
    if spec.policy is not None:
        policy_reference_position_enu = (
            (float(baseline[0]), float(baseline[1]), float(baseline[2]))
            if spec.policy_reference_is_entry
            else spec.policy_reference_position_enu
        )
    if offboard.cached_setpoint is None:
        if spec.policy is None:
            setpoint = _setpoint_for(spec=spec, baseline=baseline)
            policy_state = None
            policy_complete = False
            policy_failure_reason = None
        else:
            result = spec.policy.advance(
                policy_input=_policy_input(
                    reference_position_enu=policy_reference_position_enu,
                    observation=observation,
                ),
                state=spec.policy.initial_state,
            )
            setpoint = result.setpoint
            policy_state = result.state
            policy_complete = result.complete
            policy_failure_reason = result.failure_reason
            diagnostics = result.diagnostics
        if spec.policy is None:
            diagnostics = ()
        transmitter_cadence = _anchored_cadence(sim_time_us=now_us)
        policy_cadence = transmitter_cadence if spec.policy is not None else _CadenceState()
        requested = spec.prime_ticks <= 1
        successor = _OffboardState(
            policy_cadence=policy_cadence,
            transmitter_cadence=transmitter_cadence,
            transmissions=1,
            cached_setpoint=setpoint,
            requested=requested,
            accepted=False,
            policy_state=policy_state,
            policy_reference_position_enu=policy_reference_position_enu,
            policy_complete=policy_complete,
            policy_failure_reason=policy_failure_reason,
        )
        return successor, setpoint, command if requested else None, diagnostics

    if spec.policy is None:
        offboard = replace(
            offboard,
            cached_setpoint=_setpoint_for(spec=spec, baseline=baseline),
        )
    else:
        result = spec.policy.advance(
            policy_input=_policy_input(
                reference_position_enu=policy_reference_position_enu,
                observation=observation,
            ),
            state=spec.policy.initial_state,
        )
        offboard = replace(
            offboard,
            policy_cadence=_anchored_cadence(sim_time_us=now_us),
            cached_setpoint=result.setpoint,
            policy_state=result.state,
            policy_reference_position_enu=policy_reference_position_enu,
            policy_complete=result.complete,
            policy_failure_reason=result.failure_reason,
        )
    successor, setpoint, command_effect, diagnostics = _advance_stream_cadences(
        routine=routine,
        offboard=offboard,
        spec=spec,
        observation=observation,
        requested=offboard.requested,
        accepted=offboard.accepted,
        prime_ticks=spec.prime_ticks,
        command=command,
        retry_deferred=False,
    )
    return (
        successor,
        setpoint,
        command_effect,
        result.diagnostics if spec.policy is not None else diagnostics,
    )


def _enter_step(
    *,
    routine: VehicleRoutine,
    state: RoutineState,
    observation: RoutineObservation,
) -> tuple[RoutineState, RoutineDecision]:
    step = routine.steps[state.cursor]
    baseline = np.array(observation.vehicle_state.position_enu, dtype=np.float64, copy=True)
    baseline.setflags(write=False)
    now_us = int(observation.sim_time_us)

    offboard = state.offboard
    setpoint = None
    command = step.command
    diagnostics: tuple[str, ...] = ()
    if step.streaming is not None:
        offboard, setpoint, command, diagnostics = _enter_stream(
            routine=routine,
            offboard=offboard,
            spec=step.streaming,
            baseline=baseline,
            observation=observation,
            command=command,
        )
    elif step.command is not None:
        offboard = _OffboardState()
    elif offboard.cached_setpoint is not None:
        offboard, setpoint, _, diagnostics = _advance_stream_cadences(
            routine=routine,
            offboard=offboard,
            spec=None,
            observation=observation,
            requested=offboard.requested,
            accepted=offboard.accepted,
            prime_ticks=0,
            command=None,
            retry_deferred=True,
        )

    entered = replace(
        state,
        sent=True,
        baseline_position_enu=baseline,
        step_started_us=now_us,
        hold_since_us=None,
        offboard=offboard,
    )
    return entered, RoutineDecision.continue_(
        setpoint=setpoint,
        command=command,
        diagnostics=diagnostics,
    )


def decide(
    routine: VehicleRoutine,
    state: RoutineState,
    observation: RoutineObservation,
) -> tuple[RoutineState, RoutineDecision]:
    """Purely advance one routine cycle and describe its independent effects."""
    steps = routine.steps
    if state.cursor >= len(steps):
        return state, RoutineDecision.done()
    if not state.sent:
        return _enter_step(routine=routine, state=state, observation=observation)

    step = steps[state.cursor]
    for invariant in step.invariants:
        breach = invariant(observation, state.baseline_position_enu)
        if breach is not None:
            return state, RoutineDecision.fail(
                f"step {step.label!r} flight envelope breached: {breach}",
                failure_kind=FailureKind.ENVELOPE,
            )

    completion_enabled = (
        not (step.streaming is not None and step.streaming.policy is not None)
        or state.offboard.accepted
    )
    policy_complete = step.streaming is not None and step.streaming.policy is not None
    if policy_complete and state.offboard.policy_failure_reason is not None:
        return state, RoutineDecision.fail(
            f"step {step.label!r} policy failed: {state.offboard.policy_failure_reason}",
            failure_kind=FailureKind.COMMAND,
        )
    completed = (
        state.offboard.policy_complete
        if policy_complete
        else step.is_complete(observation, state.baseline_position_enu)
    )
    if completion_enabled and completed:
        if step.hold_s <= 0.0:
            return _advanced(
                state=state,
                policy_state=(state.offboard.policy_state if policy_complete else None),
            ), RoutineDecision.advance()
        hold_since_us = state.hold_since_us
        if hold_since_us is None:
            hold_since_us = int(observation.sim_time_us)
        if int(observation.sim_time_us) - hold_since_us >= step.hold_s * 1_000_000:
            return _advanced(state=state), RoutineDecision.advance()
        state = replace(state, hold_since_us=hold_since_us)
    elif state.hold_since_us is not None:
        state = replace(state, hold_since_us=None)

    if step.streaming is not None:
        return _decide_streaming(
            routine=routine,
            state=state,
            observation=observation,
            spec=step.streaming,
        )

    ack = observation.last_ack
    if (
        ack is not None
        and step.command is not None
        and ack.command_id == step.command.command_id
        and not ack.accepted
    ):
        if step.retry_on_temporarily_rejected and ack.result == MAV_RESULT_TEMPORARILY_REJECTED:
            return state, RoutineDecision.continue_(command=step.command)
        return state, RoutineDecision.fail(
            f"step {step.label!r} rejected: command={step.command.command_id} result={ack.result}",
            failure_kind=FailureKind.COMMAND,
        )

    step_started_us = state.step_started_us
    assert step_started_us is not None
    elapsed_us = int(observation.sim_time_us) - step_started_us
    if elapsed_us > step.timeout_s * 1_000_000:
        return state, RoutineDecision.fail(
            f"step {step.label!r} timed out after {step.timeout_s:.1f}s "
            f"(sim elapsed {elapsed_us * 1e-6:.1f}s)",
            failure_kind=FailureKind.COMMAND,
        )

    if step.command is None and state.offboard.cached_setpoint is not None:
        offboard, setpoint, _, diagnostics = _advance_stream_cadences(
            routine=routine,
            offboard=state.offboard,
            spec=None,
            observation=observation,
            requested=state.offboard.requested,
            accepted=state.offboard.accepted,
            prime_ticks=0,
            command=None,
            retry_deferred=True,
        )
        successor = state if offboard == state.offboard else replace(state, offboard=offboard)
        return successor, RoutineDecision.continue_(
            setpoint=setpoint,
            diagnostics=diagnostics,
        )
    return state, RoutineDecision.continue_()


def _decide_streaming(
    *,
    routine: VehicleRoutine,
    state: RoutineState,
    observation: RoutineObservation,
    spec: _StreamingSpec,
) -> tuple[RoutineState, RoutineDecision]:
    step = routine.steps[state.cursor]
    step_started_us = state.step_started_us
    assert step_started_us is not None
    elapsed_us = int(observation.sim_time_us) - step_started_us
    if elapsed_us > step.timeout_s * 1_000_000:
        if spec.policy is not None:
            detail = f"policy did not complete; policy state={state.offboard.policy_state!r}"
        else:
            detail = "did not reach the waypoint"
        return state, RoutineDecision.fail(
            f"step {step.label!r} timed out after {step.timeout_s:.1f}s ({detail})",
            failure_kind=FailureKind.COMMAND,
        )

    requested = state.offboard.requested
    accepted = state.offboard.accepted
    hold_since_us = state.hold_since_us
    retry_deferred = False
    if requested:
        ack = observation.last_ack
        if (
            ack is not None
            and step.command is not None
            and ack.command_id == step.command.command_id
        ):
            if ack.accepted:
                accepted = True
                if spec.policy is not None and hold_since_us is None:
                    hold_since_us = int(observation.sim_time_us)
            elif ack.result == MAV_RESULT_TEMPORARILY_REJECTED:
                retry_deferred = True
                requested = False
                accepted = False
                hold_since_us = None
            else:
                return state, RoutineDecision.fail(
                    f"step {step.label!r} rejected: command={step.command.command_id} "
                    f"result={ack.result}",
                    failure_kind=FailureKind.COMMAND,
                )

    offboard, setpoint, command, diagnostics = _advance_stream_cadences(
        routine=routine,
        offboard=state.offboard,
        spec=spec,
        observation=observation,
        requested=requested,
        accepted=accepted,
        prime_ticks=spec.prime_ticks,
        command=step.command,
        retry_deferred=retry_deferred,
    )
    if offboard == state.offboard and hold_since_us == state.hold_since_us:
        successor = state
    else:
        successor = replace(state, offboard=offboard, hold_since_us=hold_since_us)
    return successor, RoutineDecision.continue_(
        setpoint=setpoint,
        command=command,
        diagnostics=diagnostics,
    )


def _setpoint_for(
    *,
    spec: _StreamingSpec,
    baseline: Optional[np.ndarray],
) -> StreamSetpoint:
    assert spec.policy is None
    east, north, _up = spec.target_enu
    yaw_rad = math.nan
    if baseline is not None:
        de = east - float(baseline[0])
        dn = north - float(baseline[1])
        if math.hypot(de, dn) >= spec.min_heading_dist_m:
            yaw_rad = math.atan2(de, dn)
    return PositionSetpoint(
        latitude_deg=spec.latitude_deg,
        longitude_deg=spec.longitude_deg,
        altitude_m=spec.altitude_m,
        yaw_rad=yaw_rad,
    )
