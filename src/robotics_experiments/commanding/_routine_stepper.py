import math
from dataclasses import replace
from typing import Optional

import numpy as np

from robotics_experiments.commanding._cadence import _advance_cadence, _CadenceState
from robotics_experiments.commanding._routine_model import (
    FailureKind,
    RoutineDecision,
    RoutineState,
    StreamSetpoint,
    VehicleRoutine,
    _PolicySourceState,
    _SetpointStreamState,
    _StreamingSpec,
)
from robotics_experiments.commanding.commands import PositionSetpoint
from robotics_experiments.commanding.observation import (
    MAV_RESULT_TEMPORARILY_REJECTED,
    RoutineObservation,
)

__all__ = ["decide"]


def _anchored_cadence(sim_time_us: int) -> _CadenceState:
    return _CadenceState(previous_time_ns=sim_time_us * 1_000)


def _advanced(state: RoutineState) -> RoutineState:
    """Advance while retaining only the setpoint a verifier may keep alive."""
    return replace(
        state,
        cursor=state.cursor + 1,
        sent=False,
        baseline_position_enu=None,
        step_started_us=None,
        hold_since_us=None,
        policy_source=_PolicySourceState(),
        setpoint_stream=_SetpointStreamState(cached_setpoint=state.setpoint_stream.cached_setpoint),
        offboard_requested=False,
        offboard_accepted=False,
    )


def _advance_policy_source(
    *,
    routine: VehicleRoutine,
    state: RoutineState,
    spec: _StreamingSpec,
    observation: RoutineObservation,
) -> RoutineState:
    if spec.setpoint_fn is None:
        return state
    cadence, due = _advance_cadence(
        state.policy_source.cadence,
        time_ns=int(observation.sim_time_us) * 1_000,
        rate_hz=routine.policy_rate_hz,
    )
    cached = state.policy_source.cached_setpoint
    if due:
        cached = _setpoint_for(spec, state.baseline_position_enu, observation)
    return replace(state, policy_source=_PolicySourceState(cadence, cached))


def _advance_stream(
    *,
    routine: VehicleRoutine,
    state: RoutineState,
    observation: RoutineObservation,
    setpoint: StreamSetpoint,
) -> tuple[RoutineState, Optional[StreamSetpoint]]:
    cadence, due = _advance_cadence(
        state.setpoint_stream.cadence,
        time_ns=int(observation.sim_time_us) * 1_000,
        rate_hz=routine.offboard_setpoint_rate_hz,
    )
    transmissions = state.setpoint_stream.transmissions + int(due)
    stream = _SetpointStreamState(cadence, transmissions, setpoint)
    return replace(state, setpoint_stream=stream), setpoint if due else None


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
    entered = replace(
        state,
        sent=True,
        baseline_position_enu=baseline,
        step_started_us=now_us,
        hold_since_us=None,
        policy_source=_PolicySourceState(),
        setpoint_stream=_SetpointStreamState(),
        offboard_requested=False,
        offboard_accepted=False,
    )
    if step.streaming is not None:
        setpoint = _setpoint_for(step.streaming, baseline, observation)
        policy_source = _PolicySourceState(
            cadence=_anchored_cadence(now_us),
            cached_setpoint=setpoint if step.streaming.setpoint_fn is not None else None,
        )
        stream = _SetpointStreamState(
            cadence=_anchored_cadence(now_us),
            transmissions=1,
            cached_setpoint=setpoint,
        )
        command = step.command if step.streaming.prime_ticks <= 1 else None
        return replace(
            entered,
            policy_source=policy_source,
            setpoint_stream=stream,
            offboard_requested=command is not None,
        ), RoutineDecision.continue_(setpoint=setpoint, command=command)

    held_setpoint = state.setpoint_stream.cached_setpoint
    if step.command is None and held_setpoint is not None:
        return replace(
            entered,
            setpoint_stream=_SetpointStreamState(
                cadence=_anchored_cadence(now_us),
                transmissions=1,
                cached_setpoint=held_setpoint,
            ),
        ), RoutineDecision.continue_(setpoint=held_setpoint)
    if step.command is None:
        return entered, RoutineDecision.continue_()
    return entered, RoutineDecision.continue_(command=step.command)


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

    held_setpoint = state.setpoint_stream.cached_setpoint
    if step.command is None and held_setpoint is not None:
        state, setpoint = _advance_stream(
            routine=routine,
            state=state,
            observation=observation,
            setpoint=held_setpoint,
        )
        return state, RoutineDecision.continue_(setpoint=setpoint)
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
        return state, RoutineDecision.fail(
            f"step {step.label!r} timed out after {step.timeout_s:.1f}s "
            "(did not reach the waypoint)",
            failure_kind=FailureKind.COMMAND,
        )

    retry_deferred = False
    if state.offboard_requested:
        ack = observation.last_ack
        if (
            ack is not None
            and step.command is not None
            and ack.command_id == step.command.command_id
        ):
            if ack.accepted:
                hold_since_us = state.hold_since_us
                if spec.setpoint_fn is not None and hold_since_us is None:
                    hold_since_us = int(observation.sim_time_us)
                state = replace(
                    state,
                    offboard_accepted=True,
                    hold_since_us=hold_since_us,
                )
            elif ack.result == MAV_RESULT_TEMPORARILY_REJECTED:
                retry_deferred = True
                state = replace(
                    state,
                    offboard_requested=False,
                    offboard_accepted=False,
                    hold_since_us=None,
                )
            else:
                return state, RoutineDecision.fail(
                    f"step {step.label!r} rejected: command={step.command.command_id} "
                    f"result={ack.result}",
                    failure_kind=FailureKind.COMMAND,
                )

    state = _advance_policy_source(
        routine=routine,
        state=state,
        spec=spec,
        observation=observation,
    )
    if spec.setpoint_fn is None:
        setpoint = _setpoint_for(spec, state.baseline_position_enu, observation)
    else:
        policy_setpoint = state.policy_source.cached_setpoint
        assert policy_setpoint is not None
        setpoint = policy_setpoint
    state, setpoint_effect = _advance_stream(
        routine=routine,
        state=state,
        observation=observation,
        setpoint=setpoint,
    )

    command = None
    if (
        not retry_deferred
        and not state.offboard_requested
        and state.setpoint_stream.transmissions >= spec.prime_ticks
    ):
        assert step.command is not None
        command = step.command
        state = replace(state, offboard_requested=True)
    return state, RoutineDecision.continue_(setpoint=setpoint_effect, command=command)


def _setpoint_for(
    spec: _StreamingSpec,
    baseline: Optional[np.ndarray],
    observation: RoutineObservation,
) -> StreamSetpoint:
    if spec.setpoint_fn is not None:
        return spec.setpoint_fn(observation, baseline)
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
