import math
from dataclasses import replace
from typing import Optional

import numpy as np

from robotics_experiments.commanding._routine_model import (
    FailureKind,
    RoutineDecision,
    RoutineState,
    StreamSetpoint,
    VehicleRoutine,
    _StreamingSpec,
)
from robotics_experiments.commanding.commands import PositionSetpoint
from robotics_experiments.commanding.observation import (
    MAV_RESULT_TEMPORARILY_REJECTED,
    RoutineObservation,
)

__all__ = ["decide"]


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
