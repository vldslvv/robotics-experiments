"""Pure command-layer tests: the ``decide`` stepper, predicates, and gates.

These exercise the contract centerpiece without Docker, Isaac, or pymavlink:
``decide`` is driven through arm -> takeoff -> hover with synthetic
``RoutineObservation``s (invariant breaches, hold windows, ``build_vehicle_routine``)
through the package-root contract. The per-drone driver itself is covered by
``test_commanding_driver.py``.
"""

import math
from dataclasses import fields, replace

import numpy as np
import pytest

from robotics_experiments.commanding import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_REPOSITION,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF,
    MAV_DO_REPOSITION_FLAGS_CHANGE_MODE,
    MAV_RESULT_TEMPORARILY_REJECTED,
    CommandAck,
    DecisionKind,
    EnvelopeGates,
    FailureKind,
    PolicyAdapter,
    PolicyInput,
    PolicyResult,
    RoutineState,
    RoutineStep,
    VehicleRoutine,
    VelocitySetpoint,
    ack_accepted,
    arm,
    armed,
    build_vehicle_routine,
    climbed_at_least,
    cross_track_drift_below,
    cross_track_to_segment_below,
    decide,
    disarmed,
    heading_aligned,
    hovering_stable,
    moved_forward_at_least,
    reached_world_position,
    reposition,
    takeoff,
    tilt_below,
)
from tests.support.commanding_fakes import make_routine_observation

_GATES = EnvelopeGates(
    max_tilt_deg=35.0,
    max_lateral_drift_m=2.0,
    max_vertical_speed_m_s=8.0,
    max_downward_speed_m_s=-4.0,
    max_angular_speed_rad_s=8.0,
    hover_max_abs_vertical_speed_m_s=0.35,
    hover_max_abs_horizontal_speed_m_s=0.30,
)


def _arm_takeoff_routine(*, min_climb_m=1.0):
    arm_step = RoutineStep(
        command=arm(),
        is_complete=ack_accepted(MAV_CMD_COMPONENT_ARM_DISARM),
        label="arm",
        timeout_s=5.0,
        retry_on_temporarily_rejected=True,
    )
    takeoff_step = RoutineStep(
        command=takeoff(47.0, 8.0, 100.0),
        is_complete=climbed_at_least(min_climb_m),
        label="takeoff",
        timeout_s=10.0,
    )
    return VehicleRoutine(
        steps=(arm_step, takeoff_step),
        offboard_setpoint_rate_hz=50.0,
    )


def test_decide_runs_arm_takeoff_to_done():
    routine = _arm_takeoff_routine(min_climb_m=1.0)
    state = RoutineState()

    # Tick 1: first step not sent -> SEND arm, active-step baseline + timer captured.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.command.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert state.sent is True
    assert state.step_started_us == 0
    assert state.baseline_position_enu is not None
    with pytest.raises(ValueError, match="read-only"):
        state.baseline_position_enu[0] = 1.0

    # Tick 2: arm sent, no ack -> WAIT.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=10_000))
    assert decision.kind is DecisionKind.CONTINUE

    # Tick 3: arm ack accepted -> ADVANCE to takeoff step.
    arm_ack = CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, 0)
    state, decision = decide(
        routine, state, make_routine_observation(sim_time_us=20_000, last_ack=arm_ack)
    )
    assert decision.kind is DecisionKind.ADVANCE
    assert state.cursor == 1
    assert state.sent is False

    # Tick 4: takeoff step not sent -> SEND takeoff.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=30_000))
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.command.command_id == MAV_CMD_NAV_TAKEOFF

    # Tick 5: not climbed yet -> WAIT.
    state, decision = decide(
        routine, state, make_routine_observation(sim_time_us=40_000, position_enu=(0.0, 0.0, 0.2))
    )
    assert decision.kind is DecisionKind.CONTINUE

    # Tick 6: climbed past the threshold -> ADVANCE past the last step.
    state, decision = decide(
        routine, state, make_routine_observation(sim_time_us=50_000, position_enu=(0.0, 0.0, 1.5))
    )
    assert decision.kind is DecisionKind.ADVANCE
    assert state.cursor == 2

    # Tick 7: cursor past the last step -> DONE.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=60_000))
    assert decision.kind is DecisionKind.DONE


def test_decide_retries_on_temporarily_rejected():
    routine = _arm_takeoff_routine()
    state = RoutineState()
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is DecisionKind.CONTINUE

    # A temporarily-rejected arm ack re-sends rather than failing.
    temp_reject = CommandAck.from_result(
        MAV_CMD_COMPONENT_ARM_DISARM, MAV_RESULT_TEMPORARILY_REJECTED
    )
    state, decision = decide(
        routine, state, make_routine_observation(sim_time_us=5_000, last_ack=temp_reject)
    )
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.command.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert state.cursor == 0


def test_decide_fails_on_hard_rejection():
    routine = _arm_takeoff_routine()
    state = RoutineState()
    state, _ = decide(routine, state, make_routine_observation(sim_time_us=0))

    # A non-temporary, non-accepted ack fails the routine (DENIED == 2).
    denied = CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, 2)
    _, decision = decide(
        routine, state, make_routine_observation(sim_time_us=1_000, last_ack=denied)
    )
    assert decision.kind is DecisionKind.FAIL
    assert "rejected" in decision.reason


def test_decide_fails_on_timeout():
    routine = _arm_takeoff_routine()
    state = RoutineState()
    state, _ = decide(routine, state, make_routine_observation(sim_time_us=0))

    # arm step timeout_s is 5.0s; 6s of sim time with no ack -> FAIL.
    _, decision = decide(routine, state, make_routine_observation(sim_time_us=6_000_000))
    assert decision.kind is DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.COMMAND
    assert "timed out" in decision.reason


def test_armed_predicate_reads_bridge_status():
    predicate = armed()
    assert predicate(make_routine_observation(is_armed=False), None) is False
    assert predicate(make_routine_observation(is_armed=True), None) is True


def _build(steps, *, target_altitude_m=488.0, command_ack_timeout_s=10.0, policy_factory=None):
    return build_vehicle_routine(
        steps=steps,
        target_latitude_deg=47.0,
        target_longitude_deg=8.0,
        target_altitude_m=target_altitude_m,
        command_ack_timeout_s=command_ack_timeout_s,
        gates=_GATES,
        takeoff_timeout_s=30.0,
        hover_settle_s=2.0,
        home_lat_deg=47.0,
        home_lon_deg=8.0,
        home_alt_m=488.0,
        goto_position_tol_m=0.5,
        goto_heading_tol_deg=25.0,
        goto_cross_track_m=2.0,
        goto_offboard_prime_ticks=3,
        offboard_setpoint_rate_hz=50.0,
        policy_factory=policy_factory,
    )


def test_build_vehicle_routine_arm_only():
    routine = _build([{"kind": "arm"}])
    assert [s.label for s in routine.steps] == ["arm"]
    arm_step = routine.steps[0]
    assert arm_step.command.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert arm_step.timeout_s == 10.0
    # Arm retries on PX4's transient pre-arm rejection, matching the scenario.
    assert arm_step.retry_on_temporarily_rejected is True


def test_build_vehicle_routine_arm_takeoff_hover():
    routine = _build(
        [
            {"kind": "arm"},
            {"kind": "takeoff", "climb_m": 1.5, "min_climb_m": 1.0},
            {"kind": "hover", "hold_s": 5.0},
        ],
        target_altitude_m=488.0,
    )
    assert [s.label for s in routine.steps] == ["arm", "takeoff", "hover"]
    takeoff_step = routine.steps[1]
    assert takeoff_step.command.command_id == MAV_CMD_NAV_TAKEOFF
    # param5/6 are the target lat/lon, param7 the absolute altitude = base + climb.
    assert takeoff_step.command.params[4] == pytest.approx(47.0)
    assert takeoff_step.command.params[5] == pytest.approx(8.0)
    assert takeoff_step.command.params[6] == pytest.approx(489.5)
    assert takeoff_step.retry_on_temporarily_rejected is True
    # Flight gates are wired onto the takeoff and hover steps.
    assert takeoff_step.invariants
    hover_step = routine.steps[2]
    # The hover step carries no command and holds a stable hover for hold_s; its
    # timeout is hover_settle_s + hold_s.
    assert hover_step.command is None
    assert hover_step.hold_s == pytest.approx(5.0)
    assert hover_step.timeout_s == pytest.approx(7.0)
    assert hover_step.invariants


def test_build_vehicle_routine_forward_and_land():
    routine = _build(
        [
            {"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0},
            {"kind": "forward", "distance_m": 1.0, "timeout_s": 20.0},
            {"kind": "land", "timeout_s": 30.0},
        ],
        target_altitude_m=488.0,
    )
    assert [s.label for s in routine.steps] == ["takeoff", "forward", "land"]
    forward_step = routine.steps[1]
    assert forward_step.command.command_id == MAV_CMD_DO_REPOSITION
    assert forward_step.command.params[1] == pytest.approx(
        float(MAV_DO_REPOSITION_FLAGS_CHANGE_MODE)
    )
    assert forward_step.command.params[4] == pytest.approx(47.0)
    assert forward_step.command.params[5] > 8.0
    # Forward holds the last commanded flight altitude from the takeoff step.
    assert forward_step.command.params[6] == pytest.approx(490.0)
    assert forward_step.timeout_s == pytest.approx(20.0)
    assert forward_step.invariants

    land_step = routine.steps[2]
    assert land_step.command.command_id == MAV_CMD_NAV_LAND
    assert land_step.timeout_s == pytest.approx(30.0)


def test_build_vehicle_routine_consecutive_forward_steps_preserve_target_state():
    routine = _build(
        [
            {"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0},
            {"kind": "forward", "distance_m": 1.0, "timeout_s": 20.0},
            {"kind": "forward", "distance_m": 2.0, "timeout_s": 20.0},
        ],
        target_altitude_m=488.0,
    )

    first_forward = routine.steps[1].command
    second_forward = routine.steps[2].command
    first_longitude_offset = first_forward.params[5] - 8.0
    second_longitude_offset = second_forward.params[5] - first_forward.params[5]

    assert second_longitude_offset == pytest.approx(2.0 * first_longitude_offset)
    assert first_forward.params[6] == pytest.approx(490.0)
    assert second_forward.params[6] == pytest.approx(490.0)


def test_build_vehicle_routine_empty_steps_is_empty_routine():
    assert _build([]).steps == ()


def test_build_vehicle_routine_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        _build([{"kind": "orbit"}])


@pytest.mark.parametrize(
    ("descriptor", "missing"),
    [
        ({"kind": "takeoff"}, "climb_m"),
        ({"kind": "takeoff", "climb_m": 1.5}, "min_climb_m"),
        ({"kind": "hover"}, "hold_s"),
        ({"kind": "forward", "timeout_s": 20.0}, "distance_m"),
        ({"kind": "forward", "distance_m": 1.0}, "timeout_s"),
        ({"kind": "land"}, "timeout_s"),
        ({"kind": "goto", "y": 0.0, "z": 1.0, "timeout_s": 10.0}, "'x'"),
        ({"kind": "goto", "x": 1.0, "y": 0.0, "z": 1.0}, "timeout_s"),
    ],
    ids=[
        "takeoff-climb",
        "takeoff-min-climb",
        "hover-hold",
        "forward-distance",
        "forward-timeout",
        "land-timeout",
        "goto-coordinate",
        "goto-timeout",
    ],
)
def test_build_vehicle_routine_rejects_missing_fields(*, descriptor, missing):
    with pytest.raises(ValueError, match=missing):
        _build([descriptor])


# ---- goto (OFFBOARD navigation) --------------------------------------------


def _goto_routine(**overrides):
    descriptor = {"kind": "goto", "x": 3.0, "y": 0.0, "z": 2.0, "timeout_s": 20.0}
    descriptor.update(overrides)
    return _build([descriptor])


def _far(sim_time_us, **overrides):
    # On the active step's x-axis path, far from the (3, 0, 2) waypoint.
    fields = {"sim_time_us": sim_time_us * 20_000, "position_enu": (0.0, 0.0, 2.0)}
    fields.update(overrides)
    return make_routine_observation(**fields)


def _goto_state_after_offboard_request():
    routine = _goto_routine()
    state = RoutineState()
    for t in range(2):
        state, decision = decide(routine, state, _far(t))
        assert decision.kind is DecisionKind.CONTINUE
    state, decision = decide(routine, state, _far(2))
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.command.command_id == MAV_CMD_DO_SET_MODE
    assert state.offboard.requested is True
    return routine, state


def _two_goto_routine():
    return _build(
        [
            {"kind": "goto", "x": 3.0, "y": 0.0, "z": 2.0, "timeout_s": 20.0},
            {"kind": "goto", "x": 3.0, "y": 3.0, "z": 2.0, "timeout_s": 20.0},
        ]
    )


def _goto_hover_land_routine(*, hover_hold_s=1.0):
    return _build(
        [
            {"kind": "goto", "x": 3.0, "y": 0.0, "z": 2.0, "timeout_s": 20.0},
            {"kind": "hover", "hold_s": hover_hold_s},
            {"kind": "land", "timeout_s": 10.0},
        ]
    )


def _advance_first_goto(routine):
    state = RoutineState()
    for t in range(5):
        state, _decision = decide(routine, state, _far(t))
    reached = make_routine_observation(
        sim_time_us=100_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=0.0,
    )
    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.ADVANCE
    assert state.cursor == 1
    assert state.sent is False
    return state


def test_decide_goto_primes_then_engages_offboard_then_reaches():
    routine = _goto_routine()  # _build uses goto_offboard_prime_ticks=3
    state = RoutineState()

    # Entry tick streams the first (priming) setpoint, facing the leg direction:
    # NED yaw = atan2(east=3, north=0) = pi/2.
    state, decision = decide(routine, state, _far(0))
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.setpoint is not None
    assert decision.setpoint.yaw_rad == pytest.approx(math.pi / 2.0)
    assert state.offboard.transmissions == 1
    assert state.offboard.policy_generation == 0
    assert state.offboard.transmitted_policy_generation == 0

    # Two more priming setpoints (no command yet).
    state, decision = decide(routine, state, _far(1))
    assert decision.kind is DecisionKind.CONTINUE
    state, decision = decide(routine, state, _far(2))
    assert decision.kind is DecisionKind.CONTINUE
    assert state.offboard.transmissions == 3
    # The third actual transmission also engages OFFBOARD.
    assert decision.command.command_id == MAV_CMD_DO_SET_MODE
    assert decision.setpoint is not None
    assert state.offboard.requested is True

    # Then back to streaming until the waypoint is reached.
    state, decision = decide(routine, state, _far(3))
    assert decision.kind is DecisionKind.CONTINUE

    # Reached and heading-aligned (facing +east) -> ADVANCE (then DONE).
    reached = make_routine_observation(sim_time_us=5, position_enu=(3.0, 0.0, 2.0), yaw_deg=0.0)
    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.ADVANCE
    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.DONE


def test_goto_fixed_setpoint_is_cached_between_transmissions():
    routine = _goto_routine()
    state, first = decide(routine, RoutineState(), _far(0))
    cached = state.offboard.cached_setpoint

    state, second = decide(routine, state, _far(1))

    assert first.setpoint is cached
    assert second.setpoint is cached
    assert state.offboard.cached_setpoint is cached


def test_lower_offboard_rate_counts_only_actual_transmissions():
    routine = replace(_goto_routine(), offboard_setpoint_rate_hz=10.0)
    state = RoutineState()
    transmitted = 0

    for cycle in range(50):
        state, decision = decide(routine, state, _far(cycle))
        transmitted += int(decision.setpoint is not None)

    assert transmitted == 10
    assert state.offboard.transmissions == 10


def test_decide_goto_offboard_accepted_ack_continues_streaming():
    routine, state = _goto_state_after_offboard_request()
    accepted = CommandAck.from_result(MAV_CMD_DO_SET_MODE, 0)

    state, decision = decide(routine, state, _far(4, last_ack=accepted))

    assert decision.kind is DecisionKind.CONTINUE
    assert state.offboard.requested is True


def test_decide_goto_ignores_offboard_ack_before_this_step_requests_mode():
    routine = _goto_routine()
    state = RoutineState()
    state, decision = decide(routine, state, _far(0))
    assert decision.kind is DecisionKind.CONTINUE

    stale_denied = CommandAck.from_result(MAV_CMD_DO_SET_MODE, 2)
    state, decision = decide(routine, state, _far(1, last_ack=stale_denied))

    assert decision.kind is DecisionKind.CONTINUE
    assert state.offboard.transmissions == 2
    assert state.offboard.requested is False


def test_decide_goto_offboard_temporary_rejection_streams_then_retries():
    routine, state = _goto_state_after_offboard_request()
    temp_reject = CommandAck.from_result(
        MAV_CMD_DO_SET_MODE,
        MAV_RESULT_TEMPORARILY_REJECTED,
    )

    state, decision = decide(routine, state, _far(4, last_ack=temp_reject))

    assert decision.kind is DecisionKind.CONTINUE
    assert state.offboard.requested is False

    state, decision = decide(routine, state, _far(5))
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.command.command_id == MAV_CMD_DO_SET_MODE
    assert state.offboard.requested is True


def test_decide_goto_offboard_hard_rejection_fails_command():
    routine, state = _goto_state_after_offboard_request()
    denied = CommandAck.from_result(MAV_CMD_DO_SET_MODE, 2)

    _state, decision = decide(routine, state, _far(4, last_ack=denied))

    assert decision.kind is DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.COMMAND
    assert "rejected" in decision.reason


def test_decide_goto_holds_streaming_until_heading_aligned():
    routine = _goto_routine()
    state = RoutineState()
    for t in range(6):  # prime + engage offboard + steady stream
        state, _decision = decide(routine, state, _far(t))
    # At the waypoint but yawed 90 deg off the leg direction: not yet complete.
    misaligned = make_routine_observation(
        sim_time_us=120_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=90.0,
    )
    state, decision = decide(routine, state, misaligned)
    assert decision.kind is DecisionKind.CONTINUE


def test_decide_goto_cross_track_breach_fails_envelope():
    routine = _goto_routine()
    state = RoutineState()
    state, _decision = decide(routine, state, _far(0))  # entry: baseline captured
    # 3 m off the straight x-axis path exceeds goto_cross_track_m=2.0.
    off_path = make_routine_observation(sim_time_us=1, position_enu=(1.0, 3.0, 2.0))
    state, decision = decide(routine, state, off_path)
    assert decision.kind is DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.ENVELOPE


def test_consecutive_goto_uses_current_step_baseline_for_yaw():
    routine = _two_goto_routine()
    state = _advance_first_goto(routine)

    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=120_000, position_enu=(3.0, 0.0, 2.0)),
    )

    assert decision.kind is DecisionKind.CONTINUE
    # Second leg is due north from the first waypoint/current step start.
    assert decision.setpoint.yaw_rad == pytest.approx(0.0)
    assert state.baseline_position_enu is not None
    assert state.baseline_position_enu.tolist() == [3.0, 0.0, 2.0]


def test_consecutive_goto_preserves_transmitter_phase_until_next_due_emission():
    routine = replace(_two_goto_routine(), offboard_setpoint_rate_hz=10.0)
    state, _ = decide(routine, RoutineState(), _far(0))
    state, _ = decide(routine, state, _far(1))
    reached = make_routine_observation(
        sim_time_us=40_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=0.0,
    )
    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.ADVANCE

    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=60_000, position_enu=(3.0, 0.0, 2.0)),
    )
    assert decision.setpoint is None
    assert state.offboard.transmissions == 1
    assert state.offboard.cached_setpoint.yaw_rad == pytest.approx(0.0)

    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=80_000, position_enu=(3.0, 0.0, 2.0)),
    )
    assert decision.setpoint is None
    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=100_000, position_enu=(3.0, 0.0, 2.0)),
    )
    assert decision.setpoint is state.offboard.cached_setpoint
    assert state.offboard.transmissions == 2


def test_consecutive_goto_cross_track_uses_current_leg_path():
    routine = _two_goto_routine()
    state = _advance_first_goto(routine)
    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=120_000, position_enu=(3.0, 0.0, 2.0)),
    )
    assert decision.kind is DecisionKind.CONTINUE

    # This is 2.1 m east of the current northbound leg x=3 (breach), but only
    # about 1.5 m from the stale origin->target diagonal (which would pass).
    _state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=140_000, position_enu=(5.1, 3.0, 2.0)),
    )

    assert decision.kind is DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.ENVELOPE
    assert "cross-track" in decision.reason


def test_decide_goto_times_out_as_command_failure():
    routine = _goto_routine()  # timeout_s=20.0
    state = RoutineState()
    for t in range(5):  # reach the steady streaming branch
        state, _decision = decide(routine, state, _far(t))
    state, decision = decide(routine, state, _far(21_000_000))
    assert decision.kind is DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.COMMAND


def test_hover_after_goto_keeps_offboard_setpoint_stream_alive():
    routine = _goto_hover_land_routine()
    state = RoutineState()

    last_streamed = None
    for t in range(5):
        state, decision = decide(routine, state, _far(t))
        if decision.kind is DecisionKind.CONTINUE:
            last_streamed = decision.setpoint
    assert last_streamed is not None

    reached = make_routine_observation(
        sim_time_us=100_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=0.0,
    )
    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.ADVANCE
    assert state.cursor == 1
    assert state.offboard.cached_setpoint == last_streamed

    state, decision = decide(
        routine,
        state,
        make_routine_observation(
            sim_time_us=120_000,
            position_enu=(3.0, 0.0, 2.0),
            yaw_deg=0.0,
        ),
    )
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.setpoint == last_streamed

    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=500_000, position_enu=(3.0, 0.0, 2.0), yaw_deg=0.0),
    )
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.setpoint == last_streamed
    assert state.hold_since_us == 500_000


def test_explicit_command_after_offboard_hover_clears_held_setpoint():
    routine = _goto_hover_land_routine(hover_hold_s=0.0)
    state = RoutineState()
    for t in range(5):
        state, _decision = decide(routine, state, _far(t))

    reached = make_routine_observation(
        sim_time_us=100_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=0.0,
    )
    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.ADVANCE

    state, decision = decide(
        routine,
        state,
        make_routine_observation(
            sim_time_us=120_000,
            position_enu=(3.0, 0.0, 2.0),
            yaw_deg=0.0,
        ),
    )
    assert decision.kind is DecisionKind.CONTINUE
    assert state.offboard.cached_setpoint is not None

    state, decision = decide(
        routine,
        state,
        make_routine_observation(
            sim_time_us=140_000,
            position_enu=(3.0, 0.0, 2.0),
            yaw_deg=0.0,
        ),
    )
    assert decision.kind is DecisionKind.ADVANCE
    assert state.cursor == 2

    state, decision = decide(
        routine,
        state,
        make_routine_observation(
            sim_time_us=160_000,
            position_enu=(3.0, 0.0, 2.0),
            yaw_deg=0.0,
        ),
    )
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.command.command_id == MAV_CMD_NAV_LAND
    assert state.offboard.cached_setpoint is None


def test_reached_world_position_within_tolerance():
    predicate = reached_world_position(3.0, 0.0, 2.0, 0.5)
    assert predicate(make_routine_observation(position_enu=(3.3, 0.0, 2.0)), None) is True
    assert predicate(make_routine_observation(position_enu=(4.0, 0.0, 2.0)), None) is False


def test_heading_aligned_faces_target_direction():
    predicate = heading_aligned(3.0, 0.0, 20.0)
    baseline = np.zeros(3, dtype=np.float64)
    assert predicate(make_routine_observation(yaw_deg=0.0), baseline) is True
    assert predicate(make_routine_observation(yaw_deg=90.0), baseline) is False


def test_heading_aligned_trivial_for_near_vertical_leg():
    # Horizontal active-step-baseline-to-target distance below the threshold: any yaw is fine.
    predicate = heading_aligned(0.05, 0.0, 5.0)
    assert predicate(make_routine_observation(yaw_deg=90.0), np.zeros(3, dtype=np.float64)) is True


def test_cross_track_to_segment_below_measures_perpendicular_distance():
    gate = cross_track_to_segment_below(3.0, 0.0, 2.0)
    baseline = np.zeros(3, dtype=np.float64)
    # 1 m off the x-axis path -> within limit.
    assert gate(make_routine_observation(position_enu=(1.5, 1.0, 2.0)), baseline) is None
    # 3 m off -> breach (along-track progress does not matter).
    assert gate(make_routine_observation(position_enu=(1.5, 3.0, 2.0)), baseline) is not None


def test_build_vehicle_routine_goto():
    routine = _build(
        [
            {"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0},
            {"kind": "goto", "x": 3.0, "y": 1.0, "z": 2.5, "timeout_s": 25.0},
        ]
    )
    goto_step = routine.steps[1]
    assert goto_step.label.startswith("goto")
    # A streaming step engages OFFBOARD via DO_SET_MODE and carries its spec.
    assert goto_step.command.command_id == MAV_CMD_DO_SET_MODE
    assert goto_step.streaming is not None
    assert goto_step.streaming.target_enu == (3.0, 1.0, 2.5)
    assert goto_step.streaming.prime_ticks == 3
    # The geodetic target offsets the home anchor (47, 8, 488); altitude = home + up.
    assert goto_step.streaming.latitude_deg == pytest.approx(47.0, abs=1e-2)
    assert goto_step.streaming.longitude_deg != pytest.approx(8.0)
    assert goto_step.streaming.altitude_m == pytest.approx(490.5)
    assert goto_step.timeout_s == pytest.approx(25.0)
    assert goto_step.invariants


def test_reposition_serializes_default_and_authored_ground_speed():
    default_speed = reposition(47.0, 8.0, 490.0)
    authored_speed = reposition(47.0, 8.0, 490.0, 0.08)

    assert default_speed.params[0] == -1.0
    assert authored_speed.params[:3] == (
        0.08,
        float(MAV_DO_REPOSITION_FLAGS_CHANGE_MODE),
        0.0,
    )
    assert math.isnan(authored_speed.params[3])
    assert authored_speed.params[4:] == (47.0, 8.0, 490.0)


def test_build_vehicle_routine_speed_authored_goto_uses_guided_reposition():
    routine = _goto_routine(ground_speed_m_s=0.5)
    goto_step = routine.steps[0]

    assert goto_step.command.command_id == MAV_CMD_DO_REPOSITION
    assert goto_step.command.params[0] == pytest.approx(0.5)
    assert goto_step.command.params[4] == pytest.approx(47.0)
    assert goto_step.command.params[5] != pytest.approx(8.0)
    assert goto_step.command.params[6] == pytest.approx(490.0)
    assert goto_step.retry_on_temporarily_rejected is True
    assert goto_step.streaming is None
    assert goto_step.guided is not None
    assert goto_step.guided.target_enu == (3.0, 0.0, 2.0)
    assert goto_step.invariants


@pytest.mark.parametrize("ground_speed_m_s", [0.0, -0.1, math.inf, -math.inf, math.nan])
def test_build_vehicle_routine_rejects_invalid_goto_ground_speed(ground_speed_m_s):
    with pytest.raises(ValueError, match="ground_speed_m_s must be finite and positive"):
        _goto_routine(ground_speed_m_s=ground_speed_m_s)


def test_decide_speed_authored_goto_preserves_ack_completion_heading_and_gates():
    routine = _goto_routine(ground_speed_m_s=0.5)
    state, decision = decide(routine, RoutineState(), _far(0))
    assert decision.command.command_id == MAV_CMD_DO_REPOSITION
    assert decision.command.params[3] == pytest.approx(math.pi / 2.0)
    assert decision.setpoint is None

    temporary = CommandAck.from_result(
        MAV_CMD_DO_REPOSITION,
        MAV_RESULT_TEMPORARILY_REJECTED,
    )
    state, decision = decide(routine, state, _far(1, last_ack=temporary))
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.command.command_id == MAV_CMD_DO_REPOSITION
    assert decision.command.params[3] == pytest.approx(math.pi / 2.0)

    misaligned = make_routine_observation(
        sim_time_us=60_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=90.0,
    )
    state, decision = decide(routine, state, misaligned)
    assert decision.kind is DecisionKind.CONTINUE

    aligned = make_routine_observation(
        sim_time_us=80_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=0.0,
    )
    state, decision = decide(routine, state, aligned)
    assert decision.kind is DecisionKind.ADVANCE

    breach_routine = _goto_routine(ground_speed_m_s=0.5)
    breach_state, _ = decide(breach_routine, RoutineState(), _far(0))
    _, breach = decide(
        breach_routine,
        breach_state,
        make_routine_observation(
            sim_time_us=20_000,
            position_enu=(1.0, 3.0, 2.0),
        ),
    )
    assert breach.kind is DecisionKind.FAIL
    assert breach.failure_kind is FailureKind.ENVELOPE


def test_speed_authored_goto_derives_ned_yaw_from_step_baseline():
    routine = _goto_routine(x=0.0, y=3.0, ground_speed_m_s=0.5)

    _, decision = decide(routine, RoutineState(), _far(0))

    assert decision.command.params[3] == pytest.approx(0.0)


def test_speed_authored_goto_reconfirms_once_after_accepted_ack():
    routine = _goto_routine(ground_speed_m_s=0.5)
    state, _ = decide(routine, RoutineState(), _far(0))
    accepted = CommandAck.from_result(MAV_CMD_DO_REPOSITION, 0)

    state, confirmation = decide(routine, state, _far(1, last_ack=accepted))
    assert confirmation.command.command_id == MAV_CMD_DO_REPOSITION
    assert confirmation.command.params[3] == pytest.approx(math.pi / 2.0)
    assert state.guided_confirmation_sent is True

    state, successor = decide(routine, state, _far(2, last_ack=accepted))
    assert successor.kind is DecisionKind.CONTINUE
    assert successor.command is None
    assert state.guided_confirmation_sent is True


def test_decide_speed_authored_goto_preserves_hard_rejection_and_timeout():
    routine = _goto_routine(ground_speed_m_s=0.5)
    state, _ = decide(routine, RoutineState(), _far(0))
    denied = CommandAck.from_result(MAV_CMD_DO_REPOSITION, 2)
    _, rejected = decide(routine, state, _far(1, last_ack=denied))
    assert rejected.kind is DecisionKind.FAIL
    assert rejected.failure_kind is FailureKind.COMMAND
    assert "rejected" in rejected.reason

    state, _ = decide(routine, RoutineState(), _far(0))
    _, timed_out = decide(
        routine,
        state,
        make_routine_observation(
            sim_time_us=21_000_000,
            position_enu=(0.0, 0.0, 2.0),
        ),
    )
    assert timed_out.kind is DecisionKind.FAIL
    assert timed_out.failure_kind is FailureKind.COMMAND
    assert "timed out" in timed_out.reason


def test_build_vehicle_routine_takeoff_yaw_param_is_nan():
    # takeoff() leaves yaw (param4) NaN to hold current heading.
    routine = _build([{"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0}])
    assert math.isnan(routine.steps[0].command.params[3])


def test_moved_forward_at_least_uses_world_x_displacement_from_baseline():
    predicate = moved_forward_at_least(1.0)
    baseline = make_routine_observation(position_enu=(2.0, 3.0, 0.0)).vehicle_state.position_enu
    assert predicate(make_routine_observation(position_enu=(2.9, 3.0, 1.0)), baseline) is False
    assert predicate(make_routine_observation(position_enu=(3.0, 3.0, 1.0)), baseline) is True


def test_forward_invariant_allows_commanded_x_displacement():
    # Drive the forward step through decide: a large commanded x displacement
    # (with cross-track y drift inside the gate) must NOT trip the envelope. If
    # the cross-track gate wrongly measured total horizontal displacement this
    # would FAIL; instead the step advances on the completed forward distance.
    routine = _build(
        [
            {"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0},
            {"kind": "forward", "distance_m": 2.0, "timeout_s": 20.0},
        ]
    )
    forward_routine = VehicleRoutine(
        steps=(routine.steps[1],),
        offboard_setpoint_rate_hz=50.0,
    )
    state = RoutineState()

    # Entry tick captures the baseline at the pre-forward position and SENDs.
    state, decision = decide(
        forward_routine,
        state,
        make_routine_observation(sim_time_us=0, position_enu=(0.0, 2.0, 0.0)),
    )
    assert decision.kind is DecisionKind.CONTINUE

    # The commanded x displacement (and small cross-track drift) passes the gates.
    _, decision = decide(
        forward_routine,
        state,
        make_routine_observation(sim_time_us=10_000, position_enu=(2.0, 1.84, 2.0)),
    )
    assert decision.kind is not DecisionKind.FAIL


def test_cross_track_drift_gate_limits_sideways_drift():
    gate = cross_track_drift_below(2.0)
    baseline = make_routine_observation(position_enu=(0.0, 2.0, 0.0)).vehicle_state.position_enu
    assert gate(make_routine_observation(position_enu=(10.0, 3.99, 2.0)), baseline) is None
    breach = gate(make_routine_observation(position_enu=(1.0, 4.01, 2.0)), baseline)
    assert breach is not None
    assert "cross-track drift" in breach


def test_disarmed_predicate_waits_for_bridge_disarm_status():
    predicate = disarmed()
    assert predicate(make_routine_observation(is_armed=True), None) is False
    assert predicate(make_routine_observation(is_armed=False), None) is True


def test_decide_invariant_breach_fails_with_envelope_kind():
    # A takeoff step with a tilt invariant fails as an ENVELOPE event when the
    # body tilts past the limit, distinct from a command rejection/timeout.
    takeoff_step = RoutineStep(
        command=takeoff(47.0, 8.0, 100.0),
        is_complete=climbed_at_least(5.0),
        label="takeoff",
        timeout_s=30.0,
        invariants=(tilt_below(35.0),),
    )
    routine = VehicleRoutine(
        steps=(takeoff_step,),
        offboard_setpoint_rate_hz=50.0,
    )
    state = RoutineState()
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is DecisionKind.CONTINUE

    _, decision = decide(
        routine, state, make_routine_observation(sim_time_us=10_000, tilt_deg=50.0)
    )
    assert decision.kind is DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.ENVELOPE
    assert "tilt" in decision.reason


def test_decide_hover_step_holds_before_advancing():
    # Without a preceding OFFBOARD hold setpoint, a command-less hover step
    # never SENDs; it advances only once the stable hover predicate has held
    # continuously for hold_s.
    hover_step = RoutineStep(
        command=None,
        is_complete=hovering_stable(
            max_abs_vertical_speed_m_s=0.35, max_abs_horizontal_speed_m_s=0.30
        ),
        label="hover",
        timeout_s=10.0,
        hold_s=2.0,
    )
    routine = VehicleRoutine(
        steps=(hover_step,),
        offboard_setpoint_rate_hz=50.0,
    )
    state = RoutineState()

    # Entry tick: command-less step begins -> WAIT (no SEND).
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is DecisionKind.CONTINUE
    assert state.sent is True

    # Stable but the hold window has not elapsed yet -> WAIT, hold started.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=500_000))
    assert decision.kind is DecisionKind.CONTINUE
    assert state.hold_since_us == 500_000

    # A transient excursion (too fast) breaks the hold; it must restart.
    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=1_000_000, linear_velocity_enu=(0.0, 0.0, 1.0)),
    )
    assert decision.kind is DecisionKind.CONTINUE
    assert state.hold_since_us is None

    # Stable again from t=1.5s; the hold restarts here.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=1_500_000))
    assert decision.kind is DecisionKind.CONTINUE
    assert state.hold_since_us == 1_500_000

    # 2s later the hold window has elapsed -> ADVANCE past the last step.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=3_500_000))
    assert decision.kind is DecisionKind.ADVANCE
    assert state.cursor == 1


# ---- policy (OFFBOARD RL policy) --------------------------------------------


def test_policy_transition_values_have_explicit_fields():
    assert [field.name for field in fields(PolicyInput)] == [
        "vehicle_state",
        "sim_time_us",
        "task_observation",
        "reference_position_enu",
    ]
    assert [field.name for field in fields(PolicyAdapter)] == [
        "policy_rate_hz",
        "initial_state",
        "advance",
    ]
    assert [field.name for field in fields(PolicyResult)] == [
        "state",
        "setpoint",
        "complete",
        "failure_reason",
        "diagnostics",
    ]


class _FakePolicyAdvance:
    """Records narrowed inputs and completes after two seconds of policy time."""

    def __init__(self):
        self.inputs = []
        self.states = []

    def __call__(self, *, policy_input, state):
        self.inputs.append(policy_input)
        self.states.append(state)
        started_us = policy_input.sim_time_us if state == () else state
        return PolicyResult(
            state=started_us,
            setpoint=VelocitySetpoint(
                forward_mps=0.1,
                right_mps=0.0,
                down_mps=0.0,
                yaw_rate_radps=0.0,
            ),
            complete=policy_input.sim_time_us - started_us >= 2_000_000,
        )


def _policy_routine(policy_advance, *, policy_rate_hz=20.0, **overrides):
    descriptor = {"kind": "policy", "artifact": "does/not/matter", "hold_s": 2.0, "timeout_s": 10.0}
    descriptor.update(overrides)
    return _build(
        [descriptor],
        policy_factory=lambda *, artifact, hold_s: PolicyAdapter(
            policy_rate_hz=policy_rate_hz,
            initial_state=(),
            advance=policy_advance,
        ),
    )


def test_build_vehicle_routine_policy_requires_factory():
    with pytest.raises(ValueError, match="policy_factory"):
        _build([{"kind": "policy", "artifact": "a", "hold_s": 1.0, "timeout_s": 5.0}])


def test_build_vehicle_routine_policy_requires_fields():
    for missing in ("artifact", "timeout_s"):
        descriptor = {"kind": "policy", "artifact": "a", "hold_s": 1.0, "timeout_s": 5.0}
        descriptor.pop(missing)
        with pytest.raises(ValueError, match=missing):
            _build(
                [descriptor],
                policy_factory=lambda *, artifact, hold_s: PolicyAdapter(
                    policy_rate_hz=20.0,
                    initial_state=(),
                    advance=_FakePolicyAdvance(),
                ),
            )


@pytest.mark.parametrize(
    ("obsolete", "replacement", "value"),
    [
        ("name", "label", "hover"),
        ("target", "target_enu", [1.0, 2.0, 3.0]),
    ],
)
def test_build_vehicle_routine_policy_rejects_obsolete_keys(
    obsolete,
    replacement,
    value,
):
    descriptor = {
        "kind": "policy",
        "artifact": "a",
        "timeout_s": 5.0,
        obsolete: value,
    }
    with pytest.raises(ValueError, match=rf"{obsolete!r}.*{replacement!r}"):
        _build(
            [descriptor],
            policy_factory=lambda *, artifact, hold_s: PolicyAdapter(
                policy_rate_hz=20.0,
                initial_state=(),
                advance=_FakePolicyAdvance(),
            ),
        )


def test_policy_adapter_rejects_none_and_mutable_initial_state():
    with pytest.raises(ValueError, match="initial state must not be None"):
        PolicyAdapter(
            policy_rate_hz=20.0,
            initial_state=None,
            advance=_FakePolicyAdvance(),
        )
    with pytest.raises(TypeError, match="initial state must be an immutable value"):
        PolicyAdapter(
            policy_rate_hz=20.0,
            initial_state=[],
            advance=_FakePolicyAdvance(),
        )


def test_policy_result_rejects_none_and_mutable_successor_state():
    setpoint = VelocitySetpoint(0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="successor state must not be None"):
        PolicyResult(state=None, setpoint=setpoint)
    with pytest.raises(TypeError, match="successor state must be an immutable value"):
        PolicyResult(state={}, setpoint=setpoint)


def test_decide_policy_primes_engages_offboard_streams_velocity_and_completes():
    policy_advance = _FakePolicyAdvance()
    routine = _policy_routine(policy_advance)
    state = RoutineState()

    # Entry + priming ticks stream the policy's payload (a VelocitySetpoint).
    for t in range(3):
        state, decision = decide(
            routine,
            state,
            make_routine_observation(sim_time_us=t * 20_000),
        )
        assert decision.kind is DecisionKind.CONTINUE
        assert isinstance(decision.setpoint, VelocitySetpoint)
    assert state.offboard.transmissions == 3

    # The third actual transmission engages OFFBOARD on the same cycle.
    assert decision.command.command_id == MAV_CMD_DO_SET_MODE
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=60_000))
    assert decision.kind is DecisionKind.CONTINUE

    # Without an accepted OFFBOARD ack, the policy stream stays alive but the
    # hold window does not start.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=4 + 2_000_000))
    assert decision.kind is DecisionKind.CONTINUE
    assert state.cursor == 0

    accepted = CommandAck.from_result(MAV_CMD_DO_SET_MODE, 0)
    state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=5 + 2_000_000, last_ack=accepted),
    )
    assert decision.kind is DecisionKind.CONTINUE
    assert state.offboard.accepted is True

    # Adapter completion advances only after OFFBOARD acceptance.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=5 + 4_000_000))
    assert decision.kind is DecisionKind.ADVANCE
    assert state.cursor == 1
    assert state.last_policy_state == 0


def test_policy_and_setpoint_cadences_are_independent_exact_averages():
    policy_advance = _FakePolicyAdvance()
    routine = _policy_routine(policy_advance)
    state = RoutineState()
    transmitted = 0

    for cycle in range(50):
        state, decision = decide(
            routine,
            state,
            make_routine_observation(sim_time_us=cycle * 20_000),
        )
        transmitted += int(decision.setpoint is not None)

    assert len(policy_advance.inputs) == 20
    assert transmitted == 50
    assert state.offboard.transmissions == 50
    assert state.offboard.policy_generation == 20
    assert state.offboard.transmitted_policy_generation == 20


def test_policy_generation_is_not_applied_until_its_setpoint_is_transmitted():
    policy_advance = _FakePolicyAdvance()
    routine = replace(_policy_routine(policy_advance), offboard_setpoint_rate_hz=10.0)
    state = RoutineState()

    state, _decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=0),
    )
    assert state.offboard.policy_generation == 1
    assert state.offboard.transmitted_policy_generation == 1
    first_transmitted = state.offboard.transmitted_setpoint
    assert first_transmitted is state.offboard.cached_setpoint
    assert state.offboard.transmitted_setpoint_sim_time_us == 0

    for sim_time_us in (20_000, 40_000, 60_000):
        state, decision = decide(
            routine,
            state,
            make_routine_observation(sim_time_us=sim_time_us),
        )

    assert decision.setpoint is None
    assert state.offboard.policy_generation == 2
    assert state.offboard.transmitted_policy_generation == 1
    assert state.offboard.transmitted_setpoint is first_transmitted
    assert state.offboard.cached_setpoint is not first_transmitted
    assert state.offboard.transmitted_setpoint_sim_time_us == 0

    for sim_time_us in (80_000, 100_000):
        state, decision = decide(
            routine,
            state,
            make_routine_observation(sim_time_us=sim_time_us),
        )

    assert decision.setpoint is not None
    assert state.offboard.transmitted_policy_generation == state.offboard.policy_generation
    assert state.offboard.transmitted_setpoint is state.offboard.cached_setpoint
    assert state.offboard.transmitted_setpoint_sim_time_us == 100_000


def test_policy_adapter_owns_its_exact_average_cadence():
    policy_advance = _FakePolicyAdvance()
    routine = _policy_routine(policy_advance, policy_rate_hz=10.0)
    state = RoutineState()

    for cycle in range(50):
        state, _decision = decide(
            routine,
            state,
            make_routine_observation(sim_time_us=cycle * 20_000),
        )

    assert len(policy_advance.inputs) == 10


def test_policy_entry_replaces_retained_goto_payload_immediately():
    goto = _goto_routine()
    goto_state = RoutineState()
    for t in range(5):
        goto_state, _decision = decide(goto, goto_state, _far(t))
    retained_position_setpoint = goto_state.offboard.cached_setpoint

    policy_advance = _FakePolicyAdvance()
    policy = _policy_routine(policy_advance)
    policy_entry_state = replace(
        goto_state,
        cursor=0,
        sent=False,
        baseline_position_enu=None,
        step_started_us=None,
        hold_since_us=None,
    )
    successor, decision = decide(
        policy,
        policy_entry_state,
        make_routine_observation(sim_time_us=100_000),
    )

    assert len(policy_advance.inputs) == 1
    assert isinstance(decision.setpoint, VelocitySetpoint)
    assert decision.setpoint is not retained_position_setpoint
    assert successor.offboard.cached_setpoint is decision.setpoint


def test_speed_authored_goto_hover_policy_transition_primes_offboard_fresh():
    policy_advance = _FakePolicyAdvance()
    routine = _build(
        [
            {
                "kind": "goto",
                "x": 3.0,
                "y": 0.0,
                "z": 2.0,
                "ground_speed_m_s": 0.5,
                "timeout_s": 20.0,
            },
            {"kind": "hover", "hold_s": 0.0},
            {
                "kind": "policy",
                "artifact": "does/not/matter",
                "timeout_s": 10.0,
            },
        ],
        policy_factory=lambda *, artifact, hold_s: PolicyAdapter(
            policy_rate_hz=20.0,
            initial_state=(),
            advance=policy_advance,
        ),
    )
    state, decision = decide(routine, RoutineState(), _far(0))
    assert decision.command.command_id == MAV_CMD_DO_REPOSITION
    assert state.offboard.transmissions == 0
    assert state.offboard.cached_setpoint is None

    reached = make_routine_observation(
        sim_time_us=20_000,
        position_enu=(3.0, 0.0, 2.0),
        yaw_deg=0.0,
    )
    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.ADVANCE

    state, decision = decide(routine, state, reached)
    assert decision.kind is DecisionKind.CONTINUE
    assert decision.setpoint is None
    state, decision = decide(routine, state, replace(reached, sim_time_us=40_000))
    assert decision.kind is DecisionKind.ADVANCE

    state, decision = decide(routine, state, replace(reached, sim_time_us=60_000))
    assert decision.kind is DecisionKind.CONTINUE
    assert isinstance(decision.setpoint, VelocitySetpoint)
    assert decision.command is None
    assert state.offboard.transmissions == 1
    assert state.offboard.requested is False


def test_decide_policy_input_is_narrow_and_uses_step_baseline_reference():
    policy_advance = _FakePolicyAdvance()
    routine = _policy_routine(policy_advance)
    state = RoutineState()
    task_observation = ("task", 7)
    start = make_routine_observation(
        sim_time_us=0,
        position_enu=(1.0, 2.0, 3.0),
        task_observation=task_observation,
    )
    state, decision = decide(routine, state, start)
    assert decision.kind is DecisionKind.CONTINUE
    assert policy_advance.inputs[-1] == PolicyInput(
        vehicle_state=start.vehicle_state,
        sim_time_us=0,
        task_observation=task_observation,
        reference_position_enu=(1.0, 2.0, 3.0),
    )


def test_decide_policy_with_authored_target_uses_it():
    policy_advance = _FakePolicyAdvance()
    routine = _policy_routine(policy_advance, target_enu=[5.0, 6.0, 7.0])
    state = RoutineState()
    state, _decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    assert policy_advance.inputs[-1].reference_position_enu == (5.0, 6.0, 7.0)


def test_build_vehicle_routine_policy_rejects_bad_target_shape():
    for target in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [math.nan, 2.0, 3.0], "bad"):
        with pytest.raises(ValueError, match="3-item numeric sequence"):
            _policy_routine(_FakePolicyAdvance(), target_enu=target)


def test_build_vehicle_routine_policy_factory_receives_only_artifact_and_hold():
    hover_policy = _FakePolicyAdvance()
    press_policy = _FakePolicyAdvance()
    policies = {
        ".build/training/hover/latest": hover_policy,
        ".build/training/press-approach/latest": press_policy,
    }
    descriptors = [
        {
            "kind": "policy",
            "label": "hover",
            "artifact": ".build/training/hover/latest",
            "hold_s": 1.0,
            "timeout_s": 5.0,
            "target_enu": [1.0, 2.0, 3.0],
        },
        {
            "kind": "policy",
            "label": "press",
            "artifact": ".build/training/press-approach/latest",
            "timeout_s": 5.0,
            "target_enu": [4.0, 5.0, 6.0],
        },
    ]
    calls = []

    def factory(*, artifact, hold_s):
        calls.append((artifact, hold_s))
        return PolicyAdapter(
            policy_rate_hz=10.0 if "hover" in artifact else 25.0,
            initial_state=(),
            advance=policies[artifact],
        )

    routine = _build(descriptors, policy_factory=factory)

    assert [step.label for step in routine.steps] == ["policy[hover]", "policy[press]"]
    assert calls == [
        (".build/training/hover/latest", 1.0),
        (".build/training/press-approach/latest", None),
    ]
    first_stream = routine.steps[0].streaming
    second_stream = routine.steps[1].streaming
    assert first_stream is not None and first_stream.policy is not None
    assert second_stream is not None and second_stream.policy is not None
    assert first_stream.policy.policy_rate_hz == 10.0
    assert second_stream.policy.policy_rate_hz == 25.0


def test_decide_policy_envelope_breach_fails():
    routine = _policy_routine(_FakePolicyAdvance())
    state = RoutineState()
    state, _decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    tilted = make_routine_observation(sim_time_us=1, tilt_deg=60.0)
    _state, decision = decide(routine, state, tilted)
    assert decision.kind is DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.ENVELOPE


def test_decide_policy_timeout_reports_opaque_state_not_waypoint() -> None:
    routine = _policy_routine(_FakePolicyAdvance(), timeout_s=1.0)
    state, _decision = decide(
        routine,
        RoutineState(),
        make_routine_observation(sim_time_us=0),
    )

    _state, decision = decide(
        routine,
        state,
        make_routine_observation(sim_time_us=1_000_001),
    )

    assert decision.kind is DecisionKind.FAIL
    assert "policy did not complete" in str(decision.reason)
    assert "policy state=0" in str(decision.reason)
    assert "waypoint" not in str(decision.reason)
