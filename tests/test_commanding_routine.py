"""Pure command-layer tests: the ``_decide`` stepper, predicates, and gates.

These exercise the contract centerpiece without Docker, Isaac, or pymavlink:
``_decide`` is driven through _arm -> _takeoff -> hover with synthetic
``RoutineObservation``s (invariant breaches, hold windows, ``_build_vehicle_routine``)
plus a single-drone :class:`_VehicleCommander` over a fake
:class:`_AbstractCommandPort`. The per-drone driver itself is covered by
``test_commanding_driver.py``.
"""

import math

import numpy as np
import pytest
from _fakes import FakeCommandPort, make_routine_observation

from robotics_experiments.commanding import FailureKind
from robotics_experiments.commanding._commander import _VehicleCommander
from robotics_experiments.commanding._commands import (
    _MAV_CMD_COMPONENT_ARM_DISARM,
    _MAV_CMD_DO_REPOSITION,
    _MAV_CMD_DO_SET_MODE,
    _MAV_CMD_NAV_LAND,
    _MAV_CMD_NAV_TAKEOFF,
    _MAV_DO_REPOSITION_FLAGS_CHANGE_MODE,
    _arm,
    _takeoff,
)
from robotics_experiments.commanding._observation import (
    _MAV_RESULT_TEMPORARILY_REJECTED,
    CommandAck,
)
from robotics_experiments.commanding._routine import (
    _ack_accepted,
    _armed,
    _build_vehicle_routine,
    _climbed_at_least,
    _cross_track_drift_below,
    _cross_track_to_segment_below,
    _decide,
    _DecisionKind,
    _disarmed,
    _EnvelopeGates,
    _heading_aligned,
    _hovering_stable,
    _moved_forward_at_least,
    _reached_world_position,
    _RoutineState,
    _RoutineStep,
    _tilt_below,
    _VehicleRoutine,
)

_GATES = _EnvelopeGates(
    max_tilt_deg=35.0,
    max_lateral_drift_m=2.0,
    max_vertical_speed_m_s=8.0,
    max_downward_speed_m_s=-4.0,
    max_angular_speed_rad_s=8.0,
    hover_max_abs_vertical_speed_m_s=0.35,
    hover_max_abs_horizontal_speed_m_s=0.30,
)


def _arm_takeoff_routine(*, min_climb_m=1.0):
    arm_step = _RoutineStep(
        command=_arm(),
        is_complete=_ack_accepted(_MAV_CMD_COMPONENT_ARM_DISARM),
        label="arm",
        timeout_s=5.0,
        retry_on_temporarily_rejected=True,
    )
    takeoff_step = _RoutineStep(
        command=_takeoff(47.0, 8.0, 100.0),
        is_complete=_climbed_at_least(min_climb_m),
        label="takeoff",
        timeout_s=10.0,
    )
    return _VehicleRoutine(steps=(arm_step, takeoff_step))


def test_decide_runs_arm_takeoff_to_done():
    routine = _arm_takeoff_routine(min_climb_m=1.0)
    state = _RoutineState.initial()

    # Tick 1: first step not sent -> SEND _arm, baseline + timer captured.
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is _DecisionKind.SEND
    assert decision.command.command_id == _MAV_CMD_COMPONENT_ARM_DISARM
    assert state.sent is True
    assert state.step_started_us == 0
    assert state.baseline_position_enu is not None

    # Tick 2: _arm sent, no ack -> WAIT.
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=10_000))
    assert decision.kind is _DecisionKind.WAIT

    # Tick 3: _arm ack accepted -> ADVANCE to _takeoff step.
    arm_ack = CommandAck.from_result(_MAV_CMD_COMPONENT_ARM_DISARM, 0)
    state, decision = _decide(
        routine, state, make_routine_observation(sim_time_us=20_000, last_ack=arm_ack)
    )
    assert decision.kind is _DecisionKind.ADVANCE
    assert state.cursor == 1
    assert state.sent is False

    # Tick 4: _takeoff step not sent -> SEND _takeoff.
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=30_000))
    assert decision.kind is _DecisionKind.SEND
    assert decision.command.command_id == _MAV_CMD_NAV_TAKEOFF

    # Tick 5: not climbed yet -> WAIT.
    state, decision = _decide(
        routine, state, make_routine_observation(sim_time_us=40_000, position_enu=(0.0, 0.0, 0.2))
    )
    assert decision.kind is _DecisionKind.WAIT

    # Tick 6: climbed past the threshold -> ADVANCE past the last step.
    state, decision = _decide(
        routine, state, make_routine_observation(sim_time_us=50_000, position_enu=(0.0, 0.0, 1.5))
    )
    assert decision.kind is _DecisionKind.ADVANCE
    assert state.cursor == 2

    # Tick 7: cursor past the last step -> DONE.
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=60_000))
    assert decision.kind is _DecisionKind.DONE


def test_decide_retries_on_temporarily_rejected():
    routine = _arm_takeoff_routine()
    state = _RoutineState.initial()
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is _DecisionKind.SEND

    # A temporarily-rejected _arm ack re-sends rather than failing.
    temp_reject = CommandAck.from_result(
        _MAV_CMD_COMPONENT_ARM_DISARM, _MAV_RESULT_TEMPORARILY_REJECTED
    )
    state, decision = _decide(
        routine, state, make_routine_observation(sim_time_us=5_000, last_ack=temp_reject)
    )
    assert decision.kind is _DecisionKind.SEND
    assert decision.command.command_id == _MAV_CMD_COMPONENT_ARM_DISARM
    assert state.cursor == 0


def test_decide_fails_on_hard_rejection():
    routine = _arm_takeoff_routine()
    state = _RoutineState.initial()
    state, _ = _decide(routine, state, make_routine_observation(sim_time_us=0))

    # A non-temporary, non-accepted ack fails the routine (DENIED == 2).
    denied = CommandAck.from_result(_MAV_CMD_COMPONENT_ARM_DISARM, 2)
    _, decision = _decide(
        routine, state, make_routine_observation(sim_time_us=1_000, last_ack=denied)
    )
    assert decision.kind is _DecisionKind.FAIL
    assert "rejected" in decision.reason


def test_decide_fails_on_timeout():
    routine = _arm_takeoff_routine()
    state = _RoutineState.initial()
    state, _ = _decide(routine, state, make_routine_observation(sim_time_us=0))

    # _arm step timeout_s is 5.0s; 6s of sim time with no ack -> FAIL.
    _, decision = _decide(routine, state, make_routine_observation(sim_time_us=6_000_000))
    assert decision.kind is _DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.COMMAND
    assert "timed out" in decision.reason


def test_armed_predicate_reads_bridge_status():
    predicate = _armed()
    assert predicate(make_routine_observation(is_armed=False), None) is False
    assert predicate(make_routine_observation(is_armed=True), None) is True


def test_vehicle_commander_sends_then_advances_to_done():
    # One commander drives one routine over one port: SEND on entry, ADVANCE on
    # the accepted ack, DONE past the last step. Drones are independent (no fleet).
    port = FakeCommandPort()
    commander = _VehicleCommander(_arm_takeoff_routine(), port, label="d0")

    decision = commander.step(make_routine_observation(sim_time_us=0))
    assert decision.kind is _DecisionKind.SEND
    assert [c.label for c in port.sent] == ["arm"]
    assert commander.current_step_index == 0

    arm_ack = CommandAck.from_result(_MAV_CMD_COMPONENT_ARM_DISARM, 0)
    decision = commander.step(make_routine_observation(sim_time_us=10_000, last_ack=arm_ack))
    assert decision.kind is _DecisionKind.ADVANCE
    assert commander.current_step_index == 1

    # Takeoff sends, then completes on the captured climb.
    commander.step(make_routine_observation(sim_time_us=20_000))
    assert [c.label for c in port.sent] == ["arm", "takeoff"]
    decision = commander.step(
        make_routine_observation(sim_time_us=30_000, position_enu=(0.0, 0.0, 1.5))
    )
    assert decision.kind is _DecisionKind.ADVANCE
    assert commander.is_done
    assert commander.step(make_routine_observation(sim_time_us=40_000)).kind is _DecisionKind.DONE


def _build(steps, *, target_altitude_m=488.0, command_ack_timeout_s=10.0):
    return _build_vehicle_routine(
        steps,
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
    )


def test_build_vehicle_routine_arm_only():
    routine = _build([{"kind": "arm"}])
    assert [s.label for s in routine.steps] == ["arm"]
    arm_step = routine.steps[0]
    assert arm_step.command.command_id == _MAV_CMD_COMPONENT_ARM_DISARM
    assert arm_step.timeout_s == 10.0
    # Arm retries on PX4's transient pre-_arm rejection, matching the scenario.
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
    assert takeoff_step.command.command_id == _MAV_CMD_NAV_TAKEOFF
    # param5/6 are the target lat/lon, param7 the absolute altitude = base + climb.
    assert takeoff_step.command.params[4] == pytest.approx(47.0)
    assert takeoff_step.command.params[5] == pytest.approx(8.0)
    assert takeoff_step.command.params[6] == pytest.approx(489.5)
    assert takeoff_step.retry_on_temporarily_rejected is True
    # Flight gates are wired onto the _takeoff and hover steps.
    assert takeoff_step.invariants
    hover_step = routine.steps[2]
    # The hover step sends nothing and holds a stable hover for hold_s; its
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
    assert forward_step.command.command_id == _MAV_CMD_DO_REPOSITION
    assert forward_step.command.params[1] == pytest.approx(
        float(_MAV_DO_REPOSITION_FLAGS_CHANGE_MODE)
    )
    assert forward_step.command.params[4] == pytest.approx(47.0)
    assert forward_step.command.params[5] > 8.0
    # Forward holds the last commanded flight altitude from the _takeoff step.
    assert forward_step.command.params[6] == pytest.approx(490.0)
    assert forward_step.timeout_s == pytest.approx(20.0)
    assert forward_step.invariants

    land_step = routine.steps[2]
    assert land_step.command.command_id == _MAV_CMD_NAV_LAND
    assert land_step.timeout_s == pytest.approx(30.0)


def test_build_vehicle_routine_empty_steps_is_empty_routine():
    assert _build([]).steps == ()


def test_build_vehicle_routine_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        _build([{"kind": "orbit"}])


def test_build_vehicle_routine_rejects_takeoff_without_climb():
    with pytest.raises(ValueError, match="climb_m"):
        _build([{"kind": "takeoff"}])


def test_build_vehicle_routine_rejects_takeoff_without_min_climb():
    with pytest.raises(ValueError, match="min_climb_m"):
        _build([{"kind": "takeoff", "climb_m": 1.5}])


def test_build_vehicle_routine_rejects_hover_without_hold():
    with pytest.raises(ValueError, match="hold_s"):
        _build([{"kind": "hover"}])


def test_build_vehicle_routine_rejects_forward_without_distance():
    with pytest.raises(ValueError, match="distance_m"):
        _build([{"kind": "forward", "timeout_s": 20.0}])


def test_build_vehicle_routine_rejects_forward_without_timeout():
    with pytest.raises(ValueError, match="timeout_s"):
        _build([{"kind": "forward", "distance_m": 1.0}])


def test_build_vehicle_routine_rejects_land_without_timeout():
    with pytest.raises(ValueError, match="timeout_s"):
        _build([{"kind": "land"}])


# ---- goto (OFFBOARD navigation) --------------------------------------------


def _goto_routine(**overrides):
    descriptor = {"kind": "goto", "x": 3.0, "y": 0.0, "z": 2.0, "timeout_s": 20.0}
    descriptor.update(overrides)
    return _build([descriptor])


def _far(sim_time_us):
    # On the baseline->target (x-axis) path, far from the (3, 0, 2) waypoint.
    return make_routine_observation(sim_time_us=sim_time_us, position_enu=(0.0, 0.0, 2.0))


def test_decide_goto_primes_then_engages_offboard_then_reaches():
    routine = _goto_routine()  # _build uses goto_offboard_prime_ticks=3
    state = _RoutineState.initial()

    # Entry tick streams the first (priming) setpoint, facing the leg direction:
    # NED yaw = atan2(east=3, north=0) = pi/2.
    state, decision = _decide(routine, state, _far(0))
    assert decision.kind is _DecisionKind.STREAM
    assert decision.setpoint is not None
    assert decision.setpoint.yaw_rad == pytest.approx(math.pi / 2.0)
    assert state.setpoints_streamed == 1

    # Two more priming setpoints (no command yet).
    state, decision = _decide(routine, state, _far(1))
    assert decision.kind is _DecisionKind.STREAM
    state, decision = _decide(routine, state, _far(2))
    assert decision.kind is _DecisionKind.STREAM
    assert state.setpoints_streamed == 3

    # Primed: engage OFFBOARD once.
    state, decision = _decide(routine, state, _far(3))
    assert decision.kind is _DecisionKind.SEND
    assert decision.command.command_id == _MAV_CMD_DO_SET_MODE
    assert state.offboard_requested is True

    # Then back to streaming until the waypoint is reached.
    state, decision = _decide(routine, state, _far(4))
    assert decision.kind is _DecisionKind.STREAM

    # Reached and heading-aligned (facing +east) -> ADVANCE (then DONE).
    reached = make_routine_observation(sim_time_us=5, position_enu=(3.0, 0.0, 2.0), yaw_deg=0.0)
    state, decision = _decide(routine, state, reached)
    assert decision.kind is _DecisionKind.ADVANCE
    state, decision = _decide(routine, state, reached)
    assert decision.kind is _DecisionKind.DONE


def test_decide_goto_holds_streaming_until_heading_aligned():
    routine = _goto_routine()
    state = _RoutineState.initial()
    for t in range(6):  # prime + engage offboard + steady stream
        state, _decision = _decide(routine, state, _far(t))
    # At the waypoint but yawed 90 deg off the leg direction: not yet complete.
    misaligned = make_routine_observation(sim_time_us=6, position_enu=(3.0, 0.0, 2.0), yaw_deg=90.0)
    state, decision = _decide(routine, state, misaligned)
    assert decision.kind is _DecisionKind.STREAM


def test_decide_goto_cross_track_breach_fails_envelope():
    routine = _goto_routine()
    state = _RoutineState.initial()
    state, _decision = _decide(routine, state, _far(0))  # entry: baseline captured
    # 3 m off the straight x-axis path exceeds goto_cross_track_m=2.0.
    off_path = make_routine_observation(sim_time_us=1, position_enu=(1.0, 3.0, 2.0))
    state, decision = _decide(routine, state, off_path)
    assert decision.kind is _DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.ENVELOPE


def test_decide_goto_times_out_as_command_failure():
    routine = _goto_routine()  # timeout_s=20.0
    state = _RoutineState.initial()
    for t in range(5):  # reach the steady streaming branch
        state, _decision = _decide(routine, state, _far(t))
    state, decision = _decide(routine, state, _far(21_000_000))
    assert decision.kind is _DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.COMMAND


def test_reached_world_position_within_tolerance():
    predicate = _reached_world_position(3.0, 0.0, 2.0, 0.5)
    assert predicate(make_routine_observation(position_enu=(3.3, 0.0, 2.0)), None) is True
    assert predicate(make_routine_observation(position_enu=(4.0, 0.0, 2.0)), None) is False


def test_heading_aligned_faces_target_direction():
    predicate = _heading_aligned(3.0, 0.0, 20.0)
    baseline = np.zeros(3, dtype=np.float64)
    assert predicate(make_routine_observation(yaw_deg=0.0), baseline) is True
    assert predicate(make_routine_observation(yaw_deg=90.0), baseline) is False


def test_heading_aligned_trivial_for_near_vertical_leg():
    # Horizontal baseline->target distance below the heading threshold: any yaw is fine.
    predicate = _heading_aligned(0.05, 0.0, 5.0)
    assert predicate(make_routine_observation(yaw_deg=90.0), np.zeros(3, dtype=np.float64)) is True


def test_cross_track_to_segment_below_measures_perpendicular_distance():
    gate = _cross_track_to_segment_below(3.0, 0.0, 2.0)
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
    assert goto_step.command.command_id == _MAV_CMD_DO_SET_MODE
    assert goto_step.streaming is not None
    assert goto_step.streaming.target_enu == (3.0, 1.0, 2.5)
    assert goto_step.streaming.prime_ticks == 3
    # The geodetic target offsets the home anchor (47, 8, 488); altitude = home + up.
    assert goto_step.streaming.latitude_deg == pytest.approx(47.0, abs=1e-2)
    assert goto_step.streaming.longitude_deg != pytest.approx(8.0)
    assert goto_step.streaming.altitude_m == pytest.approx(490.5)
    assert goto_step.timeout_s == pytest.approx(25.0)
    assert goto_step.invariants


def test_build_vehicle_routine_rejects_goto_without_coordinate():
    with pytest.raises(ValueError, match="'x'"):
        _build([{"kind": "goto", "y": 0.0, "z": 1.0, "timeout_s": 10.0}])


def test_build_vehicle_routine_rejects_goto_without_timeout():
    with pytest.raises(ValueError, match="timeout_s"):
        _build([{"kind": "goto", "x": 1.0, "y": 0.0, "z": 1.0}])


def test_build_vehicle_routine_takeoff_yaw_param_is_nan():
    # _takeoff() leaves yaw (param4) NaN to hold current heading.
    routine = _build([{"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0}])
    assert math.isnan(routine.steps[0].command.params[3])


def test_moved_forward_at_least_uses_world_x_displacement_from_baseline():
    predicate = _moved_forward_at_least(1.0)
    baseline = make_routine_observation(position_enu=(2.0, 3.0, 0.0)).vehicle_state.position_enu
    assert predicate(make_routine_observation(position_enu=(2.9, 3.0, 1.0)), baseline) is False
    assert predicate(make_routine_observation(position_enu=(3.0, 3.0, 1.0)), baseline) is True


def test_forward_invariant_allows_commanded_x_displacement():
    # Drive the forward step through _decide: a large commanded x displacement
    # (with cross-track y drift inside the gate) must NOT trip the envelope. If
    # the cross-track gate wrongly measured total horizontal displacement this
    # would FAIL; instead the step advances on the completed forward distance.
    routine = _build(
        [
            {"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0},
            {"kind": "forward", "distance_m": 2.0, "timeout_s": 20.0},
        ]
    )
    forward_routine = _VehicleRoutine(steps=(routine.steps[1],))
    state = _RoutineState.initial()

    # Entry tick captures the baseline at the pre-forward position and SENDs.
    state, decision = _decide(
        forward_routine,
        state,
        make_routine_observation(sim_time_us=0, position_enu=(0.0, 2.0, 0.0)),
    )
    assert decision.kind is _DecisionKind.SEND

    # The commanded x displacement (and small cross-track drift) passes the gates.
    _, decision = _decide(
        forward_routine,
        state,
        make_routine_observation(sim_time_us=10_000, position_enu=(2.0, 1.84, 2.0)),
    )
    assert decision.kind is not _DecisionKind.FAIL


def test_cross_track_drift_gate_limits_sideways_drift():
    gate = _cross_track_drift_below(2.0)
    baseline = make_routine_observation(position_enu=(0.0, 2.0, 0.0)).vehicle_state.position_enu
    assert gate(make_routine_observation(position_enu=(10.0, 3.99, 2.0)), baseline) is None
    breach = gate(make_routine_observation(position_enu=(1.0, 4.01, 2.0)), baseline)
    assert breach is not None
    assert "cross-track drift" in breach


def test_disarmed_predicate_waits_for_bridge_disarm_status():
    predicate = _disarmed()
    assert predicate(make_routine_observation(is_armed=True), None) is False
    assert predicate(make_routine_observation(is_armed=False), None) is True


def test_decide_invariant_breach_fails_with_envelope_kind():
    # A _takeoff step with a tilt invariant fails as an ENVELOPE event when the
    # body tilts past the limit, distinct from a command rejection/timeout.
    takeoff_step = _RoutineStep(
        command=_takeoff(47.0, 8.0, 100.0),
        is_complete=_climbed_at_least(5.0),
        label="takeoff",
        timeout_s=30.0,
        invariants=(_tilt_below(35.0),),
    )
    routine = _VehicleRoutine(steps=(takeoff_step,))
    state = _RoutineState.initial()
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is _DecisionKind.SEND

    _, decision = _decide(
        routine, state, make_routine_observation(sim_time_us=10_000, tilt_deg=50.0)
    )
    assert decision.kind is _DecisionKind.FAIL
    assert decision.failure_kind is FailureKind.ENVELOPE
    assert "tilt" in decision.reason


def test_decide_hover_step_holds_before_advancing():
    # A command-less hover step never SENDs; it advances only once the stable
    # hover predicate has held continuously for hold_s.
    hover_step = _RoutineStep(
        command=None,
        is_complete=_hovering_stable(
            max_abs_vertical_speed_m_s=0.35, max_abs_horizontal_speed_m_s=0.30
        ),
        label="hover",
        timeout_s=10.0,
        hold_s=2.0,
    )
    routine = _VehicleRoutine(steps=(hover_step,))
    state = _RoutineState.initial()

    # Entry tick: command-less step begins -> WAIT (no SEND).
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is _DecisionKind.WAIT
    assert state.sent is True

    # Stable but the hold window has not elapsed yet -> WAIT, hold started.
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=500_000))
    assert decision.kind is _DecisionKind.WAIT
    assert state.hold_since_us == 500_000

    # A transient excursion (too fast) breaks the hold; it must restart.
    state, decision = _decide(
        routine,
        state,
        make_routine_observation(sim_time_us=1_000_000, linear_velocity_enu=(0.0, 0.0, 1.0)),
    )
    assert decision.kind is _DecisionKind.WAIT
    assert state.hold_since_us is None

    # Stable again from t=1.5s; the hold restarts here.
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=1_500_000))
    assert decision.kind is _DecisionKind.WAIT
    assert state.hold_since_us == 1_500_000

    # 2s later the hold window has elapsed -> ADVANCE past the last step.
    state, decision = _decide(routine, state, make_routine_observation(sim_time_us=3_500_000))
    assert decision.kind is _DecisionKind.ADVANCE
    assert state.cursor == 1
