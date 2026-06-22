"""Pure command-layer tests: the ``decide`` stepper and the fleet barrier.

These exercise the contract centerpiece without Docker, Isaac, or pymavlink:
``decide`` is driven through arm -> takeoff -> done with synthetic
``RoutineObservation``s, and ``FleetCommander``'s ``STEP_SYNCHRONIZED`` barrier
is checked with a fake :class:`AbstractCommandPort`.
"""

import math

import pytest
from _fakes import FakeCommandPort, make_routine_observation

from robotics_experiments.commanding import (
    DecisionKind,
    ExecutionPolicy,
    FleetCommander,
    RoutineState,
    RoutineStep,
    VehicleCommander,
    VehicleRoutine,
    ack_accepted,
    arm,
    armed,
    build_vehicle_routine,
    climbed_at_least,
    decide,
    takeoff,
)
from robotics_experiments.commanding.commands import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_NAV_TAKEOFF,
)
from robotics_experiments.commanding.observation import (
    MAV_RESULT_TEMPORARILY_REJECTED,
    CommandAck,
)


def _arm_takeoff_routine(*, baseline_altitude_m=0.0, min_climb_m=1.0):
    arm_step = RoutineStep(
        command=arm(),
        is_complete=ack_accepted(MAV_CMD_COMPONENT_ARM_DISARM),
        label="arm",
        timeout_s=5.0,
        retry_on_temporarily_rejected=True,
    )
    takeoff_step = RoutineStep(
        command=takeoff(47.0, 8.0, 100.0),
        is_complete=climbed_at_least(min_climb_m, baseline_altitude_m=baseline_altitude_m),
        label="takeoff",
        timeout_s=10.0,
    )
    return VehicleRoutine(steps=(arm_step, takeoff_step))


def test_decide_runs_arm_takeoff_to_done():
    routine = _arm_takeoff_routine(min_climb_m=1.0)
    state = RoutineState.initial()

    # Tick 1: first step not sent -> SEND arm, baseline + timer captured.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is DecisionKind.SEND
    assert decision.command.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert state.sent is True
    assert state.step_started_us == 0
    assert state.baseline_position_enu is not None

    # Tick 2: arm sent, no ack -> WAIT.
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=10_000))
    assert decision.kind is DecisionKind.WAIT

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
    assert decision.kind is DecisionKind.SEND
    assert decision.command.command_id == MAV_CMD_NAV_TAKEOFF

    # Tick 5: not climbed yet -> WAIT.
    state, decision = decide(
        routine, state, make_routine_observation(sim_time_us=40_000, position_enu=(0.0, 0.0, 0.2))
    )
    assert decision.kind is DecisionKind.WAIT

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
    state = RoutineState.initial()
    state, decision = decide(routine, state, make_routine_observation(sim_time_us=0))
    assert decision.kind is DecisionKind.SEND

    # A temporarily-rejected arm ack re-sends rather than failing.
    temp_reject = CommandAck.from_result(
        MAV_CMD_COMPONENT_ARM_DISARM, MAV_RESULT_TEMPORARILY_REJECTED
    )
    state, decision = decide(
        routine, state, make_routine_observation(sim_time_us=5_000, last_ack=temp_reject)
    )
    assert decision.kind is DecisionKind.SEND
    assert decision.command.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert state.cursor == 0


def test_decide_fails_on_hard_rejection():
    routine = _arm_takeoff_routine()
    state = RoutineState.initial()
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
    state = RoutineState.initial()
    state, _ = decide(routine, state, make_routine_observation(sim_time_us=0))

    # arm step timeout_s is 5.0s; 6s of sim time with no ack -> FAIL.
    _, decision = decide(routine, state, make_routine_observation(sim_time_us=6_000_000))
    assert decision.kind is DecisionKind.FAIL
    assert "timed out" in decision.reason


def test_armed_predicate_reads_bridge_status():
    predicate = armed()
    assert predicate(make_routine_observation(is_armed=False)) is False
    assert predicate(make_routine_observation(is_armed=True)) is True


def _make_commander(label, routine):
    port = FakeCommandPort()
    return VehicleCommander(routine, port, label=label), port


def test_fleet_step_synchronized_barrier_holds_fast_drone():
    routine_fast = _arm_takeoff_routine()
    routine_slow = _arm_takeoff_routine()
    fast, fast_port = _make_commander("fast", routine_fast)
    slow, slow_port = _make_commander("slow", routine_slow)
    fleet = FleetCommander([fast, slow], policy=ExecutionPolicy.STEP_SYNCHRONIZED)

    arm_ack = CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, 0)

    def obs(armed_ack=None):
        return make_routine_observation(sim_time_us=0, last_ack=armed_ack)

    # Tick 1: both send arm.
    decisions = fleet.step({"fast": obs(), "slow": obs()})
    assert decisions["fast"].kind is DecisionKind.SEND
    assert decisions["slow"].kind is DecisionKind.SEND

    # Tick 2: fast is armed but slow is not -> the barrier holds fast at WAIT,
    # so it must not advance to the takeoff step ahead of slow.
    decisions = fleet.step({"fast": obs(arm_ack), "slow": obs()})
    assert decisions["fast"].kind is DecisionKind.WAIT
    assert decisions["slow"].kind is DecisionKind.WAIT
    assert fast.current_step_index == 0
    assert slow.current_step_index == 0

    # Tick 3: now slow is armed too -> both advance to the takeoff step together.
    decisions = fleet.step({"fast": obs(arm_ack), "slow": obs(arm_ack)})
    assert decisions["fast"].kind is DecisionKind.ADVANCE
    assert decisions["slow"].kind is DecisionKind.ADVANCE
    assert fast.current_step_index == 1
    assert slow.current_step_index == 1

    # Each port saw exactly the arm command once (no double-send from the hold).
    assert [c.label for c in fast_port.sent] == ["arm"]
    assert [c.label for c in slow_port.sent] == ["arm"]


def test_fleet_barrier_latches_acks_arriving_on_different_ticks():
    # The realistic case: each drone's accepted ack arrives on its own tick and
    # is NOT re-presented afterwards. Without latching the accepted ack, the
    # fast drone would revert to not-complete once its ack tick passes and the barrier
    # would deadlock; with latching, both advance once the slow drone's ack
    # finally arrives.
    fast, _ = _make_commander("fast", _arm_takeoff_routine())
    slow, _ = _make_commander("slow", _arm_takeoff_routine())
    fleet = FleetCommander([fast, slow], policy=ExecutionPolicy.STEP_SYNCHRONIZED)
    arm_ack = CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, 0)

    fleet.step({"fast": make_routine_observation(), "slow": make_routine_observation()})

    # Fast's ack arrives this tick only; it must be held at the barrier.
    decisions = fleet.step(
        {"fast": make_routine_observation(last_ack=arm_ack), "slow": make_routine_observation()}
    )
    assert decisions["fast"].kind is DecisionKind.WAIT
    assert fast.current_step_index == 0

    # No ack for fast this tick (it already passed); slow's ack arrives now.
    # The latch keeps fast complete, so both advance together.
    decisions = fleet.step(
        {"fast": make_routine_observation(), "slow": make_routine_observation(last_ack=arm_ack)}
    )
    assert decisions["fast"].kind is DecisionKind.ADVANCE
    assert decisions["slow"].kind is DecisionKind.ADVANCE
    assert fast.current_step_index == 1
    assert slow.current_step_index == 1


def _build(steps, *, target_altitude_m=488.0, command_ack_timeout_s=10.0):
    return build_vehicle_routine(
        steps,
        target_latitude_deg=47.0,
        target_longitude_deg=8.0,
        target_altitude_m=target_altitude_m,
        command_ack_timeout_s=command_ack_timeout_s,
    )


def test_build_vehicle_routine_arm_only():
    routine = _build([{"kind": "arm"}])
    assert [s.label for s in routine.steps] == ["arm"]
    arm_step = routine.steps[0]
    assert arm_step.command.command_id == MAV_CMD_COMPONENT_ARM_DISARM
    assert arm_step.timeout_s == 10.0
    # Arm retries on PX4's transient pre-arm rejection, matching the scenario.
    assert arm_step.retry_on_temporarily_rejected is True


def test_build_vehicle_routine_arm_takeoff_applies_climb_to_target_altitude():
    routine = _build(
        [{"kind": "arm"}, {"kind": "takeoff", "climb_m": 1.5}], target_altitude_m=488.0
    )
    assert [s.label for s in routine.steps] == ["arm", "takeoff"]
    takeoff_step = routine.steps[1]
    assert takeoff_step.command.command_id == MAV_CMD_NAV_TAKEOFF
    # param5/6 are the target lat/lon, param7 the absolute altitude = base + climb.
    assert takeoff_step.command.params[4] == pytest.approx(47.0)
    assert takeoff_step.command.params[5] == pytest.approx(8.0)
    assert takeoff_step.command.params[6] == pytest.approx(489.5)
    assert takeoff_step.retry_on_temporarily_rejected is True


def test_build_vehicle_routine_empty_steps_is_empty_routine():
    assert _build([]).steps == ()


def test_build_vehicle_routine_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown kind"):
        _build([{"kind": "land"}])


def test_build_vehicle_routine_rejects_takeoff_without_climb():
    with pytest.raises(ValueError, match="climb_m"):
        _build([{"kind": "takeoff"}])


def test_build_vehicle_routine_takeoff_yaw_param_is_nan():
    # takeoff() leaves yaw (param4) NaN to hold current heading.
    routine = _build([{"kind": "takeoff", "climb_m": 2.0}])
    assert math.isnan(routine.steps[0].command.params[3])


def test_fleet_independent_advances_freely():
    fast, _ = _make_commander("fast", _arm_takeoff_routine())
    slow, _ = _make_commander("slow", _arm_takeoff_routine())
    fleet = FleetCommander([fast, slow], policy=ExecutionPolicy.INDEPENDENT)

    arm_ack = CommandAck.from_result(MAV_CMD_COMPONENT_ARM_DISARM, 0)
    fleet.step({"fast": make_routine_observation(), "slow": make_routine_observation()})

    # Fast gets its ack while slow does not; under INDEPENDENT fast advances.
    decisions = fleet.step(
        {
            "fast": make_routine_observation(last_ack=arm_ack),
            "slow": make_routine_observation(),
        }
    )
    assert decisions["fast"].kind is DecisionKind.ADVANCE
    assert decisions["slow"].kind is DecisionKind.WAIT
    assert fast.current_step_index == 1
    assert slow.current_step_index == 0
