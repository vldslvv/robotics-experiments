"""Pure runtime telemetry projection tests."""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robotics_experiments._cadence import _CadenceState
from robotics_experiments.commanding import (
    DriverPhase,
    RoutineObservation,
    RoutineState,
    RoutineStep,
    VehicleRoutine,
    VelocitySetpoint,
)
from robotics_experiments.commanding._routine_model import _OffboardState
from robotics_experiments.px4_bridge import BridgeStatus, VehicleState
from robotics_experiments.rl.press import PressApproachExecutionState, PressObservation
from tests.support.paths import REPO_ROOT


def _load_status_module():
    path = (
        REPO_ROOT / "isaac/runtime/extensions/experiment_runtime/ExperimentRuntime_python/status.py"
    )
    spec = importlib.util.spec_from_file_location("experiment_runtime_status_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return SimpleNamespace(
        rl=SimpleNamespace(
            press=SimpleNamespace(
                handoff=SimpleNamespace(
                    pose_confidence_min=0.85,
                    pose_age_max_s=0.1,
                    switch_range_min_m=0.44,
                    switch_range_max_m=0.52,
                    lateral_align_max_m=0.08,
                    wall_normal_speed_max_mps=0.02,
                    lateral_speed_max_mps=0.02,
                    angular_speed_max_radps=0.05,
                    relative_attitude_error_max_rad=0.10,
                    body_z_alignment_min=0.965926,
                    estimator_health_min=0.8,
                ),
                goal=SimpleNamespace(
                    range_min_m=0.40,
                    range_max_m=0.42,
                    alignment_error_max_m=0.01,
                    stable_steps=100,
                ),
            )
        )
    )


def _bridge_status():
    return BridgeStatus(
        is_connected=True,
        is_armed=True,
        first_heartbeat_received=True,
        sensor_send_count=10,
        actuator_receive_count=10,
        last_sensor_send_us=200_000,
        last_actuator_receive_us=196_000,
        actuator_lag_us=4_000,
        last_disconnect_reason=None,
    )


def _press_observation():
    return PressObservation(
        relative_switch_pos_m=np.array([0.48, 0.02, -0.01]),
        relative_switch_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pose_confidence=1.0,
        pose_age_s=0.0,
        detection_valid=True,
        body_lin_vel_mps=np.zeros(3),
        body_ang_vel_radps=np.zeros(3),
        attitude_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pusher_compression_m=0.0,
        pusher_compression_rate_mps=0.0,
        contact_valid=True,
        contact_flag=False,
        lamp_state=0,
        lamp_confidence=1.0,
        lamp_age_s=0.0,
        previous_action=np.zeros(4),
        handoff_mode=1,
        estimator_health=1.0,
        obs_mask=7,
    )


def _observation():
    return RoutineObservation(
        vehicle_state=VehicleState.hover_at(position_enu=(0.0, 17.48, 2.5), sim_time_us=200_000),
        bridge_status=_bridge_status(),
        last_ack=None,
        failsafe=False,
        sim_time_us=200_000,
        task_observation=_press_observation(),
    )


def _flying(*, descriptor, step, state, observation):
    environment = SimpleNamespace(read_status=lambda: SimpleNamespace(observation=observation))
    routine_config = SimpleNamespace(steps=(descriptor,), goto_position_tol_m=0.5)
    return SimpleNamespace(
        spawned=SimpleNamespace(
            label="x500_01",
            vehicle=SimpleNamespace(routine=routine_config),
        ),
        context=SimpleNamespace(env=environment),
        driver=SimpleNamespace(
            phase=DriverPhase.DRIVE,
            previous_cursor=0,
            failure=None,
            routine=VehicleRoutine(steps=(step,), offboard_setpoint_rate_hz=50.0),
            routine_state=state,
        ),
    )


def test_navigation_snapshot_includes_live_handoff_preview():
    module = _load_status_module()
    observation = _observation()
    descriptor = {
        "kind": "goto",
        "x": 0.0,
        "y": 17.52,
        "z": 2.5,
        "ground_speed_m_s": 0.08,
        "position_tol_m": 0.06,
    }
    step = RoutineStep(
        command=None,
        is_complete=lambda _observation, _baseline: False,
        label="goto[0.0,17.5,2.5]",
        timeout_s=20.0,
    )
    flying = _flying(
        descriptor=descriptor,
        step=step,
        state=RoutineState(sent=True, step_started_us=100_000),
        observation=observation,
    )

    snapshot = module._vehicle_runtime_snapshot(  # noqa: SLF001 - focused projection test
        config=_config(),
        flying=flying,
        host_status=SimpleNamespace(sim_time_us=204_000, bridge_status=_bridge_status()),
    )

    assert snapshot.sim_time_us == 200_000
    assert snapshot.routine_step.kind == "goto"
    assert snapshot.navigation.target_enu == (0.0, 17.52, 2.5)
    assert snapshot.navigation.remaining_distance_m == pytest.approx(0.04)
    assert snapshot.navigation.position_tolerance_m == 0.06
    assert snapshot.navigation.ground_speed_mps == 0.08
    assert snapshot.press.handoff_ready
    assert snapshot.press.handoff_reason == "ready"
    assert snapshot.policy is None


def test_policy_snapshot_separates_pending_actor_result_from_transmission():
    module = _load_status_module()
    observation = _observation()
    descriptor = {"kind": "policy", "label": "press-approach"}
    step = RoutineStep(
        command=None,
        is_complete=lambda _observation, _baseline: False,
        label="policy[press-approach]",
        timeout_s=20.0,
    )
    requested = VelocitySetpoint(0.04, -0.02, 0.01, 0.03)
    transmitted = VelocitySetpoint(0.03, -0.01, 0.0, 0.02)
    policy_state = PressApproachExecutionState(
        previous_action=(0.08, 0.04, -0.02, 0.01),
        handoff_latched=True,
        actor_evaluations=7,
        stable_goal_steps=2,
    )
    offboard = _OffboardState(
        policy_cadence=_CadenceState(previous_time_ns=200_000_000),
        policy_generation=7,
        transmitted_policy_generation=6,
        cached_setpoint=requested,
        transmitted_setpoint=transmitted,
        transmitted_setpoint_sim_time_us=180_000,
        requested=True,
        accepted=True,
        policy_state=policy_state,
    )
    flying = _flying(
        descriptor=descriptor,
        step=step,
        state=RoutineState(sent=True, step_started_us=100_000, offboard=offboard),
        observation=observation,
    )

    snapshot = module._vehicle_runtime_snapshot(  # noqa: SLF001 - focused projection test
        config=_config(),
        flying=flying,
        host_status=SimpleNamespace(sim_time_us=204_000, bridge_status=_bridge_status()),
    )

    policy = snapshot.policy
    assert policy.policy_generation == 7
    assert policy.actor_sim_time_us == 200_000
    assert policy.transmitted_policy_generation == 6
    assert policy.transmitted_sim_time_us == 180_000
    assert policy.actor_evaluations == 7
    assert policy.handoff_latched
    assert policy.stable_goal_steps == 2
    assert policy.normalized_action == (0.08, 0.04, -0.02, 0.01)
    assert policy.requested_setpoint.values != policy.transmitted_setpoint.values
