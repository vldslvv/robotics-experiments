"""Tests for the pure deploy-time approach transition."""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np

from robotics_experiments.config import (
    PressApproachGoalConfig,
    PressApproachVelocityReferenceConfig,
)
from robotics_experiments.rl.press import (
    HandoffGate,
    PressApproachExecutionParams,
    PressApproachExecutionResult,
    PressApproachExecutionState,
    PressObservation,
    PressObservationMask,
    advance_press_approach_execution,
    initial_press_approach_execution_state,
)

_GOAL = PressApproachGoalConfig(
    range_min_m=0.40,
    range_max_m=0.42,
    alignment_error_max_m=0.03,
    linear_speed_max_mps=0.10,
    angular_speed_max_radps=0.40,
    body_z_alignment_min=0.965926,
    stable_steps=5,
    velocity_reference=PressApproachVelocityReferenceConfig(
        range_center_m=0.41,
        wall_normal_prediction_horizon_s=0.4,
        wall_normal_alignment_gate_m=0.03,
        wall_normal_gain_per_s=1.25,
        wall_normal_min_mps=-0.04,
        wall_normal_max_mps=0.08,
        tangential_gain_per_s=2.0,
        tangential_speed_max_mps=0.10,
    ),
)
_GATE = HandoffGate(
    pose_confidence_min=0.85,
    pose_age_max_s=0.1,
    switch_range_min_m=0.45,
    switch_range_max_m=0.50,
    lateral_align_max_m=0.08,
    wall_normal_speed_max_mps=0.02,
    lateral_speed_max_mps=0.02,
    angular_speed_max_radps=0.05,
    body_z_alignment_min=0.965926,
    estimator_health_min=0.80,
)


def _observation(
    *,
    range_m: float = 0.48,
    lateral_m: float = 0.0,
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    contact: bool = False,
    lamp_state: int = 0,
    valid: bool = True,
) -> PressObservation:
    return PressObservation(
        relative_switch_pos_m=np.array([range_m, lateral_m, 0.0]),
        relative_switch_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pose_confidence=1.0,
        pose_age_s=0.0,
        detection_valid=valid,
        body_lin_vel_mps=np.array(linear_velocity),
        body_ang_vel_radps=np.array(angular_velocity),
        attitude_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pusher_compression_m=0.0,
        pusher_compression_rate_mps=0.0,
        contact_valid=True,
        contact_flag=contact,
        lamp_state=lamp_state,
        lamp_confidence=1.0,
        lamp_age_s=0.0,
        previous_action=np.zeros(4),
        handoff_mode=0,
        estimator_health=1.0,
        obs_mask=int(
            PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
        ),
    )


def _params(*, inference=lambda _obs: np.array([0.2, 0.0, 0.0, 0.0])):
    return PressApproachExecutionParams(
        inference=inference,
        action_dim=4,
        handoff_gate=_GATE,
        goal=_GOAL,
    )


def _advance(*, state=None, observation=None, params=None):
    return advance_press_approach_execution(
        params=_params() if params is None else params,
        state=(initial_press_approach_execution_state(action_dim=4) if state is None else state),
        observation=_observation() if observation is None else observation,
    )


def test_public_records_are_approach_specific() -> None:
    assert [field.name for field in fields(PressApproachExecutionParams)] == [
        "inference",
        "action_dim",
        "handoff_gate",
        "goal",
    ]
    assert [field.name for field in fields(PressApproachExecutionResult)] == [
        "state",
        "action",
        "complete",
        "failure_reason",
        "handoff_reason",
        "actor_observation",
        "raw_action",
    ]
    assert [field.name for field in fields(PressApproachExecutionState)] == [
        "previous_action",
        "handoff_latched",
        "stable_goal_steps",
        "initial_switch_range_m",
        "closest_switch_range_m",
        "complete",
        "contact_failure",
    ]


def test_holds_zero_until_handoff_gate_passes() -> None:
    result = _advance(observation=_observation(range_m=0.48, linear_velocity=(0.10, 0.0, 0.0)))
    np.testing.assert_array_equal(result.action, np.zeros(4))
    assert not result.state.handoff_latched
    assert result.handoff_reason == "wall_normal_speed"


def test_minimum_range_must_pass_before_actor_or_goal_progress() -> None:
    result = _advance(observation=_observation(range_m=0.44))
    np.testing.assert_array_equal(result.action, np.zeros(4))
    assert not result.state.handoff_latched
    assert result.state.stable_goal_steps == 0
    assert result.handoff_reason == "too_close"


def test_runs_inference_after_handoff_and_tracks_ranges() -> None:
    calls: list[np.ndarray] = []

    def inference(observation: np.ndarray) -> np.ndarray:
        calls.append(observation.copy())
        return np.array([0.2, -0.1, 0.0, 0.0])

    result = _advance(
        observation=_observation(range_m=0.48),
        params=_params(inference=inference),
    )
    np.testing.assert_allclose(result.action, [0.2, -0.1, 0.0, 0.0])
    assert len(calls) == 1
    assert result.state.handoff_latched
    assert result.state.initial_switch_range_m == 0.48
    assert result.state.closest_switch_range_m == 0.48
    assert len(result.actor_observation) == 34
    assert result.actor_observation[27:31] == (0.0, 0.0, 0.0, 0.0)
    assert result.actor_observation[31] == 1.0
    assert result.raw_action == (0.2, -0.1, 0.0, 0.0)

    closer = _advance(
        state=result.state,
        observation=_observation(range_m=0.44),
    )
    assert closer.state.initial_switch_range_m == 0.48
    assert closer.state.closest_switch_range_m == 0.44


def test_five_stable_steps_complete_with_zero_terminal_action() -> None:
    state = replace(
        initial_press_approach_execution_state(action_dim=4),
        handoff_latched=True,
    )
    for expected in range(1, 6):
        result = _advance(
            state=state,
            observation=_observation(range_m=0.41),
        )
        assert result.state.stable_goal_steps == expected
        assert result.complete is (expected == 5)
        state = result.state
    np.testing.assert_array_equal(result.action, np.zeros(4))
    assert result.state.complete
    assert not result.state.contact_failure


def test_unstable_sample_resets_stable_counter() -> None:
    state = replace(
        initial_press_approach_execution_state(action_dim=4),
        handoff_latched=True,
        stable_goal_steps=4,
    )
    result = _advance(
        state=state,
        observation=_observation(range_m=0.41, lateral_m=0.04),
    )
    assert result.state.stable_goal_steps == 0
    assert not result.complete


def test_contact_returns_policy_failure_and_zero_action() -> None:
    result = _advance(observation=_observation(contact=True))
    assert result.failure_reason is not None
    assert "contact" in result.failure_reason
    assert result.state.contact_failure
    np.testing.assert_array_equal(result.action, np.zeros(4))


def test_invalid_observation_returns_policy_failure() -> None:
    result = _advance(observation=_observation(valid=False))
    assert result.failure_reason is not None
    assert "invalid" in result.failure_reason
    np.testing.assert_array_equal(result.action, np.zeros(4))


def test_lamp_transition_is_invalid_for_approach_stage() -> None:
    result = _advance(observation=_observation(lamp_state=1))
    assert result.failure_reason is not None
    assert "lamp" in result.failure_reason


def test_invalid_inference_output_fails() -> None:
    result = _advance(params=_params(inference=lambda _obs: np.zeros(3)))
    assert result.failure_reason is not None
    assert "shape" in result.failure_reason
