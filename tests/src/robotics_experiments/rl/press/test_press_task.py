"""Pure tests for the minimal press-approach transition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from robotics_experiments.config import (
    PressApproachGoalConfig,
    PressApproachRewardConfig,
    PressApproachTerminationConfig,
    PressApproachVelocityReferenceConfig,
)
from robotics_experiments.flight_parameters import load_flight_parameters
from robotics_experiments.rl.press import (
    PressApproachInput,
    PressApproachRewardTerms,
    PressApproachState,
    initial_press_approach_state,
    press_approach_target_velocity,
    quaternion_geodesic_error_rad,
    reset_press_approach_state,
    step_press_approach,
)

_GOAL = PressApproachGoalConfig(
    range_min_m=0.40,
    range_max_m=0.42,
    alignment_error_max_m=0.01,
    relative_attitude_error_max_rad=np.deg2rad(1.0),
    linear_speed_max_mps=0.03,
    angular_speed_max_radps=0.05,
    body_z_alignment_min=0.965926,
    stable_steps=100,
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
_REWARD = PressApproachRewardConfig(
    hover_stabilization="boundary_weighted",
    position_progress_weight=0.1,
    stabilization_weight=0.01,
    outside_envelope_weight=0.03,
    predictive_barrier_weight=0.1,
    action_effort_weight=1.0,
    hold_progress_weight=0.01,
    terminal_success_weight=1.5,
    terminal_failure_weight=3.0,
)
_TERMINATION = PressApproachTerminationConfig(
    crash_tilt_cos=0.819152,
    min_altitude_m=0.25,
    max_wall_normal_distance_m=1.0,
    max_lateral_error_m=0.30,
    max_vertical_error_m=0.30,
    contact_data_capacity_per_body=32,
    contact_force_threshold_n=0.25,
)
_ENVELOPE = load_flight_parameters(
    Path(__file__).parents[4] / "fixtures/flight_parameters/flight_parameters.json"
)


def _quat(*, angle_rad: float) -> np.ndarray:
    return np.array([[np.sin(angle_rad / 2.0), 0.0, 0.0, np.cos(angle_rad / 2.0)]])


def _sample(
    *,
    range_m: float = 0.50,
    lateral_m: float = 0.0,
    vertical_m: float = 0.0,
    closing_speed_mps: float = 0.0,
    angular_speed_radps: float = 0.0,
    action: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    attitude_error_rad: float = 0.0,
    upright: float = 1.0,
    altitude_m: float = 1.0,
    body_collision: bool = False,
    pusher_collision: bool = False,
    time_out: bool = False,
) -> PressApproachInput[np.ndarray]:
    relative_quat_xyzw = _quat(angle_rad=attitude_error_rad)
    return PressApproachInput(
        relative_switch_pos_m=np.array([[range_m, lateral_m, vertical_m]]),
        relative_switch_quat_xyzw=relative_quat_xyzw,
        position_error_enu_m=np.array(
            [[_GOAL.velocity_reference.range_center_m - range_m, -lateral_m, -vertical_m]]
        ),
        attitude_error_flu_enu_wxyz=relative_quat_xyzw[:, [3, 0, 1, 2]],
        linear_velocity_mps=np.array([[closing_speed_mps, 0.0, 0.0]]),
        angular_velocity_radps=np.array([[angular_speed_radps, 0.0, 0.0]]),
        normalized_action=np.array([action]),
        body_z_world_z=np.array([upright]),
        altitude_m=np.array([altitude_m]),
        body_collision_flag=np.array([body_collision]),
        pusher_collision_flag=np.array([pusher_collision]),
        time_out=np.array([time_out]),
    )


def _state(*, range_m: float = 0.50) -> PressApproachState[np.ndarray]:
    sample = _sample(range_m=range_m)
    return initial_press_approach_state(
        relative_switch_pos_m=sample.relative_switch_pos_m,
        relative_switch_quat_xyzw=sample.relative_switch_quat_xyzw,
        goal_cfg=_GOAL,
        flight_parameters=_ENVELOPE,
        position_error_enu_m=sample.position_error_enu_m,
        attitude_error_flu_enu_wxyz=sample.attitude_error_flu_enu_wxyz,
    )


def _step(
    *,
    state: PressApproachState[np.ndarray] | None = None,
    sample: PressApproachInput[np.ndarray] | None = None,
    approach_enabled: bool = True,
    reward: PressApproachRewardConfig = _REWARD,
    training: bool = False,
):
    return step_press_approach(
        state=_state() if state is None else state,
        sample=_sample() if sample is None else sample,
        goal_cfg=_GOAL,
        reward_cfg=reward,
        termination_cfg=_TERMINATION,
        minimum_range_m=0.40,
        flight_parameters=_ENVELOPE,
        hold_steps=100,
        approach_enabled=approach_enabled,
        training=training,
    )


def test_quaternion_geodesic_error_matches_numpy_and_torch_and_normalizes() -> None:
    quaternion = np.array([[0.0, 0.0, 2.0 * np.sin(0.2), 2.0 * np.cos(0.2)]])
    numpy_error = quaternion_geodesic_error_rad(quaternion_xyzw=quaternion)
    torch_error = quaternion_geodesic_error_rad(
        quaternion_xyzw=torch.tensor(quaternion, dtype=torch.float64)
    )
    assert numpy_error == pytest.approx([0.4])
    assert torch_error.cpu().numpy() == pytest.approx(numpy_error)


def test_approach_rewards_only_position_progress() -> None:
    state = _state(range_m=0.70)
    closer = _step(
        state=state,
        sample=_sample(range_m=0.60, attitude_error_rad=np.deg2rad(20.0)),
    )
    farther = _step(state=state, sample=_sample(range_m=0.80))
    assert closer.rewards.position_progress[0] > 0.0
    assert farther.rewards.position_progress[0] < 0.0
    assert closer.rewards.stabilization[0] == 0.0


def test_position_ready_stage_rewards_stabilization_quality() -> None:
    settled = _step(sample=_sample(range_m=0.41))
    moving = _step(sample=_sample(range_m=0.41, closing_speed_mps=0.015))
    tilted = _step(sample=_sample(range_m=0.41, attitude_error_rad=np.deg2rad(0.5)))
    captured = _step(sample=_sample(range_m=0.43))
    outside = _step(sample=_sample(range_m=0.52))
    assert settled.rewards.stabilization[0] == pytest.approx(0.01)
    assert moving.rewards.stabilization[0] == settled.rewards.stabilization[0]
    assert 0.0 < tilted.rewards.stabilization[0] < settled.rewards.stabilization[0]
    assert 0.0 < captured.rewards.stabilization[0] < settled.rewards.stabilization[0]
    assert outside.rewards.stabilization[0] == 0.0
    assert settled.rewards.position_progress[0] == 0.0


def test_predictive_barrier_penalizes_only_projected_range_undershoot() -> None:
    safe = _step(sample=_sample(range_m=0.45, closing_speed_mps=0.05))
    unsafe = _step(sample=_sample(range_m=0.41, closing_speed_mps=0.05))
    assert safe.rewards.predictive_barrier[0] == 0.0
    assert unsafe.rewards.predictive_barrier[0] == pytest.approx(-0.025)


@pytest.mark.parametrize("choice", ["uniform", "boundary_weighted", "outside_only"])
@pytest.mark.parametrize(
    ("range_m", "attitude_error_rad", "inside"),
    [(0.41, 0.0, True), (0.45, 0.01, True), (0.511, 0.0, False), (0.41, 0.11, False)],
)
def test_hover_reward_penalizes_current_pose_without_predictive_barrier(
    *, choice, range_m, attitude_error_rad, inside
) -> None:
    result = _step(
        sample=_sample(range_m=range_m, attitude_error_rad=attitude_error_rad),
        approach_enabled=False,
        reward=replace(_REWARD, hover_stabilization=choice),
    )
    assert bool(result.hover_evaluation.inside[0]) == inside
    weight = _REWARD.stabilization_weight
    if choice == "outside_only" and inside:
        weight = 0.0
    elif choice != "uniform" and not inside:
        weight = _REWARD.outside_envelope_weight
    utilization = (
        result.hover_evaluation.position_utilization[0]
        + result.hover_evaluation.attitude_utilization[0]
    )
    assert result.rewards.stabilization[0] == pytest.approx(-weight * utilization)
    assert result.rewards.position_progress[0] == 0.0
    assert result.rewards.predictive_barrier[0] == 0.0
    if range_m == 0.41 and attitude_error_rad == 0.0:
        assert result.rewards.stabilization[0] == 0.0
    elif weight:
        assert result.rewards.stabilization[0] < 0.0


def test_hover_penalizes_action_effort_but_approach_does_not() -> None:
    action = (0.1, -0.2, 0.3, -0.4)
    hover = _step(sample=_sample(range_m=0.41, action=action), approach_enabled=False)
    approach = _step(sample=_sample(range_m=0.41, action=action), approach_enabled=True)
    assert hover.rewards.action_effort[0] == pytest.approx(-0.075)
    assert approach.rewards.action_effort[0] == 0.0


def test_hold_progress_and_success_terminal_reward() -> None:
    state = replace(
        _state(range_m=0.41),
        stable_goal_steps=np.array([99]),
        longest_stable_goal_steps=np.array([99]),
    )
    result = _step(state=state, sample=_sample(range_m=0.41))
    assert result.rewards.hold_progress[0] == 0.01
    assert result.rewards.terminal_outcome[0] == 1.5
    assert bool(result.termination.success[0])


def test_repeated_hold_below_episode_record_earns_no_progress() -> None:
    state = replace(
        _state(range_m=0.41),
        stable_goal_steps=np.array([20]),
        longest_stable_goal_steps=np.array([40]),
    )
    result = _step(state=state, sample=_sample(range_m=0.41))
    assert result.state.stable_goal_steps[0] == 21
    assert result.state.longest_stable_goal_steps[0] == 40
    assert result.rewards.hold_progress[0] == 0.0


@pytest.mark.parametrize("training", [True, False])
def test_hover_hold_qualifies_evaluation_but_training_collects_full_horizon(*, training) -> None:
    state = _state(range_m=0.415)
    expected_reward = _step(
        sample=_sample(range_m=0.415), approach_enabled=False, training=training
    ).rewards.total
    for step in range(800):
        result = step_press_approach(
            state=state,
            sample=_sample(range_m=0.415, time_out=step == 799),
            goal_cfg=_GOAL,
            reward_cfg=_REWARD,
            termination_cfg=_TERMINATION,
            minimum_range_m=0.40,
            flight_parameters=_ENVELOPE,
            hold_steps=600,
            approach_enabled=False,
            training=training,
        )
        np.testing.assert_allclose(result.rewards.total, expected_reward)
        assert result.rewards.hold_progress[0] == 0.0
        assert result.rewards.terminal_outcome[0] == 0.0
        if result.termination.terminated[0] or result.termination.time_out[0]:
            break
        state = result.state
    assert step + 1 == (800 if training else 600)
    assert bool(result.termination.success[0]) == (not training)
    assert bool(result.termination.terminated[0]) == (not training)
    assert bool(result.termination.time_out[0]) == training


@pytest.mark.parametrize("previous_record", [0, 40, 600, 800])
def test_hover_reward_does_not_depend_on_previous_hold_record(*, previous_record) -> None:
    state = replace(
        _state(range_m=0.415),
        stable_goal_steps=np.array([20]),
        longest_stable_goal_steps=np.array([previous_record]),
    )
    result = _step(state=state, sample=_sample(range_m=0.415), approach_enabled=False)
    assert result.rewards.hold_progress[0] == 0.0
    np.testing.assert_allclose(
        result.rewards.total,
        _step(sample=_sample(range_m=0.415), approach_enabled=False).rewards.total,
    )


@pytest.mark.parametrize(
    "sample",
    [
        _sample(range_m=0.39),
        _sample(altitude_m=0.1),
        _sample(upright=0.5),
        _sample(lateral_m=0.4),
        _sample(body_collision=True),
        _sample(pusher_collision=True),
    ],
)
def test_hover_physical_failure_is_terminal_even_at_collection_timeout(*, sample) -> None:
    result = _step(
        sample=replace(sample, time_out=np.array([True])),
        approach_enabled=False,
        training=True,
    )
    assert result.termination.terminated[0]
    assert not result.termination.success[0]
    assert not result.termination.time_out[0]
    assert result.rewards.terminal_outcome[0] == -3.0


@pytest.mark.parametrize(
    "sample",
    [
        _sample(range_m=0.39),
        _sample(range_m=0.50, body_collision=True),
        _sample(range_m=0.50, time_out=True),
    ],
)
def test_every_unsuccessful_termination_gets_one_terminal_penalty(
    sample: PressApproachInput[np.ndarray],
) -> None:
    result = _step(sample=sample)
    assert result.rewards.terminal_outcome[0] == -3.0


def test_target_velocity_remains_an_evaluation_reference() -> None:
    target = press_approach_target_velocity(
        relative_switch_pos_m=np.array([[0.50, 0.02, -0.01]]),
        linear_velocity_mps=np.zeros((1, 3)),
        reference_cfg=_GOAL.velocity_reference,
    )
    assert target.shape == (1, 3)
    assert target[0, 0] > 0.0
    assert target[0, 1] > 0.0
    assert target[0, 2] < 0.0


@pytest.mark.parametrize("backend", [np.asarray, torch.from_numpy])
def test_reset_replaces_only_selected_environment(backend) -> None:
    state = PressApproachState(
        normalized_position_error=backend(np.array([1.0, 2.0])),
        normalized_pose_error=backend(np.array([1.0, 2.0])),
        stable_goal_steps=backend(np.array([4, 5])),
        longest_stable_goal_steps=backend(np.array([6, 7])),
        closest_range_m=backend(np.array([0.41, 0.42])),
    )
    reset = reset_press_approach_state(
        state=state,
        env_ids=[1],
        relative_switch_pos_m=backend(np.array([[0.48, 0.0, 0.0]])),
        relative_switch_quat_xyzw=backend(_quat(angle_rad=0.0)),
        goal_cfg=_GOAL,
        flight_parameters=_ENVELOPE,
        position_error_enu_m=backend(np.array([[-0.07, 0.0, 0.0]])),
        attitude_error_flu_enu_wxyz=backend(np.array([[1.0, 0.0, 0.0, 0.0]])),
    )
    assert reset.normalized_position_error[0] == 1.0
    assert reset.normalized_pose_error[0] == 1.0
    assert reset.stable_goal_steps.tolist() == [4, 0]
    assert reset.longest_stable_goal_steps.tolist() == [6, 0]
    assert reset.closest_range_m.tolist() == [0.41, 0.48]
    assert state.stable_goal_steps.tolist() == [4, 5]
    assert state.closest_range_m.tolist() == [0.41, 0.42]


@pytest.mark.parametrize("choice", ["uniform", "boundary_weighted", "outside_only"])
@pytest.mark.parametrize("approach_enabled", [False, True])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("time_out", [False, True])
def test_numpy_and_torch_transitions_match(*, choice, approach_enabled, training, time_out) -> None:
    sample = _sample(
        range_m=0.415,
        lateral_m=0.004,
        closing_speed_mps=0.01,
        angular_speed_radps=0.02,
        attitude_error_rad=0.01,
    )
    sample = replace(sample, time_out=np.array([time_out]))
    reward = replace(_REWARD, hover_stabilization=choice)
    numpy_result = _step(
        sample=sample, reward=reward, approach_enabled=approach_enabled, training=training
    )
    torch_sample = PressApproachInput(
        **{
            field.name: torch.as_tensor(getattr(sample, field.name))
            for field in fields(PressApproachInput)
        }
    )
    numpy_state = _state()
    torch_state = PressApproachState(
        **{
            field.name: torch.as_tensor(getattr(numpy_state, field.name))
            for field in fields(PressApproachState)
        }
    )
    torch_result = step_press_approach(
        state=torch_state,
        sample=torch_sample,
        goal_cfg=_GOAL,
        reward_cfg=reward,
        termination_cfg=_TERMINATION,
        minimum_range_m=0.40,
        flight_parameters=_ENVELOPE,
        hold_steps=100,
        approach_enabled=approach_enabled,
        training=training,
    )
    for field in fields(PressApproachRewardTerms):
        np.testing.assert_allclose(
            getattr(numpy_result.rewards, field.name),
            getattr(torch_result.rewards, field.name).cpu().numpy(),
        )
    for field in fields(numpy_result.termination):
        np.testing.assert_array_equal(
            getattr(numpy_result.termination, field.name),
            getattr(torch_result.termination, field.name).cpu().numpy(),
        )
    for field in fields(PressApproachState):
        np.testing.assert_allclose(
            getattr(numpy_result.state, field.name),
            getattr(torch_result.state, field.name).cpu().numpy(),
        )


def test_task_records_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        _state().stable_goal_steps = np.array([3])
