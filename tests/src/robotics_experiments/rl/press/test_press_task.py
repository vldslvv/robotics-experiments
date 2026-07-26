"""Pure tests for the press approach curriculum transition."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest
import torch

from robotics_experiments.config import (
    PressApproachGoalConfig,
    PressApproachRewardConfig,
    PressApproachTerminationConfig,
)
from robotics_experiments.rl.press import (
    PressApproachInput,
    PressApproachRewardTerms,
    initial_press_approach_state,
    reset_press_approach_state,
    step_press_approach,
)

_GOAL = PressApproachGoalConfig(
    range_min_m=0.40,
    range_max_m=0.42,
    alignment_error_max_m=0.03,
    linear_speed_max_mps=0.10,
    angular_speed_max_radps=0.40,
    body_z_alignment_min=0.965926,
    stable_steps=5,
    closing_speed_far_mps=0.20,
    closing_speed_far_range_m=0.50,
    closing_speed_zero_range_m=0.42,
)
_REWARD = PressApproachRewardConfig(
    range_progress_weight=50.0,
    alignment_error_weight=50.0,
    closing_speed_error_weight=20.0,
    upright_error_weight=5.0,
    action_delta_weight=0.004,
    success_weight=150.0,
    failure_weight=100.0,
    timeout_weight=25.0,
)
_TERMINATION = PressApproachTerminationConfig(
    crash_tilt_cos=0.819152,
    min_altitude_m=0.25,
    max_wall_normal_distance_m=1.0,
    max_lateral_error_m=0.30,
    max_vertical_error_m=0.30,
    contact_force_threshold_n=0.25,
)


def _state(*, count: int = 1, range_m: float = 0.50):
    return initial_press_approach_state(
        relative_switch_range_m=np.full(count, range_m),
        action=np.zeros((count, 4)),
    )


def _sample(
    *,
    range_m: float = 0.50,
    lateral_m: float = 0.0,
    vertical_m: float = 0.0,
    closing_speed_mps: float = 0.20,
    lateral_speed_mps: float = 0.0,
    angular_speed_radps: float = 0.0,
    upright: float = 1.0,
    altitude_m: float = 1.0,
    contact: bool = False,
    action: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
    time_out: bool = False,
) -> PressApproachInput[np.ndarray]:
    return PressApproachInput(
        relative_switch_pos_m=np.array([[range_m, lateral_m, vertical_m]]),
        linear_velocity_mps=np.array([[closing_speed_mps, lateral_speed_mps, 0.0]]),
        angular_velocity_radps=np.array([[angular_speed_radps, 0.0, 0.0]]),
        body_z_world_z=np.array([upright]),
        altitude_m=np.array([altitude_m]),
        contact_flag=np.array([contact]),
        action=np.array([action]),
        time_out=np.array([time_out]),
    )


def _step(*, state=None, sample=None):
    return step_press_approach(
        state=_state() if state is None else state,
        sample=_sample() if sample is None else sample,
        goal_cfg=_GOAL,
        reward_cfg=_REWARD,
        termination_cfg=_TERMINATION,
    )


def test_reward_record_has_exactly_eight_signed_terms() -> None:
    assert [field.name for field in fields(PressApproachRewardTerms)] == [
        "range_progress",
        "alignment_error",
        "closing_speed_error",
        "upright_error",
        "action_delta",
        "success",
        "failure",
        "timeout",
    ]


def test_range_progress_is_signed() -> None:
    closer = _step(sample=_sample(range_m=0.48, closing_speed_mps=0.15))
    farther = _step(sample=_sample(range_m=0.52, closing_speed_mps=0.20))
    assert closer.rewards.range_progress[0] == pytest.approx(1.0)
    assert farther.rewards.range_progress[0] == pytest.approx(-1.0)


def test_alignment_and_upright_errors_are_quadratic() -> None:
    result = _step(
        sample=_sample(
            lateral_m=0.03,
            vertical_m=0.04,
            upright=0.9,
        )
    )
    assert result.rewards.alignment_error[0] == pytest.approx(-50.0 * 0.05**2)
    assert result.rewards.upright_error[0] == pytest.approx(-5.0 * 0.1**2)


@pytest.mark.parametrize(
    ("range_m", "target_mps"),
    [(0.50, 0.20), (0.46, 0.10), (0.42, 0.0), (0.40, 0.0), (0.60, 0.20)],
)
def test_closing_speed_target_tapers_linearly(
    range_m: float,
    target_mps: float,
) -> None:
    result = _step(
        state=_state(range_m=range_m),
        sample=_sample(range_m=range_m, closing_speed_mps=target_mps),
    )
    assert result.rewards.closing_speed_error[0] == pytest.approx(0.0)
    wrong = _step(
        state=_state(range_m=range_m),
        sample=_sample(range_m=range_m, closing_speed_mps=target_mps + 0.1),
    )
    assert wrong.rewards.closing_speed_error[0] == pytest.approx(-0.2)


def test_action_delta_uses_mean_square() -> None:
    result = _step(sample=_sample(action=(1.0, -1.0, 0.0, 0.0)))
    assert result.rewards.action_delta[0] == pytest.approx(-0.002)


def test_success_requires_five_consecutive_stable_steps() -> None:
    state = _state(range_m=0.41)
    for expected_count in range(1, 6):
        result = _step(
            state=state,
            sample=_sample(range_m=0.41, closing_speed_mps=0.0),
        )
        assert result.state.stable_goal_steps[0] == expected_count
        assert bool(result.termination.success[0]) is (expected_count == 5)
        state = result.state
    assert result.rewards.success[0] == 150.0
    assert result.termination.terminated[0]


def test_unstable_step_resets_success_counter() -> None:
    state = replace(_state(range_m=0.41), stable_goal_steps=np.array([4]))
    result = _step(
        state=state,
        sample=_sample(range_m=0.41, lateral_m=0.031, closing_speed_mps=0.0),
    )
    assert result.state.stable_goal_steps[0] == 0
    assert not result.termination.success[0]


@pytest.mark.parametrize(
    ("sample", "flag"),
    [
        (_sample(contact=True), "contact"),
        (_sample(upright=0.7), "crash"),
        (_sample(altitude_m=0.1), "altitude"),
        (_sample(lateral_m=0.4), "relative_envelope"),
    ],
)
def test_contact_and_flight_envelope_failures(sample, flag: str) -> None:
    result = _step(sample=sample)
    assert getattr(result.termination, flag)[0]
    assert result.termination.terminated[0]
    assert result.rewards.failure[0] == -100.0
    assert not result.termination.success[0]


def test_timeout_is_truncation_only_and_mutually_exclusive_with_success() -> None:
    timed_out = _step(sample=_sample(time_out=True))
    assert timed_out.termination.time_out[0]
    assert not timed_out.termination.terminated[0]
    assert timed_out.rewards.timeout[0] == -25.0

    state = replace(_state(range_m=0.41), stable_goal_steps=np.array([4]))
    success = _step(
        state=state,
        sample=_sample(range_m=0.41, closing_speed_mps=0.0, time_out=True),
    )
    assert success.termination.success[0]
    assert not success.termination.time_out[0]
    assert success.rewards.timeout[0] == 0.0


def test_reward_total_is_sum_of_all_contributions() -> None:
    result = _step(
        sample=_sample(
            range_m=0.48,
            lateral_m=0.02,
            closing_speed_mps=0.10,
            upright=0.98,
            action=(0.5, 0.0, 0.0, 0.0),
        )
    )
    expected = sum(
        float(getattr(result.rewards, field.name)[0]) for field in fields(PressApproachRewardTerms)
    )
    assert result.rewards.total[0] == pytest.approx(expected)


def test_state_is_frozen_and_reset_is_selective() -> None:
    state = _state(count=2)
    with pytest.raises(FrozenInstanceError):
        state.stable_goal_steps = np.ones(2, dtype=np.int64)
    advanced = replace(
        state,
        previous_range_m=np.array([0.41, 0.42]),
        previous_action=np.ones((2, 4)),
        stable_goal_steps=np.array([4, 3]),
        closest_range_m=np.array([0.40, 0.39]),
    )
    reset = reset_press_approach_state(
        state=advanced,
        env_ids=np.array([1]),
        relative_switch_range_m=np.array([0.48]),
        action=np.zeros((1, 4)),
    )
    assert reset.stable_goal_steps.tolist() == [4, 0]
    assert reset.previous_range_m.tolist() == [0.41, 0.48]
    assert reset.closest_range_m.tolist() == [0.40, 0.48]
    assert np.all(reset.previous_action[0] == 1.0)
    assert np.all(reset.previous_action[1] == 0.0)


def test_numpy_and_torch_transitions_match() -> None:
    numpy_state = _state(count=2)
    numpy_sample = PressApproachInput(
        relative_switch_pos_m=np.array([[0.41, 0.01, 0.0], [0.55, 0.02, 0.01]]),
        linear_velocity_mps=np.array([[0.0, 0.0, 0.0], [0.18, 0.01, 0.0]]),
        angular_velocity_radps=np.array([[0.1, 0.0, 0.0], [0.2, 0.1, 0.0]]),
        body_z_world_z=np.array([1.0, 0.98]),
        altitude_m=np.array([1.0, 1.0]),
        contact_flag=np.array([False, False]),
        action=np.array([[0.1, 0.0, 0.0, 0.0], [0.2, -0.1, 0.0, 0.0]]),
        time_out=np.array([False, True]),
    )
    numpy_result = _step(state=numpy_state, sample=numpy_sample)
    torch_state = initial_press_approach_state(
        relative_switch_range_m=torch.tensor([0.50, 0.50], dtype=torch.float64),
        action=torch.zeros((2, 4), dtype=torch.float64),
    )
    torch_sample = PressApproachInput(
        **{
            field.name: torch.as_tensor(
                getattr(numpy_sample, field.name),
                dtype=(
                    torch.bool
                    if getattr(numpy_sample, field.name).dtype == np.bool_
                    else torch.float64
                ),
            )
            for field in fields(PressApproachInput)
        }
    )
    torch_result = step_press_approach(
        state=torch_state,
        sample=torch_sample,
        goal_cfg=_GOAL,
        reward_cfg=_REWARD,
        termination_cfg=_TERMINATION,
    )
    for field in fields(PressApproachRewardTerms):
        np.testing.assert_allclose(
            getattr(numpy_result.rewards, field.name),
            getattr(torch_result.rewards, field.name).cpu().numpy(),
        )
    np.testing.assert_allclose(
        numpy_result.rewards.total,
        torch_result.rewards.total.cpu().numpy(),
    )
