"""Host tests for the matched Isaac Lab press-entry reset transform."""

from __future__ import annotations

import pytest
import torch

from robotics_experiments.config import PressApproachTrainingResetConfig
from robotics_experiments.rl.press import HandoffGate
from robotics_experiments.workflows.isaac_lab.tasks._press_reset import (
    _press_entry_state_from_unit_samples,
    _press_training_entry_state_from_unit_samples,
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
_TRAINING_RESET = PressApproachTrainingResetConfig(
    hard_case_fraction=0.5,
    range_min_m=0.45,
    range_max_m=0.47,
    tangential_error_min_m=0.06,
    tangential_error_max_m=0.08,
)


def test_press_entry_reset_is_deterministic_and_stays_inside_shared_gate() -> None:
    unit_samples = torch.tensor(
        [
            [0.0] * 9,
            [1.0] * 9,
            [0.3, 0.8, 0.2, 0.7, 0.6, 0.4, 0.9, 0.1, 0.75],
        ],
        dtype=torch.float64,
    )

    first = _press_entry_state_from_unit_samples(
        unit_samples=unit_samples,
        gate=_GATE,
    )
    second = _press_entry_state_from_unit_samples(
        unit_samples=unit_samples.clone(),
        gate=_GATE,
    )

    for first_value, second_value in zip(first, second, strict=True):
        torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)

    wall_normal, lateral_vertical, linear_velocity, angular_velocity = first
    assert torch.all(wall_normal >= _GATE.switch_range_min_m)
    assert torch.all(wall_normal <= _GATE.switch_range_max_m)
    assert torch.all(torch.linalg.vector_norm(lateral_vertical, dim=-1) <= 0.08 + 1e-12)
    assert torch.all(torch.abs(linear_velocity[:, 0]) <= 0.02 + 1e-12)
    assert torch.all(torch.linalg.vector_norm(linear_velocity[:, 1:3], dim=-1) <= 0.02 + 1e-12)
    assert torch.all(torch.linalg.vector_norm(angular_velocity, dim=-1) <= 0.05 + 1e-12)


def test_press_training_reset_mixes_full_envelope_with_area_uniform_annulus() -> None:
    unit_samples = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [1.0, 1.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.49],
            [1.0, 1.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
        ],
        dtype=torch.float64,
    )

    wall_normal, lateral_vertical, linear_velocity, angular_velocity = (
        _press_training_entry_state_from_unit_samples(
            unit_samples=unit_samples,
            gate=_GATE,
            training_reset=_TRAINING_RESET,
            hard_case_fraction=0.5,
        )
    )

    torch.testing.assert_close(
        wall_normal,
        torch.tensor([0.45, 0.47, 0.50], dtype=torch.float64),
    )
    radii = torch.linalg.vector_norm(lateral_vertical, dim=-1)
    torch.testing.assert_close(radii, torch.tensor([0.06, 0.08, 0.08], dtype=torch.float64))
    torch.testing.assert_close(linear_velocity[:, 0], torch.zeros(3, dtype=torch.float64))
    torch.testing.assert_close(angular_velocity, torch.zeros((3, 3), dtype=torch.float64))


def test_press_training_reset_rejects_wrong_sample_width() -> None:
    with pytest.raises(ValueError, match=r"shape \(count, 10\)"):
        _press_training_entry_state_from_unit_samples(
            unit_samples=torch.zeros((1, 9)),
            gate=_GATE,
            training_reset=_TRAINING_RESET,
            hard_case_fraction=0.5,
        )
