"""Host tests for the shared full-envelope press-entry reset transform."""

from __future__ import annotations

import pytest
import torch

from robotics_experiments.rl.press import HandoffGate, quaternion_geodesic_error_rad
from robotics_experiments.workflows.isaac_lab.tasks._press_reset import (
    _press_entry_state_from_unit_samples,
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
    relative_attitude_error_max_rad=0.10,
    body_z_alignment_min=0.965926,
    estimator_health_min=0.80,
)


def test_press_entry_reset_is_deterministic_and_covers_shared_gate() -> None:
    unit_samples = torch.tensor(
        [
            [0.0] * 12,
            [1.0] * 12,
            [0.3, 0.8, 0.2, 0.7, 0.6, 0.4, 0.9, 0.1, 0.75, 0.6, 0.3, 0.8],
        ],
        dtype=torch.float64,
    )
    first = _press_entry_state_from_unit_samples(unit_samples=unit_samples, gate=_GATE)
    second = _press_entry_state_from_unit_samples(unit_samples=unit_samples.clone(), gate=_GATE)
    for first_value, second_value in zip(first, second, strict=True):
        torch.testing.assert_close(first_value, second_value, rtol=0.0, atol=0.0)

    wall_normal, lateral_vertical, linear_velocity, angular_velocity, relative_quat_wxyz = first
    assert wall_normal[[0, 1]].tolist() == pytest.approx([0.45, 0.50])
    assert torch.all(torch.linalg.vector_norm(lateral_vertical, dim=-1) <= 0.08 + 1e-12)
    assert torch.all(torch.abs(linear_velocity[:, 0]) <= 0.02 + 1e-12)
    assert torch.all(torch.linalg.vector_norm(linear_velocity[:, 1:3], dim=-1) <= 0.02 + 1e-12)
    assert torch.all(torch.linalg.vector_norm(angular_velocity, dim=-1) <= 0.05 + 1e-12)
    attitude_error = quaternion_geodesic_error_rad(
        quaternion_xyzw=relative_quat_wxyz[:, [1, 2, 3, 0]]
    )
    assert torch.all(attitude_error <= 0.10 + 1e-12)


def test_press_entry_reset_rejects_wrong_sample_width() -> None:
    with pytest.raises(ValueError, match=r"shape \(count, 12\)"):
        _press_entry_state_from_unit_samples(unit_samples=torch.zeros((1, 13)), gate=_GATE)


def test_rotation_sampling_is_volume_uniform() -> None:
    generator = torch.Generator().manual_seed(42)
    samples = torch.rand((50_000, 12), generator=generator, dtype=torch.float64)
    relative_quat_wxyz = _press_entry_state_from_unit_samples(
        unit_samples=samples,
        gate=_GATE,
    )[4]
    angles = quaternion_geodesic_error_rad(quaternion_xyzw=relative_quat_wxyz[:, [1, 2, 3, 0]])
    assert float(angles.mean()) == pytest.approx(0.075, abs=5.0e-4)
