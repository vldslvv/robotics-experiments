"""Host-safe deterministic transforms for matched press-entry reset samples."""

from __future__ import annotations

import math

import torch

from robotics_experiments.config import PressApproachTrainingResetConfig
from robotics_experiments.rl.press import HandoffGate

__all__: list[str] = []


def _disk_from_unit_samples(
    *,
    radius_sample: torch.Tensor,
    angle_sample: torch.Tensor,
    radius_max: float,
) -> torch.Tensor:
    radius = torch.sqrt(radius_sample) * radius_max
    angle = angle_sample * (2.0 * math.pi)
    return torch.stack((radius * torch.cos(angle), radius * torch.sin(angle)), dim=-1)


def _press_entry_state_from_unit_samples(
    *,
    unit_samples: torch.Tensor,
    gate: HandoffGate,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map nine independent unit samples into the complete actor-entry envelope."""
    if unit_samples.ndim != 2 or unit_samples.shape[1] != 9:
        raise ValueError("press entry reset requires unit_samples with shape (count, 9)")

    wall_normal = gate.switch_range_min_m + unit_samples[:, 0] * (
        gate.switch_range_max_m - gate.switch_range_min_m
    )
    lateral_vertical = _disk_from_unit_samples(
        radius_sample=unit_samples[:, 1],
        angle_sample=unit_samples[:, 2],
        radius_max=gate.lateral_align_max_m,
    )
    wall_normal_speed = (2.0 * unit_samples[:, 3] - 1.0) * (gate.wall_normal_speed_max_mps)
    lateral_speed = _disk_from_unit_samples(
        radius_sample=unit_samples[:, 4],
        angle_sample=unit_samples[:, 5],
        radius_max=gate.lateral_speed_max_mps,
    )
    linear_velocity_contact = torch.cat((wall_normal_speed[:, None], lateral_speed), dim=-1)

    angular_radius = torch.pow(unit_samples[:, 6], 1.0 / 3.0) * gate.angular_speed_max_radps
    angular_cos_polar = 2.0 * unit_samples[:, 7] - 1.0
    angular_sin_polar = torch.sqrt(torch.clamp(1.0 - angular_cos_polar.square(), min=0.0))
    angular_azimuth = unit_samples[:, 8] * (2.0 * math.pi)
    angular_velocity = angular_radius[:, None] * torch.stack(
        (
            angular_sin_polar * torch.cos(angular_azimuth),
            angular_sin_polar * torch.sin(angular_azimuth),
            angular_cos_polar,
        ),
        dim=-1,
    )
    return wall_normal, lateral_vertical, linear_velocity_contact, angular_velocity


def _press_training_entry_state_from_unit_samples(
    *,
    unit_samples: torch.Tensor,
    gate: HandoffGate,
    training_reset: PressApproachTrainingResetConfig,
    hard_case_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix the full handoff envelope with the configured hard-case annulus."""
    if unit_samples.ndim != 2 or unit_samples.shape[1] != 10:
        raise ValueError("press training reset requires unit_samples with shape (count, 10)")

    wall_normal, lateral_vertical, linear_velocity, angular_velocity = (
        _press_entry_state_from_unit_samples(
            unit_samples=unit_samples[:, :9],
            gate=gate,
        )
    )
    hard_wall_normal = training_reset.range_min_m + unit_samples[:, 0] * (
        training_reset.range_max_m - training_reset.range_min_m
    )
    radius_squared = training_reset.tangential_error_min_m**2 + unit_samples[:, 1] * (
        training_reset.tangential_error_max_m**2 - training_reset.tangential_error_min_m**2
    )
    angle = unit_samples[:, 2] * (2.0 * math.pi)
    radius = torch.sqrt(radius_squared)
    hard_lateral_vertical = torch.stack(
        (radius * torch.cos(angle), radius * torch.sin(angle)),
        dim=-1,
    )
    hard_case = unit_samples[:, 9] < hard_case_fraction
    return (
        torch.where(hard_case, hard_wall_normal, wall_normal),
        torch.where(hard_case[:, None], hard_lateral_vertical, lateral_vertical),
        linear_velocity,
        angular_velocity,
    )
