"""Host-safe deterministic transforms for matched press-entry reset samples."""

from __future__ import annotations

import math

import torch

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


def _rotation_ball_from_unit_samples(
    *,
    radius_sample: torch.Tensor,
    polar_sample: torch.Tensor,
    azimuth_sample: torch.Tensor,
    radius_max_rad: float,
) -> torch.Tensor:
    """Map independent unit samples to a volume-uniform axis-angle ball."""
    angle = torch.pow(radius_sample, 1.0 / 3.0) * radius_max_rad
    cos_polar = 2.0 * polar_sample - 1.0
    sin_polar = torch.sqrt(torch.clamp(1.0 - cos_polar.square(), min=0.0))
    azimuth = azimuth_sample * (2.0 * math.pi)
    axis = torch.stack(
        (
            sin_polar * torch.cos(azimuth),
            sin_polar * torch.sin(azimuth),
            cos_polar,
        ),
        dim=-1,
    )
    half_angle = 0.5 * angle
    return torch.cat(
        (torch.cos(half_angle)[:, None], axis * torch.sin(half_angle)[:, None]),
        dim=-1,
    )


def _press_entry_state_from_unit_samples(
    *,
    unit_samples: torch.Tensor,
    gate: HandoffGate,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map twelve independent unit samples into the complete actor-entry envelope."""
    if unit_samples.ndim != 2 or unit_samples.shape[1] != 12:
        raise ValueError("press entry reset requires unit_samples with shape (count, 12)")

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
    relative_quat_wxyz = _rotation_ball_from_unit_samples(
        radius_sample=unit_samples[:, 9],
        polar_sample=unit_samples[:, 10],
        azimuth_sample=unit_samples[:, 11],
        radius_max_rad=gate.relative_attitude_error_max_rad,
    )
    return (
        wall_normal,
        lateral_vertical,
        linear_velocity_contact,
        angular_velocity,
        relative_quat_wxyz,
    )
