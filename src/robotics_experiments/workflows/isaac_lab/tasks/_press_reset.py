"""Host-safe deterministic transforms for matched press-entry reset samples."""

from __future__ import annotations

import math

import torch

from robotics_experiments.flight_parameters import FlightParameters
from robotics_experiments.rl.flight_parameters import evaluate_flight_parameters
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


def _quaternion_from_rpy(*, rpy_rad: torch.Tensor) -> torch.Tensor:
    half = 0.5 * rpy_rad
    cr, cp, cy = torch.cos(half).unbind(dim=-1)
    sr, sp, sy = torch.sin(half).unbind(dim=-1)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _hover_pose_from_unit_samples(
    *, unit_samples: torch.Tensor, flight_parameters: FlightParameters
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample target-local poses from the measured hover parameters."""
    if unit_samples.ndim != 2 or unit_samples.shape[1] != 6:
        raise ValueError("hover reset requires unit_samples with shape (count, 6)")

    hover = flight_parameters.hover
    position_lower = unit_samples.new_tensor(hover.position_error_enu_m.lower)
    position_upper = unit_samples.new_tensor(hover.position_error_enu_m.upper)
    position_error = position_lower + unit_samples[:, :3] * (position_upper - position_lower)
    radial_utilization = torch.maximum(
        torch.linalg.vector_norm(position_error[:, :2], dim=-1)
        / hover.horizontal_position_error_m.upper,
        torch.linalg.vector_norm(position_error, dim=-1) / hover.position_error_m.upper,
    )
    position_error /= radial_utilization.clamp_min(1.0)[:, None]
    interior_scale = 1.0 - math.sqrt(torch.finfo(unit_samples.dtype).eps)
    position_error *= interior_scale

    attitude_lower = unit_samples.new_tensor(hover.attitude_error_rpy_rad.lower)
    attitude_upper = unit_samples.new_tensor(hover.attitude_error_rpy_rad.upper)
    rpy = attitude_lower + unit_samples[:, 3:] * (attitude_upper - attitude_lower)
    attitude = _quaternion_from_rpy(rpy_rad=rpy)
    for _ in range(2):
        utilization = evaluate_flight_parameters(
            flight_parameters=flight_parameters,
            position_error_enu_m=torch.zeros_like(position_error),
            attitude_error_flu_enu_wxyz=attitude,
        ).attitude_utilization
        rpy /= utilization.clamp_min(1.0)[:, None]
        attitude = _quaternion_from_rpy(rpy_rad=rpy)
    rpy *= interior_scale
    attitude = _quaternion_from_rpy(rpy_rad=rpy)
    return position_error, attitude


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
