"""Vectorized (torch) in-Lab flight actuation + force sink for DirectRLEnv.

Torch/isaacsim leaf: kept off the core import path (imported only by the env).
State lives in the frozen ``_FlightActuationParams`` (device tensors built once
by :func:`build_flight_actuation_params`) and behavior in pure top-level
functions, the batched twin of two pure, host-tested references it must match
term-for-term:

- the actuation cascade exported by ``robotics_experiments.rl``
  (numpy single-env oracle: ``ActuationParams`` + ``action_to_rotor_omega``) --
  this module batches the velocity/ctbr cascade over ``[num_envs]``; schema 5
  reproduces PX4's normalized allocator and legacy schemas use the physical
  allocation pseudoinverse;
- the per-rotor -> body-wrench reduction exported as
  ``robotics_experiments.rl.rotor_forces_to_body_wrench`` --
  ``_rotor_forces_to_body_wrench`` batches it.

``apply_body_wrench`` is the sole side-effecting sink, the vectorized analogue
of package-root ``IsaacArticulationIO.apply_dynamics``:
per-rotor thrust ``[0, 0, T_i]`` at the body-FLU rotor offsets, the body-Z yaw
reaction, and linear drag, reduced to one equivalent wrench per env applied at
the body origin in the body frame (Isaac Lab's
``set_external_force_and_torque`` applies in the body frame, so no body->world
rotation is needed -- unlike the bridge's ``is_global=True`` path).

Normalized controls pass through the bridge-owned affine motor mapping. The
realized thrust is then ``T_i = rotor_constant_i * omega_i^2`` clipped to
``[min, max]`` rotor velocity, exactly as the PX4 bridge's
``PegasusQuadraticThrustModel`` does.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import torch
from numpy.typing import ArrayLike

from robotics_experiments.alignment import ControllerSettings
from robotics_experiments.config import RlActionConfig
from robotics_experiments.rl import PolicyContract

__all__ = [
    "action_to_rotor_omega",
    "apply_body_wrench",
    "build_flight_actuation_params",
    "initial_flight_controller_state",
    "reset_flight_controller_state",
    "rotor_omega_to_body_wrench",
    "rotor_thrust_to_body_wrench",
    "seed_flight_controller_state",
]

_GRAVITY_M_S2 = 9.81
_EPS = 1e-9
_SATURATION_EPS = 1e-5


@dataclass(frozen=True)
class _FlightActuationParams:
    """Immutable batched actuation state: geometry + allocation + gains + limits.

    Device-tensor twin of ``robotics_experiments.rl.ActuationParams``; built once per
    env by :func:`build_flight_actuation_params`. The velocity-cascade gains and
    limits are populated for both kinds (``ctbr`` ignores them).
    """

    kind: str
    mass: float
    controller_dt_s: float
    device: torch.device | str
    positions_body: torch.Tensor
    rot_dir: torch.Tensor
    rotor_constant: torch.Tensor
    rolling_moment_coefficient: torch.Tensor
    min_rotor_velocity: torch.Tensor
    max_rotor_velocity: torch.Tensor
    input_offset: torch.Tensor
    input_scaling: torch.Tensor
    zero_position_armed: torch.Tensor
    allocation_pinv: torch.Tensor
    normalized_allocation_mix: torch.Tensor
    normalized_allocator: bool
    max_total_thrust: float
    kp_rate: torch.Tensor
    ki_rate: torch.Tensor
    kd_rate: torch.Tensor
    kff_rate: torch.Tensor
    rate_output_scale_nm: torch.Tensor
    rate_derivative_filter_hz: float | torch.Tensor
    rate_integral_limit_nm: torch.Tensor
    max_body_rate_radps: float
    hover_thrust_fraction: float | torch.Tensor
    hover_thrust_normalized: float | torch.Tensor
    kp_vel: torch.Tensor
    ki_vel: torch.Tensor
    kd_vel: torch.Tensor
    velocity_filter_hz: float | torch.Tensor
    velocity_derivative_filter_hz: float | torch.Tensor
    velocity_integral_limit_mps2: torch.Tensor
    kp_att: torch.Tensor
    attitude_yaw_weight: float | torch.Tensor
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_tilt_rad: float | torch.Tensor
    min_collective_thrust_fraction: float | torch.Tensor
    max_collective_thrust_fraction: float | torch.Tensor
    horizontal_thrust_margin_fraction: float | torch.Tensor


@dataclass(frozen=True)
class _FlightControllerState:
    """Batched state of the PX4-like velocity and rate cascade."""

    filtered_velocity_world: torch.Tensor
    acceleration_world: torch.Tensor
    velocity_integral_world: torch.Tensor
    previous_angular_velocity_body: torch.Tensor
    angular_acceleration_body: torch.Tensor
    rate_integral_body: torch.Tensor
    initialized: torch.Tensor


def _matrix_from_quat_wxyz(quat: torch.Tensor) -> torch.Tensor:
    """Batched body-FLU -> world-ENU rotation matrix from ``[..., 4]`` wxyz quats."""
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    two = 2.0
    row0 = torch.stack(
        [1 - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y)], dim=-1
    )
    row1 = torch.stack(
        [two * (x * y + w * z), 1 - two * (x * x + z * z), two * (y * z - w * x)], dim=-1
    )
    row2 = torch.stack(
        [two * (x * z - w * y), two * (y * z + w * x), 1 - two * (x * x + y * y)], dim=-1
    )
    return torch.stack([row0, row1, row2], dim=-2)


def _canonical_quat(quat: torch.Tensor, /) -> torch.Tensor:
    normalized = quat / quat.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return torch.where(normalized[:, :1] < 0.0, -normalized, normalized)


def _quat_multiply(first: torch.Tensor, second: torch.Tensor, /) -> torch.Tensor:
    aw, ax, ay, az = first.unbind(dim=-1)
    bw, bx, by, bz = second.unbind(dim=-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_conjugate(quat: torch.Tensor, /) -> torch.Tensor:
    return torch.cat([quat[:, :1], -quat[:, 1:]], dim=-1)


def _quat_from_two_vectors(source: torch.Tensor, target: torch.Tensor, /) -> torch.Tensor:
    source = source / source.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    target = target / target.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    dot = (source * target).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    ordinary = torch.cat([1.0 + dot, torch.cross(source, target, dim=-1)], dim=-1)
    ordinary = ordinary / ordinary.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    basis_index = source.abs().argmin(dim=-1)
    basis = torch.nn.functional.one_hot(basis_index, num_classes=3).to(source.dtype)
    opposite_axis = torch.cross(source, basis, dim=-1)
    opposite_axis = opposite_axis / opposite_axis.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    opposite = torch.cat([torch.zeros_like(dot), opposite_axis], dim=-1)
    return torch.where(dot < -1.0 + 1e-6, opposite, ordinary)


def _quat_from_rotation(rotation: torch.Tensor, /) -> torch.Tensor:
    m00, m01, m02 = rotation[:, 0].unbind(dim=-1)
    m10, m11, m12 = rotation[:, 1].unbind(dim=-1)
    m20, m21, m22 = rotation[:, 2].unbind(dim=-1)
    candidates = torch.stack(
        [
            torch.stack([1.0 + m00 + m11 + m22, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, 1.0 + m00 - m11 - m22, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, 1.0 - m00 + m11 - m22, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m02 + m20, m12 + m21, 1.0 - m00 - m11 + m22], dim=-1),
        ],
        dim=1,
    )
    magnitudes = torch.sqrt(torch.diagonal(candidates, dim1=-2, dim2=-1).clamp_min(0.0))
    index = magnitudes.argmax(dim=-1)
    selected = candidates[
        torch.arange(len(candidates), device=rotation.device),
        index,
    ]
    denominator = (2.0 * magnitudes.gather(1, index.unsqueeze(-1))).clamp_min(0.2)
    quat = selected / denominator
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def _attitude_rate_setpoint(
    *,
    attitude_wxyz: torch.Tensor,
    rotation: torch.Tensor,
    desired_rotation: torch.Tensor,
    proportional_gain: torch.Tensor,
    yaw_weight: float | torch.Tensor,
    yaw_rate_radps: torch.Tensor,
) -> torch.Tensor:
    """Batched PX4 reduced-quaternion attitude law."""
    attitude = attitude_wxyz / attitude_wxyz.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    desired_attitude = _quat_from_rotation(desired_rotation)
    reduced_desired = _quat_from_two_vectors(rotation[:, :, 2], desired_rotation[:, :, 2])
    corner = (reduced_desired[:, 1].abs() > 1.0 - 1e-5) | (reduced_desired[:, 2].abs() > 1.0 - 1e-5)
    reduced_desired = torch.where(
        corner.unsqueeze(-1),
        desired_attitude,
        _quat_multiply(reduced_desired, attitude),
    )
    delta_yaw = _canonical_quat(_quat_multiply(_quat_conjugate(reduced_desired), desired_attitude))
    weight = _environment_values(yaw_weight, delta_yaw[:, 0])
    weighted_yaw = torch.stack(
        [
            torch.cos(weight * torch.arccos(delta_yaw[:, 0].clamp(-1.0, 1.0))),
            torch.zeros_like(weight),
            torch.zeros_like(weight),
            torch.sin(weight * torch.arcsin(delta_yaw[:, 3].clamp(-1.0, 1.0))),
        ],
        dim=-1,
    )
    weighted_desired = _quat_multiply(reduced_desired, weighted_yaw)
    error = _canonical_quat(_quat_multiply(_quat_conjugate(attitude), weighted_desired))
    gain_values = (
        proportional_gain
        if proportional_gain.ndim == 2
        else proportional_gain.unsqueeze(0).expand(len(attitude), -1)
    )
    yaw_gain = torch.where(
        weight > 1e-4,
        gain_values[:, 2] / weight.clamp_min(1e-4),
        gain_values[:, 2],
    )
    gain = torch.cat([gain_values[:, :2], yaw_gain.unsqueeze(-1)], dim=-1)
    rate_setpoint = 2.0 * error[:, 1:] * gain
    return rate_setpoint + rotation[:, 2, :] * yaw_rate_radps.unsqueeze(-1)


def _clamp_tilt(
    thrust_world: torch.Tensor,
    max_tilt_rad: float | torch.Tensor,
) -> torch.Tensor:
    """Batched unit desired body-Z from a world thrust vector, tilt-limited from up."""
    norm = thrust_world.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 0.0, 1.0], device=thrust_world.device).expand_as(thrust_world)
    safe = norm.squeeze(-1) > _EPS
    direction = torch.where(norm > _EPS, thrust_world / norm.clamp_min(_EPS), up)
    cos_tilt = direction[:, 2].clamp(-1.0, 1.0)
    tilt = torch.arccos(cos_tilt)
    horizontal = direction - cos_tilt.unsqueeze(-1) * up
    horizontal = horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    if isinstance(max_tilt_rad, torch.Tensor):
        tilt_limit = max_tilt_rad
        clamped = (
            torch.cos(tilt_limit).unsqueeze(-1) * up
            + torch.sin(tilt_limit).unsqueeze(-1) * horizontal
        )
    else:
        tilt_limit = torch.full_like(tilt, max_tilt_rad)
        clamped = np.cos(max_tilt_rad) * up + np.sin(max_tilt_rad) * horizontal
    over_tilt = (tilt > tilt_limit) & safe
    return torch.where(over_tilt.unsqueeze(-1), clamped, direction)


def _rotor_forces_to_body_wrench(
    positions_body: torch.Tensor,
    forces_n: torch.Tensor,
    body_torque: torch.Tensor,
    body_drag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-rotor thrust + body effects to one body wrench per env.

    Batched analogue of ``robotics_experiments.rl.rotor_forces_to_body_wrench``. ``forces_n``
    is ``[B, N]``; ``positions_body`` is ``[N, 3]``; ``body_torque`` /
    ``body_drag`` are ``[B, 3]``. Returns ``(force_body, torque_body)``, both
    ``[B, 3]`` in body-FLU.
    """
    x = positions_body[:, 0]
    y = positions_body[:, 1]
    total_thrust = forces_n.sum(dim=-1)
    zeros = torch.zeros_like(total_thrust)
    force_body = torch.stack([zeros, zeros, total_thrust], dim=-1) + body_drag
    roll = (y.unsqueeze(0) * forces_n).sum(dim=-1)
    pitch = -(x.unsqueeze(0) * forces_n).sum(dim=-1)
    torque_body = torch.stack([roll, pitch, torch.zeros_like(roll)], dim=-1) + body_torque
    return force_body, torque_body


def _wrench_to_omega(
    params: _FlightActuationParams, collective: torch.Tensor, body_torque: torch.Tensor
) -> torch.Tensor:
    wrench = torch.cat([collective.unsqueeze(-1), body_torque], dim=-1)  # [B, 4]
    thrust = wrench @ params.allocation_pinv.T  # [B, N]
    thrust = thrust.clamp_min(0.0)
    omega = torch.sqrt(thrust / params.rotor_constant)
    return torch.clamp(omega, params.min_rotor_velocity, params.max_rotor_velocity)


def _desaturation_gain(
    actuator_sp: torch.Tensor,
    desaturation_vector: torch.Tensor,
    *,
    actuator_max: float = 1.0,
) -> torch.Tensor:
    valid = desaturation_vector.abs() >= 0.2
    safe_vector = torch.where(valid, desaturation_vector, 1.0).unsqueeze(0)
    below_gain = torch.where(
        valid.unsqueeze(0) & (actuator_sp < 0.0),
        -actuator_sp / safe_vector,
        torch.zeros_like(actuator_sp),
    )
    above_gain = torch.where(
        valid.unsqueeze(0) & (actuator_sp > actuator_max),
        (actuator_max - actuator_sp) / safe_vector,
        torch.zeros_like(actuator_sp),
    )
    gains = below_gain + above_gain
    zeros = torch.zeros(len(actuator_sp), dtype=actuator_sp.dtype, device=actuator_sp.device)
    return torch.minimum(gains.amin(dim=-1), zeros) + torch.maximum(
        gains.amax(dim=-1),
        zeros,
    )


def _desaturate(
    actuator_sp: torch.Tensor,
    desaturation_vector: torch.Tensor,
    *,
    increase_only: bool = False,
    actuator_max: float = 1.0,
) -> torch.Tensor:
    gain = _desaturation_gain(
        actuator_sp,
        desaturation_vector,
        actuator_max=actuator_max,
    )
    blocked = torch.zeros_like(gain, dtype=torch.bool)
    if increase_only:
        blocked = gain < 0.0
        gain = gain.clamp_min(0.0)
    actuator_sp = actuator_sp + gain.unsqueeze(-1) * desaturation_vector
    gain = 0.5 * _desaturation_gain(
        actuator_sp,
        desaturation_vector,
        actuator_max=actuator_max,
    )
    gain = torch.where(blocked, torch.zeros_like(gain), gain)
    return actuator_sp + gain.unsqueeze(-1) * desaturation_vector


def _normalized_controls_to_omega(
    params: _FlightActuationParams,
    collective: torch.Tensor,
    body_torque: torch.Tensor,
) -> torch.Tensor:
    mix = params.normalized_allocation_mix
    control = torch.stack(
        [
            body_torque[:, 0],
            -body_torque[:, 1],
            -body_torque[:, 2],
            -collective,
        ],
        dim=-1,
    )
    actuator_sp = (
        control[:, 0:1] * mix[:, 0] + control[:, 1:2] * mix[:, 1] + control[:, 3:4] * mix[:, 3]
    )
    actuator_sp = _desaturate(actuator_sp, mix[:, 3], increase_only=True)
    actuator_sp = _desaturate(actuator_sp, mix[:, 0])
    actuator_sp = _desaturate(actuator_sp, mix[:, 1])
    actuator_sp = actuator_sp + control[:, 2:3] * mix[:, 2]
    actuator_sp = _desaturate(actuator_sp, mix[:, 2], actuator_max=1.15)
    actuator_sp = _desaturate(actuator_sp, mix[:, 3], increase_only=True)
    actuator_sp = actuator_sp.clamp(0.0, 1.0)
    omega = (actuator_sp + params.input_offset) * params.input_scaling + params.zero_position_armed
    return torch.clamp(omega, params.min_rotor_velocity, params.max_rotor_velocity)


def _low_pass_alpha(
    *,
    cutoff_hz: float | torch.Tensor,
    dt_s: float,
) -> float | torch.Tensor:
    if isinstance(cutoff_hz, torch.Tensor):
        tensor_scaled = 2.0 * math.pi * cutoff_hz * dt_s
        filtered = tensor_scaled / (1.0 + tensor_scaled)
        return torch.where(cutoff_hz > 0.0, filtered, torch.ones_like(filtered))
    if cutoff_hz <= 0.0:
        return 1.0
    scalar_scaled = 2.0 * math.pi * cutoff_hz * dt_s
    return scalar_scaled / (1.0 + scalar_scaled)


def _environment_values(
    value: float | torch.Tensor,
    reference: torch.Tensor,
    /,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.full_like(reference, value)


def _limited_thrust_world(
    *,
    params: _FlightActuationParams,
    desired_acceleration: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = torch.tensor([0.0, 0.0, _GRAVITY_M_S2], device=params.device)
    physical_requested = params.mass * (desired_acceleration + gravity)
    desired_z = torch.cat(
        [
            desired_acceleration[:, :2],
            torch.full_like(desired_acceleration[:, 2:3], _GRAVITY_M_S2),
        ],
        dim=-1,
    )
    desired_z = desired_z / desired_z.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    hover_fraction = _environment_values(
        params.hover_thrust_fraction,
        desired_acceleration[:, 2],
    )
    collective_fraction = (
        hover_fraction
        * (1.0 + desired_acceleration[:, 2] / _GRAVITY_M_S2)
        / desired_z[:, 2].clamp_min(_EPS)
    )
    px4_requested = desired_z * collective_fraction.unsqueeze(-1) * params.max_total_thrust
    requested = torch.where(
        (hover_fraction > 0.0).unsqueeze(-1),
        px4_requested,
        physical_requested,
    )
    maximum = _environment_values(
        params.max_total_thrust * params.max_collective_thrust_fraction,
        requested[:, 2],
    )
    minimum = _environment_values(
        params.max_total_thrust * params.min_collective_thrust_fraction,
        requested[:, 2],
    )
    vertical = torch.minimum(torch.maximum(requested[:, 2], minimum), maximum)
    radial_limit = torch.sqrt((maximum * maximum - vertical.square()).clamp_min(0.0))
    margin_limit = _environment_values(
        params.max_total_thrust
        * (params.max_collective_thrust_fraction - params.horizontal_thrust_margin_fraction),
        requested[:, 2],
    )
    horizontal_limit = torch.minimum(radial_limit, margin_limit)
    horizontal = requested[:, :2]
    horizontal_norm = horizontal.norm(dim=-1, keepdim=True)
    horizontal_scale = torch.minimum(
        torch.ones_like(horizontal_norm),
        horizontal_limit.unsqueeze(-1) / horizontal_norm.clamp_min(_EPS),
    )
    limited = torch.cat(
        [horizontal * horizontal_scale, vertical.unsqueeze(-1)],
        dim=-1,
    )
    achieved_acceleration = limited / params.mass - gravity
    return limited, achieved_acceleration


def _normalized_thrust_world(
    *,
    params: _FlightActuationParams,
    desired_acceleration: torch.Tensor,
    velocity_error: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = torch.full_like(desired_acceleration[:, 2:3], _GRAVITY_M_S2)
    desired_z = torch.cat([desired_acceleration[:, :2], gravity], dim=-1)
    desired_z = _clamp_tilt(desired_z, params.max_tilt_rad)
    hover = _environment_values(
        params.hover_thrust_normalized,
        desired_acceleration[:, 2],
    )
    minimum = _environment_values(
        params.min_collective_thrust_fraction,
        desired_acceleration[:, 2],
    )
    maximum = _environment_values(
        params.max_collective_thrust_fraction,
        desired_acceleration[:, 2],
    )
    collective = torch.maximum(
        hover
        * (1.0 + desired_acceleration[:, 2] / _GRAVITY_M_S2)
        / desired_z[:, 2].clamp_min(_EPS),
        minimum,
    )
    requested = desired_z * collective.unsqueeze(-1)
    block_vertical = ((requested[:, 2] <= minimum) & (velocity_error[:, 2] <= 0.0)) | (
        (requested[:, 2] >= maximum) & (velocity_error[:, 2] >= 0.0)
    )
    adjusted_vertical_error = torch.where(
        block_vertical,
        torch.zeros_like(velocity_error[:, 2]),
        velocity_error[:, 2],
    )

    horizontal = requested[:, :2]
    horizontal_norm = horizontal.norm(dim=-1)
    margin = _environment_values(
        params.horizontal_thrust_margin_fraction,
        horizontal_norm,
    )
    allocated_horizontal = torch.minimum(horizontal_norm, margin)
    vertical_max = torch.sqrt((maximum.square() - allocated_horizontal.square()).clamp_min(0.0))
    vertical = torch.minimum(requested[:, 2], vertical_max)
    horizontal_max = torch.sqrt((maximum.square() - vertical.square()).clamp_min(0.0))
    horizontal_scale = torch.minimum(
        torch.ones_like(horizontal_norm),
        horizontal_max / horizontal_norm.clamp_min(_EPS),
    )
    limited_horizontal = horizontal * horizontal_scale.unsqueeze(-1)
    limited = torch.cat([limited_horizontal, vertical.unsqueeze(-1)], dim=-1)
    produced_acceleration = limited_horizontal * (_GRAVITY_M_S2 / hover).unsqueeze(-1)
    arw_active = desired_acceleration[:, :2].square().sum(
        dim=-1
    ) > produced_acceleration.square().sum(dim=-1)
    gain = 2.0 / (params.kp_vel[:, 0] if params.kp_vel.ndim == 2 else params.kp_vel[0])
    adjusted_horizontal_error = velocity_error[:, :2] - gain.unsqueeze(-1) * (
        desired_acceleration[:, :2] - produced_acceleration
    )
    adjusted_horizontal_error = torch.where(
        arw_active.unsqueeze(-1),
        adjusted_horizontal_error,
        velocity_error[:, :2],
    )
    return limited, torch.cat(
        [adjusted_horizontal_error, adjusted_vertical_error.unsqueeze(-1)],
        dim=-1,
    )


def _rate_to_omega(
    params: _FlightActuationParams,
    controller_state: _FlightControllerState,
    collective: torch.Tensor,
    body_rate_setpoint: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    dt_s: float,
    /,
) -> tuple[_FlightControllerState, torch.Tensor]:
    rate = torch.clamp(body_rate_setpoint, -params.max_body_rate_radps, params.max_body_rate_radps)
    raw_angular_acceleration = (
        angular_velocity_body - controller_state.previous_angular_velocity_body
    ) / dt_s
    alpha = _low_pass_alpha(
        cutoff_hz=params.rate_derivative_filter_hz,
        dt_s=dt_s,
    )
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.unsqueeze(-1)
    filtered_angular_acceleration = controller_state.angular_acceleration_body + alpha * (
        raw_angular_acceleration - controller_state.angular_acceleration_body
    )
    angular_acceleration = torch.where(
        controller_state.initialized.unsqueeze(-1),
        filtered_angular_acceleration,
        torch.zeros_like(filtered_angular_acceleration),
    )
    rate_error = rate - angular_velocity_body
    normalized_torque = (
        params.kp_rate * rate_error
        + controller_state.rate_integral_body
        - params.kd_rate * angular_acceleration
        + params.kff_rate * rate
    )
    body_torque = (
        normalized_torque
        if params.normalized_allocator
        else params.rate_output_scale_nm * normalized_torque
    )
    integral_factor = torch.clamp(
        1.0 - torch.square(rate_error / math.radians(400.0)),
        min=0.0,
    )
    rate_integral = torch.clamp(
        controller_state.rate_integral_body + integral_factor * params.ki_rate * rate_error * dt_s,
        -params.rate_integral_limit_nm,
        params.rate_integral_limit_nm,
    )
    successor = _FlightControllerState(
        filtered_velocity_world=controller_state.filtered_velocity_world,
        acceleration_world=controller_state.acceleration_world,
        velocity_integral_world=controller_state.velocity_integral_world,
        previous_angular_velocity_body=angular_velocity_body,
        angular_acceleration_body=angular_acceleration,
        rate_integral_body=rate_integral,
        initialized=torch.ones_like(controller_state.initialized),
    )
    if params.normalized_allocator:
        omega = _normalized_controls_to_omega(params, collective, body_torque)
    else:
        omega = _wrench_to_omega(params, collective, body_torque)
    return successor, omega


def _ctbr(
    params: _FlightActuationParams,
    controller_state: _FlightControllerState,
    action: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    dt_s: float,
    /,
) -> tuple[_FlightControllerState, torch.Tensor]:
    fraction = torch.clamp(params.hover_thrust_fraction * (1.0 + action[:, 0]), 0.0, 1.0)
    collective = fraction * params.max_total_thrust
    body_rate_setpoint = action[:, 1:4] * params.max_body_rate_radps
    return _rate_to_omega(
        params,
        controller_state,
        collective,
        body_rate_setpoint,
        angular_velocity_body,
        dt_s,
    )


def _velocity(
    params: _FlightActuationParams,
    controller_state: _FlightControllerState,
    action: torch.Tensor,
    attitude_wxyz: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    dt_s: float,
    /,
) -> tuple[_FlightControllerState, torch.Tensor]:
    rotation = _matrix_from_quat_wxyz(attitude_wxyz)  # [B, 3, 3]
    yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])  # [B]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    velocity_command_body = action[:, :3] * params.max_body_velocity_mps
    velocity_setpoint_world = torch.stack(
        [
            cos_yaw * velocity_command_body[:, 0] - sin_yaw * velocity_command_body[:, 1],
            sin_yaw * velocity_command_body[:, 0] + cos_yaw * velocity_command_body[:, 1],
            velocity_command_body[:, 2],
        ],
        dim=-1,
    )
    velocity_alpha = _low_pass_alpha(
        cutoff_hz=params.velocity_filter_hz,
        dt_s=dt_s,
    )
    if isinstance(velocity_alpha, torch.Tensor):
        velocity_alpha = velocity_alpha.unsqueeze(-1)
    candidate_filtered_velocity = controller_state.filtered_velocity_world + velocity_alpha * (
        linear_velocity_world - controller_state.filtered_velocity_world
    )
    filtered_velocity = torch.where(
        controller_state.initialized.unsqueeze(-1),
        candidate_filtered_velocity,
        linear_velocity_world,
    )
    raw_acceleration = (filtered_velocity - controller_state.filtered_velocity_world) / dt_s
    derivative_alpha = _low_pass_alpha(
        cutoff_hz=params.velocity_derivative_filter_hz,
        dt_s=dt_s,
    )
    if isinstance(derivative_alpha, torch.Tensor):
        derivative_alpha = derivative_alpha.unsqueeze(-1)
    candidate_acceleration = controller_state.acceleration_world + derivative_alpha * (
        raw_acceleration - controller_state.acceleration_world
    )
    acceleration = torch.where(
        controller_state.initialized.unsqueeze(-1),
        candidate_acceleration,
        torch.zeros_like(candidate_acceleration),
    )
    velocity_error = velocity_setpoint_world - filtered_velocity
    desired_acceleration = (
        params.kp_vel * velocity_error
        + controller_state.velocity_integral_world
        - params.kd_vel * acceleration
    )
    if params.normalized_allocator:
        thrust_world, integration_error = _normalized_thrust_world(
            params=params,
            desired_acceleration=desired_acceleration,
            velocity_error=velocity_error,
        )
        velocity_integral = (
            controller_state.velocity_integral_world + params.ki_vel * integration_error * dt_s
        )
    else:
        thrust_world, achieved_acceleration = _limited_thrust_world(
            params=params,
            desired_acceleration=desired_acceleration,
        )
        saturation_error = achieved_acceleration - desired_acceleration
        may_integrate = (saturation_error.abs() < _SATURATION_EPS) | (
            velocity_error * saturation_error >= 0.0
        )
        velocity_integral = torch.where(
            may_integrate,
            controller_state.velocity_integral_world + params.ki_vel * velocity_error * dt_s,
            controller_state.velocity_integral_world,
        )
    velocity_integral = torch.clamp(
        velocity_integral,
        -params.velocity_integral_limit_mps2,
        params.velocity_integral_limit_mps2,
    )
    body_z = rotation[:, :, 2]
    collective = (thrust_world * body_z).sum(dim=-1).clamp_min(0.0)

    desired_z = _clamp_tilt(thrust_world, params.max_tilt_rad)
    heading_x = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(cos_yaw)], dim=-1)
    desired_y = torch.cross(desired_z, heading_x, dim=-1)
    desired_y = desired_y / desired_y.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    desired_x = torch.cross(desired_y, desired_z, dim=-1)
    desired_rotation = torch.stack([desired_x, desired_y, desired_z], dim=-1)  # columns

    body_rate_setpoint = _attitude_rate_setpoint(
        attitude_wxyz=attitude_wxyz,
        rotation=rotation,
        desired_rotation=desired_rotation,
        proportional_gain=params.kp_att,
        yaw_weight=params.attitude_yaw_weight,
        yaw_rate_radps=action[:, 3] * params.max_yaw_rate_radps,
    )
    outer_successor = _FlightControllerState(
        filtered_velocity_world=filtered_velocity,
        acceleration_world=acceleration,
        velocity_integral_world=velocity_integral,
        previous_angular_velocity_body=controller_state.previous_angular_velocity_body,
        angular_acceleration_body=controller_state.angular_acceleration_body,
        rate_integral_body=controller_state.rate_integral_body,
        initialized=controller_state.initialized,
    )
    return _rate_to_omega(
        params,
        outer_successor,
        collective,
        body_rate_setpoint,
        angular_velocity_body,
        dt_s,
    )


def build_flight_actuation_params(
    *,
    kind: str,
    positions_body: ArrayLike,
    rot_dir: ArrayLike,
    rotor_constant: ArrayLike,
    rolling_moment_coefficient: ArrayLike,
    min_rotor_velocity: ArrayLike,
    max_rotor_velocity: ArrayLike,
    input_offset: ArrayLike,
    input_scaling: ArrayLike,
    zero_position_armed: ArrayLike,
    action_cfg: RlActionConfig,
    controller_settings: ControllerSettings,
    policy_contract: PolicyContract | None,
    mass: float,
    controller_dt_s: float,
    device,
) -> _FlightActuationParams:
    """Build actuation params from YAML controllers plus optional artifact scales.

    Precomputes the device tensors, physical and normalized allocation maps, and
    maximum thrust once (mirroring
    ``robotics_experiments.rl.build_actuation_params``).

    Raises:
        ValueError: for an unknown or not-yet-implemented action level.
    """
    if kind not in ("velocity", "ctbr"):
        raise ValueError(
            f"unsupported rl.action.kind={kind!r}; implemented levels are " "'velocity' and 'ctbr'"
        )

    def _tensor(values):
        return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

    positions = _tensor(positions_body)
    constants = _tensor(rotor_constant)
    rolling = _tensor(rolling_moment_coefficient)
    directions = _tensor(rot_dir)
    max_velocity = _tensor(max_rotor_velocity)

    num_rotors = constants.shape[0]
    x = positions[:, 0]
    y = positions[:, 1]
    yaw = (rolling / constants) * directions
    allocation = torch.stack([torch.ones(num_rotors, device=device), y, -x, yaw], dim=0)
    normalized_effectiveness = torch.stack(
        [
            y,
            x,
            -yaw,
            -torch.ones(num_rotors, device=device),
        ],
        dim=0,
    )
    normalized_mix = torch.linalg.pinv(normalized_effectiveness)
    roll_count = int((normalized_mix[:, 0].abs() > 1e-3).sum().item())
    pitch_count = int((normalized_mix[:, 1].abs() > 1e-3).sum().item())
    roll_scale = (
        torch.sqrt(normalized_mix[:, 0].square().sum() / (roll_count / 2.0))
        if roll_count
        else torch.tensor(1.0, device=device)
    )
    pitch_scale = (
        torch.sqrt(normalized_mix[:, 1].square().sum() / (pitch_count / 2.0))
        if pitch_count
        else torch.tensor(1.0, device=device)
    )
    roll_pitch_scale = torch.maximum(roll_scale, pitch_scale)
    normalized_mix[:, :2] /= roll_pitch_scale
    normalized_mix[:, 2] /= normalized_mix[:, 2].max()
    thrust_values = normalized_mix[:, 3].abs()
    normalized_mix[:, 3] /= thrust_values[thrust_values > torch.finfo(torch.float32).eps].mean()
    normalized_mix = torch.where(
        normalized_mix.abs() < 1e-3,
        torch.zeros_like(normalized_mix),
        normalized_mix,
    )

    return _FlightActuationParams(
        kind=kind,
        mass=float(mass),
        controller_dt_s=float(controller_dt_s),
        device=device,
        positions_body=positions,
        rot_dir=directions,
        rotor_constant=constants,
        rolling_moment_coefficient=rolling,
        min_rotor_velocity=_tensor(min_rotor_velocity),
        max_rotor_velocity=max_velocity,
        input_offset=_tensor(input_offset),
        input_scaling=_tensor(input_scaling),
        zero_position_armed=_tensor(zero_position_armed),
        allocation_pinv=torch.linalg.pinv(allocation),  # (N, 4)
        normalized_allocation_mix=normalized_mix,
        normalized_allocator=kind == "velocity"
        and controller_settings.hover_thrust_normalized > 0.0,
        max_total_thrust=float((constants * max_velocity.square()).sum().item()),
        kp_rate=_tensor(controller_settings.kp_rate),
        ki_rate=_tensor(controller_settings.ki_rate),
        kd_rate=_tensor(controller_settings.kd_rate),
        kff_rate=_tensor(controller_settings.kff_rate),
        rate_output_scale_nm=_tensor(controller_settings.rate_output_scale_nm),
        rate_derivative_filter_hz=float(controller_settings.rate_derivative_filter_hz),
        rate_integral_limit_nm=_tensor(controller_settings.rate_integral_limit_nm),
        max_body_rate_radps=float(
            action_cfg.max_body_rate_radps
            if policy_contract is None
            else policy_contract.max_body_rate_radps
        ),
        hover_thrust_fraction=float(
            controller_settings.hover_thrust_fraction
            if kind == "velocity"
            else (
                action_cfg.hover_thrust_fraction
                if policy_contract is None
                else policy_contract.hover_thrust_fraction
            )
        ),
        hover_thrust_normalized=float(controller_settings.hover_thrust_normalized),
        kp_vel=_tensor(controller_settings.kp_vel),
        ki_vel=_tensor(controller_settings.ki_vel),
        kd_vel=_tensor(controller_settings.kd_vel),
        velocity_filter_hz=float(controller_settings.velocity_filter_hz),
        velocity_derivative_filter_hz=float(controller_settings.velocity_derivative_filter_hz),
        velocity_integral_limit_mps2=_tensor(controller_settings.velocity_integral_limit_mps2),
        kp_att=_tensor(controller_settings.kp_att),
        attitude_yaw_weight=float(controller_settings.attitude_yaw_weight),
        max_body_velocity_mps=float(
            action_cfg.max_body_velocity_mps
            if policy_contract is None
            else policy_contract.max_body_velocity_mps
        ),
        max_yaw_rate_radps=float(
            action_cfg.max_yaw_rate_radps
            if policy_contract is None
            else policy_contract.max_yaw_rate_radps
        ),
        max_tilt_rad=float(np.radians(controller_settings.max_tilt_cmd_deg)),
        min_collective_thrust_fraction=float(controller_settings.min_collective_thrust_fraction),
        max_collective_thrust_fraction=float(controller_settings.max_collective_thrust_fraction),
        horizontal_thrust_margin_fraction=float(
            controller_settings.horizontal_thrust_margin_fraction
        ),
    )


def _replace_controller_settings_batch(
    *,
    params: _FlightActuationParams,
    controller_settings: Sequence[ControllerSettings],
    environments_per_setting: int,
) -> _FlightActuationParams:
    """Replace cascade values with candidate-major per-environment tensors."""
    if environments_per_setting < 1 or not controller_settings:
        raise ValueError("batched controller settings require settings and environments")

    def _vectors(name: str) -> torch.Tensor:
        values = [
            getattr(settings, name)
            for settings in controller_settings
            for _environment in range(environments_per_setting)
        ]
        return torch.as_tensor(values, dtype=torch.float32, device=params.device)

    def _scalars(name: str, *, radians: bool = False) -> torch.Tensor:
        values = [
            float(getattr(settings, name))
            for settings in controller_settings
            for _environment in range(environments_per_setting)
        ]
        result = torch.as_tensor(values, dtype=torch.float32, device=params.device)
        return torch.deg2rad(result) if radians else result

    return replace(
        params,
        kp_rate=_vectors("kp_rate"),
        ki_rate=_vectors("ki_rate"),
        kd_rate=_vectors("kd_rate"),
        kff_rate=_vectors("kff_rate"),
        rate_output_scale_nm=_vectors("rate_output_scale_nm"),
        rate_derivative_filter_hz=_scalars("rate_derivative_filter_hz"),
        rate_integral_limit_nm=_vectors("rate_integral_limit_nm"),
        kp_vel=_vectors("kp_vel"),
        ki_vel=_vectors("ki_vel"),
        kd_vel=_vectors("kd_vel"),
        velocity_filter_hz=_scalars("velocity_filter_hz"),
        velocity_derivative_filter_hz=_scalars("velocity_derivative_filter_hz"),
        velocity_integral_limit_mps2=_vectors("velocity_integral_limit_mps2"),
        hover_thrust_fraction=_scalars("hover_thrust_fraction"),
        hover_thrust_normalized=_scalars("hover_thrust_normalized"),
        kp_att=_vectors("kp_att"),
        attitude_yaw_weight=_scalars("attitude_yaw_weight"),
        max_tilt_rad=_scalars("max_tilt_cmd_deg", radians=True),
        min_collective_thrust_fraction=_scalars("min_collective_thrust_fraction"),
        max_collective_thrust_fraction=_scalars("max_collective_thrust_fraction"),
        horizontal_thrust_margin_fraction=_scalars("horizontal_thrust_margin_fraction"),
    )


def initial_flight_controller_state(
    *,
    num_envs: int,
    device,
) -> _FlightControllerState:
    """Return reset state for every batched environment."""
    zeros = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    return _FlightControllerState(
        filtered_velocity_world=zeros.clone(),
        acceleration_world=zeros.clone(),
        velocity_integral_world=zeros.clone(),
        previous_angular_velocity_body=zeros.clone(),
        angular_acceleration_body=zeros.clone(),
        rate_integral_body=zeros.clone(),
        initialized=torch.zeros(num_envs, dtype=torch.bool, device=device),
    )


def reset_flight_controller_state(
    *,
    state: _FlightControllerState,
    env_ids,
) -> _FlightControllerState:
    """Purely clear selected batched controller rows."""
    values = {}
    for name in (
        "filtered_velocity_world",
        "acceleration_world",
        "velocity_integral_world",
        "previous_angular_velocity_body",
        "angular_acceleration_body",
        "rate_integral_body",
        "initialized",
    ):
        tensor = getattr(state, name).clone()
        tensor[env_ids] = 0
        values[name] = tensor
    return _FlightControllerState(**values)


def seed_flight_controller_state(
    *,
    state: _FlightControllerState,
    params: _FlightActuationParams,
    linear_velocity_world: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    acceleration_setpoint_world: torch.Tensor,
    linear_acceleration_world: torch.Tensor,
    angular_acceleration_body: torch.Tensor,
    rate_integral_body: torch.Tensor,
    valid: torch.Tensor,
) -> _FlightControllerState:
    """Restore per-environment neutral-command PX4 controller boundaries."""
    if params.kind != "velocity":
        raise ValueError("controller-state seeding requires velocity actuation")
    expected = state.filtered_velocity_world.shape
    vectors = {
        "linear_velocity_world": linear_velocity_world,
        "angular_velocity_body": angular_velocity_body,
        "acceleration_setpoint_world": acceleration_setpoint_world,
        "linear_acceleration_world": linear_acceleration_world,
        "angular_acceleration_body": angular_acceleration_body,
        "rate_integral_body": rate_integral_body,
    }
    for name, value in vectors.items():
        if value.shape != expected:
            raise ValueError(f"{name} has shape {tuple(value.shape)}, expected {expected}")
    if valid.shape != state.initialized.shape:
        raise ValueError(
            f"valid has shape {tuple(valid.shape)}, expected {tuple(state.initialized.shape)}"
        )
    selected = valid.unsqueeze(-1)
    velocity_error = -linear_velocity_world
    velocity_integral = torch.clamp(
        acceleration_setpoint_world
        - params.kp_vel * velocity_error
        + params.kd_vel * linear_acceleration_world,
        -params.velocity_integral_limit_mps2,
        params.velocity_integral_limit_mps2,
    )
    clipped_rate_integral = torch.clamp(
        rate_integral_body,
        -params.rate_integral_limit_nm,
        params.rate_integral_limit_nm,
    )
    return _FlightControllerState(
        filtered_velocity_world=torch.where(
            selected,
            linear_velocity_world,
            state.filtered_velocity_world,
        ),
        acceleration_world=torch.where(
            selected,
            linear_acceleration_world,
            state.acceleration_world,
        ),
        velocity_integral_world=torch.where(
            selected,
            velocity_integral,
            state.velocity_integral_world,
        ),
        previous_angular_velocity_body=torch.where(
            selected,
            angular_velocity_body,
            state.previous_angular_velocity_body,
        ),
        angular_acceleration_body=torch.where(
            selected,
            angular_acceleration_body,
            state.angular_acceleration_body,
        ),
        rate_integral_body=torch.where(
            selected,
            clipped_rate_integral,
            state.rate_integral_body,
        ),
        initialized=state.initialized | valid,
    )


def action_to_rotor_omega(
    params: _FlightActuationParams,
    controller_state: _FlightControllerState,
    action: torch.Tensor,
    attitude_wxyz: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    angular_velocity_body: torch.Tensor,
    dt_s: float,
    /,
) -> tuple[_FlightControllerState, torch.Tensor]:
    """Advance the batched controller and return per-rotor omega ``[B, N]``."""
    if params.kind == "ctbr":
        return _ctbr(
            params,
            controller_state,
            action,
            angular_velocity_body,
            dt_s,
        )
    return _velocity(
        params,
        controller_state,
        action,
        attitude_wxyz,
        linear_velocity_world,
        angular_velocity_body,
        dt_s,
    )


def rotor_omega_to_body_wrench(
    params: _FlightActuationParams,
    omega: torch.Tensor,
    body_velocity_flu: torch.Tensor,
    linear_drag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realize omega -> per-rotor thrust -> body wrench (mirrors the thrust model).

    ``omega`` is ``[B, N]``; ``body_velocity_flu`` / ``linear_drag`` are
    ``[B, 3]`` / ``[3]``. Includes the body-Z yaw reaction
    ``sum(rolling_moment_coefficient_i * omega_i^2 * rot_dir_i)`` -- in this
    model yaw authority comes solely from the rotor reaction torque, so it
    must not be dropped.
    """
    omega_squared = omega.square()
    forces = params.rotor_constant * omega_squared
    return rotor_thrust_to_body_wrench(
        params,
        forces,
        body_velocity_flu,
        linear_drag,
    )


def rotor_thrust_to_body_wrench(
    params: _FlightActuationParams,
    forces: torch.Tensor,
    body_velocity_flu: torch.Tensor,
    linear_drag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Realize official-actuator per-rotor thrust as one body-frame wrench."""
    omega_squared = forces / params.rotor_constant
    yaw_reaction = (params.rolling_moment_coefficient * omega_squared * params.rot_dir).sum(dim=-1)
    zeros = torch.zeros_like(yaw_reaction)
    body_torque = torch.stack([zeros, zeros, yaw_reaction], dim=-1)
    body_drag = -linear_drag * body_velocity_flu
    return _rotor_forces_to_body_wrench(params.positions_body, forces, body_torque, body_drag)


def apply_body_wrench(
    articulation,
    body_id: int,
    force_body: torch.Tensor,
    torque_body: torch.Tensor,
) -> None:
    """Write one body-frame wrench per env to the articulation base body.

    ``force_body`` / ``torque_body`` are ``[B, 3]`` in body-FLU. Isaac Lab's
    ``set_external_force_and_torque`` stores the wrench (applied every sim step
    until changed) at the body origin in the body frame; the env's
    ``write_data_to_sim`` flushes it. Implementation note: verify the installed
    Isaac Lab signature -- some versions take ``body_ids`` positionally.
    """
    forces = force_body.unsqueeze(1)
    torques = torque_body.unsqueeze(1)
    articulation.set_external_force_and_torque(forces, torques, body_ids=[body_id])
