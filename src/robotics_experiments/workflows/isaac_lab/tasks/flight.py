"""Vectorized (torch) in-Lab flight actuation + force sink for DirectRLEnv.

Torch/isaacsim leaf: kept off the core import path (imported only by the env).
State lives in the frozen ``_FlightActuationParams`` (device tensors built once
by :func:`build_flight_actuation_params`) and behavior in pure top-level
functions, the batched twin of two pure, host-tested references it must match
term-for-term:

- the actuation cascade exported by ``robotics_experiments.rl``
  (numpy single-env oracle: ``ActuationParams`` + ``action_to_rotor_omega``) --
  this module batches the velocity/ctbr cascade over ``[num_envs]``; both
  builders precompute the allocation pseudoinverse;
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

The realized thrust is ``T_i = rotor_constant_i * omega_i^2`` clipped to
``[min, max]`` rotor velocity, so an action reproduces the commanded wrench (up
to the clips), exactly as the PX4 bridge's ``PegasusQuadraticThrustModel`` does.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    "rotor_omega_to_body_wrench",
    "rotor_thrust_to_body_wrench",
]

_GRAVITY_M_S2 = 9.81
_EPS = 1e-9


@dataclass(frozen=True)
class _FlightActuationParams:
    """Immutable batched actuation state: geometry + allocation + gains + limits.

    Device-tensor twin of ``robotics_experiments.rl.ActuationParams``; built once per
    env by :func:`build_flight_actuation_params`. The velocity-cascade gains and
    limits are populated for both kinds (``ctbr`` ignores them).
    """

    kind: str
    mass: float
    device: torch.device | str
    positions_body: torch.Tensor
    rot_dir: torch.Tensor
    rotor_constant: torch.Tensor
    rolling_moment_coefficient: torch.Tensor
    min_rotor_velocity: torch.Tensor
    max_rotor_velocity: torch.Tensor
    allocation_pinv: torch.Tensor
    max_total_thrust: float
    kp_rate: torch.Tensor
    max_body_rate_radps: float
    hover_thrust_fraction: float
    kp_vel: torch.Tensor
    kd_vel: torch.Tensor
    kp_att: torch.Tensor
    max_body_velocity_mps: float
    max_yaw_rate_radps: float
    max_tilt_rad: float


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


def _vee(matrix: torch.Tensor) -> torch.Tensor:
    """Batched inverse-hat of a skew-symmetric ``[..., 3, 3]`` matrix."""
    return torch.stack([matrix[..., 2, 1], matrix[..., 0, 2], matrix[..., 1, 0]], dim=-1)


def _clamp_tilt(thrust_world: torch.Tensor, max_tilt_rad: float) -> torch.Tensor:
    """Batched unit desired body-Z from a world thrust vector, tilt-limited from up."""
    norm = thrust_world.norm(dim=-1, keepdim=True)
    up = torch.tensor([0.0, 0.0, 1.0], device=thrust_world.device).expand_as(thrust_world)
    safe = norm.squeeze(-1) > _EPS
    direction = torch.where(norm > _EPS, thrust_world / norm.clamp_min(_EPS), up)
    cos_tilt = direction[:, 2].clamp(-1.0, 1.0)
    tilt = torch.arccos(cos_tilt)
    horizontal = direction - cos_tilt.unsqueeze(-1) * up
    horizontal = horizontal / horizontal.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    clamped = np.cos(max_tilt_rad) * up + np.sin(max_tilt_rad) * horizontal
    over_tilt = (tilt > max_tilt_rad) & safe
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


def _rate_to_omega(
    params: _FlightActuationParams,
    collective: torch.Tensor,
    body_rate_setpoint: torch.Tensor,
    angular_velocity_body: torch.Tensor,
) -> torch.Tensor:
    rate = torch.clamp(body_rate_setpoint, -params.max_body_rate_radps, params.max_body_rate_radps)
    body_torque = params.kp_rate * (rate - angular_velocity_body)
    return _wrench_to_omega(params, collective, body_torque)


def _ctbr(
    params: _FlightActuationParams, action: torch.Tensor, angular_velocity_body: torch.Tensor
) -> torch.Tensor:
    fraction = torch.clamp(params.hover_thrust_fraction * (1.0 + action[:, 0]), 0.0, 1.0)
    collective = fraction * params.max_total_thrust
    body_rate_setpoint = action[:, 1:4] * params.max_body_rate_radps
    return _rate_to_omega(params, collective, body_rate_setpoint, angular_velocity_body)


def _velocity(
    params: _FlightActuationParams,
    action: torch.Tensor,
    attitude_wxyz: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    angular_velocity_body: torch.Tensor,
) -> torch.Tensor:
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
    velocity_error = velocity_setpoint_world - linear_velocity_world
    desired_acceleration = params.kp_vel * velocity_error - params.kd_vel * linear_velocity_world

    gravity = torch.tensor([0.0, 0.0, _GRAVITY_M_S2], device=params.device)
    thrust_world = params.mass * (desired_acceleration + gravity)  # [B, 3]
    body_z = rotation[:, :, 2]
    collective = (thrust_world * body_z).sum(dim=-1).clamp_min(0.0)

    desired_z = _clamp_tilt(thrust_world, params.max_tilt_rad)
    heading_x = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(cos_yaw)], dim=-1)
    desired_y = torch.cross(desired_z, heading_x, dim=-1)
    desired_y = desired_y / desired_y.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    desired_x = torch.cross(desired_y, desired_z, dim=-1)
    desired_rotation = torch.stack([desired_x, desired_y, desired_z], dim=-1)  # columns

    error_matrix = desired_rotation.transpose(-1, -2) @ rotation
    error_matrix = error_matrix - rotation.transpose(-1, -2) @ desired_rotation
    attitude_error = 0.5 * _vee(error_matrix)
    body_rate_setpoint = -params.kp_att * attitude_error
    body_rate_setpoint = body_rate_setpoint.clone()
    body_rate_setpoint[:, 2] = body_rate_setpoint[:, 2] + action[:, 3] * params.max_yaw_rate_radps
    return _rate_to_omega(params, collective, body_rate_setpoint, angular_velocity_body)


def build_flight_actuation_params(
    *,
    kind: str,
    positions_body: ArrayLike,
    rot_dir: ArrayLike,
    rotor_constant: ArrayLike,
    rolling_moment_coefficient: ArrayLike,
    min_rotor_velocity: ArrayLike,
    max_rotor_velocity: ArrayLike,
    action_cfg: RlActionConfig,
    controller_settings: ControllerSettings,
    policy_contract: PolicyContract | None,
    mass: float,
    device,
) -> _FlightActuationParams:
    """Build actuation params from YAML controllers plus optional artifact scales.

    Precomputes the device tensors, the allocation pseudoinverse, and the max
    total thrust once (mirroring ``robotics_experiments.rl.build_actuation_params``).

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

    return _FlightActuationParams(
        kind=kind,
        mass=float(mass),
        device=device,
        positions_body=positions,
        rot_dir=directions,
        rotor_constant=constants,
        rolling_moment_coefficient=rolling,
        min_rotor_velocity=_tensor(min_rotor_velocity),
        max_rotor_velocity=max_velocity,
        allocation_pinv=torch.linalg.pinv(allocation),  # (N, 4)
        max_total_thrust=float((constants * max_velocity.square()).sum().item()),
        kp_rate=_tensor(controller_settings.kp_rate),
        max_body_rate_radps=float(
            action_cfg.max_body_rate_radps
            if policy_contract is None
            else policy_contract.max_body_rate_radps
        ),
        hover_thrust_fraction=float(
            action_cfg.hover_thrust_fraction
            if policy_contract is None
            else policy_contract.hover_thrust_fraction
        ),
        kp_vel=_tensor(controller_settings.kp_vel),
        kd_vel=_tensor(controller_settings.kd_vel),
        kp_att=_tensor(controller_settings.kp_att),
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
    )


def action_to_rotor_omega(
    params: _FlightActuationParams,
    action: torch.Tensor,
    attitude_wxyz: torch.Tensor,
    linear_velocity_world: torch.Tensor,
    angular_velocity_body: torch.Tensor,
) -> torch.Tensor:
    """Map a batched action ``[B, 4]`` to per-rotor omega ``[B, N]``."""
    if params.kind == "ctbr":
        return _ctbr(params, action, angular_velocity_body)
    return _velocity(params, action, attitude_wxyz, linear_velocity_world, angular_velocity_body)


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
