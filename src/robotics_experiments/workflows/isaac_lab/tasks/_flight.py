"""Vectorized (torch) in-Lab flight actuation + force sink for DirectRLEnv.

Torch/isaacsim leaf: kept off the core import path (imported only by the env). It
is the batched analogue of two pure, host-tested references, and must match them
term-for-term:

- the actuation controllers in ``robotics_experiments.press._actuation``
  (numpy single-env oracle) -- ``VectorizedActuation`` mirrors the velocity/ctbr
  cascade over ``[num_envs]``;
- the per-rotor -> body-wrench reduction in
  ``press._actuation.reduce_to_body_wrench`` -- ``rotor_forces_to_body_wrench``
  batches it.

``apply_body_wrench`` is the vectorized analogue of
``px4_bridge.isaac_articulation.IsaacArticulationIO.apply_dynamics``: per-rotor
thrust ``[0, 0, T_i]`` at the body-FLU rotor offsets, the body-Z yaw reaction,
and linear drag, reduced to one equivalent wrench per env applied at the body
origin in the body frame (Isaac Lab's ``set_external_force_and_torque`` applies
in the body frame, so no body->world rotation is needed -- unlike the bridge's
``is_global=True`` path).

The realized thrust is ``T_i = rotor_constant_i * omega_i^2`` clipped to
``[min, max]`` rotor velocity, so an action reproduces the commanded wrench (up
to the clips), exactly as the PX4 bridge's ``PegasusQuadraticThrustModel`` does.
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import ArrayLike

__all__ = [
    "VectorizedActuation",
    "apply_body_wrench",
    "rotor_forces_to_body_wrench",
]

_GRAVITY_M_S2 = 9.81
_EPS = 1e-9


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


def rotor_forces_to_body_wrench(
    positions_body: torch.Tensor,
    forces_n: torch.Tensor,
    body_torque: torch.Tensor,
    body_drag: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-rotor thrust + body effects to one body wrench per env.

    Batched analogue of ``press._actuation.reduce_to_body_wrench``. ``forces_n``
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


class VectorizedActuation:
    """Batched velocity/ctbr actuation matching ``press._actuation`` over N envs.

    Constructed once per env from the resolved airframe geometry + ``press.action``
    config; ``action_to_rotor_omega`` maps a batched normalized action to per-rotor
    omega references the force sink turns into thrust.
    """

    def __init__(
        self,
        *,
        kind: str,
        positions_body: ArrayLike,
        rot_dir: ArrayLike,
        rotor_constant: ArrayLike,
        rolling_moment_coefficient: ArrayLike,
        min_rotor_velocity: ArrayLike,
        max_rotor_velocity: ArrayLike,
        action_cfg,
        mass: float,
        device,
    ) -> None:
        if kind not in ("velocity", "ctbr"):
            raise ValueError(
                f"unsupported press.action.kind={kind!r}; implemented levels are "
                "'velocity' and 'ctbr'"
            )
        self._kind = kind
        self._device = device
        self._mass = float(mass)

        def _tensor(values):
            return torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

        self._positions_body = _tensor(positions_body)
        self._rotor_constant = _tensor(rotor_constant)
        self._rolling_moment_coefficient = _tensor(rolling_moment_coefficient)
        self._rot_dir = _tensor(rot_dir)
        self._min_rotor_velocity = _tensor(min_rotor_velocity)
        self._max_rotor_velocity = _tensor(max_rotor_velocity)

        num_rotors = self._rotor_constant.shape[0]
        x = self._positions_body[:, 0]
        y = self._positions_body[:, 1]
        yaw = (self._rolling_moment_coefficient / self._rotor_constant) * self._rot_dir
        allocation = torch.stack([torch.ones(num_rotors, device=device), y, -x, yaw], dim=0)
        self._allocation_pinv = torch.linalg.pinv(allocation)  # (N, 4)
        self._max_total_thrust = float(
            (self._rotor_constant * self._max_rotor_velocity.square()).sum().item()
        )

        self._kp_rate = _tensor(action_cfg.controller_kp_rate)
        self._max_body_rate = float(action_cfg.max_body_rate_radps)
        self._hover_fraction = float(action_cfg.hover_thrust_fraction)
        # velocity-only gains/limits (unused by ctbr but cheap to hold).
        self._kp_vel = _tensor(action_cfg.controller_kp_vel)
        self._kd_vel = _tensor(action_cfg.controller_kd_vel)
        self._kp_att = _tensor(action_cfg.controller_kp_att)
        self._max_velocity = float(action_cfg.max_body_velocity_mps)
        self._max_yaw_rate = float(action_cfg.max_yaw_rate_radps)
        self._max_tilt = float(np.radians(action_cfg.max_tilt_cmd_deg))

    def _wrench_to_omega(self, collective: torch.Tensor, body_torque: torch.Tensor) -> torch.Tensor:
        wrench = torch.cat([collective.unsqueeze(-1), body_torque], dim=-1)  # [B, 4]
        thrust = wrench @ self._allocation_pinv.T  # [B, N]
        thrust = thrust.clamp_min(0.0)
        omega = torch.sqrt(thrust / self._rotor_constant)
        return torch.clamp(omega, self._min_rotor_velocity, self._max_rotor_velocity)

    def _rate_to_omega(
        self,
        collective: torch.Tensor,
        body_rate_setpoint: torch.Tensor,
        angular_velocity_body: torch.Tensor,
    ) -> torch.Tensor:
        rate = torch.clamp(body_rate_setpoint, -self._max_body_rate, self._max_body_rate)
        body_torque = self._kp_rate * (rate - angular_velocity_body)
        return self._wrench_to_omega(collective, body_torque)

    def action_to_rotor_omega(
        self,
        action: torch.Tensor,
        attitude_wxyz: torch.Tensor,
        linear_velocity_world: torch.Tensor,
        angular_velocity_body: torch.Tensor,
    ) -> torch.Tensor:
        """Map a batched action ``[B, 4]`` to per-rotor omega ``[B, N]``."""
        if self._kind == "ctbr":
            return self._ctbr(action, angular_velocity_body)
        return self._velocity(action, attitude_wxyz, linear_velocity_world, angular_velocity_body)

    def _ctbr(self, action: torch.Tensor, angular_velocity_body: torch.Tensor) -> torch.Tensor:
        fraction = torch.clamp(self._hover_fraction * (1.0 + action[:, 0]), 0.0, 1.0)
        collective = fraction * self._max_total_thrust
        body_rate_setpoint = action[:, 1:4] * self._max_body_rate
        return self._rate_to_omega(collective, body_rate_setpoint, angular_velocity_body)

    def _velocity(
        self,
        action: torch.Tensor,
        attitude_wxyz: torch.Tensor,
        linear_velocity_world: torch.Tensor,
        angular_velocity_body: torch.Tensor,
    ) -> torch.Tensor:
        rotation = _matrix_from_quat_wxyz(attitude_wxyz)  # [B, 3, 3]
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])  # [B]
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

        velocity_command_body = action[:, :3] * self._max_velocity
        velocity_setpoint_world = torch.stack(
            [
                cos_yaw * velocity_command_body[:, 0] - sin_yaw * velocity_command_body[:, 1],
                sin_yaw * velocity_command_body[:, 0] + cos_yaw * velocity_command_body[:, 1],
                velocity_command_body[:, 2],
            ],
            dim=-1,
        )
        velocity_error = velocity_setpoint_world - linear_velocity_world
        desired_acceleration = self._kp_vel * velocity_error - self._kd_vel * linear_velocity_world

        gravity = torch.tensor([0.0, 0.0, _GRAVITY_M_S2], device=self._device)
        thrust_world = self._mass * (desired_acceleration + gravity)  # [B, 3]
        body_z = rotation[:, :, 2]
        collective = (thrust_world * body_z).sum(dim=-1).clamp_min(0.0)

        desired_z = _clamp_tilt(thrust_world, self._max_tilt)
        heading_x = torch.stack([cos_yaw, sin_yaw, torch.zeros_like(cos_yaw)], dim=-1)
        desired_y = torch.cross(desired_z, heading_x, dim=-1)
        desired_y = desired_y / desired_y.norm(dim=-1, keepdim=True).clamp_min(_EPS)
        desired_x = torch.cross(desired_y, desired_z, dim=-1)
        desired_rotation = torch.stack([desired_x, desired_y, desired_z], dim=-1)  # columns

        error_matrix = desired_rotation.transpose(-1, -2) @ rotation
        error_matrix = error_matrix - rotation.transpose(-1, -2) @ desired_rotation
        attitude_error = 0.5 * _vee(error_matrix)
        body_rate_setpoint = -self._kp_att * attitude_error
        body_rate_setpoint = body_rate_setpoint.clone()
        body_rate_setpoint[:, 2] = body_rate_setpoint[:, 2] + action[:, 3] * self._max_yaw_rate
        return self._rate_to_omega(collective, body_rate_setpoint, angular_velocity_body)

    def rotor_omega_to_body_wrench(
        self, omega: torch.Tensor, body_velocity_flu: torch.Tensor, linear_drag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Realize omega -> per-rotor thrust -> body wrench (mirrors the thrust model).

        ``omega`` is ``[B, N]``; ``body_velocity_flu`` / ``linear_drag`` are
        ``[B, 3]`` / ``[3]``. Includes the body-Z yaw reaction
        ``sum(rolling_moment_coefficient_i * omega_i^2 * rot_dir_i)`` -- in this
        model yaw authority comes solely from the rotor reaction torque, so it
        must not be dropped.
        """
        omega_squared = omega.square()
        forces = self._rotor_constant * omega_squared
        yaw_reaction = (self._rolling_moment_coefficient * omega_squared * self._rot_dir).sum(
            dim=-1
        )
        zeros = torch.zeros_like(yaw_reaction)
        body_torque = torch.stack([zeros, zeros, yaw_reaction], dim=-1)
        body_drag = -linear_drag * body_velocity_flu
        return rotor_forces_to_body_wrench(self._positions_body, forces, body_torque, body_drag)


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
