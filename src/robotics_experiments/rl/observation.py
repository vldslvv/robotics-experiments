"""Shared body-relative policy observations for NumPy and Isaac Lab tensors.

Goals and state are supplied by callers. Packing has no scene or task knowledge.
World coordinates are ENU; body coordinates are forward, left, up (FLU).
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import torch

from robotics_experiments.px4_bridge import VehicleState, quat_conjugate_wxyz, rotate_vector_wxyz
from robotics_experiments.rl._interfaces import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
)

__all__ = [
    "HOVER_OBSERVATION_DIM",
    "HOVER_OBSERVATION_SCHEMA_VERSION",
    "hover_observation_from_state",
    "pack_hover_observation",
    "pack_pose_observation",
]

_ArrayT = TypeVar("_ArrayT", np.ndarray, torch.Tensor)


def _rotation_matrix(quat_wxyz: _ArrayT, /) -> _ArrayT:
    """Convert unit-equivalent quaternion signs to the same rotation matrix."""
    if isinstance(quat_wxyz, torch.Tensor):
        q = quat_wxyz / torch.linalg.vector_norm(quat_wxyz, dim=-1, keepdim=True)
    else:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
        q = quat_wxyz / np.linalg.norm(quat_wxyz, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    entries = (
        1 - 2 * (y * y + z * z),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x * x + y * y),
    )
    if isinstance(quat_wxyz, torch.Tensor):
        return torch.stack(entries, dim=-1).reshape(*q.shape[:-1], 3, 3)
    return np.stack(entries, axis=-1).reshape(*q.shape[:-1], 3, 3)


def pack_hover_observation(
    *,
    goal_offset_enu_m: _ArrayT,
    quat_flu_enu_wxyz: _ArrayT,
    linear_velocity_body_flu: _ArrayT,
    angular_velocity_body_flu: _ArrayT,
) -> _ArrayT:
    """Pack hover v2: body goal offset (m), velocity, body rate, unit gravity.

    The goal offset is desired minus actual position. Both NumPy and Torch
    accept one state or a batch of states, retaining Torch dtype and device.
    """
    world_to_body = _rotation_matrix(quat_flu_enu_wxyz).swapaxes(-1, -2)
    goal_body = (world_to_body @ goal_offset_enu_m[..., None])[..., 0]
    gravity_body = -world_to_body[..., :, 2]
    if isinstance(goal_offset_enu_m, torch.Tensor):
        return torch.cat(
            (goal_body, linear_velocity_body_flu, angular_velocity_body_flu, gravity_body), dim=-1
        )
    return np.concatenate(
        (goal_body, linear_velocity_body_flu, angular_velocity_body_flu, gravity_body), axis=-1
    )


def pack_pose_observation(
    *,
    goal_offset_enu_m: _ArrayT,
    quat_flu_enu_wxyz: _ArrayT,
    goal_quat_world_wxyz: _ArrayT,
    linear_velocity_body_flu: _ArrayT,
    angular_velocity_body_flu: _ArrayT,
) -> _ArrayT:
    """Append desired orientation to the shared 12-value flight observation.

    Values 12:18 are columns 0 then 1 of the goal-to-body rotation matrix.
    These six values retain full relative orientation without quaternion signs.
    """
    flight = pack_hover_observation(
        goal_offset_enu_m=goal_offset_enu_m,
        quat_flu_enu_wxyz=quat_flu_enu_wxyz,
        linear_velocity_body_flu=linear_velocity_body_flu,
        angular_velocity_body_flu=angular_velocity_body_flu,
    )
    goal_to_body = _rotation_matrix(quat_flu_enu_wxyz).swapaxes(-1, -2) @ _rotation_matrix(
        goal_quat_world_wxyz
    )
    if isinstance(goal_offset_enu_m, torch.Tensor):
        return torch.cat((flight, goal_to_body[..., :, 0], goal_to_body[..., :, 1]), dim=-1)
    return np.concatenate((flight, goal_to_body[..., :, 0], goal_to_body[..., :, 1]), axis=-1)


def hover_observation_from_state(*, state: VehicleState, target_enu: np.ndarray) -> np.ndarray:
    """Assemble the shared hover observation from a deployment snapshot."""
    return pack_hover_observation(
        goal_offset_enu_m=np.asarray(target_enu, dtype=np.float64) - state.position_enu,
        quat_flu_enu_wxyz=state.orientation_flu_enu_wxyz,
        linear_velocity_body_flu=rotate_vector_wxyz(
            quat_conjugate_wxyz(state.orientation_flu_enu_wxyz), state.linear_velocity_enu
        ),
        angular_velocity_body_flu=state.angular_velocity_flu,
    )
