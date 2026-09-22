"""Versioned, pure observation layout for the press RL task.

Each layout is a documented, versioned field order packed by a pure function
(NumPy/torch + stdlib only; on the core import path). The schema version is recorded
in the exported policy artifact's metadata and validated at load time, so a
policy can never silently run against a layout it was not trained on.

PARITY INVARIANT: the same packing function is fed from Lab tensors in training
and from deployment state plus task-provider measurements. Shared flight
features are calculated by the task-neutral pose packer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import TypeVar

import numpy as np
import torch

from robotics_experiments.px4_bridge import (
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_wxyz_to_xyzw,
    rotate_vector_wxyz,
)
from robotics_experiments.rl import pack_pose_observation
from robotics_experiments.rl._interfaces import (
    PRESS_OBSERVATION_DIM,
    PRESS_OBSERVATION_SCHEMA_VERSION,
)

__all__ = [
    "OBSERVATION_SCHEMA_VERSION",
    "PRESS_OBSERVATION_DIM",
    "PressObservation",
    "PressObservationMask",
    "pack_press_observation",
    "to_vector",
    "world_switch_to_contact_frame",
]

# ---- Contact frame -----------------------------------------------------------
#
# The contact frame is fixed by the switch (plate) pose: x is the wall-normal
# PRESS axis (pointing from the free space in front of the plate INTO the
# surface -- the direction the drone moves to press), y is the plate horizontal
# tangent, z the plate vertical tangent. Convention for the switch pose
# quaternion: the switch's local +X is the OUTWARD plate normal (out of the
# wall, toward an approaching drone), local +Z its vertical tangent. The
# contact frame is therefore the switch frame rotated pi about its local Z
# (x_c = -x_s, y_c = -y_s, z_c = z_s), which keeps it right-handed with the
# vertical tangent unchanged. Every pose provider must agree on this frame.

# Rotation of pi about local Z as a wxyz quaternion.
_CONTACT_FROM_SWITCH_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])


def world_switch_to_contact_frame(
    drone_pose: tuple[np.ndarray, np.ndarray],
    switch_pose: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Express the switch relative to the drone in the contact frame.

    Args:
        drone_pose: ``(position_enu, quat_wxyz)`` of the drone body (FLU->ENU).
        switch_pose: ``(position_enu, quat_wxyz)`` of the switch plate, local
            ``+X`` being the outward plate normal.

    Returns:
        ``(rel_pos, rel_quat_xyzw)``: the switch position relative to the drone
        expressed in the contact frame (so a switch straight ahead at 0.2 m is
        ``(0.2, 0, 0)``), and the drone attitude expressed in the contact frame
        as an xyzw quaternion.
    """
    drone_pos, drone_quat = drone_pose
    switch_pos, switch_quat = switch_pose
    contact_quat = quat_multiply_wxyz(normalize_quat_wxyz(switch_quat), _CONTACT_FROM_SWITCH_WXYZ)
    offset_enu = np.asarray(switch_pos, dtype=np.float64) - np.asarray(drone_pos, dtype=np.float64)
    rel_pos = rotate_vector_wxyz(quat_conjugate_wxyz(contact_quat), offset_enu)
    rel_quat_wxyz = quat_multiply_wxyz(
        quat_conjugate_wxyz(contact_quat), normalize_quat_wxyz(drone_quat)
    )
    # Canonicalize the double cover (q and -q are the same rotation) to a
    # non-negative w for physical pose measurements near identity.
    if rel_quat_wxyz[0] < 0.0:
        rel_quat_wxyz = -rel_quat_wxyz
    return rel_pos, quat_wxyz_to_xyzw(rel_quat_wxyz)


# ---- Press observation (the policy input for the press task family) ----------

# Bump the private interface contract whenever this field set, order, units, or
# frames change.
OBSERVATION_SCHEMA_VERSION = PRESS_OBSERVATION_SCHEMA_VERSION
ArrayT = TypeVar("ArrayT", np.ndarray, torch.Tensor)


class PressObservationMask(IntFlag):
    """Availability bits for optional press-observation field groups."""

    POSE = 1 << 0
    CONTACT = 1 << 1
    LAMP = 1 << 2


@dataclass(frozen=True)
class PressObservation:
    """Task-relative, provider-agnostic policy input for the press task.

    These physical measurements feed shared body-relative actor packing and
    task-specific handoff/completion checks. The pose source can change from
    ground truth to perception without changing the actor layout.
    Perception uncertainty is explicit control state
    (``pose_confidence`` / ``pose_age_s`` / ``detection_valid``).

    Fields and frames (times in seconds, repo convention):
        relative_switch_pos_m: Switch position relative to the drone in the
            contact frame (3,); x is the wall-normal press axis.
        relative_switch_quat_xyzw: Drone attitude in the contact frame (4,).
        pose_confidence: Provider confidence in [0, 1].
        pose_age_s: Age of the pose estimate, seconds.
        detection_valid: Whether the pose estimate is currently valid.
        body_lin_vel_mps: Body-FLU linear velocity (3,).
        body_ang_vel_radps: Body-FLU angular velocity (3,).
        attitude_quat_xyzw: Body-FLU -> world-ENU attitude (4,), xyzw.
        pusher_compression_m: Pusher prismatic-joint compression, meters.
        pusher_compression_rate_mps: Compression rate, m/s.
        contact_valid: Whether the contact signal is available.
        contact_flag: Whether pad contact is currently detected.
        lamp_state: -1 unknown / 0 off / 1 on.
        lamp_confidence: Confidence in ``lamp_state``, [0, 1].
        lamp_age_s: Age of the lamp estimate, seconds.
        previous_action: The previous normalized policy action (action_dim,).
        handoff_mode: 0 nav / 1 precontact / 2 contact.
        estimator_health: Aggregate estimator health in [0, 1].
        obs_mask: Bitmask of populated optional field groups.
    """

    relative_switch_pos_m: np.ndarray
    relative_switch_quat_xyzw: np.ndarray
    pose_confidence: float
    pose_age_s: float
    detection_valid: bool
    body_lin_vel_mps: np.ndarray
    body_ang_vel_radps: np.ndarray
    attitude_quat_xyzw: np.ndarray
    pusher_compression_m: float
    pusher_compression_rate_mps: float
    contact_valid: bool
    contact_flag: bool
    lamp_state: int
    lamp_confidence: float
    lamp_age_s: float
    previous_action: np.ndarray
    handoff_mode: int
    estimator_health: float
    obs_mask: int


def to_vector(
    *,
    obs: PressObservation,
    position_error_enu_m: np.ndarray,
) -> np.ndarray:
    """Pack the shared body-relative pose and press sensor fields (schema 3)."""
    body_quat = np.asarray(obs.attitude_quat_xyzw)[[3, 0, 1, 2]]
    body_in_goal = np.asarray(obs.relative_switch_quat_xyzw)[[3, 0, 1, 2]]
    goal_quat = quat_multiply_wxyz(body_quat, quat_conjugate_wxyz(body_in_goal))
    return pack_press_observation(
        position_error_enu_m=position_error_enu_m,
        goal_quat_world_wxyz=goal_quat,
        pose_confidence=obs.pose_confidence,
        pose_age_s=obs.pose_age_s,
        detection_valid=obs.detection_valid,
        body_lin_vel_mps=obs.body_lin_vel_mps,
        body_ang_vel_radps=obs.body_ang_vel_radps,
        attitude_quat_xyzw=obs.attitude_quat_xyzw,
        pusher_compression_m=obs.pusher_compression_m,
        pusher_compression_rate_mps=obs.pusher_compression_rate_mps,
        contact_valid=obs.contact_valid,
        contact_flag=obs.contact_flag,
        lamp_state=obs.lamp_state,
        lamp_confidence=obs.lamp_confidence,
        lamp_age_s=obs.lamp_age_s,
        previous_action=obs.previous_action,
        handoff_mode=obs.handoff_mode,
        estimator_health=obs.estimator_health,
        obs_mask=obs.obs_mask,
    )


def pack_press_observation(
    *,
    position_error_enu_m: ArrayT,
    goal_quat_world_wxyz: ArrayT,
    pose_confidence: ArrayT | float,
    pose_age_s: ArrayT | float,
    detection_valid: ArrayT | bool,
    body_lin_vel_mps: ArrayT,
    body_ang_vel_radps: ArrayT,
    attitude_quat_xyzw: ArrayT,
    pusher_compression_m: ArrayT | float,
    pusher_compression_rate_mps: ArrayT | float,
    contact_valid: ArrayT | bool,
    contact_flag: ArrayT | bool,
    lamp_state: ArrayT | int,
    lamp_confidence: ArrayT | float,
    lamp_age_s: ArrayT | float,
    previous_action: ArrayT,
    handoff_mode: ArrayT | int,
    estimator_health: ArrayT | float,
    obs_mask: ArrayT | int,
) -> ArrayT:
    """Pack the press schema for either NumPy values or batched torch tensors."""
    pose = pack_pose_observation(
        goal_offset_enu_m=-position_error_enu_m,
        quat_flu_enu_wxyz=attitude_quat_xyzw[..., [3, 0, 1, 2]],
        goal_quat_world_wxyz=goal_quat_world_wxyz,
        linear_velocity_body_flu=body_lin_vel_mps,
        angular_velocity_body_flu=body_ang_vel_radps,
    )
    if isinstance(position_error_enu_m, torch.Tensor):
        reference = position_error_enu_m

        def column(value: torch.Tensor | bool | int | float) -> torch.Tensor:
            return torch.as_tensor(value, dtype=reference.dtype, device=reference.device).reshape(
                *reference.shape[:-1], 1
            )

        return torch.cat(
            (
                pose,
                column(pose_confidence),
                column(pose_age_s),
                column(detection_valid),
                column(pusher_compression_m),
                column(pusher_compression_rate_mps),
                column(contact_valid),
                column(contact_flag),
                column(lamp_state),
                column(lamp_confidence),
                column(lamp_age_s),
                previous_action,
                column(handoff_mode),
                column(estimator_health),
                column(obs_mask),
            ),
            dim=-1,
        )

    batched = pose.ndim == 2

    def array(value, width: int):
        result = np.asarray(value, dtype=np.float64)
        return result.reshape((-1, width) if batched else (width,))

    axis = -1 if batched else 0
    vector = np.concatenate(
        (
            array(pose, 18),
            array(pose_confidence, 1),
            array(pose_age_s, 1),
            array(detection_valid, 1),
            array(pusher_compression_m, 1),
            array(pusher_compression_rate_mps, 1),
            array(contact_valid, 1),
            array(contact_flag, 1),
            array(lamp_state, 1),
            array(lamp_confidence, 1),
            array(lamp_age_s, 1),
            array(previous_action, np.asarray(previous_action).shape[-1]),
            array(handoff_mode, 1),
            array(estimator_health, 1),
            array(obs_mask, 1),
        ),
        axis=axis,
    )
    expected_dim = 31 + np.asarray(previous_action).shape[-1]
    if vector.shape[-1] != expected_dim:
        raise ValueError(
            f"press observation has {vector.shape[-1]} values, expected {expected_dim}"
        )
    return vector
