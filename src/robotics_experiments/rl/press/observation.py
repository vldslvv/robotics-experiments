"""Versioned, pure observation layouts for the press RL tasks.

Each layout is a documented, versioned field order packed by a pure function
(numpy + stdlib only; on the core import path). The schema version is recorded
in the exported policy artifact's metadata and validated at load time, so a
policy can never silently run against a layout it was not trained on.

PARITY INVARIANT: the same packing function is fed from Lab tensors in training
and from :class:`~robotics_experiments.px4_bridge.VehicleState` (plus a target
position) at deploy. Any field a deploy-side caller cannot assemble from
``VehicleState`` is forbidden in a layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from robotics_experiments.px4_bridge import (
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_wxyz_to_xyzw,
    rotate_vector_wxyz,
)

if TYPE_CHECKING:
    from robotics_experiments.px4_bridge import VehicleState

__all__ = [
    "HOVER_OBSERVATION_DIM",
    "HOVER_OBSERVATION_SCHEMA_VERSION",
    "OBSERVATION_SCHEMA_VERSION",
    "PressObservation",
    "hover_observation_from_state",
    "pack_hover_observation",
    "to_vector",
    "world_switch_to_contact_frame",
]

# Bump whenever the hover layout below changes shape, order, units, or frame.
HOVER_OBSERVATION_SCHEMA_VERSION = 1

# Hover layout v1: [pos_err_enu(3), quat_flu_enu_wxyz(4), lin_vel_enu(3),
# ang_vel_body_flu(3)] -- 13 floats. The angular velocity is body-FLU (matching
# VehicleState.angular_velocity_flu and the actuation controller input); the
# position error and linear velocity are world-ENU.
HOVER_OBSERVATION_DIM = 13


def pack_hover_observation(
    position_err_enu: np.ndarray,
    quat_flu_enu_wxyz: np.ndarray,
    linear_velocity_enu: np.ndarray,
    angular_velocity_body_flu: np.ndarray,
) -> np.ndarray:
    """Pack the hover layout (schema v1) into a float64 vector of length 13.

    Args:
        position_err_enu: World-ENU position minus the hover target, meters (3,).
        quat_flu_enu_wxyz: Body-FLU -> world-ENU attitude quaternion, wxyz (4,).
        linear_velocity_enu: World-ENU linear velocity, m/s (3,).
        angular_velocity_body_flu: Body-FLU angular velocity, rad/s (3,).
    """
    vector = np.concatenate(
        (
            np.asarray(position_err_enu, dtype=np.float64).reshape(3),
            np.asarray(quat_flu_enu_wxyz, dtype=np.float64).reshape(4),
            np.asarray(linear_velocity_enu, dtype=np.float64).reshape(3),
            np.asarray(angular_velocity_body_flu, dtype=np.float64).reshape(3),
        )
    )
    return vector


def hover_observation_from_state(state: VehicleState, target_enu: np.ndarray) -> np.ndarray:
    """Assemble the hover layout from a live ``VehicleState`` (deploy side).

    The deploy-time counterpart of the env's batched observation: identical
    field order and frames, so a Lab-trained policy consumes it unchanged.
    """
    return pack_hover_observation(
        state.position_enu - np.asarray(target_enu, dtype=np.float64).reshape(3),
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
        state.angular_velocity_flu,
    )


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
    # non-negative w so the policy input is continuous across identity.
    if rel_quat_wxyz[0] < 0.0:
        rel_quat_wxyz = -rel_quat_wxyz
    return rel_pos, quat_wxyz_to_xyzw(rel_quat_wxyz)


# ---- Press observation (the policy input for the press task family) ----------

# Bump whenever the press layout (field set, order, units, or frames) changes.
OBSERVATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PressObservation:
    """Task-relative, provider-agnostic policy input for the press task.

    The policy never sees raw sensors or simulator internals -- only this
    normalized observation, so the pose source can change (ground truth ->
    synthetic noise -> real perception) with "policy API change: none".
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


def to_vector(obs: PressObservation) -> np.ndarray:
    """Flatten a :class:`PressObservation` to the schema-v1 float vector.

    Field order is the dataclass declaration order; bools and ints are cast to
    float. Length is ``30 + len(previous_action)``. Swapping the pose provider
    must never change this length or order (asserted by tests).
    """
    previous_action = np.asarray(obs.previous_action, dtype=np.float64).reshape(-1)
    return np.concatenate(
        (
            np.asarray(obs.relative_switch_pos_m, dtype=np.float64).reshape(3),
            np.asarray(obs.relative_switch_quat_xyzw, dtype=np.float64).reshape(4),
            np.array(
                [obs.pose_confidence, obs.pose_age_s, float(obs.detection_valid)],
                dtype=np.float64,
            ),
            np.asarray(obs.body_lin_vel_mps, dtype=np.float64).reshape(3),
            np.asarray(obs.body_ang_vel_radps, dtype=np.float64).reshape(3),
            np.asarray(obs.attitude_quat_xyzw, dtype=np.float64).reshape(4),
            np.array(
                [
                    obs.pusher_compression_m,
                    obs.pusher_compression_rate_mps,
                    float(obs.contact_valid),
                    float(obs.contact_flag),
                    float(obs.lamp_state),
                    obs.lamp_confidence,
                    obs.lamp_age_s,
                ],
                dtype=np.float64,
            ),
            previous_action,
            np.array(
                [float(obs.handoff_mode), obs.estimator_health, float(obs.obs_mask)],
                dtype=np.float64,
            ),
        )
    )
