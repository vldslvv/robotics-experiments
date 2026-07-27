"""The nav -> contact handoff gate: a pure predicate over the observation.

Mirrors the command layer's continuous envelope invariants
(`commanding/_routine_predicates.py`: ``tilt_below``, drift/speed gates) in style -- pure,
config-built thresholds, no magic numbers -- but is a separate predicate: the
handoff decides whether the contact behavior may START (different thresholds,
contact frame), while the envelope gates a flight in progress.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Optional

import numpy as np

from robotics_experiments.px4_bridge import rotate_vector_wxyz
from robotics_experiments.rl.press.observation import PressObservation

__all__ = ["HandoffGate", "HandoffReason", "HandoffResult", "handoff_ready"]


class HandoffReason(enum.Enum):
    """Why the handoff gate rejected (first failed check, in check order)."""

    NOT_VALID = "not_valid"
    LOW_CONFIDENCE = "low_confidence"
    STALE = "stale"
    TOO_CLOSE = "too_close"
    TOO_FAR = "too_far"
    LATERAL = "lateral"
    WALL_NORMAL_SPEED = "wall_normal_speed"
    LATERAL_SPEED = "lateral_speed"
    ANGULAR_SPEED = "angular_speed"
    NOT_UPRIGHT = "not_upright"
    CONTACT_UNAVAILABLE = "contact_unavailable"
    CONTACT = "contact"
    PREVIOUS_ACTION = "previous_action"
    MODE = "mode"
    ESTIMATOR = "estimator"


@dataclass(frozen=True)
class HandoffGate:
    """Handoff thresholds, built from the ``rl.press.handoff`` config."""

    pose_confidence_min: float
    pose_age_max_s: float
    switch_range_min_m: float
    switch_range_max_m: float
    lateral_align_max_m: float
    wall_normal_speed_max_mps: float
    lateral_speed_max_mps: float
    angular_speed_max_radps: float
    body_z_alignment_min: float
    estimator_health_min: float

    @classmethod
    def from_config(cls, cfg) -> "HandoffGate":
        """Build from a ``PressHandoffConfig``-shaped object."""
        return cls(
            pose_confidence_min=float(cfg.pose_confidence_min),
            pose_age_max_s=float(cfg.pose_age_max_s),
            switch_range_min_m=float(cfg.switch_range_min_m),
            switch_range_max_m=float(cfg.switch_range_max_m),
            lateral_align_max_m=float(cfg.lateral_align_max_m),
            wall_normal_speed_max_mps=float(cfg.wall_normal_speed_max_mps),
            lateral_speed_max_mps=float(cfg.lateral_speed_max_mps),
            angular_speed_max_radps=float(cfg.angular_speed_max_radps),
            body_z_alignment_min=float(cfg.body_z_alignment_min),
            estimator_health_min=float(cfg.estimator_health_min),
        )


@dataclass(frozen=True)
class HandoffResult:
    """Gate outcome: ready, or the first failed check's reason."""

    ready: bool
    reason: Optional[HandoffReason]


def _contact_frame_velocity(obs: PressObservation) -> np.ndarray:
    # The drone's velocity expressed in the contact frame: rel_quat is the
    # drone attitude in the contact frame (xyzw), so rotating the body-FLU
    # velocity by it yields the contact-frame velocity.
    q = np.asarray(obs.relative_switch_quat_xyzw, dtype=np.float64)
    q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    return rotate_vector_wxyz(q_wxyz, np.asarray(obs.body_lin_vel_mps, dtype=np.float64))


def _body_z_world_z(obs: PressObservation) -> float:
    x, y = np.asarray(obs.attitude_quat_xyzw, dtype=np.float64)[:2]
    return float(1.0 - 2.0 * (x * x + y * y))


def handoff_ready(obs: PressObservation, gate: HandoffGate) -> HandoffResult:
    """Decide whether the nav -> contact handoff may proceed.

    Checks run in a fixed order and the first violation is the reason:
    validity, confidence, age, wall-normal range, lateral alignment,
    wall-normal speed, lateral speed, angular speed, attitude, contact,
    previous action, pre-contact mode, and estimator health.
    """
    if not obs.detection_valid:
        return HandoffResult(False, HandoffReason.NOT_VALID)
    if obs.pose_confidence < gate.pose_confidence_min:
        return HandoffResult(False, HandoffReason.LOW_CONFIDENCE)
    if obs.pose_age_s > gate.pose_age_max_s:
        return HandoffResult(False, HandoffReason.STALE)
    rel_pos = np.asarray(obs.relative_switch_pos_m, dtype=np.float64)
    if rel_pos[0] < gate.switch_range_min_m:
        return HandoffResult(False, HandoffReason.TOO_CLOSE)
    if rel_pos[0] > gate.switch_range_max_m:
        return HandoffResult(False, HandoffReason.TOO_FAR)
    if float(np.linalg.norm(rel_pos[1:3])) > gate.lateral_align_max_m:
        return HandoffResult(False, HandoffReason.LATERAL)
    velocity_contact = _contact_frame_velocity(obs)
    if abs(float(velocity_contact[0])) > gate.wall_normal_speed_max_mps:
        return HandoffResult(False, HandoffReason.WALL_NORMAL_SPEED)
    if float(np.linalg.norm(velocity_contact[1:3])) > gate.lateral_speed_max_mps:
        return HandoffResult(False, HandoffReason.LATERAL_SPEED)
    angular_speed = float(np.linalg.norm(np.asarray(obs.body_ang_vel_radps, dtype=np.float64)))
    if angular_speed > gate.angular_speed_max_radps:
        return HandoffResult(False, HandoffReason.ANGULAR_SPEED)
    if _body_z_world_z(obs) < gate.body_z_alignment_min:
        return HandoffResult(False, HandoffReason.NOT_UPRIGHT)
    if not obs.contact_valid:
        return HandoffResult(False, HandoffReason.CONTACT_UNAVAILABLE)
    if obs.contact_flag:
        return HandoffResult(False, HandoffReason.CONTACT)
    if np.any(np.asarray(obs.previous_action, dtype=np.float64) != 0.0):
        return HandoffResult(False, HandoffReason.PREVIOUS_ACTION)
    if obs.handoff_mode != 1:
        return HandoffResult(False, HandoffReason.MODE)
    if obs.estimator_health < gate.estimator_health_min:
        return HandoffResult(False, HandoffReason.ESTIMATOR)
    return HandoffResult(True, None)
