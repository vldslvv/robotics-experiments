"""Pure tests for the nav -> contact handoff gate."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from robotics_experiments.rl.hover import HoverResetSamples
from robotics_experiments.rl.press import (
    HandoffGate,
    HandoffReason,
    PressObservation,
    handoff_ready,
)

_GATE = HandoffGate(
    pose_confidence_min=0.85,
    pose_age_max_s=0.1,
    switch_range_min_m=0.45,
    switch_range_max_m=0.50,
    lateral_align_max_m=0.08,
    wall_normal_speed_max_mps=0.02,
    lateral_speed_max_mps=0.02,
    angular_speed_max_radps=0.05,
    relative_attitude_error_max_rad=np.deg2rad(5.0),
    body_z_alignment_min=0.965926,
    estimator_health_min=0.80,
)


def _calm_close_observation(**overrides: Any) -> PressObservation:
    base: dict[str, Any] = dict(
        relative_switch_pos_m=np.array([0.48, 0.02, -0.03]),
        relative_switch_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pose_confidence=0.95,
        pose_age_s=0.02,
        detection_valid=True,
        body_lin_vel_mps=np.array([0.005, 0.002, -0.001]),
        body_ang_vel_radps=np.array([0.005, -0.005, 0.002]),
        attitude_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pusher_compression_m=0.0,
        pusher_compression_rate_mps=0.0,
        contact_valid=True,
        contact_flag=False,
        lamp_state=-1,
        lamp_confidence=0.0,
        lamp_age_s=0.0,
        previous_action=np.zeros(4),
        handoff_mode=1,
        estimator_health=0.95,
        obs_mask=0,
    )
    base.update(overrides)
    return PressObservation(**base)


def test_calm_close_confident_fresh_is_ready() -> None:
    result = handoff_ready(_calm_close_observation(), _GATE)
    assert result.ready
    assert result.reason is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"detection_valid": False}, HandoffReason.NOT_VALID),
        ({"pose_confidence": 0.5}, HandoffReason.LOW_CONFIDENCE),
        ({"pose_age_s": 0.5}, HandoffReason.STALE),
        ({"relative_switch_pos_m": np.array([0.6, 0.0, 0.0])}, HandoffReason.TOO_FAR),
        ({"relative_switch_pos_m": np.array([0.44, 0.0, 0.0])}, HandoffReason.TOO_CLOSE),
        ({"relative_switch_pos_m": np.array([0.48, 0.09, 0.0])}, HandoffReason.LATERAL),
        (
            {"body_lin_vel_mps": np.array([0.03, 0.0, 0.0])},
            HandoffReason.WALL_NORMAL_SPEED,
        ),
        (
            {"body_lin_vel_mps": np.array([0.0, 0.03, 0.0])},
            HandoffReason.LATERAL_SPEED,
        ),
        (
            {"body_ang_vel_radps": np.array([0.0, 0.0, 0.06])},
            HandoffReason.ANGULAR_SPEED,
        ),
        (
            {
                "relative_switch_quat_xyzw": np.array(
                    [np.sin(np.deg2rad(5.01) / 2.0), 0.0, 0.0, np.cos(np.deg2rad(5.01) / 2.0)]
                )
            },
            HandoffReason.ATTITUDE,
        ),
        (
            {
                "attitude_quat_xyzw": np.array(
                    [np.sin(np.deg2rad(10.0)), 0.0, 0.0, np.cos(np.deg2rad(10.0))]
                )
            },
            HandoffReason.NOT_UPRIGHT,
        ),
        ({"contact_valid": False}, HandoffReason.CONTACT_UNAVAILABLE),
        ({"contact_flag": True}, HandoffReason.CONTACT),
        ({"previous_action": np.array([0.01, 0.0, 0.0, 0.0])}, HandoffReason.PREVIOUS_ACTION),
        ({"handoff_mode": 0}, HandoffReason.MODE),
        ({"estimator_health": 0.5}, HandoffReason.ESTIMATOR),
    ],
)
def test_each_violation_returns_its_reason(overrides, reason) -> None:
    result = handoff_ready(_calm_close_observation(**overrides), _GATE)
    assert not result.ready
    assert result.reason is reason


def test_check_order_reports_first_violation() -> None:
    # An invalid detection wins over every other simultaneous violation.
    observation = _calm_close_observation(
        detection_valid=False, pose_confidence=0.0, estimator_health=0.0
    )
    assert handoff_ready(observation, _GATE).reason is HandoffReason.NOT_VALID


def test_boundary_values_are_inclusive() -> None:
    # Thresholds are inclusive: exactly-at-limit values pass.
    observation = _calm_close_observation(
        pose_confidence=0.85,
        pose_age_s=0.1,
        relative_switch_pos_m=np.array([0.45, 0.08, 0.0]),
        body_lin_vel_mps=np.array([0.02, 0.02, 0.0]),
        body_ang_vel_radps=np.array([0.05, 0.0, 0.0]),
        estimator_health=0.80,
    )
    result = handoff_ready(observation, _GATE)
    assert result.ready, result.reason

    observation = _calm_close_observation(relative_switch_pos_m=np.array([0.50, 0.08, 0.0]))
    result = handoff_ready(observation, _GATE)
    assert result.ready, result.reason


def test_contact_frame_velocity_uses_relative_attitude() -> None:
    # Drone yawed 90 deg in the contact frame: body-forward velocity becomes
    # lateral in the contact frame, so it trips the LATERAL_SPEED gate, not
    # WALL_NORMAL_SPEED.
    yaw90_xyzw = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    observation = _calm_close_observation(
        relative_switch_quat_xyzw=yaw90_xyzw,
        body_lin_vel_mps=np.array([0.03, 0.0, 0.0]),
    )
    assert handoff_ready(observation, _GATE).reason is HandoffReason.LATERAL_SPEED


def test_gate_from_config_duck_types() -> None:
    class Cfg:
        pose_confidence_min = 0.9
        pose_age_max_s = 0.06
        switch_range_min_m = 0.20
        switch_range_max_m = 0.25
        lateral_align_max_m = 0.03
        wall_normal_speed_max_mps = 0.08
        lateral_speed_max_mps = 0.10
        angular_speed_max_radps = 0.20
        relative_attitude_error_max_rad = 0.10
        body_z_alignment_min = 0.95
        estimator_health_min = 0.90

    gate = HandoffGate.from_config(Cfg())
    assert gate.pose_confidence_min == 0.9
    assert gate.angular_speed_max_radps == 0.20


def test_hover_handoff_uses_reset_extrema_in_world_frame() -> None:
    samples = HoverResetSamples(
        position_error_enu_m=((-0.03, -0.01, -0.02), (0.04, 0.02, 0.03)),
        attitude_error_flu_enu_wxyz=(
            (np.cos(0.015), 0.0, 0.0, -np.sin(0.015)),
            (np.cos(0.02), 0.0, 0.0, np.sin(0.02)),
        ),
        linear_velocity_enu_mps=((-0.01, -0.03, -0.02), (0.02, 0.04, 0.01)),
        angular_velocity_enu_radps=((-0.01, -0.02, -0.03), (0.03, 0.02, 0.01)),
    )
    gate = replace(_GATE, hover_reset_bounds=samples.bounds())
    # +90 degree body yaw: body-forward velocity is world +Y. The old
    # contact-speed limit rejects this 0.03 m/s state; the recorded reset allows it.
    obs = _calm_close_observation(
        relative_switch_pos_m=np.array([0.52, 0.0, 0.0]),
        attitude_quat_xyzw=np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        body_lin_vel_mps=np.array([0.03, 0.0, 0.0]),
        body_ang_vel_radps=np.zeros(3),
    )
    assert handoff_ready(obs, gate, np.zeros(3)).ready
    for position in (np.array([-0.0301, 0.0, 0.0]), np.array([0.0401, 0.0, 0.0])):
        assert handoff_ready(obs, gate, position).reason is HandoffReason.POSITION
    fast = replace(obs, body_lin_vel_mps=np.array([0.0401, 0.0, 0.0]))
    assert handoff_ready(fast, gate, np.zeros(3)).reason is HandoffReason.LINEAR_SPEED
    tilted = replace(
        obs, relative_switch_quat_xyzw=np.array([0.0, 0.0, np.sin(0.025), np.cos(0.025)])
    )
    assert handoff_ready(tilted, gate, np.zeros(3)).reason is HandoffReason.ATTITUDE
    with pytest.raises(ValueError, match="world-ENU"):
        handoff_ready(obs, gate)
