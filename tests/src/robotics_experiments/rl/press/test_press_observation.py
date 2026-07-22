"""Pure tests for the versioned press observation layouts and contact frame."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from robotics_experiments.px4_bridge import VehicleState
from robotics_experiments.rl.press import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    PressObservation,
    PressObservationMask,
    hover_observation_from_state,
    pack_hover_observation,
    pack_press_observation,
    to_vector,
    world_switch_to_contact_frame,
)

_IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
# Rotation of pi about world Z (switch outward normal facing world -X).
_Z180_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])


def _press_observation(**overrides: Any) -> PressObservation:
    base: dict[str, Any] = dict(
        relative_switch_pos_m=np.array([0.2, 0.0, 0.0]),
        relative_switch_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pose_confidence=1.0,
        pose_age_s=0.0,
        detection_valid=True,
        body_lin_vel_mps=np.zeros(3),
        body_ang_vel_radps=np.zeros(3),
        attitude_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        pusher_compression_m=0.0,
        pusher_compression_rate_mps=0.0,
        contact_valid=False,
        contact_flag=False,
        lamp_state=-1,
        lamp_confidence=0.0,
        lamp_age_s=0.0,
        previous_action=np.zeros(4),
        handoff_mode=0,
        estimator_health=1.0,
        obs_mask=0,
    )
    base.update(overrides)
    return PressObservation(**base)


def test_hover_layout_length_and_order() -> None:
    vector = pack_hover_observation(
        position_err_enu=np.array([1.0, 2.0, 3.0]),
        quat_flu_enu_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_enu=np.array([4.0, 5.0, 6.0]),
        angular_velocity_body_flu=np.array([7.0, 8.0, 9.0]),
    )
    assert vector.shape == (HOVER_OBSERVATION_DIM,)
    assert vector.dtype == np.float64
    np.testing.assert_allclose(vector[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(vector[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(vector[7:10], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(vector[10:13], [7.0, 8.0, 9.0])


def test_hover_layout_field_changes_move_expected_slots() -> None:
    base = pack_hover_observation(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3)
    )
    changed = pack_hover_observation(
        np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.array([0.0, 0.0, 0.5])
    )
    diff = np.nonzero(changed - base)[0]
    assert diff.tolist() == [12]


def test_hover_layout_schema_version_pinned() -> None:
    # Bump HOVER_OBSERVATION_SCHEMA_VERSION whenever the layout changes; this
    # pin forces that bump to be a conscious, test-visible act.
    assert HOVER_OBSERVATION_SCHEMA_VERSION == 1
    assert HOVER_OBSERVATION_DIM == 13


def test_hover_observation_from_state_matches_pack() -> None:
    state = VehicleState(
        position_enu=np.array([2.0, 1.0, 3.0]),
        orientation_flu_enu_wxyz=np.array([0.9238795, 0.0, 0.0, 0.3826834]),
        linear_velocity_enu=np.array([0.1, -0.2, 0.3]),
        angular_velocity_flu=np.array([0.01, 0.02, -0.03]),
        sim_time_us=1_000_000,
    )
    target = np.array([2.0, 1.0, 2.0])
    from_state = hover_observation_from_state(state, target)
    packed = pack_hover_observation(
        state.position_enu - target,
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
        state.angular_velocity_flu,
    )
    np.testing.assert_allclose(from_state, packed)
    np.testing.assert_allclose(from_state[0:3], [0.0, 0.0, 1.0])


def test_contact_frame_switch_straight_ahead() -> None:
    # Drone at origin facing +X; switch 0.2 m ahead with its outward plate
    # normal (local +X) facing back toward the drone (world -X).
    rel_pos, rel_quat = world_switch_to_contact_frame(
        drone_pose=(np.zeros(3), _IDENTITY_WXYZ),
        switch_pose=(np.array([0.2, 0.0, 0.0]), _Z180_WXYZ),
    )
    np.testing.assert_allclose(rel_pos, [0.2, 0.0, 0.0], atol=1e-12)
    # Contact frame == world frame here (press axis = +X), so the drone's
    # identity attitude is identity in the contact frame too.
    np.testing.assert_allclose(rel_quat, [0.0, 0.0, 0.0, 1.0], atol=1e-12)


def test_contact_frame_lateral_and_vertical_offsets() -> None:
    # Same geometry (contact frame == world frame), drone displaced 0.05 m in
    # +Y and 0.1 m in -Z: the switch appears 0.2 ahead, -0.05 along the plate
    # horizontal tangent, +0.1 along the vertical tangent.
    rel_pos, _ = world_switch_to_contact_frame(
        drone_pose=(np.array([0.0, 0.05, -0.1]), _IDENTITY_WXYZ),
        switch_pose=(np.array([0.2, 0.0, 0.0]), _Z180_WXYZ),
    )
    np.testing.assert_allclose(rel_pos, [0.2, -0.05, 0.1], atol=1e-12)


def test_press_observation_to_vector_length_and_slots() -> None:
    assert OBSERVATION_SCHEMA_VERSION == 1
    obs = _press_observation()
    vector = to_vector(obs)
    assert vector.shape == (PRESS_OBSERVATION_DIM,)
    np.testing.assert_allclose(vector[0:3], [0.2, 0.0, 0.0])
    # detection_valid occupies slot 9; flipping it changes exactly that slot.
    flipped = to_vector(_press_observation(detection_valid=False))
    diff = np.nonzero(flipped - vector)[0]
    assert diff.tolist() == [9]
    # lamp_state occupies slot 24.
    lamp = to_vector(_press_observation(lamp_state=1))
    assert np.nonzero(lamp - vector)[0].tolist() == [24]


def test_press_observation_mask_bits_are_named_and_composable() -> None:
    mask = PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
    assert int(mask) == 7


def test_press_observation_torch_numpy_packing_parity() -> None:
    torch = pytest.importorskip("torch")
    obs = _press_observation(
        detection_valid=True,
        contact_valid=True,
        contact_flag=True,
        lamp_state=1,
        lamp_confidence=1.0,
        handoff_mode=2,
        obs_mask=7,
    )
    expected = to_vector(obs)
    actual = pack_press_observation(
        relative_switch_pos_m=torch.tensor(obs.relative_switch_pos_m).reshape(1, 3),
        relative_switch_quat_xyzw=torch.tensor(obs.relative_switch_quat_xyzw).reshape(1, 4),
        pose_confidence=torch.tensor([obs.pose_confidence]),
        pose_age_s=torch.tensor([obs.pose_age_s]),
        detection_valid=torch.tensor([obs.detection_valid]),
        body_lin_vel_mps=torch.tensor(obs.body_lin_vel_mps).reshape(1, 3),
        body_ang_vel_radps=torch.tensor(obs.body_ang_vel_radps).reshape(1, 3),
        attitude_quat_xyzw=torch.tensor(obs.attitude_quat_xyzw).reshape(1, 4),
        pusher_compression_m=torch.tensor([obs.pusher_compression_m]),
        pusher_compression_rate_mps=torch.tensor([obs.pusher_compression_rate_mps]),
        contact_valid=torch.tensor([obs.contact_valid]),
        contact_flag=torch.tensor([obs.contact_flag]),
        lamp_state=torch.tensor([obs.lamp_state]),
        lamp_confidence=torch.tensor([obs.lamp_confidence]),
        lamp_age_s=torch.tensor([obs.lamp_age_s]),
        previous_action=torch.tensor(obs.previous_action).reshape(1, 4),
        handoff_mode=torch.tensor([obs.handoff_mode]),
        estimator_health=torch.tensor([obs.estimator_health]),
        obs_mask=torch.tensor([obs.obs_mask]),
    )
    np.testing.assert_allclose(actual.cpu().numpy()[0], expected)


def test_press_observation_field_order_is_stable() -> None:
    # The vector packs fields in dataclass declaration order; a reorder must
    # show up here as a failure, forcing a schema-version bump.
    names = [field.name for field in dataclasses.fields(PressObservation)]
    assert names == [
        "relative_switch_pos_m",
        "relative_switch_quat_xyzw",
        "pose_confidence",
        "pose_age_s",
        "detection_valid",
        "body_lin_vel_mps",
        "body_ang_vel_radps",
        "attitude_quat_xyzw",
        "pusher_compression_m",
        "pusher_compression_rate_mps",
        "contact_valid",
        "contact_flag",
        "lamp_state",
        "lamp_confidence",
        "lamp_age_s",
        "previous_action",
        "handoff_mode",
        "estimator_health",
        "obs_mask",
    ]
