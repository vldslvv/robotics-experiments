"""Pure tests for the versioned press observation and contact frame."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from robotics_experiments.rl.press import (
    OBSERVATION_SCHEMA_VERSION,
    PRESS_OBSERVATION_DIM,
    PressObservation,
    PressObservationMask,
    pack_press_observation,
    to_vector,
    world_switch_to_contact_frame,
)

_IDENTITY_WXYZ = np.array([1.0, 0.0, 0.0, 0.0])
# Rotation of pi about world Z (switch outward normal facing world -X).
_Z180_WXYZ = np.array([0.0, 0.0, 0.0, 1.0])
_POSITION_ERROR_ENU_M = np.array([0.05, -0.02, 0.1])


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
    assert OBSERVATION_SCHEMA_VERSION == 3
    obs = _press_observation()
    vector = to_vector(
        obs=obs,
        position_error_enu_m=_POSITION_ERROR_ENU_M,
    )
    assert vector.shape == (PRESS_OBSERVATION_DIM,)
    np.testing.assert_allclose(vector[0:3], [-0.05, 0.02, -0.1])
    # detection_valid occupies slot 20; flipping it changes exactly that slot.
    flipped = to_vector(
        obs=_press_observation(detection_valid=False),
        position_error_enu_m=_POSITION_ERROR_ENU_M,
    )
    diff = np.nonzero(flipped - vector)[0]
    assert diff.tolist() == [20]
    # lamp_state occupies slot 25.
    lamp = to_vector(
        obs=_press_observation(lamp_state=1),
        position_error_enu_m=_POSITION_ERROR_ENU_M,
    )
    assert np.nonzero(lamp - vector)[0].tolist() == [25]


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
    expected = to_vector(
        obs=obs,
        position_error_enu_m=_POSITION_ERROR_ENU_M,
    )
    actual = pack_press_observation(
        position_error_enu_m=torch.tensor(_POSITION_ERROR_ENU_M).reshape(1, 3),
        goal_quat_world_wxyz=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64).reshape(1, 4),
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


@pytest.mark.parametrize("yaw", [0.0, 90.0, -90.0, 180.0])
@pytest.mark.parametrize("batched", [False, True])
def test_press_provider_and_lab_pose_packing_agree_across_world_frames(yaw, batched):
    import torch
    from scipy.spatial.transform import Rotation

    goal = Rotation.from_euler("zyx", [yaw, 3.0, -2.0], degrees=True)
    body = goal * Rotation.from_euler("xyz", [0.02, -0.01, 0.03])
    drone_position = np.array([7.0, -4.0, 3.0])
    contact_position = np.array([0.49, -0.02, 0.01])
    switch_position = drone_position + goal.apply(contact_position)
    switch = goal * Rotation.from_euler("z", np.pi)
    relative_position, relative_quat = world_switch_to_contact_frame(
        drone_pose=(drone_position, body.as_quat()[[3, 0, 1, 2]]),
        switch_pose=(switch_position, switch.as_quat()[[3, 0, 1, 2]]),
    )
    position_error = drone_position - (switch_position - goal.apply([0.48, 0.0, 0.0]))
    obs = _press_observation(
        relative_switch_pos_m=relative_position,
        relative_switch_quat_xyzw=relative_quat,
        attitude_quat_xyzw=body.as_quat(),
        body_lin_vel_mps=np.array([0.01, 0.02, -0.03]),
    )
    expected = to_vector(obs=obs, position_error_enu_m=position_error)
    np.testing.assert_allclose(expected[:3], body.inv().apply(-position_error), atol=1e-12)
    kwargs = dataclasses.asdict(obs)
    del kwargs["relative_switch_pos_m"]
    del kwargs["relative_switch_quat_xyzw"]
    kwargs.update(
        position_error_enu_m=position_error, goal_quat_world_wxyz=goal.as_quat()[[3, 0, 1, 2]]
    )
    tensors = {name: torch.as_tensor(value, dtype=torch.float64) for name, value in kwargs.items()}
    if batched:
        tensors = {name: tensor.unsqueeze(0) for name, tensor in tensors.items()}
    actual = pack_press_observation(**tensors).numpy()
    np.testing.assert_allclose(actual[0] if batched else actual, expected, atol=1e-12)
    flipped = dataclasses.replace(
        obs,
        attitude_quat_xyzw=-obs.attitude_quat_xyzw,
        relative_switch_quat_xyzw=-obs.relative_switch_quat_xyzw,
    )
    np.testing.assert_allclose(
        to_vector(obs=flipped, position_error_enu_m=position_error), expected, atol=1e-12
    )
