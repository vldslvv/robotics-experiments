"""Physical frame and NumPy/Torch parity tests for shared observations."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from robotics_experiments.px4_bridge import VehicleState
from robotics_experiments.rl import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    hover_observation_from_state,
    pack_hover_observation,
    pack_pose_observation,
)


def test_hover_layout_length_order_and_world_to_body_conversion():
    state = VehicleState(
        position_enu=np.array([2.0, 1.0, 3.0]),
        orientation_flu_enu_wxyz=np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]),
        linear_velocity_enu=np.array([4.0, 5.0, 6.0]),
        angular_velocity_flu=np.array([7.0, 8.0, 9.0]),
        sim_time_us=0,
    )
    actual = hover_observation_from_state(state=state, target_enu=np.array([3.0, 3.0, 6.0]))
    assert HOVER_OBSERVATION_SCHEMA_VERSION == 2
    assert HOVER_OBSERVATION_DIM == 12
    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, [2, -1, 3, 5, -4, 6, 7, 8, 9, 0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(
        actual,
        pack_hover_observation(
            goal_offset_enu_m=np.array([1.0, 2.0, 3.0]),
            quat_flu_enu_wxyz=state.orientation_flu_enu_wxyz,
            linear_velocity_body_flu=np.array([5.0, -4.0, 6.0]),
            angular_velocity_body_flu=state.angular_velocity_flu,
        ),
        atol=1e-12,
    )


@pytest.mark.parametrize("yaw_degrees", [0, 90, -90, 180, 37])
@pytest.mark.parametrize("batched", [False, True])
def test_pose_packing_parity_signs_and_global_yaw_invariance(yaw_degrees, batched):
    rng = np.random.default_rng(42)
    body = Rotation.random(16, random_state=rng)
    goal = Rotation.random(16, random_state=rng)
    offset = rng.normal(size=(16, 3))
    velocity = rng.normal(size=(16, 3))
    rate = rng.normal(size=(16, 3))
    global_yaw = Rotation.from_euler("z", yaw_degrees, degrees=True)
    original = dict(
        goal_offset_enu_m=offset,
        quat_flu_enu_wxyz=body.as_quat()[:, [3, 0, 1, 2]],
        goal_quat_world_wxyz=goal.as_quat()[:, [3, 0, 1, 2]],
        linear_velocity_body_flu=velocity,
        angular_velocity_body_flu=rate,
    )
    moved = dict(
        original,
        goal_offset_enu_m=global_yaw.apply(offset),
        quat_flu_enu_wxyz=(global_yaw * body).as_quat()[:, [3, 0, 1, 2]],
        goal_quat_world_wxyz=(global_yaw * goal).as_quat()[:, [3, 0, 1, 2]],
    )
    if not batched:
        original = {key: value[0] for key, value in original.items()}
        moved = {key: value[0] for key, value in moved.items()}
    expected = pack_pose_observation(**original)
    np.testing.assert_allclose(pack_pose_observation(**moved), expected, atol=2e-14)
    for key in ("quat_flu_enu_wxyz", "goal_quat_world_wxyz"):
        flipped = dict(original)
        flipped[key] = -flipped[key]
        np.testing.assert_allclose(pack_pose_observation(**flipped), expected, atol=2e-14)
    for dtype in (torch.float32, torch.float64):
        actual = pack_pose_observation(
            **{key: torch.as_tensor(value, dtype=dtype) for key, value in original.items()}
        )
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6)
    matrix = (body.inv() * goal).as_matrix()
    reconstructed = np.stack((expected[..., 12:15], expected[..., 15:18]), axis=-1)
    np.testing.assert_allclose(
        reconstructed, matrix[..., :2] if batched else matrix[0, :, :2], atol=2e-14
    )


def test_body_only_rotation_changes_goal_and_gravity_without_quaternion_discontinuity():
    def observation(*, angle):
        return pack_pose_observation(
            goal_offset_enu_m=np.array([1.0, 0.0, 0.0]),
            quat_flu_enu_wxyz=Rotation.from_euler("x", angle).as_quat()[[3, 0, 1, 2]],
            goal_quat_world_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity_body_flu=np.zeros(3),
            angular_velocity_body_flu=np.zeros(3),
        )

    before = observation(angle=np.pi - 1e-8)
    after = observation(angle=np.pi + 1e-8)
    np.testing.assert_allclose(before, after, atol=3e-8)
    np.testing.assert_allclose(before[9:12], [0, 0, 1], atol=2e-8)
    assert not np.allclose(before, observation(angle=0.0))
