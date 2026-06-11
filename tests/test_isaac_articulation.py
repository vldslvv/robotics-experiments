import math

import numpy as np

from robotics_experiments.px4_bridge.isaac_articulation import IsaacArticulationIO


class _FakeRigidBody:
    def __init__(self, *, position, orientation_wxyz):
        self._position = np.asarray(position, dtype=np.float64)
        self._orientation_wxyz = np.asarray(orientation_wxyz, dtype=np.float64)
        self.calls = []

    def get_world_pose(self):
        return self._position, self._orientation_wxyz

    def apply_forces_and_torques_at_pos(self, **kwargs):
        self.calls.append(kwargs)


def test_rotor_forces_to_world_rotates_positions_and_keeps_vertical_force_under_yaw():
    q_yaw_90_wxyz = np.array(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
        dtype=np.float64,
    )

    forces_world, positions_world = IsaacArticulationIO._rotor_forces_to_world(
        body_position_world=np.array([10.0, 20.0, 30.0], dtype=np.float64),
        body_orientation_wxyz=q_yaw_90_wxyz,
        rotor_positions_body=np.array([[1.0, 2.0, 0.25]], dtype=np.float64),
        forces_n=np.array([4.0], dtype=np.float64),
    )

    np.testing.assert_allclose(positions_world, [[8.0, 21.0, 30.25]], atol=1e-12)
    np.testing.assert_allclose(forces_world, [[0.0, 0.0, 4.0]], atol=1e-12)


def test_equal_symmetric_rotor_forces_are_torque_neutral_after_world_transform():
    q_yaw_90_wxyz = np.array(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
        dtype=np.float64,
    )
    body_position = np.array([10.0, 20.0, 30.0], dtype=np.float64)

    forces_world, positions_world = IsaacArticulationIO._rotor_forces_to_world(
        body_position_world=body_position,
        body_orientation_wxyz=q_yaw_90_wxyz,
        rotor_positions_body=np.array(
            [
                [0.25, -0.25, 0.04],
                [-0.25, 0.25, 0.04],
                [0.25, 0.25, 0.04],
                [-0.25, -0.25, 0.04],
            ],
            dtype=np.float64,
        ),
        forces_n=np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64),
    )

    net_torque = np.sum(np.cross(positions_world - body_position, forces_world), axis=0)
    np.testing.assert_allclose(net_torque[:2], [0.0, 0.0], atol=1e-12)


def test_apply_forces_calls_isaac_with_world_space_forces_and_positions():
    q_yaw_90_wxyz = np.array(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
        dtype=np.float64,
    )
    body = _FakeRigidBody(position=[10.0, 20.0, 30.0], orientation_wxyz=q_yaw_90_wxyz)

    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=["m0"])
    io._body = body
    io._body_path = "/World/Drone/body"
    io._rotor_positions_body = np.array([[1.0, 2.0, 0.25]], dtype=np.float64)

    io.apply_forces(np.array([4.0], dtype=np.float64))

    assert len(body.calls) == 1
    call = body.calls[0]
    assert call["is_global"] is True
    np.testing.assert_allclose(call["forces"], [[0.0, 0.0, 4.0]], atol=1e-12)
    np.testing.assert_allclose(call["positions"], [[8.0, 21.0, 30.25]], atol=1e-12)
