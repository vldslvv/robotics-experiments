import math

import numpy as np

from robotics_experiments.px4_bridge.isaac_articulation import IsaacArticulationIO


class _FakeRigidBody:
    def __init__(
        self,
        *,
        position,
        orientation_wxyz,
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
    ):
        self._position = np.asarray(position, dtype=np.float64)
        self._orientation_wxyz = np.asarray(orientation_wxyz, dtype=np.float64)
        self._linear_velocity = np.asarray(linear_velocity, dtype=np.float64)
        self._angular_velocity = np.asarray(angular_velocity, dtype=np.float64)
        self.calls = []

    def get_world_pose(self):
        return self._position, self._orientation_wxyz

    def get_linear_velocity(self):
        return self._linear_velocity

    def get_angular_velocity(self):
        return self._angular_velocity

    def apply_forces_and_torques_at_pos(self, **kwargs):
        self.calls.append(kwargs)


class _FakeBodyMissingVelocity:
    def get_world_pose(self):
        return np.zeros(3, dtype=np.float64), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class _FakeBatchedRigidBody:
    def __init__(self, *, position, orientation_wxyz, linear_velocity, angular_velocity):
        self._position = np.asarray([position], dtype=np.float64)
        self._orientation_wxyz = np.asarray([orientation_wxyz], dtype=np.float64)
        self._linear_velocity = np.asarray([linear_velocity], dtype=np.float64)
        self._angular_velocity = np.asarray([angular_velocity], dtype=np.float64)

    def get_world_poses(self):
        return self._position, self._orientation_wxyz

    def get_linear_velocities(self):
        return self._linear_velocity

    def get_angular_velocities(self):
        return self._angular_velocity


class _FakeArticulation:
    def __init__(self, *, position, orientation_wxyz, linear_velocity, angular_velocity):
        self._position = np.asarray(position, dtype=np.float64)
        self._orientation_wxyz = np.asarray(orientation_wxyz, dtype=np.float64)
        self._linear_velocity = np.asarray(linear_velocity, dtype=np.float64)
        self._angular_velocity = np.asarray(angular_velocity, dtype=np.float64)

    def get_world_pose(self):
        return self._position, self._orientation_wxyz

    def get_linear_velocity(self):
        return self._linear_velocity

    def get_angular_velocity(self):
        return self._angular_velocity


def test_read_state_rotates_world_angular_velocity_into_body_frame():
    q_yaw_90_wxyz = np.array(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
        dtype=np.float64,
    )
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=[])
    io._articulation = _FakeArticulation(
        position=[0.0, 0.0, 1.0],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
    )
    io._body = _FakeRigidBody(
        position=[0.0, 0.0, 1.0],
        orientation_wxyz=q_yaw_90_wxyz,
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[1.0, 0.0, 0.0],
    )

    state = io.read_state(sim_time_us=1_000)

    np.testing.assert_allclose(state.angular_velocity_flu, [0.0, -1.0, 0.0], atol=1e-12)


def test_read_state_samples_body_pose_when_root_pose_differs():
    q_root_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    q_body_wxyz = np.array(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
        dtype=np.float64,
    )
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=[])
    io._articulation = _FakeArticulation(
        position=[100.0, 200.0, 300.0],
        orientation_wxyz=q_root_wxyz,
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
    )
    io._body = _FakeRigidBody(
        position=[1.0, 2.0, 3.0],
        orientation_wxyz=q_body_wxyz,
    )

    state = io.read_state(sim_time_us=1_000)

    np.testing.assert_allclose(state.position_enu, [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(state.orientation_flu_enu_wxyz, q_body_wxyz, atol=1e-12)


def test_read_state_uses_body_linear_velocity_for_velocity_and_acceleration():
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=[])
    io._articulation = _FakeArticulation(
        position=[0.0, 0.0, 0.0],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        linear_velocity=[100.0, 200.0, 300.0],
        angular_velocity=[0.0, 0.0, 0.0],
    )
    body = _FakeRigidBody(
        position=[0.0, 0.0, 0.0],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        linear_velocity=[1.0, 2.0, 3.0],
    )
    io._body = body

    first = io.read_state(sim_time_us=1_000)
    body._linear_velocity = np.array([1.5, 3.0, 4.5], dtype=np.float64)
    second = io.read_state(sim_time_us=501_000)

    np.testing.assert_allclose(first.linear_velocity_enu, [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(second.linear_velocity_enu, [1.5, 3.0, 4.5], atol=1e-12)
    np.testing.assert_allclose(second.linear_acceleration_enu, [1.0, 2.0, 3.0], atol=1e-12)


def test_read_state_uses_body_angular_velocity_when_root_differs():
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=[])
    io._articulation = _FakeArticulation(
        position=[0.0, 0.0, 0.0],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[100.0, 200.0, 300.0],
    )
    io._body = _FakeRigidBody(
        position=[0.0, 0.0, 0.0],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        angular_velocity=[1.0, 2.0, 3.0],
    )

    state = io.read_state(sim_time_us=1_000)

    np.testing.assert_allclose(state.angular_velocity_flu, [1.0, 2.0, 3.0], atol=1e-12)


def test_read_state_supports_batched_body_rigid_prim_api():
    q_yaw_90_wxyz = np.array(
        [math.cos(math.pi / 4.0), 0.0, 0.0, math.sin(math.pi / 4.0)],
        dtype=np.float64,
    )
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=[])
    io._articulation = _FakeArticulation(
        position=[100.0, 200.0, 300.0],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        linear_velocity=[100.0, 200.0, 300.0],
        angular_velocity=[100.0, 200.0, 300.0],
    )
    io._body = _FakeBatchedRigidBody(
        position=[1.0, 2.0, 3.0],
        orientation_wxyz=q_yaw_90_wxyz,
        linear_velocity=[4.0, 5.0, 6.0],
        angular_velocity=[1.0, 0.0, 0.0],
    )

    state = io.read_state(sim_time_us=1_000)

    np.testing.assert_allclose(state.position_enu, [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(state.orientation_flu_enu_wxyz, q_yaw_90_wxyz, atol=1e-12)
    np.testing.assert_allclose(state.linear_velocity_enu, [4.0, 5.0, 6.0], atol=1e-12)
    np.testing.assert_allclose(state.angular_velocity_flu, [0.0, -1.0, 0.0], atol=1e-12)


def test_read_state_requires_body_velocity_api():
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=[])
    io._articulation = _FakeArticulation(
        position=[0.0, 0.0, 0.0],
        orientation_wxyz=[1.0, 0.0, 0.0, 0.0],
        linear_velocity=[100.0, 200.0, 300.0],
        angular_velocity=[100.0, 200.0, 300.0],
    )
    io._body = _FakeBodyMissingVelocity()
    io._body_path = "/World/Drone/body"

    try:
        io.read_state(sim_time_us=1_000)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("read_state() should require body velocity APIs")

    assert "/World/Drone/body" in message
    assert "get_linear_velocity()/get_linear_velocities()" in message


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
