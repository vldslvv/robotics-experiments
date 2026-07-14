import logging
import math
import sys
import types

import numpy as np
import pytest

from robotics_experiments.px4_bridge import (
    IsaacArticulationIO,
    RootBodyPoseDiagnosticParams,
    RotorDynamics,
)


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
    for array in (
        state.position_enu,
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
        state.angular_velocity_flu,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 1.0


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


def test_read_state_uses_body_linear_velocity_when_root_differs():
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


def test_apply_dynamics_applies_rotor_forces_then_body_torque_and_drag():
    body = _FakeRigidBody(position=[0.0, 0.0, 0.0], orientation_wxyz=[1.0, 0.0, 0.0, 0.0])

    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=["m0"])
    io._body = body
    io._body_path = "/World/Drone/body"
    io._rotor_positions_body = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)

    io.apply_dynamics(
        RotorDynamics(
            forces_n=np.array([4.0], dtype=np.float64),
            body_torque_n_m=np.array([0.0, 0.0, 0.5], dtype=np.float64),
            body_drag_n=np.array([-0.2, 0.0, 0.0], dtype=np.float64),
            rotor_omega_rad_s=np.array([700.0], dtype=np.float64),
        )
    )

    # One per-rotor world-space force, then body torque and body drag at the
    # body origin (is_global=False), in that order.
    assert len(body.calls) == 3
    assert body.calls[0]["is_global"] is True
    np.testing.assert_allclose(body.calls[0]["forces"], [[0.0, 0.0, 4.0]], atol=1e-12)

    torque_call = body.calls[1]
    assert torque_call["is_global"] is False
    np.testing.assert_allclose(torque_call["torques"], [[0.0, 0.0, 0.5]], atol=1e-12)
    np.testing.assert_allclose(torque_call["positions"], [[0.0, 0.0, 0.0]], atol=1e-12)

    drag_call = body.calls[2]
    assert drag_call["is_global"] is False
    np.testing.assert_allclose(drag_call["forces"], [[-0.2, 0.0, 0.0]], atol=1e-12)


class _FakeRotateZOp:
    def __init__(self):
        self.value = 0.0

    def Set(self, value):  # noqa: N802 - mirrors the USD xformOp API
        self.value = float(value)


class _FakeAttr:
    def __init__(self, value):
        self._value = value

    def HasValue(self):  # noqa: N802 - mirrors the USD attr API
        return self._value is not None

    def Get(self):  # noqa: N802 - mirrors the USD attr API
        return self._value


class _FakeRotorPrim:
    def __init__(self, *, spin_dir):
        self.spin_dir = spin_dir

    def GetAttribute(self, name):  # noqa: N802 - mirrors the USD prim API
        assert name == "drone:spinDir"
        return _FakeAttr(self.spin_dir) if self.spin_dir is not None else None


class _FakeStage:
    def __init__(self, prims):
        self._prims = prims

    def GetPrimAtPath(self, path):  # noqa: N802 - mirrors the USD stage API
        return self._prims[path]


def _install_fake_pxr(monkeypatch, ops_by_prim):
    pxr = types.ModuleType("pxr")
    usd_geom = types.ModuleType("pxr.UsdGeom")

    class XformOp:
        TypeRotateZ = "RotateZ"

    class Xformable:
        def __init__(self, prim):
            self._prim = prim

        def GetOrderedXformOps(self):  # noqa: N802 - mirrors the USD xformable API
            return ops_by_prim[self._prim]

    usd_geom.XformOp = XformOp
    usd_geom.Xformable = Xformable
    pxr.UsdGeom = usd_geom
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", usd_geom)


class _FakeUsdRotateZOp(_FakeRotateZOp):
    def GetOpType(self):  # noqa: N802 - mirrors the USD xformOp API
        return "RotateZ"


def test_update_rotor_spin_accumulates_signed_angle_and_wraps():
    # Two rotors with opposite spin directions, driven for two ticks. The visual
    # angle accumulates by sign * degrees(omega) * visual_scale * dt and wraps to
    # [0, 360). CCW (+1) advances positive; CW (-1) wraps to just under 360.
    io = IsaacArticulationIO(
        articulation_path="/World/Drone",
        rotor_link_paths=["m0", "m1"],
        rotor_spin_visual_scale=0.5,
    )
    ops = [_FakeRotateZOp(), _FakeRotateZOp()]
    io._rotor_spin_ops = ops
    io._rotor_spin_signs = np.array([1.0, -1.0], dtype=np.float64)
    io._rotor_angle_deg = np.zeros(2, dtype=np.float64)

    omega = np.array([4.0, 4.0], dtype=np.float64)
    step_deg = math.degrees(4.0) * 0.5 * 0.25  # visual_scale=0.5, dt=0.25

    io.update_rotor_spin(omega, dt=0.25)
    np.testing.assert_allclose(ops[0].value, step_deg)
    np.testing.assert_allclose(ops[1].value, (-step_deg) % 360.0)

    io.update_rotor_spin(omega, dt=0.25)
    np.testing.assert_allclose(ops[0].value, (2.0 * step_deg) % 360.0)
    np.testing.assert_allclose(ops[1].value, (-2.0 * step_deg) % 360.0)
    # CW rotor sits just below a full turn (negative angle wrapped into range).
    assert ops[1].value > 180.0


def test_update_rotor_spin_angle_magnitude_follows_omega():
    io = IsaacArticulationIO(
        articulation_path="/World/Drone",
        rotor_link_paths=["m0", "m1"],
        rotor_spin_visual_scale=1.0,
    )
    ops = [_FakeRotateZOp(), _FakeRotateZOp()]
    io._rotor_spin_ops = ops
    io._rotor_spin_signs = np.array([1.0, 1.0], dtype=np.float64)
    io._rotor_angle_deg = np.zeros(2, dtype=np.float64)

    io.update_rotor_spin(np.array([525.0, 1100.0], dtype=np.float64), dt=0.001)

    np.testing.assert_allclose(ops[0].value, math.degrees(525.0) * 0.001)
    np.testing.assert_allclose(ops[1].value, math.degrees(1100.0) * 0.001)
    assert ops[0].value < ops[1].value


def test_update_rotor_spin_noop_before_initialize():
    # No spin handles acquired yet (initialize not run): the seam is a silent
    # no-op rather than raising, so a pre-init tick cannot crash the sim.
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=["m0"])
    io.update_rotor_spin(np.array([500.0], dtype=np.float64), dt=0.004)


def test_injected_rotor_spin_signs_override_usd_spin_dir_in_px4_order(monkeypatch):
    prim0 = _FakeRotorPrim(spin_dir="BROKEN")
    prim1 = _FakeRotorPrim(spin_dir="BROKEN")
    ops = {
        prim0: [_FakeUsdRotateZOp()],
        prim1: [_FakeUsdRotateZOp()],
    }
    _install_fake_pxr(monkeypatch, ops)
    stage = _FakeStage({"m0": prim0, "m1": prim1})
    io = IsaacArticulationIO(
        articulation_path="/World/Drone",
        rotor_link_paths=["m0", "m1"],
        rotor_spin_signs=[-1.0, 1.0],
    )

    io._resolve_rotor_spin_ops(stage)

    np.testing.assert_allclose(io._rotor_spin_signs, [-1.0, 1.0])
    assert io._rotor_spin_ops == [ops[prim0][0], ops[prim1][0]]


def test_usd_rotor_spin_dir_fallback_rejects_missing_or_invalid_values(monkeypatch):
    prim = _FakeRotorPrim(spin_dir=None)
    _install_fake_pxr(monkeypatch, {prim: [_FakeUsdRotateZOp()]})
    io = IsaacArticulationIO(articulation_path="/World/Drone", rotor_link_paths=["m0"])

    with pytest.raises(RuntimeError, match="invalid drone:spinDir"):
        io._resolve_rotor_spin_ops(_FakeStage({"m0": prim}))

    prim.spin_dir = "SIDEWAYS"
    with pytest.raises(RuntimeError, match="expected 'CW' or 'CCW'"):
        io._resolve_rotor_spin_ops(_FakeStage({"m0": prim}))


def test_initialize_runs_root_body_diagnostics_before_rotor_resolution(monkeypatch, caplog):
    order = []

    class InitPrim:
        def __init__(self, path, *, is_rigid_body):
            self.path = path
            self._is_rigid_body = is_rigid_body

        def IsValid(self):  # noqa: N802 - mirrors the USD prim API
            return True

        def HasAPI(self, _api):  # noqa: N802 - mirrors the USD prim API
            return self._is_rigid_body

    class InitStage:
        def __init__(self):
            self._prims = {
                "/Drone": InitPrim("/Drone", is_rigid_body=False),
                "/Drone/body": InitPrim("/Drone/body", is_rigid_body=True),
            }

        def GetPrimAtPath(self, path):  # noqa: N802 - mirrors the USD stage API
            return self._prims[path]

    class InitQuat:
        def GetReal(self):  # noqa: N802 - mirrors the USD quat API
            return 1.0

        def GetImaginary(self):  # noqa: N802 - mirrors the USD quat API
            return (0.0, 0.0, 0.0)

    class InitMatrix:
        def ExtractTranslation(self):  # noqa: N802 - mirrors the USD matrix API
            return (0.0, 0.0, 0.0)

        def ExtractRotationQuat(self):  # noqa: N802 - mirrors the USD matrix API
            return InitQuat()

        def GetInverse(self):  # noqa: N802 - mirrors the USD matrix API
            return self

        def Transform(self, point):  # noqa: N802 - mirrors the USD matrix API
            return point

    class InitXformable:
        def __init__(self, _prim):
            pass

        def ComputeLocalToWorldTransform(self, _time_code):  # noqa: N802
            return InitMatrix()

        def GetOrderedXformOps(self):  # noqa: N802 - mirrors the USD xformable API
            return []

    stage = InitStage()

    class SingleArticulation:
        def __init__(self, *, prim_path):
            self.prim_path = prim_path

        def initialize(self):
            order.append("articulation")

    class RigidPrim:
        def __init__(self, *, prim_paths_expr):
            self.prim_paths_expr = prim_paths_expr

        def initialize(self):
            order.append("body")

    omni = types.ModuleType("omni")
    omni_usd = types.ModuleType("omni.usd")
    omni_usd.get_context = lambda: types.SimpleNamespace(get_stage=lambda: stage)
    omni.usd = omni_usd

    isaacsim = types.ModuleType("isaacsim")
    isaacsim_core = types.ModuleType("isaacsim.core")
    isaacsim_core_prims = types.ModuleType("isaacsim.core.prims")
    isaacsim_core_prims.RigidPrim = RigidPrim
    isaacsim_core_prims.SingleArticulation = SingleArticulation

    usd_physics = types.ModuleType("pxr.UsdPhysics")
    usd_physics.RigidBodyAPI = object()
    usd = types.ModuleType("pxr.Usd")
    usd.TimeCode = types.SimpleNamespace(Default=lambda: object())
    usd_geom = types.ModuleType("pxr.UsdGeom")
    usd_geom.Xformable = InitXformable
    pxr = types.ModuleType("pxr")
    pxr.Usd = usd
    pxr.UsdGeom = usd_geom
    pxr.UsdPhysics = usd_physics

    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    monkeypatch.setitem(sys.modules, "isaacsim.core", isaacsim_core)
    monkeypatch.setitem(sys.modules, "isaacsim.core.prims", isaacsim_core_prims)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", usd_geom)
    monkeypatch.setitem(sys.modules, "pxr.UsdPhysics", usd_physics)

    def resolve_rotors(self, stage_arg, body_path):
        order.append("rotors")
        assert stage_arg is stage
        assert body_path == "/Drone/body"
        return np.zeros((0, 3), dtype=np.float64)

    monkeypatch.setattr(IsaacArticulationIO, "_resolve_rotor_positions_body", resolve_rotors)

    io = IsaacArticulationIO(
        articulation_path="/Drone",
        rotor_link_paths=[],
        root_body_pose_diagnostics=RootBodyPoseDiagnosticParams(
            policy="warn",
            translation_threshold_m=0.5,
            orientation_threshold_rad=0.25,
        ),
    )

    with caplog.at_level(logging.INFO, logger="robotics_experiments.px4_bridge.isaac_articulation"):
        io.initialize()

    assert order == ["articulation", "body", "rotors"]
    assert "PX4 root/body pose diagnostic" in caplog.text
    assert "articulation=/Drone" in caplog.text
    assert "body=/Drone/body" in caplog.text
