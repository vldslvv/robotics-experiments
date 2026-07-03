import logging
import math
import sys
import types

import numpy as np
import pytest

from robotics_experiments.px4_bridge._asset_diagnostics import (
    RootBodyPoseDiagnosticParams,
    compute_root_body_pose_diagnostics,
    validate_root_body_pose,
)


class _FakeQuat:
    def __init__(self, quat_wxyz):
        self._quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)

    def GetReal(self):
        return float(self._quat_wxyz[0])

    def GetImaginary(self):
        return self._quat_wxyz[1:]


class _FakeMatrix:
    def __init__(self, translation, quat_wxyz):
        self._translation = np.asarray(translation, dtype=np.float64)
        self._quat_wxyz = _normalize_quat(np.asarray(quat_wxyz, dtype=np.float64))

    def ExtractTranslation(self):
        return self._translation

    def ExtractRotationQuat(self):
        return _FakeQuat(self._quat_wxyz)

    def GetInverse(self):
        q_inv = _quat_conjugate(self._quat_wxyz)
        return _FakeMatrix(-_rotate(q_inv, self._translation), q_inv)

    def Transform(self, point):
        return self._translation + _rotate(self._quat_wxyz, np.asarray(point, dtype=np.float64))


class _FakePrim:
    def __init__(self, path, matrix=None):
        self.path = path
        self.matrix = matrix

    def IsValid(self):
        return self.matrix is not None


class _FakeStage:
    def __init__(self, matrices):
        self._matrices = matrices

    def GetPrimAtPath(self, path):
        return _FakePrim(path, self._matrices.get(path))


class _FakeXformable:
    def __init__(self, prim):
        self._prim = prim

    def ComputeLocalToWorldTransform(self, _time_code):
        return self._prim.matrix


@pytest.fixture(autouse=True)
def _fake_pxr(monkeypatch):
    pxr = types.ModuleType("pxr")
    usd = types.ModuleType("pxr.Usd")
    usd_geom = types.ModuleType("pxr.UsdGeom")
    usd.TimeCode = types.SimpleNamespace(Default=lambda: object())
    usd_geom.Xformable = _FakeXformable
    pxr.Usd = usd
    pxr.UsdGeom = usd_geom
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "pxr.Usd", usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdGeom", usd_geom)


def test_coincident_root_body_has_identity_relative_pose():
    stage = _FakeStage(
        {
            "/World/Drone": _FakeMatrix([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0]),
            "/World/Drone/body": _FakeMatrix([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0]),
        }
    )

    diagnostics = compute_root_body_pose_diagnostics(stage, "/World/Drone", "/World/Drone/body")

    assert not diagnostics.has_mismatch
    np.testing.assert_allclose(diagnostics.body_relative_to_root_pose[0], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(diagnostics.body_relative_to_root_pose[1], [1.0, 0.0, 0.0, 0.0])


def test_translated_body_reports_translation_delta():
    stage = _FakeStage(
        {
            "/World/Drone": _FakeMatrix([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            "/World/Drone/body": _FakeMatrix([0.0, 0.0, 0.25], [1.0, 0.0, 0.0, 0.0]),
        }
    )

    diagnostics = compute_root_body_pose_diagnostics(stage, "/World/Drone", "/World/Drone/body")

    assert diagnostics.has_mismatch
    assert diagnostics.translation_delta_m == pytest.approx(0.25)
    np.testing.assert_allclose(diagnostics.body_relative_to_root_pose[0], [0.0, 0.0, 0.25])


def test_rotated_body_reports_angular_delta():
    angle = math.pi / 2.0
    stage = _FakeStage(
        {
            "/World/Drone": _FakeMatrix([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            "/World/Drone/body": _FakeMatrix(
                [0.0, 0.0, 0.0],
                [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)],
            ),
        }
    )

    diagnostics = compute_root_body_pose_diagnostics(stage, "/World/Drone", "/World/Drone/body")

    assert diagnostics.has_mismatch
    assert diagnostics.orientation_delta_rad == pytest.approx(angle)
    np.testing.assert_allclose(
        diagnostics.body_relative_to_root_pose[1],
        [math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)],
        atol=1e-12,
    )


def test_warn_policy_logs_warning_and_does_not_raise(caplog):
    stage = _stage_with_translated_body(0.25)

    with caplog.at_level(logging.INFO, logger="px4-diagnostics-test"):
        diagnostics = validate_root_body_pose(
            stage,
            "/World/Drone",
            "/World/Drone/body",
            RootBodyPoseDiagnosticParams(policy="warn"),
            logger=logging.getLogger("px4-diagnostics-test"),
        )

    assert diagnostics is not None and diagnostics.has_mismatch
    assert "PX4 root/body pose diagnostic" in caplog.text
    assert "PX4 root/body pose mismatch" in caplog.text
    assert "articulation=/World/Drone" in caplog.text
    assert "body=/World/Drone/body" in caplog.text
    assert "translation_delta_m=0.25" in caplog.text
    assert "translation_threshold_m=0.0001" in caplog.text
    assert "orientation_delta_rad=0" in caplog.text
    assert "orientation_threshold_rad=0.0001" in caplog.text
    assert "body_relative_to_root_pose=" in caplog.text


def test_fail_policy_logs_error_and_raises(caplog):
    stage = _stage_with_translated_body(0.25)

    with caplog.at_level(logging.INFO, logger="px4-diagnostics-test"):
        with pytest.raises(RuntimeError, match="PX4 root/body pose mismatch"):
            validate_root_body_pose(
                stage,
                "/World/Drone",
                "/World/Drone/body",
                RootBodyPoseDiagnosticParams(policy="fail"),
                logger=logging.getLogger("px4-diagnostics-test"),
            )

    assert "PX4 root/body pose diagnostic" in caplog.text
    assert "PX4 root/body pose mismatch" in caplog.text
    assert "translation_delta_m=0.25" in caplog.text
    assert "body_relative_to_root_pose=" in caplog.text


def test_unknown_policy_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported root/body mismatch policy"):
        validate_root_body_pose(
            _FakeStage({}),
            "/World/Drone",
            "/World/Drone/body",
            RootBodyPoseDiagnosticParams(policy="explode"),
        )


def test_validate_ignore_policy_skips_diagnostic_work(caplog):
    with caplog.at_level(logging.INFO, logger="px4-diagnostics-test"):
        diagnostics = validate_root_body_pose(
            _FakeStage({}),
            "/World/Drone",
            "/World/Drone/body",
            RootBodyPoseDiagnosticParams(policy="ignore"),
            logger=logging.getLogger("px4-diagnostics-test"),
        )

    assert diagnostics is None
    assert caplog.text == ""


def _stage_with_translated_body(translation_delta_m: float) -> _FakeStage:
    return _FakeStage(
        {
            "/World/Drone": _FakeMatrix([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            "/World/Drone/body": _FakeMatrix(
                [translation_delta_m, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ),
        }
    )


def _normalize_quat(q):
    return q / np.linalg.norm(q)


def _quat_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_multiply(a, b):
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _rotate(q, value):
    vector = np.array([0.0, *value], dtype=np.float64)
    return _quat_multiply(_quat_multiply(q, vector), _quat_conjugate(q))[1:]
