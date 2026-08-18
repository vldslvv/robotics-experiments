"""Private pose mathematics shared by derivation and evaluation."""

from __future__ import annotations

import math

import numpy as np

from robotics_experiments.flight_parameters._models import Quaternion

__all__: list[str] = []


def _normalized_quaternion(value: Quaternion, /) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if (
        quaternion.shape != (4,)
        or not math.isfinite(norm)
        or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-5)
    ):
        raise ValueError("attitude must be a finite normalized wxyz quaternion")
    return quaternion / norm


def _conjugate(value: np.ndarray, /) -> np.ndarray:
    return value * np.asarray((1.0, -1.0, -1.0, -1.0), dtype=np.float64)


def _multiply(left: np.ndarray, right: np.ndarray, /) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float64,
    )


def _relative_attitude_errors(
    target: Quaternion, actual: Quaternion, /
) -> tuple[np.ndarray, float, float]:
    target_q = _normalized_quaternion(target)
    actual_q = _normalized_quaternion(actual)
    relative = _multiply(_conjugate(target_q), actual_q)
    relative /= np.linalg.norm(relative)
    if relative[0] < 0.0:
        relative = -relative
    w, x, y, z = relative
    rpy = np.asarray(
        (
            math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))),
            math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dtype=np.float64,
    )
    body_z_alignment = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    tilt = math.acos(body_z_alignment)
    orientation = 2.0 * math.acos(max(0.0, min(1.0, abs(float(relative[0])))))
    return rpy, tilt, orientation


def _yaw_quaternion(yaw_rad: float, /) -> Quaternion:
    return (math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad))
