"""Small fake-data helpers used across subsystem test suites."""

import math

import numpy as np

__all__: list[str] = []


def _quat_pitch_deg(angle_deg: float) -> np.ndarray:
    """Return a [w, x, y, z] quaternion for a pitch rotation of angle_deg."""
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)


def _quat_yaw_deg(angle_deg: float) -> np.ndarray:
    """Return a [w, x, y, z] quaternion for a yaw rotation of angle_deg."""
    half = math.radians(angle_deg) / 2.0
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)
