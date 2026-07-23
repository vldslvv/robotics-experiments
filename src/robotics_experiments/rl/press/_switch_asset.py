"""Canonical pose conversion for the source LightSwitch asset."""

from __future__ import annotations

import math

import numpy as np

from robotics_experiments.px4_bridge import quat_multiply_wxyz

__all__: list[str] = []

SQRT_HALF = math.sqrt(0.5)

# The source asset's local -Y is the outward plate normal. The policy contract
# defines canonical local +X as outward, so canonical pose = source pose * Rz(-pi/2).
ASSET_TO_CANONICAL_WXYZ = np.array([SQRT_HALF, 0.0, 0.0, -SQRT_HALF])


def canonical_switch_quat_wxyz(quat_wxyz: np.ndarray, /) -> np.ndarray:
    """Convert a raw source-asset root orientation to the policy convention."""
    return quat_multiply_wxyz(quat_wxyz, ASSET_TO_CANONICAL_WXYZ)
