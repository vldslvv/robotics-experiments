"""Ground-truth switch-pose provider reading a live USD stage.

This module is intentionally Isaac-only: it imports ``pxr`` at import time
because it reads prim world poses from a live stage. The core import path never
loads it directly; ``rl/press/__init__.py`` exposes ``GroundTruthPoseProvider``
lazily via ``__getattr__`` (same pattern as ``environment/stage.py``), and
``tests/test_core_imports.py`` guards the boundary.
"""

from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

from robotics_experiments.rl.press._provider import PoseSample

__all__ = ["GroundTruthPoseProvider"]


class GroundTruthPoseProvider:
    """Exact switch pose from the stage: ``confidence=1``, ``age=0``, valid.

    The switch prim's local ``+X`` must be the outward plate normal (the
    contact-frame convention in :mod:`robotics_experiments.rl.press.observation`).
    """

    def __init__(self, stage, switch_prim_path: str) -> None:
        prim = stage.GetPrimAtPath(switch_prim_path)
        if not prim or not prim.IsValid():
            raise ValueError(f"switch prim not found at {switch_prim_path!r}")
        self._xformable = UsdGeom.Xformable(prim)

    def sample(self, sim_time_s: float) -> PoseSample:
        """Return the exact current world pose of the switch prim."""
        transform = self._xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        rotation = transform.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        quat_xyzw = np.array(
            [imaginary[0], imaginary[1], imaginary[2], rotation.GetReal()],
            dtype=np.float64,
        )
        return PoseSample(
            world_switch_pos_m=np.array(
                [translation[0], translation[1], translation[2]], dtype=np.float64
            ),
            world_switch_quat_xyzw=quat_xyzw,
            confidence=1.0,
            age_s=0.0,
            valid=True,
        )
