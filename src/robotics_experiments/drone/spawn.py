"""Isaac-only implementation for the package-level drone stage contract.

This module is intentionally Isaac-only: it imports ``carb``, ``pxr``, and
``isaacsim`` at import time because every function operates on a live USD stage.
The core import path never loads it directly; ``drone/__init__.py`` exposes the
public API (``spawn_drone`` / ``despawn_drone``) lazily via ``__getattr__``, so
``import robotics_experiments`` works outside the Isaac runtime. See
``robotics_experiments.simulation`` for the same pattern and
``tests/test_core_imports.py`` for the boundary guard.
"""

import math
from typing import Optional

import carb
import numpy as np
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robotics_experiments.config import DroneSpawnConfig

__all__ = ["spawn_drone", "despawn_drone"]


def _resolve_parent(stage: Usd.Stage) -> str:
    """Return the stage parent path used for spawner children.

    Uses the stage default prim when set, otherwise falls back to ``/World``
    and emits a warning.

    Args:
        stage: USD stage to inspect.

    Returns:
        Absolute prim path string of the resolved parent.
    """
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        parent = default_prim.GetPath().pathString
        carb.log_info(f"[DroneSpawner] Default prim found: {parent}")
        return parent
    parent = "/World"
    carb.log_warn(f"[DroneSpawner] No default prim on stage, falling back to {parent}")
    return parent


def _apply_pose(stage: Usd.Stage, prim_path: str, pos, yaw: float) -> None:
    """Overwrite a prim's xform ops with a translate and Z-rotation.

    Clears any existing xform op order before writing, so repeated calls do
    not stack transforms.

    Args:
        stage: USD stage that owns the prim.
        prim_path: Absolute path to the prim to transform.
        pos: ``(x, y, z)`` world translation.
        yaw: Rotation about the Z axis, in radians.
    """
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xform.AddRotateZOp().Set(math.degrees(yaw))


def _computed_stationary_z(stage: Usd.Stage, prim_path: str, config: DroneSpawnConfig) -> float:
    """Compute root Z that places the drone's lower bound above the ground.

    The caller must first apply the desired X/Y/yaw with Z set to zero. The
    resulting world bound then gives a root-relative lower Z for that yaw.
    """
    try:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise ValueError(f"invalid prim: {prim_path}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
            useExtentsHint=True,
        )
        bound = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if bound.IsEmpty():
            raise ValueError("empty composed bound")
        min_z = float(bound.GetMin()[2])
        return (
            float(config.stationary_ground_z) + float(config.stationary_ground_clearance_m) - min_z
        )
    except Exception as exc:
        fallback = float(config.stationary_spawn_z_fallback)
        carb.log_warn(
            f"[DroneSpawner] Could not compute stationary spawn height for {prim_path}: "
            f"{exc}; using fallback z={fallback:.3f}"
        )
        return fallback


def _find_rigid_body_path(stage: Usd.Stage, prim_path: str) -> Optional[str]:
    root = stage.GetPrimAtPath(prim_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim.GetPath().pathString
    return None


def _clear_rigid_body_velocities(stage: Usd.Stage, prim_path: str) -> None:
    """Best-effort velocity reset for the spawned drone rigid body."""
    body_path = _find_rigid_body_path(stage, prim_path)
    if body_path is None:
        carb.log_warn(
            f"[DroneSpawner] No rigid body found under {prim_path}; velocity reset skipped"
        )
        return

    try:
        from isaacsim.core.prims import RigidPrim

        body = RigidPrim(prim_paths_expr=body_path)
        if hasattr(body, "initialize"):
            body.initialize()
        zeros = np.zeros(3, dtype=np.float64)
        if hasattr(body, "set_linear_velocity"):
            body.set_linear_velocity(zeros)
        if hasattr(body, "set_angular_velocity"):
            body.set_angular_velocity(zeros)
        carb.log_info(f"[DroneSpawner] Cleared rigid body velocity for {body_path}")
    except Exception as exc:
        carb.log_warn(f"[DroneSpawner] Could not clear rigid body velocity for {body_path}: {exc}")


def spawn_drone(
    *,
    stage: Usd.Stage,
    config: DroneSpawnConfig,
) -> str:
    """Spawn or re-pose one configured drone and return its prim path."""
    parent = _resolve_parent(stage)

    drone_path = f"{parent}/{config.prim_name}"
    if not stage.GetPrimAtPath(drone_path).IsValid():
        add_reference_to_stage(config.usd_path, drone_path)

    x, y = config.position_xy
    pos = (float(x), float(y), 0.0)
    yaw = float(config.yaw_rad)

    if config.spawn_stationary:
        # Place X/Y/yaw with Z=0 so the world bound gives a root-relative lower
        # Z, then resolve the ground-resting height and place at the final pose.
        x, y, _ = pos
        _apply_pose(stage, drone_path, (x, y, 0.0), yaw)
        pos = (x, y, _computed_stationary_z(stage, drone_path, config))

        # Reposition with updated Z pos
        _apply_pose(stage, drone_path, pos, yaw)
        _clear_rigid_body_velocities(stage, drone_path)
        mode = "stationary"
    else:
        _apply_pose(stage, drone_path, pos, yaw)
        mode = "configured"

    carb.log_info(f"[DroneSpawner] Spawned mode={mode} at pos={pos} yaw={math.degrees(yaw):.1f}deg")
    return drone_path


def despawn_drone(*, stage: Usd.Stage, config: DroneSpawnConfig) -> None:
    """Remove the configured drone prim if it exists."""
    if stage is None:
        return
    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    drone_path = f"{parent}/{config.prim_name}"
    if stage.GetPrimAtPath(drone_path).IsValid():
        stage.RemovePrim(drone_path)
        carb.log_info(f"[DroneSpawner] Despawned {drone_path}")
