"""Drone spawning and pose randomization on a USD stage.

This module is intentionally Isaac-only: it imports ``carb``, ``pxr``, and
``isaacsim`` at import time because every function operates on a live USD stage.
The core import path never loads it directly; ``drone/__init__.py`` exposes the
public API (``spawn_drone`` / ``despawn_drone``) lazily via ``__getattr__``, so
``import robotics_experiments`` works outside the Isaac runtime. See
``robotics_experiments.simulation._isaac_settings`` for the same pattern and
``tests/test_core_imports.py`` for the boundary guard.
"""

import math
import random
from typing import Optional

import carb
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robotics_experiments.config._resolve import resolve_section
from robotics_experiments.config.types import DroneConfig

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


def _ensure_guide_cube(stage: Usd.Stage, parent: str, config: DroneConfig) -> None:
    """Create a guide-purpose cube that visualizes the spawn volume.

    Idempotent: returns immediately if the cube prim already exists. The cube
    is marked with USD ``guide`` purpose so it does not render in ``default``
    or ``render`` passes.

    Args:
        stage: USD stage to mutate.
        parent: Absolute prim path under which to place the cube.
        config: Provides the cube name, center, and half-extents.
    """
    path = f"{parent}/{config.spawn_volume_prim_name}"
    if stage.GetPrimAtPath(path).IsValid():
        return
    cube = UsdGeom.Cube.Define(stage, path)
    cx, cy, cz = config.spawn_volume_center
    hx, hy, hz = config.spawn_volume_half_extents
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
    xform.AddScaleOp().Set(Gf.Vec3f(hx, hy, hz))
    UsdGeom.Imageable(cube).CreatePurposeAttr(UsdGeom.Tokens.guide)


def _random_pose(config: DroneConfig):
    """Sample a uniform random pose inside the configured spawn volume.

    Args:
        config: Provides the spawn volume center and half-extents.

    Returns:
        Tuple ``((x, y, z), yaw)`` where the position is in world units and
        ``yaw`` is an angle in radians on ``[-pi, pi]``.
    """
    cx, cy, cz = config.spawn_volume_center
    hx, hy, hz = config.spawn_volume_half_extents
    x = cx + random.uniform(-hx, hx)
    y = cy + random.uniform(-hy, hy)
    z = cz + random.uniform(-hz, hz)
    yaw = random.uniform(-math.pi, math.pi)
    return (x, y, z), yaw


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


def _computed_stationary_z(stage: Usd.Stage, prim_path: str, config: DroneConfig) -> float:
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
        import numpy as np
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
    stage: Usd.Stage,
    config: DroneConfig | None = None,
    *,
    position: tuple[float, float] | None = None,
    yaw: float = 0.0,
) -> str:
    """Spawn (or re-pose) the drone prim and return its prim path.

    Adds the drone's USD asset as a reference under the stage default prim on
    first call, then assigns a translation and yaw.

    When ``position`` is given (the multi-vehicle path), the drone is placed at
    that explicit world X/Y with the given ``yaw`` (no random sampling, no guide
    cube). When ``position`` is ``None`` (the legacy single-drone path), X/Y/yaw
    are sampled inside the configured spawn volume and the guide cube is ensured.
    In both cases, when ``config.spawn_stationary`` is true, Z is computed from
    the referenced asset bounds so PX4 starts from a stationary ground pose and
    velocities are cleared.

    Args:
        stage: Target USD stage. Must be non-None and writable.
        config: Drone spawner configuration; when ``None``, resolved via
            :func:`.config.get_config` at call time.
        position: Explicit world ``(x, y)`` spawn position, or ``None`` to sample
            randomly inside the spawn volume.
        yaw: Fixed yaw (radians) used when ``position`` is given; ignored when
            sampling randomly.

    Returns:
        The drone prim path (``<parent>/<config.prim_name>``).
    """
    config = resolve_section("drone", config)
    parent = _resolve_parent(stage)

    drone_path = f"{parent}/{config.prim_name}"
    if not stage.GetPrimAtPath(drone_path).IsValid():
        add_reference_to_stage(config.usd_path, drone_path)

    if position is not None:
        pos = (float(position[0]), float(position[1]), 0.0)
        yaw = float(yaw)
    else:
        _ensure_guide_cube(stage, parent, config)
        pos, yaw = _random_pose(config)

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
        mode = "volume"

    carb.log_info(f"[DroneSpawner] Spawned mode={mode} at pos={pos} yaw={math.degrees(yaw):.1f}deg")
    return drone_path


def despawn_drone(stage: Usd.Stage, config: DroneConfig | None = None) -> None:
    """Remove the drone prim from the stage if present.

    Safe to call when the stage is ``None`` or the prim does not exist; the
    guide cube is intentionally left in place.

    Args:
        stage: USD stage to mutate, or ``None`` for a no-op.
        config: Drone spawner configuration; when ``None``, resolved via
            :func:`.config.get_config` at call time.
    """
    if stage is None:
        return
    config = resolve_section("drone", config)
    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    drone_path = f"{parent}/{config.prim_name}"
    if stage.GetPrimAtPath(drone_path).IsValid():
        stage.RemovePrim(drone_path)
        carb.log_info(f"[DroneSpawner] Despawned {drone_path}")
