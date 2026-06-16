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


def _random_pose(config: DroneConfig, *, include_z: bool = True):
    """Sample a uniform random pose inside the configured spawn volume.

    Args:
        config: Provides the spawn volume center and half-extents.
        include_z: When false, use ``0`` for Z. This is used before dynamic
            stationary placement computes the correct ground-resting height.

    Returns:
        Tuple ``((x, y, z), yaw)`` where the position is in world units and
        ``yaw`` is an angle in radians on ``[-pi, pi]``.
    """
    cx, cy, cz = config.spawn_volume_center
    hx, hy, hz = config.spawn_volume_half_extents
    x = cx + random.uniform(-hx, hx)
    y = cy + random.uniform(-hy, hy)
    z = cz + random.uniform(-hz, hz) if include_z else 0.0
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


def spawn_drone(stage: Usd.Stage, config: DroneConfig | None = None) -> None:
    """Spawn (or re-pose) the drone prim.

    Adds the drone's USD asset as a reference under the stage default prim on
    first call, ensures a guide cube exists to visualize the spawn volume, then
    assigns a sampled translation and yaw. When ``config.spawn_stationary`` is
    true, X/Y/yaw are sampled but Z is computed from the referenced asset bounds
    so PX4 starts from a stationary ground pose.

    Args:
        stage: Target USD stage. Must be non-None and writable.
        config: Drone spawner configuration; when ``None``, resolved via
            :func:`.config.get_config` at call time.
    """
    config = resolve_section("drone", config)
    parent = _resolve_parent(stage)
    _ensure_guide_cube(stage, parent, config)

    drone_path = f"{parent}/{config.prim_name}"
    if not stage.GetPrimAtPath(drone_path).IsValid():
        add_reference_to_stage(config.usd_path, drone_path)

    if config.spawn_stationary:
        sampled_pos, yaw = _random_pose(config, include_z=False)
        _apply_pose(stage, drone_path, sampled_pos, yaw)
        pos = (
            sampled_pos[0],
            sampled_pos[1],
            _computed_stationary_z(stage, drone_path, config),
        )
        mode = "stationary"
    else:
        pos, yaw = _random_pose(config)
        mode = "volume"

    _apply_pose(stage, drone_path, pos, yaw)
    if config.spawn_stationary:
        _clear_rigid_body_velocities(stage, drone_path)
    carb.log_info(f"[DroneSpawner] Spawned mode={mode} at pos={pos} yaw={math.degrees(yaw):.1f}deg")


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
