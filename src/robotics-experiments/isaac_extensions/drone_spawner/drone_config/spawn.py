import math
import random

import carb
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Usd, UsdGeom

from .config import DRONE_CONFIG, DroneConfig


def spawn_drone(stage: Usd.Stage, config: DroneConfig = DRONE_CONFIG) -> None:
    parent = _resolve_parent(stage)
    _ensure_guide_cube(stage, parent, config)

    drone_path = f"{parent}/{config.prim_name}"
    if not stage.GetPrimAtPath(drone_path).IsValid():
        add_reference_to_stage(config.usd_path, drone_path)

    pos, yaw = _random_pose(config)
    _apply_pose(stage, drone_path, pos, yaw)
    carb.log_info(f"[DroneSpawner] Spawned at pos={pos} yaw={math.degrees(yaw):.1f}deg")


def despawn_drone(stage: Usd.Stage, config: DroneConfig = DRONE_CONFIG) -> None:
    if stage is None:
        return
    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    drone_path = f"{parent}/{config.prim_name}"
    if stage.GetPrimAtPath(drone_path).IsValid():
        stage.RemovePrim(drone_path)
        carb.log_info(f"[DroneSpawner] Despawned {drone_path}")


def _resolve_parent(stage: Usd.Stage) -> str:
    default_prim = stage.GetDefaultPrim()
    if default_prim:
        parent = default_prim.GetPath().pathString
        carb.log_info(f"[DroneSpawner] Default prim found: {parent}")
        return parent
    parent = "/World"
    carb.log_warn(f"[DroneSpawner] No default prim on stage, falling back to {parent}")
    return parent


def _ensure_guide_cube(stage: Usd.Stage, parent: str, config: DroneConfig) -> None:
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
    cx, cy, cz = config.spawn_volume_center
    hx, hy, hz = config.spawn_volume_half_extents
    x = cx + random.uniform(-hx, hx)
    y = cy + random.uniform(-hy, hy)
    z = cz + random.uniform(-hz, hz)
    yaw = random.uniform(-math.pi, math.pi)
    return (x, y, z), yaw


def _apply_pose(stage: Usd.Stage, prim_path: str, pos, yaw: float) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
    xform.AddRotateZOp().Set(math.degrees(yaw))
