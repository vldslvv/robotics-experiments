import math
import random

import carb
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Usd, UsdGeom

from robotics_experiments.config import DroneConfig
from robotics_experiments.config import get_config as _get_experiment_config

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


def spawn_drone(stage: Usd.Stage, config: DroneConfig | None = None) -> None:
    """Spawn (or re-pose) the drone prim at a random pose within the spawn volume.

    Adds the drone's USD asset as a reference under the stage default prim on
    first call, ensures a guide cube exists to visualize the spawn volume, then
    assigns a uniformly sampled translation and yaw.

    Args:
        stage: Target USD stage. Must be non-None and writable.
        config: Drone spawner configuration; when ``None``, resolved via
            :func:`.config.get_config` at call time.
    """
    if config is None:
        config = _get_experiment_config().drone
    parent = _resolve_parent(stage)
    _ensure_guide_cube(stage, parent, config)

    drone_path = f"{parent}/{config.prim_name}"
    if not stage.GetPrimAtPath(drone_path).IsValid():
        add_reference_to_stage(config.usd_path, drone_path)

    pos, yaw = _random_pose(config)
    _apply_pose(stage, drone_path, pos, yaw)
    carb.log_info(f"[DroneSpawner] Spawned at pos={pos} yaw={math.degrees(yaw):.1f}deg")


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
    if config is None:
        config = _get_experiment_config().drone
    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    drone_path = f"{parent}/{config.prim_name}"
    if stage.GetPrimAtPath(drone_path).IsValid():
        stage.RemovePrim(drone_path)
        carb.log_info(f"[DroneSpawner] Despawned {drone_path}")
