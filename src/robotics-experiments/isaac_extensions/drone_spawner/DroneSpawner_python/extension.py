import math
import random

import carb
import omni.ext
import omni.timeline
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage
from omni.timeline import TimelineEventType
from omni.usd import StageEventType
from pxr import Gf, UsdGeom

from .global_variables import (
    DRONE_PRIM_NAME,
    DRONE_USD_PATH,
    SPAWN_VOLUME_CENTER,
    SPAWN_VOLUME_HALF_EXTENTS,
    SPAWN_VOLUME_PRIM_NAME,
)


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._usd_context = omni.usd.get_context()
        self._stage_event_sub = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event
        )
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_event_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )
        if self._usd_context.get_stage() is not None:
            self._spawn()

    def on_shutdown(self):
        self._stage_event_sub = None
        self._timeline_event_sub = None

    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            self._spawn()

    def _on_timeline_event(self, event):
        if event.type == int(TimelineEventType.STOP):
            self._despawn()
        elif event.type == int(TimelineEventType.PLAY):
            self._spawn()

    def _spawn(self):
        stage = self._usd_context.get_stage()
        # Spawn under the stage's default prim so the drone is included when
        # this stage is referenced from another stage. Prims that sit as
        # siblings of the default prim are outside its subtree and get
        # dropped by references.
        default_prim = stage.GetDefaultPrim()
        if default_prim:
            parent = default_prim.GetPath().pathString
            carb.log_info(f"[DroneSpawner] Default prim found: {parent}")
        else:
            parent = "/World"
            carb.log_warn(f"[DroneSpawner] No default prim on stage, falling back to {parent}")

        self._ensure_guide_cube(stage, parent)

        drone_path = f"{parent}/{DRONE_PRIM_NAME}"
        if not stage.GetPrimAtPath(drone_path).IsValid():
            add_reference_to_stage(DRONE_USD_PATH, drone_path)

        pos, yaw = self._random_pose()
        self._apply_pose(stage, drone_path, pos, yaw)
        carb.log_info(f"[DroneSpawner] Spawned at pos={pos} yaw={math.degrees(yaw):.1f}deg")

    def _despawn(self):
        stage = self._usd_context.get_stage()
        if stage is None:
            return
        default_prim = stage.GetDefaultPrim()
        parent = default_prim.GetPath().pathString if default_prim else "/World"
        drone_path = f"{parent}/{DRONE_PRIM_NAME}"
        if stage.GetPrimAtPath(drone_path).IsValid():
            stage.RemovePrim(drone_path)
            carb.log_info(f"[DroneSpawner] Despawned {drone_path}")

    def _ensure_guide_cube(self, stage, parent):
        path = f"{parent}/{SPAWN_VOLUME_PRIM_NAME}"
        if stage.GetPrimAtPath(path).IsValid():
            return
        cube = UsdGeom.Cube.Define(stage, path)
        cx, cy, cz = SPAWN_VOLUME_CENTER
        hx, hy, hz = SPAWN_VOLUME_HALF_EXTENTS
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        # UsdGeom.Cube is a unit cube (size=2) centered at origin, so scaling
        # by half-extents produces the requested AABB.
        xform.AddTranslateOp().Set(Gf.Vec3d(cx, cy, cz))
        xform.AddScaleOp().Set(Gf.Vec3f(hx, hy, hz))
        # purpose=guide => viewport-only; cameras, RTX renders, and sensors
        # ignore this prim.
        UsdGeom.Imageable(cube).CreatePurposeAttr(UsdGeom.Tokens.guide)
        # Translucent so it reads as a volume, not a solid box.
        # cube.CreateDisplayColorPrimvar().Set([Gf.Vec3f(0.2, 0.6, 1.0)])
        # cube.CreateDisplayOpacityPrimvar().Set([0.15])

    def _random_pose(self):
        cx, cy, cz = SPAWN_VOLUME_CENTER
        hx, hy, hz = SPAWN_VOLUME_HALF_EXTENTS
        x = cx + random.uniform(-hx, hx)
        y = cy + random.uniform(-hy, hy)
        z = cz + random.uniform(-hz, hz)
        yaw = random.uniform(-math.pi, math.pi)
        return (x, y, z), yaw

    def _apply_pose(self, stage, prim_path, pos, yaw):
        prim = stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        # Clear first so re-randomizing on stage re-open doesn't stack ops.
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddRotateZOp().Set(math.degrees(yaw))
