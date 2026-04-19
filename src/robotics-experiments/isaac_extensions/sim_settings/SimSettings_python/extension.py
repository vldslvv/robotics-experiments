import carb
import omni.ext
import omni.usd
from omni.usd import StageEventType
from pxr import Gf, PhysxSchema, UsdPhysics
from sim_config.config import SimSettings

from .global_variables import SIM_SETTINGS


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._usd_context = omni.usd.get_context()
        self._stage_event_sub = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event
        )
        stage = self._usd_context.get_stage()
        if stage is not None:
            self._apply(stage, SIM_SETTINGS)

    def on_shutdown(self):
        self._stage_event_sub = None

    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            stage = self._usd_context.get_stage()
            if stage is not None:
                self._apply(stage, SIM_SETTINGS)

    def _apply(self, stage, settings: SimSettings) -> None:
        physics_scene = self._get_or_create_physics_scene(stage)

        gx, gy, gz = settings.gravity
        magnitude = (gx * gx + gy * gy + gz * gz) ** 0.5
        if magnitude > 0.0:
            physics_scene.CreateGravityDirectionAttr().Set(
                Gf.Vec3f(gx / magnitude, gy / magnitude, gz / magnitude)
            )
            physics_scene.CreateGravityMagnitudeAttr().Set(magnitude)

        physx_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene.GetPrim())
        physx_api.CreateTimeStepsPerSecondAttr().Set(int(settings.physics_hz))
        physx_api.CreateSolverTypeAttr().Set(settings.solver_type)

        carb_settings = carb.settings.get_settings()
        carb_settings.set("/app/runLoops/main/rateLimitEnabled", True)
        carb_settings.set("/app/runLoops/main/rateLimitFrequency", int(settings.rendering_hz))

        carb.log_info(
            f"[SimSettings] physics={settings.physics_hz}Hz "
            f"rendering={settings.rendering_hz}Hz solver={settings.solver_type} "
            f"gravity={settings.gravity}"
        )

    def _get_or_create_physics_scene(self, stage) -> UsdPhysics.Scene:
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                return UsdPhysics.Scene(prim)
        default_prim = stage.GetDefaultPrim()
        parent = default_prim.GetPath().pathString if default_prim else "/World"
        return UsdPhysics.Scene.Define(stage, f"{parent}/physicsScene")
