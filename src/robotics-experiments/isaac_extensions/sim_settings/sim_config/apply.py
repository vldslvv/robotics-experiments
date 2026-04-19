import carb
from pxr import Gf, PhysxSchema, Usd, UsdPhysics

from .config import SIM_SETTINGS, SimSettings


def apply_settings(stage: Usd.Stage, settings: SimSettings = SIM_SETTINGS) -> None:
    scene = _get_or_create_physics_scene(stage)
    _apply_gravity(scene, settings.gravity)
    _apply_physx(scene, settings.physics_hz, settings.solver_type)
    _apply_render_rate(settings.rendering_hz)

    carb.log_info(
        f"[SimSettings] physics={settings.physics_hz}Hz "
        f"rendering={settings.rendering_hz}Hz solver={settings.solver_type} "
        f"gravity={settings.gravity}"
    )


def _get_or_create_physics_scene(stage: Usd.Stage) -> UsdPhysics.Scene:
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return UsdPhysics.Scene(prim)
    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    return UsdPhysics.Scene.Define(stage, f"{parent}/physicsScene")


def _apply_gravity(scene: UsdPhysics.Scene, gravity: tuple) -> None:
    gx, gy, gz = gravity
    magnitude = (gx * gx + gy * gy + gz * gz) ** 0.5
    if magnitude <= 0.0:
        return
    scene.CreateGravityDirectionAttr().Set(
        Gf.Vec3f(gx / magnitude, gy / magnitude, gz / magnitude)
    )
    scene.CreateGravityMagnitudeAttr().Set(magnitude)


def _apply_physx(scene: UsdPhysics.Scene, physics_hz: float, solver_type: str) -> None:
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_api.CreateTimeStepsPerSecondAttr().Set(int(physics_hz))
    physx_api.CreateSolverTypeAttr().Set(solver_type)


def _apply_render_rate(rendering_hz: float) -> None:
    carb_settings = carb.settings.get_settings()
    carb_settings.set("/app/runLoops/main/rateLimitEnabled", True)
    carb_settings.set("/app/runLoops/main/rateLimitFrequency", int(rendering_hz))
