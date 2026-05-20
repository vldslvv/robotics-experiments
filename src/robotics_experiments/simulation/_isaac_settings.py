"""Isaac Sim backend for applying simulation settings.

This module is intentionally allowed to import Isaac runtime dependencies such
as ``carb`` and ``pxr``. Keep caller-facing imports pointed at
``robotics_experiments.simulation.apply_settings`` so non-Isaac code can import
the package without loading simulator-only modules.
"""

import carb
from pxr import Gf, PhysxSchema, Usd, UsdPhysics

from robotics_experiments.config import SimSettings
from robotics_experiments.config import get_config as _get_experiment_config

__all__ = ["apply_isaac_settings"]


def _get_or_create_physics_scene(stage: Usd.Stage) -> UsdPhysics.Scene:
    """Return the first ``UsdPhysics.Scene`` on the stage, creating one if none exists.

    A newly created scene is placed under the stage default prim, falling
    back to ``/World`` when no default prim is set.

    Args:
        stage: USD stage to inspect or mutate.

    Returns:
        The existing or newly defined physics scene.
    """
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return UsdPhysics.Scene(prim)
    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    return UsdPhysics.Scene.Define(stage, f"{parent}/physicsScene")


def _apply_gravity(scene: UsdPhysics.Scene, gravity: tuple) -> None:
    """Write gravity direction and magnitude attributes on a physics scene.

    Zero-magnitude gravity vectors are skipped because the direction is
    undefined and writing zeros would disable gravity in a way USD does not
    round-trip cleanly.

    Args:
        scene: Target ``UsdPhysics.Scene``.
        gravity: ``(gx, gy, gz)`` world gravity vector in m/s^2.
    """
    gx, gy, gz = gravity
    magnitude = (gx * gx + gy * gy + gz * gz) ** 0.5
    if magnitude <= 0.0:
        return
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(gx / magnitude, gy / magnitude, gz / magnitude))
    scene.CreateGravityMagnitudeAttr().Set(magnitude)


def _apply_physx(scene: UsdPhysics.Scene, physics_hz: float, solver_type: str) -> None:
    """Configure PhysX time stepping and solver type on a physics scene.

    Args:
        scene: Target ``UsdPhysics.Scene``; the ``PhysxSceneAPI`` schema is
            applied to its prim if not already present.
        physics_hz: PhysX steps per second; truncated to int by PhysX schema.
        solver_type: PhysX solver identifier (e.g. ``"TGS"`` or ``"PGS"``).
    """
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_api.CreateTimeStepsPerSecondAttr().Set(int(physics_hz))
    physx_api.CreateSolverTypeAttr().Set(solver_type)


def _apply_render_rate(rendering_hz: float) -> None:
    """Enable and set the Kit main run-loop rate limit.

    Args:
        rendering_hz: Target render frequency in Hz; truncated to int.
    """
    carb_settings = carb.settings.get_settings()
    carb_settings.set("/app/runLoops/main/rateLimitEnabled", True)
    carb_settings.set("/app/runLoops/main/rateLimitFrequency", int(rendering_hz))


def apply_isaac_settings(stage: Usd.Stage, settings: SimSettings | None = None) -> None:
    """Apply gravity, PhysX, and render-rate settings to a stage.

    Ensures a ``UsdPhysics.Scene`` exists, updates its gravity and PhysX
    attributes, and configures the Kit main-loop render rate via carb
    settings.

    Args:
        stage: USD stage to configure.
        settings: Simulation settings to apply; when ``None``, resolved via
            :func:`.config.get_config` at call time.
    """
    if settings is None:
        settings = _get_experiment_config().sim
    scene = _get_or_create_physics_scene(stage)
    _apply_gravity(scene, settings.gravity)
    _apply_physx(scene, settings.physics_hz, settings.solver_type)
    _apply_render_rate(settings.rendering_hz)

    carb.log_info(
        f"[SimSettings] physics={settings.physics_hz}Hz "
        f"rendering={settings.rendering_hz}Hz solver={settings.solver_type} "
        f"gravity={settings.gravity}"
    )
