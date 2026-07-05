"""Standalone Isaac Sim test for the ``apply_settings`` simulation API.

This test uses the real standalone Isaac workflow and Hydra-loaded project
config. The shared runtime loads a real USD scene, then this module calls the
production ``robotics_experiments.simulation.apply_settings(stage)`` API with
no manually constructed settings object (the same API the live
``experiment_runtime`` orchestrator and this scenario both use).

Run inside the Isaac Sim container:
    /isaac-sim/python.sh \
        /isaac-sim/proj/isaac/runtime/standalone/tests/test_sim_settings.py
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from _runtime import SceneTestContext, run_scene_test

from robotics_experiments.config import get_config
from robotics_experiments.simulation import apply_settings

if TYPE_CHECKING:
    from pxr import Usd, UsdPhysics

DEFAULT_SCENE = "@isaac/assets/TestScenes/MinimalIntegration.usda"
TEST_NAME = "sim_settings"


def _find_physics_scenes(stage: Usd.Stage) -> list[UsdPhysics.Scene]:
    from pxr import UsdPhysics

    return [UsdPhysics.Scene(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-5):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def _assert_vec3_close(actual: Any, expected: tuple[float, float, float], label: str) -> None:
    actual_tuple = tuple(float(value) for value in actual)
    for axis, (actual_value, expected_value) in enumerate(zip(actual_tuple, expected, strict=True)):
        _assert_close(actual_value, expected_value, f"{label}[{axis}]")


def _assert_settings_applied(context: SceneTestContext, original_scene_paths: list[str]) -> None:
    # carb / pxr.PhysxSchema follow Isaac's create-app-first import convention and
    # stay deferred until after SimulationApp has started.
    import carb
    from pxr import PhysxSchema

    expected = get_config().sim
    scenes = _find_physics_scenes(context.stage)
    if not scenes:
        raise AssertionError("Expected at least one UsdPhysics.Scene after applying settings")

    scene_paths = [scene.GetPrim().GetPath().pathString for scene in scenes]
    if original_scene_paths:
        if scene_paths != original_scene_paths:
            raise AssertionError(
                "Expected existing physics scenes to be reused: "
                f"{original_scene_paths}, got {scene_paths}"
            )
    elif len(scene_paths) != 1:
        raise AssertionError(f"Expected one created physics scene, got {scene_paths}")

    scene = scenes[0]
    context.report(f"{TEST_NAME}: physics scene under test: {scene.GetPrim().GetPath().pathString}")
    gx, gy, gz = expected.gravity
    gravity_magnitude = (gx * gx + gy * gy + gz * gz) ** 0.5
    if gravity_magnitude > 0.0:
        expected_direction = (
            gx / gravity_magnitude,
            gy / gravity_magnitude,
            gz / gravity_magnitude,
        )
        _assert_close(
            float(scene.GetGravityMagnitudeAttr().Get()),
            gravity_magnitude,
            "gravity magnitude",
        )
        _assert_vec3_close(
            scene.GetGravityDirectionAttr().Get(),
            expected_direction,
            "gravity direction",
        )

    physx_api = PhysxSchema.PhysxSceneAPI(scene.GetPrim())
    if int(physx_api.GetTimeStepsPerSecondAttr().Get()) != int(expected.physics_hz):
        raise AssertionError(
            "PhysX timestep mismatch: "
            f"expected {int(expected.physics_hz)}, "
            f"got {physx_api.GetTimeStepsPerSecondAttr().Get()}"
        )
    if physx_api.GetSolverTypeAttr().Get() != expected.solver_type:
        raise AssertionError(
            f"PhysX solver mismatch: expected {expected.solver_type}, "
            f"got {physx_api.GetSolverTypeAttr().Get()}"
        )

    carb_settings = carb.settings.get_settings()
    if not carb_settings.get("/app/runLoops/main/rateLimitEnabled"):
        raise AssertionError("Expected Kit main run-loop rate limit to be enabled")
    actual_rendering_hz = int(carb_settings.get("/app/runLoops/main/rateLimitFrequency"))
    if actual_rendering_hz != int(expected.rendering_hz):
        raise AssertionError(
            "Render rate mismatch: "
            f"expected {int(expected.rendering_hz)}, got {actual_rendering_hz}"
        )
    context.report(
        f"{TEST_NAME}: verified Hydra settings: "
        f"physics_hz={expected.physics_hz}, "
        f"rendering_hz={expected.rendering_hz}, "
        f"solver_type={expected.solver_type}, "
        f"gravity={tuple(expected.gravity)}"
    )


def _run_test(context: SceneTestContext) -> None:
    original_scene_paths = [
        scene.GetPrim().GetPath().pathString for scene in _find_physics_scenes(context.stage)
    ]
    context.report(
        f"{TEST_NAME}: physics scenes before apply_settings: "
        f"{original_scene_paths if original_scene_paths else 'none'}"
    )
    context.report(f"{TEST_NAME}: calling robotics_experiments.simulation.apply_settings(stage)")
    apply_settings(context.stage)
    _assert_settings_applied(context, original_scene_paths)


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
        test_body=_run_test,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
