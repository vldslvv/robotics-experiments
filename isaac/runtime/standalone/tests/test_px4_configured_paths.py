"""Standalone Isaac Sim smoke test for discovered PX4 bridge prim paths.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh \
        /isaac-sim/proj/isaac/runtime/standalone/tests/test_px4_configured_paths.py
"""

from __future__ import annotations

from _runtime import SceneTestContext, run_scene_test

from robotics_experiments.config import get_config
from robotics_experiments.px4_bridge import discover_px4_rotor_mapping

DEFAULT_SCENE = "@isaac/assets/TestScenes/MinimalIntegration.usda"
TEST_NAME = "px4_discovered_paths"


def _run_test(context: SceneTestContext) -> None:
    # The drone spawn helpers import isaacsim at module scope, so they need the
    # running Kit app and stay deferred here.
    from robotics_experiments.drone import despawn_drone, spawn_drone

    env = get_config()
    vehicle = next(iter(env.vehicles.values()))
    spawn_drone(stage=context.stage, config=vehicle.spawn)

    default_prim = context.stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    articulation_path = vehicle.bridge.articulation_path or f"{parent}/{vehicle.spawn.prim_name}"
    articulation_prim = context.stage.GetPrimAtPath(articulation_path)
    if not articulation_prim.IsValid():
        raise AssertionError(f"Configured articulation path is invalid: {articulation_path}")

    rotor_mapping = discover_px4_rotor_mapping(context.stage, articulation_path)
    missing = [
        path
        for path in rotor_mapping.rotor_link_paths
        if not context.stage.GetPrimAtPath(path).IsValid()
    ]

    despawn_drone(stage=context.stage, config=vehicle.spawn)
    if missing:
        raise AssertionError(f"Configured PX4 rotor paths are invalid: {missing}")

    context.report(
        f"{TEST_NAME}: verified articulation and {rotor_mapping.num_rotors} discovered rotors"
    )


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
        test_body=_run_test,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
