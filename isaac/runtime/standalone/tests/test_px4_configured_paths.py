"""Standalone Isaac Sim smoke test for configured PX4 bridge prim paths.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh \
        /isaac-sim/proj/isaac/runtime/standalone/tests/test_px4_configured_paths.py
"""

from __future__ import annotations

from _runtime import SceneTestContext, run_scene_test

DEFAULT_SCENE = "@isaac/assets/LightSwitchWarehouse/Warehouse.usd"
TEST_NAME = "px4_configured_paths"


def _run_test(context: SceneTestContext) -> None:
    from robotics_experiments.config import get_config
    from robotics_experiments.drone import despawn_drone, spawn_drone

    env = get_config()
    spawn_drone(context.stage, env.drone)

    default_prim = context.stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    articulation_path = env.px4.articulation_path or f"{parent}/{env.drone.prim_name}"
    articulation_prim = context.stage.GetPrimAtPath(articulation_path)
    if not articulation_prim.IsValid():
        raise AssertionError(f"Configured articulation path is invalid: {articulation_path}")

    missing = []
    for path in env.px4.rotor_link_paths:
        rotor_path = path if path.startswith("/") else f"{articulation_path}/{path}"
        if not context.stage.GetPrimAtPath(rotor_path).IsValid():
            missing.append(rotor_path)

    despawn_drone(context.stage, env.drone)
    if missing:
        raise AssertionError(f"Configured PX4 rotor paths are invalid: {missing}")

    context.report(f"{TEST_NAME}: verified configured articulation and rotor paths")


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
        test_body=_run_test,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
