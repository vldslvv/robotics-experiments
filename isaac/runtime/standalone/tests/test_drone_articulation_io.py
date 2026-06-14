"""Standalone Isaac Sim smoke test for real drone articulation I/O.

This test verifies the Isaac-side half of the PX4 bridge contract: a spawned
configured drone can be opened through ``IsaacArticulationIO``, sampled into a
finite ``VehicleState``, and addressed with per-rotor force commands. It does
not connect to PX4 SITL; ``test_px4_bridge_host.py`` owns the live MAVLink
coverage.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh \
        /isaac-sim/proj/isaac/runtime/standalone/tests/test_drone_articulation_io.py
"""

from __future__ import annotations

import math

import numpy as np
from _runtime import SceneTestContext, run_scene_test

DEFAULT_SCENE = "@isaac/assets/LightSwitchWarehouse/Warehouse.usd"
TEST_NAME = "drone_articulation_io"
PHYSICS_DT_S = 0.004
SETTLE_UPDATES = 8


def _assert_finite_array(value, *, shape: tuple[int, ...], label: str) -> None:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise AssertionError(f"{label}: expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{label}: expected finite values, got {array}")


def _resolve_drone_paths(stage, env) -> tuple[str, list[str]]:
    from robotics_experiments.px4_bridge import discover_px4_rotor_mapping

    default_prim = stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    articulation_path = env.px4.articulation_path or f"{parent}/{env.drone.prim_name}"
    rotor_mapping = discover_px4_rotor_mapping(stage, articulation_path)
    return articulation_path, rotor_mapping.rotor_link_paths


def _assert_configured_prims_exist(stage, articulation_path: str, rotor_paths: list[str]) -> None:
    missing = []
    if not stage.GetPrimAtPath(articulation_path).IsValid():
        missing.append(articulation_path)
    for path in rotor_paths:
        if not stage.GetPrimAtPath(path).IsValid():
            missing.append(path)
    if missing:
        raise AssertionError(f"Configured drone articulation I/O paths are invalid: {missing}")


def _advance(context: SceneTestContext, steps: int) -> None:
    for _ in range(steps):
        context.simulation_app.update()


def _assert_vehicle_state(state, *, sim_time_us: int) -> None:
    _assert_finite_array(state.position_enu, shape=(3,), label="position_enu")
    _assert_finite_array(
        state.orientation_flu_enu_wxyz,
        shape=(4,),
        label="orientation_flu_enu_wxyz",
    )
    _assert_finite_array(state.linear_velocity_enu, shape=(3,), label="linear_velocity_enu")
    _assert_finite_array(state.angular_velocity_flu, shape=(3,), label="angular_velocity_flu")
    _assert_finite_array(
        state.linear_acceleration_enu,
        shape=(3,),
        label="linear_acceleration_enu",
    )
    quat_norm = float(np.linalg.norm(np.asarray(state.orientation_flu_enu_wxyz)))
    if not math.isclose(quat_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise AssertionError(f"orientation quaternion is not normalized: norm={quat_norm}")
    if int(state.sim_time_us) != sim_time_us:
        raise AssertionError(f"sim_time_us: expected {sim_time_us}, got {state.sim_time_us}")


def _run_test(context: SceneTestContext) -> None:
    import omni.timeline

    from robotics_experiments.config import get_config
    from robotics_experiments.drone import despawn_drone, spawn_drone
    from robotics_experiments.px4_bridge import IsaacArticulationIO
    from robotics_experiments.simulation import apply_settings

    env = get_config()
    context.report(f"{TEST_NAME}: applying configured simulation settings")
    apply_settings(context.stage)

    articulation_io = None
    timeline = omni.timeline.get_timeline_interface()
    context.report(f"{TEST_NAME}: spawning configured drone")
    spawn_drone(context.stage, env.drone)
    try:
        articulation_path, rotor_paths = _resolve_drone_paths(context.stage, env)
        context.report(f"{TEST_NAME}: using drone articulation path {articulation_path}")
        _assert_configured_prims_exist(context.stage, articulation_path, rotor_paths)

        _advance(context, SETTLE_UPDATES)
        timeline.play()
        _advance(context, SETTLE_UPDATES)

        context.report(f"{TEST_NAME}: constructing real IsaacArticulationIO")
        articulation_io = IsaacArticulationIO(
            articulation_path=articulation_path,
            rotor_link_paths=rotor_paths,
        )
        articulation_io.initialize()
        if not articulation_io.is_initialized:
            raise AssertionError("Isaac articulation I/O did not initialize")
        if articulation_io.num_rotors != len(rotor_paths):
            raise AssertionError(
                f"Isaac articulation I/O has {articulation_io.num_rotors} rotors; "
                f"expected {len(rotor_paths)}"
            )

        context.report(f"{TEST_NAME}: reading spawned drone state")
        state = articulation_io.read_state(sim_time_us=int(PHYSICS_DT_S * 1_000_000))
        _assert_vehicle_state(state, sim_time_us=int(PHYSICS_DT_S * 1_000_000))

        forces_n = np.full(len(rotor_paths), 0.25, dtype=np.float64)
        context.report(f"{TEST_NAME}: applying per-rotor forces through Isaac")
        articulation_io.apply_forces(forces_n)
        _advance(context, SETTLE_UPDATES)

        state_after_force = articulation_io.read_state(
            sim_time_us=int(2 * PHYSICS_DT_S * 1_000_000)
        )
        _assert_vehicle_state(state_after_force, sim_time_us=int(2 * PHYSICS_DT_S * 1_000_000))
    finally:
        context.report(f"{TEST_NAME}: stopping timeline and despawning drone")
        timeline.stop()
        if articulation_io is not None:
            articulation_io.shutdown()
        despawn_drone(context.stage, env.drone)

    context.report(f"{TEST_NAME}: verified real drone articulation read/write path")


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
        test_body=_run_test,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
