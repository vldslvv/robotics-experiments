"""Standalone Isaac/PX4 absolute-waypoint navigation routine.

This test runs the same routine as:
    make run-all HYDRA_CONFIG=configs/runtime_drone_navigate.yaml

The strict integration harness starts a PX4 SITL instance, launches Isaac
headless, spawns the configured drone, starts the production PX4 bridge, and
drives the configured routine to completion. It specifically protects OFFBOARD
setpoint continuity across any configured ``goto`` -> ``hover`` transition,
where PX4 will failsafe if the setpoint stream stops while OFFBOARD is still
active.

Run through the host Make target so PX4 lifecycle and logs are owned outside
Isaac:
    make test-integration-px4-navigation
"""

from __future__ import annotations

from typing import Any

from _px4_takeoff_scenario import (
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import SceneTestContext, run_scene_test

TEST_NAME = "px4_navigation_routine"
PX4_CONTAINER_LOG_NAME = "test_px4_navigation_routine_px4.log"
PX4_BRIDGE_EXTENSION_PATH = "isaac/runtime/extensions/px4_sitl_brigde"


def _step_kind(step: Any) -> str:
    return str(step["kind"])


def _step_waypoint(step: Any) -> tuple[float, float, float]:
    return (float(step["x"]), float(step["y"]), float(step["z"]))


def _goto_waypoints(steps: list[Any]) -> tuple[tuple[float, float, float], ...]:
    return tuple(_step_waypoint(step) for step in steps if _step_kind(step) == "goto")


def _has_goto_hover_transition(steps: list[Any]) -> bool:
    kinds = [_step_kind(step) for step in steps]
    return any(
        current == "goto" and following == "hover"
        for current, following in zip(kinds, kinds[1:], strict=False)
    )


def _assert_navigation_routine_config(context: SceneTestContext) -> None:
    from robotics_experiments.config import get_config

    env = get_config()
    flying = [vehicle for vehicle in env.vehicles if vehicle.routine.steps]
    if not flying:
        raise AssertionError(f"{TEST_NAME}: active Hydra config has no flying vehicles")

    for vehicle in flying:
        steps = list(vehicle.routine.steps)
        kinds = tuple(_step_kind(step) for step in steps)
        waypoints = _goto_waypoints(steps)
        if not waypoints:
            raise AssertionError(
                f"{TEST_NAME}: vehicle {vehicle.id} has no configured goto steps: {kinds!r}"
            )
        if not _has_goto_hover_transition(steps):
            raise AssertionError(
                f"{TEST_NAME}: vehicle {vehicle.id} has no goto -> hover transition: {kinds!r}"
            )

        context.report(
            f"{TEST_NAME}: validated vehicle {vehicle.id} navigation routine "
            f"kinds={kinds!r} goto_waypoints={waypoints!r}"
        )


def _run_navigation_routine(context: SceneTestContext) -> None:
    _assert_navigation_routine_config(context)
    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name=TEST_NAME,
            px4_container_log_name=PX4_CONTAINER_LOG_NAME,
            success_label="absolute-waypoint navigation routine",
        ),
    )


def main() -> None:
    # default_scene omitted: the target passes SCENE=@config so this executes the
    # scene authored by configs/runtime_drone_navigate.yaml.
    run_scene_test(
        test_name=TEST_NAME,
        manage_stage=False,
        extra_python_paths=[PX4_BRIDGE_EXTENSION_PATH],
        test_body=_run_navigation_routine,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
