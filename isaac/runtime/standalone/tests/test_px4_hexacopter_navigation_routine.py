"""Standalone Isaac/PX4 absolute-waypoint navigation routine for a hexacopter.

This test runs the same routine as:
    make run-all HYDRA_CONFIG=configs/runtime_hexacopter_navigate.yaml

It is the hexacopter analogue of ``test_px4_navigation_routine.py``: the strict
integration harness starts a PX4 SITL instance (launched with the generic
hexarotor-x airframe derived from the hexacopter_s550.usda asset), launches Isaac
headless, spawns the configured six-rotor drone, starts the production PX4
bridge, and drives the configured navigation routine to completion. As in the
quad case it specifically protects OFFBOARD setpoint continuity across the
``goto`` -> ``hover`` transition, where PX4 will failsafe if the setpoint stream
stops while OFFBOARD is still active.

Run through the host Make target so PX4 lifecycle and logs are owned outside
Isaac:
    make test-integration-px4-hexacopter-navigation
"""

from __future__ import annotations

from _px4_navigation_scenario import run_px4_navigation_scenario
from _runtime import SceneTestContext, run_scene_test

TEST_NAME = "px4_hexacopter_navigation_routine"
PX4_CONTAINER_LOG_NAME = "test_px4_hexacopter_navigation_routine_px4.log"
PX4_BRIDGE_EXTENSION_PATH = "isaac/runtime/extensions/px4_sitl_brigde"


def _run_navigation_routine(context: SceneTestContext) -> None:
    run_px4_navigation_scenario(
        context,
        test_name=TEST_NAME,
        px4_container_log_name=PX4_CONTAINER_LOG_NAME,
        success_label="hexacopter absolute-waypoint navigation routine",
    )


def main() -> None:
    # default_scene omitted: the target passes SCENE=@config so this executes the
    # scene authored by configs/runtime_hexacopter_navigate.yaml.
    run_scene_test(
        test_name=TEST_NAME,
        manage_stage=False,
        extra_python_paths=[PX4_BRIDGE_EXTENSION_PATH],
        test_body=_run_navigation_routine,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
