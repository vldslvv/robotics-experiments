"""Standalone takeoff -> goto -> trained approach -> return -> land gate."""

from __future__ import annotations

from _px4_press_approach_routine_scenario import (
    run_px4_press_approach_routine_scenario,
)
from _runtime import run_scene_test

TEST_NAME = "px4_press_approach_routine"
PX4_BRIDGE_EXTENSION_PATH = "isaac/runtime/extensions/px4_sitl_brigde"


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        manage_stage=False,
        extra_python_paths=[PX4_BRIDGE_EXTENSION_PATH],
        test_body=lambda context: run_px4_press_approach_routine_scenario(context=context),
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
