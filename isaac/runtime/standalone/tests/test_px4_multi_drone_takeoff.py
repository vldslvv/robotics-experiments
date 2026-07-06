"""Standalone Isaac/PX4 dual-drone takeoff scenario.

This test runs the multi-vehicle takeoff loop against N independent PX4 SITL
instances: it composes the ``takeoff_quad_quad`` Hydra
experiment (two drones, distinct spawn positions, id 0/1 -> derived simulator
ports 4560/4561 and GCS links 14550/14551), spawns every vehicle, starts one
production PX4 bridge host per drone, commands arm and takeoff over each drone's
own GCS MAVLink endpoint, and asserts that *every* drone climbs and holds a
stable hover within its configured envelope. The scene comes from the top-level
``scene`` config and the routine (steps + flight-envelope gates) from the Hydra
``routine`` group; ``--strict`` (set by the Make target) makes a breach fail.

Run through the host Make target so the N PX4 SITL containers and logs are owned
outside Isaac:
    make test-integration-px4-multi-drone-takeoff
"""

from __future__ import annotations

from _px4_takeoff_scenario import (
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import run_scene_test

TEST_NAME = "px4_multi_drone_takeoff"
PX4_CONTAINER_LOG_NAME = "test_px4_multi_drone_takeoff_px4.log"
PX4_BRIDGE_EXTENSION_PATH = "isaac/runtime/extensions/px4_sitl_brigde"


def main() -> None:
    # default_scene omitted: the scene is read from the Hydra scene config
    # (the make target passes SCENE=@config as the sentinel).
    run_scene_test(
        test_name=TEST_NAME,
        manage_stage=False,
        extra_python_paths=[PX4_BRIDGE_EXTENSION_PATH],
        test_body=lambda context: run_px4_takeoff_scenario(
            context,
            Px4TakeoffScenarioOptions(
                test_name=TEST_NAME,
                px4_container_log_name=PX4_CONTAINER_LOG_NAME,
            ),
        ),
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
