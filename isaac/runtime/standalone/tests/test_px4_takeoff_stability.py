"""Standalone Isaac/PX4 takeoff stability scenario.

This test runs a full single-vehicle loop against a live PX4 SITL instance:
spawn the configured drone, start the production PX4 bridge host, command arm
and takeoff over PX4's GCS MAVLink endpoint, and assert that the vehicle climbs
without excessive tilt, lateral drift, bridge disconnects, stale actuator
lag, or PX4 failsafe state.

Run through the host Make target so PX4 lifecycle and logs are owned outside
Isaac:
    make test-integration-px4-takeoff
"""

from __future__ import annotations

from _px4_takeoff_scenario import (
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import run_scene_test

DEFAULT_SCENE = "@isaac/assets/LightSwitchWarehouse/Warehouse.usd"
TEST_NAME = "px4_takeoff_stability"
PX4_CONTAINER_LOG_NAME = "test_px4_takeoff_stability_px4.log"


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
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
