"""Standalone Isaac/PX4 negative takeoff stability scenario.

This test intentionally swaps two PX4 rotor marker paths in the local bridge
configuration, then runs the same arm/takeoff scenario as the positive
stability test. It passes only if the scenario reaches command flow and detects
the resulting instability through the stability envelope.

Run through the host Make target so PX4 lifecycle and logs are owned outside
Isaac:
    make test-integration-px4-takeoff-instability
"""

from __future__ import annotations

from dataclasses import replace

from _px4_takeoff_scenario import (
    ExpectedInstabilityDetected,
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import SceneTestContext, run_scene_test

DEFAULT_SCENE = "@isaac/assets/LightSwitchWarehouse/Warehouse.usd"
TEST_NAME = "px4_takeoff_instability_detection"
PX4_CONTAINER_LOG_NAME = "test_px4_takeoff_instability_px4.log"


def _swap_front_right_and_front_left_rotor_paths(px4_config):
    rotor_paths = list(px4_config.rotor_link_paths)
    if len(rotor_paths) < 4:
        raise AssertionError(
            "PX4 rotor path swap negative test requires at least four rotor paths"
        )
    rotor_paths[0], rotor_paths[2] = rotor_paths[2], rotor_paths[0]
    return replace(px4_config, rotor_link_paths=rotor_paths)


def _run_test(context: SceneTestContext) -> None:
    try:
        run_px4_takeoff_scenario(
            context,
            Px4TakeoffScenarioOptions(
                test_name=TEST_NAME,
                px4_container_log_name=PX4_CONTAINER_LOG_NAME,
                px4_config_transform=_swap_front_right_and_front_left_rotor_paths,
            ),
        )
    except ExpectedInstabilityDetected as exc:
        context.report(f"{TEST_NAME}: detected expected instability: {exc}")
        return
    raise AssertionError(
        "Rotor-path swap negative test did not trigger the takeoff stability envelope"
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
