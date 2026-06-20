# Standalone Isaac Tests

Standalone Isaac tests run through `/isaac-sim/python.sh` inside the Isaac
container. They should exercise real USD stages and Hydra-loaded project
configuration, with mocks kept to the minimum needed for dependencies that
cannot run in standalone Isaac.

Use `_runtime.py` for shared setup and teardown. Each extension should get its
own test module when it has real standalone coverage to add, for example
`test_sim_settings.py`.

Integration Make targets default to the minimal
`@isaac/assets/TestScenes/MinimalIntegration.usda` scene because the current
tests do not validate warehouse geometry, lights, pusher mechanisms, or
navigation through obstacles. Use `SCENE=@isaac/assets/LightSwitchWarehouse/Warehouse.usd`
on a Make invocation, or pass `--scene` to a standalone script, when manually
running against the warehouse.

PX4-backed scenario tests should be run from their host Make targets so Docker
lifecycle, mounted EEPROM state, and persisted `px4/logs` diagnostics stay
owned by the repository workflow. For example, use
`make test-integration-px4-takeoff` for the arm/takeoff stability scenario.
Use `make test-integration-px4-takeoff-instability` for the opt-in negative
scenario that intentionally swaps rotor paths and expects instability detection.
The positive takeoff scenario commands a 2.5 m global takeoff over MAVLink
without changing PX4 takeoff-height parameters. It currently passes after the
vehicle reaches the observed airborne envelope, waits 2 seconds for hover
settle, then holds 5 seconds of hover with no more than 0.35 m of height range
and no more than 0.35 m/s vertical speed in magnitude.
