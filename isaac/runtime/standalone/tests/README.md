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

`--scene` is optional. When omitted, or when passed the sentinel `@config`, the
scene is read from the experiment's top-level `scene` config. The takeoff
scenario is just the shared `routine` (`[arm, takeoff, hover]`) driven by
`_px4_takeoff_scenario.py`: the flight verification (real climb, sustained stable
hover, and the continuous tilt/drift/speed envelope gates) lives in the routine
itself, configured by the `routine` Hydra group (`RoutineConfig`); there is no
separate `integration_test` group. `configs/experiment/routine/takeoff.yaml` is
the shared routine, selected by `configs/integration_test_takeoff.yaml`,
`configs/integration_test_multi_drone_takeoff.yaml`, and
`configs/runtime_arm_takeoff.yaml` alike.

The same scenario can run two ways. With `--strict` it is an **integration test**:
a flight-envelope breach raises `ExpectedInstabilityDetected` and any other
routine failure raises `AssertionError` (non-zero exit). Without `--strict` it is
a **regular run**: a breach is logged and the routine is stopped, like the live
extension. The integration Make targets pass `--strict` (via `STRICT=1`); the
generic infra checks (bridge connectivity, the heartbeat/handshake/telemetry
handshakes, the pre-arm start-tilt check, the bridge-log scan) always apply.

PX4-backed scenario tests should be run from their host Make targets so Docker
lifecycle, mounted EEPROM state, and persisted `px4/logs` diagnostics stay
owned by the repository workflow. For example, use
`make test-integration-px4-takeoff` for the arm/takeoff stability scenario.
Use `make test-integration-px4-takeoff-instability` for the opt-in negative
scenario that intentionally faults a rotor after takeoff and passes only when the
routine's hover-step envelope catches the resulting instability. The positive
takeoff scenario commands a 2.5 m global takeoff over MAVLink without changing PX4
takeoff-height parameters; the takeoff step completes once the vehicle climbs at
least 1.5 m and the hover step then holds 5 seconds of stable hover within
0.35 m/s vertical and 0.30 m/s horizontal speed.

`make test-integration-px4-multi-drone-takeoff` runs the same routine for two
drones at once (the `integration_test_multi_drone_takeoff` Hydra experiment). The
shared scenario in `_px4_takeoff_scenario.py` is driven by the `vehicles` list:
one entry is the single-vehicle baseline, N entries drive N drones, each bound to
its own PX4 SITL instance (the resolved simulator port `4560 + id`, GCS link the
resolved `vehicle.gcs_port = px4.gcs_baseport + id`). All bridges tick on the same
PhysX steps, so one `simulation_app.update()` advances every drone's lockstep
exchange; each drone runs its own independent `RoutineDriver` (one driver per
drone, no fleet barrier) and its routine gates are evaluated against its own spawn
baseline - every drone must
pass.
