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

`--scene` is optional. When omitted, or when passed the sentinel `@scenario`,
the scene is read from the Hydra `scenario.scene` config (see `ScenarioConfig`).
The takeoff scenario's tunables - takeoff target, command-flow timeouts, and the
flight-envelope gates - also come from the `scenario` Hydra group rather than
being hardcoded; `configs/experiment/scenario/default.yaml` holds the baseline
values.

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

`make test-integration-px4-multi-drone-takeoff` runs the same command flow for
two drones at once (the `multi_drone_takeoff` Hydra config). The shared scenario
in `_px4_takeoff_scenario.py` is driven by the `vehicles` list: one entry is the
single-vehicle baseline, N entries drive N drones, each bound to its own PX4 SITL
instance (the resolved simulator port `4560 + id`, GCS link
`scenario.gcs_baseport + id`). All bridges tick on the same PhysX steps,
so one `simulation_app.update()` advances every drone's lockstep exchange; each
phase runs across all drones over the shared clock, and the envelope gates are
evaluated per drone (drone-id-tagged) against its own spawn position - every
drone must pass.
