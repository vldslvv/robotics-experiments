# Standalone Isaac Tests

Standalone Isaac tests run through `/isaac-sim/python.sh` inside the Isaac
container. They should exercise real USD stages and Hydra-loaded project
configuration, with mocks kept to the minimum needed for dependencies that
cannot run in standalone Isaac.

Use `_runtime.py` for shared setup and teardown. Each extension should get its
own test module when it has real standalone coverage to add, for example
`test_sim_settings.py`.

`test_press_hover_goto.py` is the exception to the `_runtime.py` pattern: it is
the F1 in-Lab actuation gate and exercises the Isaac Lab `DirectRLEnv` surface
(`Isaac-Drone-Hover-Direct-v0`), not the single-env `IsaacArticulationIO`, so it
launches its own `AppLauncher` and builds the env directly. It drives the env's
`velocity` action with an analytic outer position controller (no RL) and asserts
every parallel env holds a hover (< 0.1 m) then tracks a 2 m goto (< 0.2 m) for
the final 1.0 s validation window. The make gate runs with a fixed seed for
reproducibility; direct diagnostic runs can pass `--diag --seed <N> --num-envs 1
--log-every <N>` to inspect the response. Its `--scene` argument is accepted for
the shared integration runner but ignored, and the script logs that explicitly.
Run it with `make test-integration-press-hover-goto`; it needs no PX4 container.
It prints the same `[isaac-standalone-test] ERROR:` failure sentinel the harness
greps for.

The same script's `--contact-chain` mode is the deterministic F3 real-asset
gate, invoked by `make test-integration-press-contact-chain` with one
environment. It spawns the real paired switch/lamp, orients the switch into the
canonical contact frame, and first verifies the pusher's fixed mount/boom,
visible plunger, forward-facing collidable pad, fully deployed joint, enabled
arm and swept-propeller colliders, and the static wall collider. It then
commands a slow press, verifies pad contact plus pusher and button displacement,
advances the production N-step latch, requires exactly one lamp transition, and
retreats within the stability envelope. A test-local `SwitchPressEnv` subclass
returns no dones and no approach reward, so Stage 1's contact-failure
termination cannot reset this physical measurement path.

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
the shared routine, selected by `configs/takeoff_quad.yaml`,
`configs/takeoff_quad_quad.yaml`, and regular `make run-all` invocations alike.

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
The Isaac core upgrade workflow adds `--benchmark` through the same harness;
use `make benchmark-isaac-core` rather than passing that internal flag manually
so cache warm-up, phased repetitions, the separate Tracy capture, comparison,
and artifacts remain consistent. See `docs/ISAAC_CORE_BENCHMARK.md`.
Use `make test-integration-px4-takeoff-instability` for the opt-in negative
scenario that intentionally faults a rotor after takeoff and passes only when the
routine's hover-step envelope catches the resulting instability. The positive
takeoff scenario commands a 2.5 m global takeoff over MAVLink without changing PX4
takeoff-height parameters; the takeoff step completes once the vehicle climbs at
least 1.5 m and the hover step then holds 5 seconds of stable hover within
0.35 m/s vertical and 0.30 m/s horizontal speed.

`make test-integration-px4-multi-drone-takeoff` runs the same routine for two
drones at once (the `takeoff_quad_quad` Hydra experiment). The
shared scenario in `_px4_takeoff_scenario.py` is driven by the `vehicles` list:
one entry is the single-vehicle baseline, N entries drive N drones, each bound to
its own PX4 SITL instance (the resolved simulator port `4560 + id`, GCS link the
resolved `vehicle.gcs_port = px4.gcs_baseport + id`). All bridges tick on the same
PhysX steps, so one `simulation_app.update()` advances every drone's lockstep
exchange; each drone runs its own independent `DriverContext` plus
`RoutineDriverState` composition (one progress state per drone, no fleet barrier),
and its routine gates are evaluated against its own spawn baseline - every drone
must pass.

`make test-integration-px4-navigation` runs the absolute-waypoint `goto`
navigation routine as one multi-drone scene
(`configs/navigate_quad_hex.yaml`): a quad (id 0) and a hexacopter
(id 1, the `hexacopter_s550.usda` asset, so its PX4 instance launches the derived
generic hexarotor-x airframe) fly the same routine concurrently in parallel lanes
(the hex `goto` waypoints offset by its spawn so the absolute waypoints never
converge), and the test passes only when both hold their envelope. The single
thin entrypoint `test_px4_navigation_routine.py` runs the shared
`_px4_navigation_scenario.py`: its `_assert_navigation_routine_config` asserts
each flying vehicle's routine carries a `goto` leg and a `goto -> hover`
transition (where PX4 failsafes if the OFFBOARD setpoint stream stops while
OFFBOARD is active), then drives the same strict `_px4_takeoff_scenario.py` flight
to completion.

`make test-integration-press-approach-routine` runs the artifact-dependent
Stage 1 mission in `configs/policy_press_approach_quad.yaml`: takeoff, navigate
to about 0.48 m, execute the `press-v1` / `approach` artifact through the
generic `policy` step, return, land, and disarm. The strict hook requires stable
contact-free completion and no lamp transition. Run it only after selection
publishes `.build/training/press-approach/latest` and `make eval-press`
independently accepts it; it is excluded from `test-integration-all`.
