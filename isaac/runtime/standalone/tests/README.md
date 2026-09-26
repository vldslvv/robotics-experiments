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
(`Isaac-Drone-Hover-Direct-v1`), not the single-env `IsaacArticulationIO`, so it
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

`flight_parameters_capture_response.py` is the PX4-backed flight-parameters
response capture shell. It
uses the shared fleet scenario and ordinary `{kind: policy}` OFFBOARD
lifecycle, with a test-only injected adapter that emits the controller
definition's linear-velocity, yaw-rate, and combined probes. It writes exact
command timing, initial state, position, attitude, body/world velocity, and
angular velocity to the selected flight-parameters response directory. Before each profile,
the adapter holds a neutral setpoint through the configured minimum and
continuous speed/rate/tilt stability window. This excludes OFFBOARD priming
from the recorded policy-execution response and gives each reset Lab replay a
near-equilibrium initial state. Trace yaw remains body-FLU +CCW while the
outbound PX4 setpoint is converted to NED +clockwise. Its response routine
ends with the probe policy and lets teardown stop the ephemeral simulator;
landing and auto-disarm are outside this controller target. It is not a
press-policy scenario and adds no commanding routine kind.

`flight_parameters_capture.py` is the isolated hover-evidence leaf owned by
`make measure-flight-parameters`. Each fresh PX4/Isaac invocation uses the
ordinary arm, takeoff, and hover routine and records a long raw simulator
ground-truth pose candidate at the configured cadence. It does not infer hover
from the routine cursor or kinematics. The host coordinator decodes the
corresponding completed PX4 ULog inside `flight_parameters`, proves the PX4
mode/control/goal contract, and only then selects the settled measurement
window. The raw trace, ULog, verification record, selected trace, and bridge
log remain together as trial evidence. Statistics derivation and publication
remain in the host-safe `robotics_experiments.flight_parameters` module.

The same script's `--contact-chain` mode is the deterministic F3 real-asset
gate, invoked by `make test-integration-press-contact-chain` with one
environment. It spawns the real paired switch/lamp, orients the switch into the
canonical contact frame, and first verifies the pusher's fixed mount/boom,
visible plunger, forward-facing collidable pad, fully deployed joint, enabled
arm and swept-propeller colliders, and the static wall collider. It then
commands a slow press, verifies pad contact plus pusher and button displacement,
advances the production N-step latch, requires exactly one lamp transition, and
retreats within the stability envelope for the complete configured deployment
hold. Before constructing the environment it resolves the production config's
controller and flight-parameters selectors to immutable run paths, validates
their shared qualification lineage, and reports those canonical inputs. A
test-local `SwitchPressEnv` subclass returns no dones and no approach reward, so
Stage 1's privileged collision termination cannot reset this physical
measurement path. Its test-local press/retreat speed derives from the
stable-flight speed limit rather than the approach reward's dense velocity
reference.

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
the shared routine, selected by `configs/routines/basic_takeoff_quad.yaml`,
`configs/routines/fleet_takeoff_quad_quad.yaml`, and regular `make run-all` invocations alike.

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
drones at once (the `fleet_takeoff_quad_quad` Hydra experiment). The
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
(`configs/routines/fleet_navigate_quad_hex.yaml`): a quad (id 0) and a hexacopter
(id 1, the `hexacopter_s550.usda` asset, so its PX4 instance launches the derived
generic hexarotor-x airframe) fly the same routine concurrently in parallel lanes
(the hex `goto` waypoints offset by its spawn so the absolute waypoints never
converge), and the test passes only when both hold their envelope. The first
leg uses PX4 guided repositioning at 0.5 m/s and is followed by hover so the
test can report requested, peak, and average horizontal speed, braking
distance, final error, and overshoot in `GOTO_SPEED_DIAGNOSTIC` JSON. The
second leg remains a speedless OFFBOARD goto followed by hover, preserving the
stream-continuity integration check. The single thin entrypoint
`test_px4_navigation_routine.py` runs the shared
`_px4_navigation_scenario.py`: its `_assert_navigation_routine_config` asserts
each flying vehicle's routine carries a `goto` leg and a `goto -> hover`
transition. Its speed collector additionally requires one guided goto followed
by hover, while the other transition checks that OFFBOARD streaming remains
continuous. It then drives the same strict `_px4_takeoff_scenario.py` flight to
completion.

`make test-integration-press-approach-routine` runs the artifact-dependent
Stage 1 mission in `configs/routines/policy_press_approach_quad.yaml`: takeoff, navigate
to a safe 0.75 m staging waypoint at a requested 0.5 m/s, settle, make the short
move to the 0.60 m handoff waypoint at 0.08 m/s with a measured
0.06 m completion radius, settle for two seconds, and execute the
`press` / `approach` artifact through the generic `policy` step. The
strict hook ends after the handoff latches, one valid real actor evaluation is
returned, and that evaluation's first setpoint is transmitted. It does not
require later actor success, contact-free completion, return, landing, or lamp
state. It requires a strict artifact at
`.build/training/press-approach/latest`; an accepted campaign normally publishes
that label, while an operator may select a diagnostic unqualified artifact
without turning this smoke result into qualification evidence. The target is
excluded from `test-integration-all`. Policy quality remains a separate
held-out/transfer contract.

`make measure-press-navigation-handoff` repeats the same production config at
least ten times but stops on the first exact observation after waypoint settling,
before actor gating. It summarizes switch-relative entry-position noise beneath
`.build/benchmarks/press-navigation-handoff-campaigns/`; rerun it after changes
to navigation sensors, PX4 estimation/navigation, the controller, route, or
warehouse geometry, then update the Hydra measurement evidence and shared
training/handoff envelope.

`make test-integration-press-policy-transfer` is the separate post-training PX4
acceptance in the production warehouse. It runs the trained press actor at
three entry ranges with physical wall and switch contact enabled, and requires
no body contact, pusher contact, or lamp transition. Its observations and
actions are not controller-fitting inputs.
