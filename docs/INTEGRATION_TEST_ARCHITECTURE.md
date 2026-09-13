# Integration-test architecture

This is the architecture and rationale for the container-backed integration
tests. Use the Make targets as the public entrypoints; run harness scripts
directly only when debugging the harness itself. The orchestration, caching,
artifact-isolation, and timing details live here. The authoritative definitions
are `make/integration-tests.mk`, `make/alignment.mk`, `make/rl.mk`,
`make/config.mk`,
`make/scripts/run_isaac_integration_test.sh`,
`make/scripts/run_px4_integration_test.sh`,
`make/scripts/run_isaac_test_in_container.sh`,
`make/scripts/run_integration_suite.sh`, and
`make/scripts/warm_isaac_cache.sh` -- if this doc disagrees with them, those
files win and this doc should be corrected.

The standalone Isaac test scripts these targets run live under
`isaac/runtime/standalone/tests/`; see that directory's `README.md` for
scenario-specific notes.

Inside the container, `run_isaac_test_in_container.sh` launches the Python test
as a distinct process group while a separate pipeline filters and persists its
output. Container HUP, INT, and TERM signals are forwarded to that complete
Python process group as SIGINT, and the runner waits for it before exiting with
the corresponding shell status. This lets campaign parents and workers finish
their atomic interruption reports instead of leaving a live-looking partial
report when a host wrapper is canceled.

`make/integration-tests.mk` owns the general integration harness variables,
aggregate suite, and non-RL target definitions. `make/rl.mk` owns merged
training/evaluation, independent press evaluation, and all press/Isaac Lab task
targets. `make/alignment.mk` exclusively owns controller capture, target sweep,
fit/publication, and the alignment target-envelope gate.
`INTEGRATION_TEST_TARGETS` deliberately remains in
`make/integration-tests.mk`; GNU Make resolves the included definitions from
`make/rl.mk` when the aggregate runs.

## Targets

`make measure-flight-parameters` is an owning campaign rather than part of the
aggregate integration suite. It invokes the internal
`test-integration-flight-parameters-sample` target once per configured trial.
Each invocation uses the ordinary strict PX4/Isaac harness with isolated
processes and run artifacts, then copies its single ULog beside the raw
ground-truth pose candidate and bridge log. The host-safe `flight_parameters`
coordinator decodes that ULog, rejects any trial that does not prove the PX4
hover mode/control/goal contract, and writes the selected neutral trace and
verification record. After every configured trial succeeds, it derives and
validates the immutable hover-envelope bundle and atomically advances its
relative `latest` symlink. See `docs/FLIGHT_PARAMETERS.md`.

- `make test-integration` runs `test-integration-isaac` and
  `test-integration-px4-takeoff`.
- `make test-integration-all` runs every target in `INTEGRATION_TEST_TARGETS`:
  `test-integration-isaac`, `test-integration-drone-articulation`,
  `test-integration-press-hover-goto`, `test-integration-press-contact-chain`,
  `test-integration-px4-takeoff`,
  `test-integration-px4-navigation`, `test-integration-px4-takeoff-instability`,
  and `test-integration-px4-multi-drone-takeoff`. It prints an aggregate
  passed/failed summary after all targets finish.
- `make test-integration-press-hover-goto` runs the F1 in-Lab actuation gate: the
  Isaac Lab `DroneHoverEnv` driven by an analytic velocity controller (no PX4
  container) that must hold a hover and track a 2 m goto across all parallel envs
  for the final 1.0 s validation windows. It runs on the
  Isaac Lab image (`ISAAC_LAB_IMAGE`, the Dockerfile `isaac-lab` stage layered
  `FROM isaac-sim`) instead of the default `isaac-sim` image (`ISAAC_IMAGE`),
  because it imports `isaaclab`. The target sets the override with a
  target-specific `ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)`;
  `ISAAC_INTEGRATION_COMMON_ENV` expands it at recipe time, so the rest of the
  runner is unchanged. Build the image first with `make build-isaac-lab` (or
  `make build`).
- `make test-integration-press-contact-chain` runs the F3 deterministic
  real-asset gate on the Isaac Lab image with one environment. It spawns the
  paired switch/lamp; requires enabled arm, swept-propeller, and wall collision
  geometry; and verifies the complete visible mount/boom/plunger/pad tool plus
  full initial pusher deployment. It then drives a slow press, requires real
  forward-pad contact and pusher/button displacement, advances the production
  N-step latch, verifies exactly one lamp transition, and retreats within the
  stability envelope for the complete deployment hold. Its time budget combines
  the selected curriculum stage's interaction duration with that configured
  hold; a test-local no-dones environment keeps these physical measurements
  independent of Stage 1's contact-failure termination. The gate
  selects the production press experiment config so its scripted post-contact
  ramp retains the authority required to cross the real latch threshold; it
  resolves the config's accepted controller and complete flight-parameters
  bundle to immutable run paths, validates their shared qualification lineage,
  and reports those canonical inputs before constructing the environment. It
  does not load or execute the policy named by that config. It is part of
  `test-integration-all` and requires no PX4 container or trained policy.
- `make test-integration-press-policy` runs the F2b trained-policy gate on the
  Isaac Lab image. It needs the automatically published artifact from
  `make train-rl`, so it is deliberately not part of
  `test-integration-all`; see `docs/RL_WORKFLOW.md` for the policy artifact
  contract and routine config. Policy metadata includes the active policy
  rate and controller provenance; any artifact outside the strict v1 contract
  must be retrained.
- `make test-integration-press-approach-routine` runs the Stage 1 deployment
  flight from `configs/policy_press_approach_quad.yaml`: takeoff, settle at a
  safe staging point about 0.59 m from the switch at a requested 0.5 m/s, make
  the short move to the unchanged 0.47 m handoff waypoint at 0.08 m/s, settle
  for two seconds, and enter the artifact-selected approach policy. It passes
  only after the unchanged handoff predicate latches, real artifact inference
  returns a valid action, and the resulting first policy setpoint is actually
  transmitted. It then ends the isolated invocation smoke test without judging
  later actor behavior, return navigation, landing, contact rate, or task
  success. Like the hover artifact gate, it is excluded from
  `test-integration-all` because it needs an independently accepted artifact
  declaring `policy_interface: press-approach` and `curriculum_stage: approach`.
  Accepted training advances `latest` after the immutable selected actor passes
  the disjoint final test. An operator-selected diagnostic `latest` may be flown
  by this target, but that execution is diagnostic and does not become policy
  qualification evidence; this integration target remains an independent
  workflow gate. Policy quality remains owned by held-out evaluation and
  `test-integration-press-policy-transfer`.
- `make measure-press-navigation-handoff` runs at least ten independent copies
  of that exact production config through the final waypoint and settle, ending
  each before handoff gating can bias the pose sample. It writes schema-2 raw
  samples and a summary beneath
  `.build/benchmarks/press-navigation-handoff-campaigns/`. The summary records
  the identical final route, switch-relative position extrema, normalized
  relative quaternions, maximum quaternion-geodesic attitude error, and the
  configured 0.005 m / 0.25 degree margins required of the shared
  Lab-reset/live-handoff envelope.
  Re-run it whenever the sensor models, PX4 estimator/navigation behavior,
  controller, routine, or warehouse geometry changes. Its internal one-sample
  target is harness plumbing, not a public acceptance gate.
- `make test-integration-alignment-target` runs the analytic hover/goto
  stability envelope with the selected alignment `best` artifact. It is owned
  by `make/alignment.mk` and excluded from `test-integration-all` because it
  requires an alignment publication. During `align-controllers`, the same target
  tests the provisional run-local controller and writes `integration.json` through
  the existing script's `--alignment-report` option. The report binds pass/fail to
  the selected settings and flight-parameters fingerprints; finalization requires
  that matching passing report.
- `make test-integration-press-policy-transfer` runs the trained press actor
  through three PX4 policy stages at distinct entry ranges in the production
  warehouse. Physical wall and switch contact remain enabled, and the gate
  requires no body contact, pusher contact, or lamp transition. This is a
  downstream transfer gate and never fits or publishes controller settings. It
  is artifact-dependent and excluded from `test-integration-all`. The shared
  transfer routine derives its artifact from the active curriculum stage and
  gives each policy step 45 seconds, enough for the 30-second/600-sample
  completion hold. The approach config remains the target default; qualify the
  first press stage with:

  ```bash
  make test-integration-press-policy-transfer \
    HYDRA_CONFIG=configs/policy_transfer_press_hover_quad.yaml
  ```
- `make test-integration-px4-takeoff` starts a temporary PX4 SITL container, runs
  the standalone Isaac PX4 arm/takeoff stability scenario, preserves isolated
  artifacts, and removes the temporary PX4 container.
- `make test-integration-px4-takeoff-instability` runs the same scenario with a
  test-local PX4 rotor mapping fault and passes only when the stability envelope
  detects the resulting instability.
- `make test-integration-px4-multi-drone-takeoff` runs the multi-drone takeoff
  scenario from `configs/takeoff_quad_quad.yaml` (the target sets
  `HYDRA_CONFIG` itself). It starts one PX4 SITL container per PX4-enabled
  vehicle, shares one network namespace, and passes only when every drone holds
  its takeoff/hover envelope. See "Multi-instance PX4" below.
- `make test-integration-px4-navigation` runs the absolute-waypoint `goto`
  navigation routine as **one multi-drone scene**
  (`configs/navigate_quad_hex.yaml`, scene via `@config`): a quad (id 0, x500)
  and a hexacopter (id 1, `hexacopter_s550.usda`) fly the same routine
  **concurrently** in parallel lanes (the hex `goto` waypoints offset +4 in x by
  its spawn, so the absolute world-ENU waypoints never converge), each on its
  own PX4 SITL instance with the per-asset airframe derived from its USD asset
  (the quad's `none_x500_v2`/`22000` and the hex's generated Hex X
  `none_hexa_x`/`22001`). Each first leg is a wall-free guided goto at 0.5 m/s
  followed by hover; the scenario emits one `GOTO_SPEED_DIAGNOSTIC` JSON record
  per vehicle with requested, peak, and average horizontal speed, braking
  distance, completion/final error, overshoot, and cross-track values. The
  braking distance starts at the last sample at or above the requested speed
  (or the peak when the vehicle never reaches it) and ends at maximum
  along-track progress after that sample. The
  second leg remains speedless and protects OFFBOARD setpoint continuity across
  `goto` -> `hover`. The target passes only when **both** drones complete
  without breaching their position, cross-track, or flight-parameters gates. It
  reuses the generic multi-PX4 launch path
  (`run_px4_integration_test.sh` over the enabled vehicles returned by
  `load_config(path)`, like
  the multi-drone takeoff) and the shared `_px4_navigation_scenario.py` driver,
  whose `_assert_navigation_routine_config` validates the `goto`/`goto ->
  hover` config per flying vehicle before running the strict takeoff scenario.

## Concurrency

`make test-integration-all` delegates aggregation to
`make/scripts/run_integration_suite.sh`, which fans the targets out with
`xargs -P "${INTEGRATION_TEST_JOBS}"`. **`INTEGRATION_TEST_JOBS` defaults to
`1`** (serial; `make/config.mk`). Raise it (for example
`INTEGRATION_TEST_JOBS=3`) when the host can sustain more parallelism. Higher
concurrency risks Isaac/GPU resource starvation and the host-load-dependent PX4
SITL startup actuator backlog that destabilizes takeoff, so increase it only on
a host that can keep up.

## Warm-cache pre-stage

`make test-integration-all` depends on `test-integration-warm-isaac-cache`, which
runs first via `make/scripts/warm_isaac_cache.sh`. In host mode, that script
creates the isolated run root and starts the Isaac container; in container mode,
it performs a headless `SimulationApp` start/stop. Hydra, OmegaConf, pymavlink,
and their pinned closures are baked into the Isaac image, so neither cache
warm-up nor individual scenarios install packages or require network access at
test time. This makes cold or cleared shared-cache initialization single-writer,
then lets the real test fan-out use warmed caches instead of each parallel
target racing to initialize identical caches.

## Shared cache vs isolated artifacts

Integration Isaac runs share the persistent Isaac cache mounts under
`isaac/isaac-sim/cache/` (`kit`, `ov`, `pip`, `glcache`, `computecache`; see
`make/config.mk`) to avoid repeating identical Kit, OV, pip, GL, and Compute
cache initialization.

Per-run logs, data, PX4 logs, and timing artifacts are isolated under
`.build/docker/integration/<target>.<run-id>/`, where `<run-id>` is a UTC
timestamp plus the runner shell PID
(`<target>.$(date -u +%Y%m%dT%H%M%SZ).$$`), so parallel runs never share PX4
state or diagnostics. PX4 runs additionally get isolated `px4/logs` and
`px4/eeprom` subdirectories and a per-run PX4 container name.

## Structured timing lines

Logs include structured timing lines for performance triage:

- Host/container stages:
  `[integration-timing] target=... stage=... elapsed_s=... total_s=... status=...`
- Standalone Isaac stages:
  `[isaac-standalone-test] TIMING test=... stage=... elapsed_s=... total_s=...`

PX4 scenario stage timings also include `sim_elapsed_s=...` when the stage
advances simulated time. Each per-run artifact directory persists the
host/container timing lines in `integration-timing.log`.

Standalone integration logs that report drone state use the shared `_runtime.py`
vector formatter and include full ENU position and linear velocity
(`pos_enu_m=(x=..., y=..., z=...)`, `vel_enu_m_s=(x=..., y=..., z=...)`) for
flight-performance triage.

The integration runners also accept `ISAAC_TEST_EXTRA_ARGS` for an owning
workflow to pass standalone-script flags. The Isaac core performance workflow
uses `--benchmark` to add phased Kit/render/physics/articulation measurements to
the existing drone articulation scenario. Its separate one-shot trace also uses
`--benchmark-profile` and `ISAAC_PROFILE_TRACE_NAME`; ordinary integration
targets do not capture samples or profiler data. The methodology, gate, and
baseline workflow are documented in `docs/ISAAC_CORE_BENCHMARK.md`.

## Scene selection

Integration tests default to the minimal
`@isaac/assets/TestScenes/MinimalIntegration.usda` scene (`SCENE` in
`make/config.mk`) because the current scenarios do not validate warehouse
interactions. Override with
`SCENE=@isaac/assets/LightSwitchWarehouse/Warehouse.usd` when a manual warehouse
run is needed.

`--scene` can also be the sentinel `@config`, which makes the standalone runner
read the scene from the experiment's top-level `scene` config instead of a path.
The multi-drone target sets `SCENE=@config` so its scene and integration-test
tunables come from the active Hydra experiment.

## Multi-instance PX4

`make/scripts/run_px4_integration_test.sh` computes the per-run artifact/log
paths, starts one PX4 SITL container per PX4-enabled vehicle in the active Hydra
config, and delegates the Isaac-side test body to
`make/scripts/run_isaac_test_in_container.sh` inside the Isaac container. It
iterates the enabled vehicles from `load_config(path)` and uses each
`vehicle.id` as the instance index rather than a blind `0..count` loop, so
non-contiguous ids keep their derived ports. The same resolved launch tuple
threads each vehicle's PX4 simulator model/autostart plus
`px4.failsafe.offboard_loss_action` and timeout into the container as pinned PX4
startup parameters. The same resolved tuple carries `px4.logging_profile` into
PX4's `SDLOG_PROFILE`; ordinary tests retain profile `1`, while alignment opts
into full-rate controller topics. The first instance in that list owns
the network namespace. Each additional instance `i` joins that namespace
(`--network container:<primary>`), runs with `PX4_INSTANCE=i` so
`run_px4_sitl_ci.sh` launches PX4 `-i i` (simulator TCP `4560 + i`, GCS UDP
`14550 + i`), and uses an isolated per-instance rootfs plus per-instance
`px4/logs/<i>` / `px4/eeprom/<i>` mounts so non-volatile state never collides.
The runner waits on each instance's `TCP port 456i` readiness line, then the
Isaac container joins the primary namespace and connects all N bridges. The
cleanup trap collects each container's `docker logs` (`<px4-log>`,
`<px4-log>.<i>`) and removes every container. To keep `collect_logs` cheap on a
long passing flight, the trap captures only the last `INTEGRATION_LOG_TAIL_LINES`
of the PX4 console on **success** (the scrollback is ANSI redraw noise; the real
artifacts are the persisted ULog/bridge logs) and the **full** console only on
failure or with `INTEGRATION_VERBOSE=1`. The active Hydra config must carry
a `{vehicle}` token in `px4.debug_log_path` for multi-vehicle bridge logs (the
multi-drone runner injects a `px4.debug_log_path` override carrying the token;
config validation rejects two vehicles resolving to the same template). Keep
`INTEGRATION_TEST_JOBS=1`: a 2-drone run already loads the host with two PX4
boots and two serial lockstep exchanges per tick.

`PX4_INSTANCE` is the per-container index threaded into `run_px4_sitl_ci.sh` /
`reset_px4_state.sh` (see `px4/docker/README.md`). Every instance (including
`0`) is launched uniformly by `px4_sitl_common.sh::launch_px4_sitl`: the
prebuilt binary run with `-i ${PX4_INSTANCE}` and an interactive `pxh>` console
(no `-d`), from its own per-instance rootfs subdir, with no `instance == 0`
special case in the launch, reset, or mount logic. Keep the launch interactive
(`-d` omitted) so every per-vehicle terminal accepts commands.

Strict routine failures do not abort the callback immediately. The shared driver
first attempts `MAV_CMD_NAV_LAND`, escalates a rejected or unacknowledged command
by stopping OFFBOARD streaming so PX4's configured offboard-loss action can run,
and monitors until disarm or recovery timeout. The standalone strict shell then
raises the preserved original failure, so the integration result remains
failure-typed without preventing landing. Its final dead-callback backstop follows
the live per-driver phase deadline and is checked after the physics callback has
one opportunity to complete that deadline. The DRIVE allowance comes from the
fully built routine's summed step budgets and Hydra-owned
`drive_budget_multiplier`; the shell does not maintain a second estimate of
routine duration.
