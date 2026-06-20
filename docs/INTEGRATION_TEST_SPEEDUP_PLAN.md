# Integration Test Speedup Plan

## Summary

The latest integration test logs show that the suite is dominated by repeated
Isaac Sim process startup and full warehouse USD stage loading. The test bodies
are short except for the PX4 takeoff scenarios, where the largest deliberate
waits are pre-arm estimator settle and hover observation.

Latest analyzed artifacts:

- `.build/docker/integration/test-integration-isaac.20260619T194348Z.100000/`
- `.build/docker/integration/test-integration-drone-articulation.20260619T194348Z.99999/`
- `.build/docker/integration/test-integration-px4-takeoff.20260619T194615Z.102266/`
- `.build/docker/integration/test-integration-px4-takeoff-instability.20260619T194620Z.102845/`

Observed standalone Isaac timings:

| Test | SimulationApp start | Stage open | Test body |
| --- | ---: | ---: | ---: |
| `sim_settings` | 91.048s | 58.085s | 0.114s |
| `drone_articulation_io` | 90.524s | 53.772s | 0.686s |
| `px4_takeoff_stability` | 88.585s | 48.586s | 30.934s |
| `px4_takeoff_instability_detection` | 87.801s | 50.193s | 20.284s |

PX4 scenario timing inside `test_body`:

| Stage | Positive | Instability |
| --- | ---: | ---: |
| `ground_settle` | 1.768s | 1.743s |
| `gcs_heartbeat_wait` | 1.456s | 0.559s |
| `px4_prearm_settle` | 10.279s | 10.199s |
| `arm_command_ack` | 3.080s | 2.569s |
| `takeoff_hover_observation` | 14.131s | 5.027s |

## Step 1: Persist Host-Level Integration Timing

### Goal

Make future speed investigations complete by preserving `[integration-timing]`
lines in the per-run artifact directory, not only in the terminal output.

### Rationale

The Isaac logs now contain detailed standalone timings, but the host-level
Make/script timings for Docker startup, pip install, PX4 readiness, log
collection, and cleanup are printed to stdout and are not stored in the
artifact logs. That prevents precise analysis of container overhead after the
fact.

### Implementation Plan

- Add a host timing log path in both integration runners:
  - For `RUN_ISAAC_INTEGRATION_TEST` in `make/integration-tests.mk`, create
    `$$run_root/integration-timing.log`.
  - For `make/run_px4_integration_test.sh`, create
    `${ISAAC_RUN_ROOT}/../integration-timing.log` or an explicit
    `HOST_TIMING_LOG` environment variable from the Make target.
- Introduce a shell helper that prints each `[integration-timing]` line to both
  stdout and the timing log.
  - In `make/run_px4_integration_test.sh`, update `integration_timing()`.
  - In `RUN_ISAAC_INTEGRATION_TEST`, replace the inline `printf` timing helper
    with the same behavior inside the container command and around the host
    container stage.
- Preserve the current stdout format exactly:
  - `[integration-timing] target=... stage=... elapsed_s=... total_s=... status=...`
- Ensure timing log creation happens before any stage that can fail, including
  `prepare_logs` and Docker startup.
- Document the new artifact in `AGENTS.md` under integration logs.

### Acceptance Criteria

- Every integration run directory contains `integration-timing.log`.
- The file includes host stages and inner container stages:
  - Isaac-only targets: `pip_install`, `test_script`, `failure_scan`,
    `isaac_container`.
  - PX4 targets: `prepare_logs`, `remove_existing_px4_container`,
    `start_px4_container`, `px4_readiness_wait`, `pip_install`,
    `test_script`, `failure_scan`, `isaac_container`, `collect_logs`,
    `remove_px4_container`, `total`.
- Existing terminal output remains unchanged.

### Test Plan

- `bash -n make/run_px4_integration_test.sh`
- `make -n test-integration-isaac`
- `make -n test-integration-drone-articulation`
- `make -n test-integration-px4-takeoff`
- `make -n test-integration-px4-takeoff-instability`
- Run one cheap or targeted integration test if Docker/GPU is available, then
  verify `integration-timing.log` exists and contains timing lines.

### Risk

Low. This is diagnostic-only if the stdout format is preserved and failures do
not depend on writing the timing log.

## Step 2: Add a Minimal USD Scene for Integration Tests

### Goal

Reduce `stage_open` time for integration tests that do not need the full
`LightSwitchWarehouse` scene.

### Rationale

`sim_settings` spends 58.085s opening the warehouse stage and then 0.114s in
the actual test body. `drone_articulation_io` spends 53.772s opening the same
stage and then 0.686s in the test body. The PX4 takeoff scenarios also validate
flight behavior, not warehouse interaction. The current integration tests need
a valid stage, physics scene behavior, and a ground collider for the configured
drone spawn, but they do not require warehouse geometry, lights, pusher
mechanisms, materials, or imported environment assets.

### Implementation Plan

- Add a minimal ASCII USD stage under `isaac/assets/`:
  - `isaac/assets/TestScenes/MinimalIntegration.usda`
- The scene should include only:
  - `/World` default prim.
  - `/World/physicsScene`.
  - `/World/GroundPlane` as a static collision cube with its top surface at
    `z=0`.
  - No warehouse geometry, lights, pusher mechanisms, materials, or imported
    environment assets.
- Update the global integration scene default in `make/config.mk`:
  - From `@isaac/assets/LightSwitchWarehouse/Warehouse.usd`.
  - To `@isaac/assets/TestScenes/MinimalIntegration.usda`.
- Keep existing `--scene` and `SCENE=...` override behavior unchanged, so any
  target can still run against the warehouse manually.
- Do not move scene selection into Hydra in this step. `_runtime.py` opens the
  stage before project Hydra config is loaded, so scene selection remains a
  test harness concern until that startup order is redesigned.
- Update `isaac/runtime/standalone/tests/README.md` with which tests use the
  minimal scene and why.
- Update `AGENTS.md` so the default minimal scene and manual warehouse override
  are part of the project workflow instructions.

### Acceptance Criteria

- All current integration targets default to the minimal scene through
  `make/config.mk`.
- `SCENE=@isaac/assets/LightSwitchWarehouse/Warehouse.usd` still works as a
  manual override.
- `sim_settings` still verifies configured physics settings and whether an
  existing physics scene is reused.
- `drone_articulation_io` still spawns the configured drone, initializes real
  `IsaacArticulationIO`, reads finite pose/velocity/angular velocity, and
  applies rotor forces.
- The PX4 takeoff scenarios still use the same Make/CLI scene override
  mechanism as the Isaac-only tests.
- The minimal scene has a valid ground collider at `z=0` for stationary drone
  spawn.
- `stage_open` drops materially from the current warehouse baseline.

### Test Plan

- `uv run ruff check isaac/runtime/standalone/tests/`
- `uv run ruff format --check isaac/runtime/standalone/tests/`
- `make -n test-integration-isaac`
- `make -n test-integration-drone-articulation`
- `make -n test-integration-px4-takeoff`
- `make -n test-integration-px4-takeoff-instability`
- `make -n test-integration-all`
- If Docker/GPU is available:
  - `make test-integration-isaac`
  - `make test-integration-drone-articulation`
  - `make test-integration-px4-takeoff`
  - `make test-integration-px4-takeoff-instability`
- Compare new `[isaac-standalone-test] TIMING ... stage=stage_open` values
  against the current warehouse baseline.
- Confirm each run still writes `integration-timing.log`.

### Risk

Medium. Isaac/USD authoring details may make a minimal scene subtly different
from the production warehouse scene. Keep `SCENE=...` overrides available for
manual warehouse validation.

## Future Step: Create a Hydra Integration Profile

### Goal

Add deterministic integration-test runtime behavior through Hydra overrides,
without moving scene selection into Hydra.

### Rationale

The minimal scene is selected before Hydra config is loaded. Hydra should still
be useful later for runtime determinism once the exact test overrides are
chosen, but it should not own the stage path unless a future refactor
introduces a typed runtime/test harness config that is loaded before stage open.

### Candidate Overrides

- `drone.spawn_volume_center=[0.0,0.0,0.0]` or another fixed spawn center
  aligned with the ground plane.
- `drone.spawn_volume_half_extents=[0.0,0.0,0.0]` to remove random X/Y/Z spawn
  spread.
- Consider a deterministic yaw path if needed. The current config has no yaw
  field, so do not add one unless test flakiness requires it.
- Consider `px4.sensor_noise_enabled=false` for repeatable PX4 tests, with an
  explicit note that this trades realism for determinism.
- Keep `sim.physics_hz`, PX4 lockstep timeouts, rotor constants, thrust model,
  and safety gating at production defaults.

### Implementation Plan

- Create a Hydra integration profile after deciding the exact override set.
- Wire the profile through `EXPERIMENT_CONFIG_OVERRIDES` in integration Make
  targets.
- Keep scene path out of Hydra unless startup order is refactored so a typed
  runtime/test harness config is loaded before stage open.

## Step 3: Run Isaac-Only Smoke Tests in One Isaac Process

### Goal

Avoid paying `SimulationApp` startup and stage loading once per Isaac-only smoke
test.

### Rationale

The latest `sim_settings` and `drone_articulation_io` runs each spend about
145s before their sub-second test bodies complete. Combining them into one
standalone script can remove one full Isaac startup and one full stage load from
the Isaac-only portion of the suite.

### Implementation Plan

- Add a combined standalone test script under
  `isaac/runtime/standalone/tests/`, for example:
  - `test_isaac_smoke_suite.py`
- Reuse `run_scene_test()` once.
- Refactor current test body functions so they can be called from both:
  - Their existing individual scripts.
  - The combined smoke suite.
- Keep each logical test report label distinct:
  - `sim_settings`
  - `drone_articulation_io`
- In the combined script:
  - Load the scene once.
  - Run sim settings assertions first.
  - Run drone articulation assertions second.
  - Ensure drone cleanup happens even if the second body fails.
- Add a new Make target, for example:
  - `test-integration-isaac-smoke`
- Decide whether `test-integration-all` should use the combined smoke target
  instead of separate `test-integration-isaac` and
  `test-integration-drone-articulation`.
  - Default recommendation: keep individual targets available, but use the
    combined target in `INTEGRATION_TEST_TARGETS`.
- Update `AGENTS.md` and `isaac/runtime/standalone/tests/README.md`.

### Acceptance Criteria

- Individual targets still work for focused debugging.
- `test-integration-all` runs one combined Isaac-only smoke target instead of
  two separate Isaac-only startup cycles.
- The combined target emits per-logical-test timing lines so regressions are
  still attributable.
- A failure in one logical test is reported clearly with its logical test name.

### Test Plan

- `uv run ruff check isaac/runtime/standalone/tests/`
- `uv run ruff format --check isaac/runtime/standalone/tests/`
- `make -n test-integration-isaac-smoke`
- `make -n test-integration-all`
- If Docker/GPU is available:
  - `make test-integration-isaac-smoke`
  - Compare total time against the sum of current `test-integration-isaac` and
    `test-integration-drone-articulation`.

### Risk

Medium. Running tests in one Isaac process can create state leakage. The suite
must explicitly reset or clean up any stage, timeline, config, and drone state
between logical test bodies.

## Step 4: Make PX4 Pre-Arm Settle Readiness-Driven

### Goal

Replace the fixed 10s PX4 pre-arm settle with a readiness condition plus a
short minimum settle window.

### Rationale

Both PX4 tests spend about 10.2s in `px4_prearm_settle`. This is a deliberate
stability wait, but the logs show it is one of the largest remaining costs
inside the PX4 test body. A readiness-driven gate can reduce best-case runtime
while keeping the old timeout as a fallback.

### Implementation Plan

- Add test-local constants in `_px4_takeoff_scenario.py`:
  - `_PREARM_MIN_SETTLE_S`, default initially 3.0s.
  - `_PREARM_MAX_SETTLE_S`, default initially 10.0s.
- Change `_settle_px4_prearm()` to loop until both are true:
  - Minimum simulated settle time has elapsed.
  - PX4 appears command-ready.
- Use available MAVLink signals already drained during settle:
  - Heartbeat remains non-failsafe.
  - No failsafe status is observed.
  - Candidate readiness message, if available in drained messages, indicates
    navigation/estimator/local-position readiness.
- If no reliable readiness message is available from the current stream, do not
  guess. Keep the fixed 10s behavior and add logging that readiness evidence is
  missing. In that case, the implementation step should stop after adding
  better diagnostic logging rather than shortening the wait.
- Emit a timing/reason log when settle exits:
  - elapsed simulated time
  - readiness reason
  - whether it exited on readiness or max settle
- Keep `_PREARM_MAX_SETTLE_S=10.0` until several local runs prove shorter
  settle is stable.

### Acceptance Criteria

- PX4 tests still pass repeatedly with no increase in failsafe flakes.
- Logs show why pre-arm settle ended.
- Best-case `px4_prearm_settle` falls below 10s only when readiness evidence is
  present.
- If readiness evidence is not available, behavior remains equivalent to the
  current fixed 10s settle.

### Test Plan

- `uv run ruff check isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`
- `uv run ruff format --check isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`
- `make -n test-integration-px4-takeoff`
- `make -n test-integration-px4-takeoff-instability`
- If Docker/GPU is available, run at least three repetitions:
  - `make test-integration-px4-takeoff`
  - `make test-integration-px4-takeoff-instability`
- Compare `px4_prearm_settle`, `arm_command_ack`, and flake/failsafe outcomes
  against the current baseline.

### Risk

High if readiness is guessed. PX4 startup/estimator readiness is load
dependent, and this wait probably hides real initialization variance.

## Step 5: Split Positive PX4 Takeoff Into Fast and Full Stability Modes

### Goal

Keep a fast integration target for command-flow and climb regression detection,
while retaining the full hover-quality target for deeper validation.

### Rationale

The positive PX4 scenario spends 14.131s in `takeoff_hover_observation`.
Approximately 7s of that is deliberate settle plus hover observation after the
vehicle reaches the takeoff envelope. This is useful, but not every developer
iteration needs the full hover-quality window.

### Implementation Plan

- Add an option to `Px4TakeoffScenarioOptions` for observation profile:
  - `observation_profile: Literal["fast", "full"] = "full"`
- Implement profile-specific constants in `_px4_takeoff_scenario.py`:
  - Full: current behavior.
  - Fast: stop shortly after reaching the takeoff altitude envelope and
    checking immediate stability for a short window.
- Recommended initial fast values:
  - Hover settle: 0.5s.
  - Hover observation: 1.0s.
  - Keep the same tilt, lateral drift, vertical speed, and angular speed guards.
- Add a new standalone entrypoint or Make target:
  - `test-integration-px4-takeoff-fast`
- Keep existing `test-integration-px4-takeoff` as the full profile until there
  is enough confidence to change defaults.
- Do not weaken the negative instability target in this step.

### Acceptance Criteria

- Full positive PX4 target is behaviorally unchanged.
- Fast positive PX4 target passes and reports its profile in logs.
- Fast target reduces `takeoff_hover_observation` materially.
- The fast target still fails on gross instability, tilt, drift, or vertical
  speed divergence.

### Test Plan

- `uv run ruff check isaac/runtime/standalone/tests/ make/config.mk make/integration-tests.mk`
- `uv run ruff format --check isaac/runtime/standalone/tests/`
- `bash -n make/run_px4_integration_test.sh`
- `make -n test-integration-px4-takeoff-fast`
- If Docker/GPU is available:
  - `make test-integration-px4-takeoff-fast`
  - `make test-integration-px4-takeoff`
- Compare `takeoff_hover_observation` and final stability metrics.

### Risk

Medium. A fast profile may miss slow hover regressions. Keep the full profile
available and do not replace it until the desired test matrix is explicit.

## Step 6: Evaluate Reusing Warm Isaac/PX4 Containers

### Goal

Explore whether persistent test containers can avoid repeated startup costs for
local iteration.

### Rationale

Per-test transient containers give clean state and good artifact isolation, but
they force every target through Isaac startup and stage load. A local-only
developer target that reuses a warm Isaac process or container could be much
faster for repeated debugging.

### Implementation Plan

- Add this only as an opt-in developer workflow, not as the default integration
  target.
- Create a design spike first:
  - Determine whether standalone Isaac scripts can be run in a persistent
    container one after another without stale stage/timeline state.
  - Determine whether PX4 EEPROM/log isolation can be reset reliably between
    runs.
  - Measure first-run and subsequent-run timings.
- If feasible, add explicit Make targets such as:
  - `make run-isaac-test-shell`
  - `make test-integration-isaac-warm`
- Keep transient targets as the correctness baseline.
- Document that warm targets are for local iteration only.

### Acceptance Criteria

- Warm workflow never replaces the transient default targets.
- State reset is explicit and logged.
- A stale previous failure cannot make a later test pass.
- Speedup is measured and documented before keeping the workflow.

### Test Plan

- Run the warm target twice in a row and compare first vs second timings.
- Intentionally fail one logical test, then run a passing test and confirm
  artifacts and status are isolated.
- Run the corresponding transient target to confirm behavior matches.

### Risk

High. Persistent simulator processes can leak USD, timeline, physics tensor,
MAVLink, and config state. Treat this as a local-iteration optimization only.

## Recommended Order

1. Persist host-level timing. This improves all later analysis and has low risk.
2. Add a minimal USD scene for integration tests. This attacks the largest
   low-risk scene-loading cost while preserving `SCENE=...` overrides.
3. Add a Hydra integration profile for deterministic runtime overrides after
   deciding the exact override set.
4. Combine Isaac-only smoke tests in one Isaac process. This removes a full
   startup cycle from `test-integration-all`.
5. Investigate PX4 pre-arm readiness. This can save about 10s per PX4 test, but
   only if readiness can be proven from logs/messages.
6. Add a fast PX4 takeoff profile for local iteration while preserving the full
   profile.
7. Consider warm containers only after the transient path is optimized.

## Current Baseline To Compare Against

Use these values as the first comparison point after each speedup step:

- `sim_settings`: total through test body 149.306s.
- `drone_articulation_io`: total through test body 145.043s.
- `px4_takeoff_stability`: total through test body 168.169s,
  scenario body 30.928s.
- `px4_takeoff_instability_detection`: total through test body 158.336s,
  scenario body 20.284s.

The largest expected wins are:

- Minimal smoke-test scene: potentially tens of seconds per Isaac-only target.
- Combined Isaac-only smoke target: roughly one avoided Isaac startup and stage
  open cycle.
- PX4 readiness-driven pre-arm settle: up to about 10s per PX4 target if safe.
- Fast positive PX4 profile: several seconds for local iteration only.
