# Integration-test architecture

This is the architecture and rationale for the container-backed integration
tests. Use the Make targets as the public entrypoints; run harness scripts
directly only when debugging the harness itself. The orchestration, caching,
artifact-isolation, and timing details live here. The authoritative definitions
are `make/integration-tests.mk`, `make/config.mk`,
`make/scripts/run_isaac_integration_test.sh`,
`make/scripts/run_px4_integration_test.sh`,
`make/scripts/run_isaac_test_in_container.sh`,
`make/scripts/run_integration_suite.sh`, and
`make/scripts/warm_isaac_cache.sh` -- if this doc disagrees with them, those
files win and this doc should be corrected.

The standalone Isaac test scripts these targets run live under
`isaac/runtime/standalone/tests/`; see that directory's `README.md` for
scenario-specific notes.

## Targets

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
  for the final 1.0 s validation windows. It is the one target that runs on the
  Isaac Lab image (`ISAAC_LAB_IMAGE`, the Dockerfile `isaac-lab` stage layered
  `FROM isaac-sim`) instead of the default `isaac-sim` image (`ISAAC_IMAGE`),
  because it imports `isaaclab`. The target sets the override with a
  target-specific `ISAAC_IMAGE = $(ISAAC_LAB_IMAGE)`;
  `ISAAC_INTEGRATION_COMMON_ENV` expands it at recipe time, so the rest of the
  runner is unchanged. Build the image first with `make build-isaac-lab` (or
  `make build`).
- `make test-integration-press-contact-chain` runs the F3 deterministic
  real-asset gate on the Isaac Lab image with one environment. It spawns the
  paired switch/lamp, drives a slow pusher press, requires real pad contact and
  pusher/button displacement, advances the production N-step latch, verifies
  exactly one lamp transition, then retreats within the stability envelope.
  It is part of `test-integration-all` and requires no PX4 container or trained
  policy.
- `make test-integration-press-policy` runs the F2b trained-policy gate on the
  Isaac Lab image. It needs a prior `make train-rl`, so it is deliberately not
  part of `test-integration-all`; see `docs/RL_WORKFLOW.md` for the policy
  artifact contract and routine config. Policy metadata includes the active
  policy rate, so artifacts trained before that field was introduced must be
  retrained.
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
  `none_hexa_x`/`22001`). It protects OFFBOARD setpoint and transmitter-phase
  continuity across `goto` -> `goto` and `goto` -> `hover` transitions and
  passes only when **both** drones hold their
  envelope. It reuses the generic multi-PX4 launch path
  (`run_px4_integration_test.sh` over the enabled vehicles returned by
  `load_config(path)`, like
  the multi-drone takeoff) and the shared `_px4_navigation_scenario.py` driver,
  whose
  `_assert_navigation_routine_config` validates the `goto`/`goto -> hover`
  config per flying vehicle before running the strict takeoff scenario.

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
non-contiguous ids keep their derived ports. The first instance in that list owns
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
