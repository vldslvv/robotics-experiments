# AGENTS.md - Robotics stack (Isaac Sim + ROS 2 + PX4)

> This file is the single source of project instructions. `CLAUDE.md` is a
> symlink to `AGENTS.md`, so never edit `CLAUDE.md` directly (and do not try to
> "sync" the two) -- always make changes here in `AGENTS.md` and they apply to
> both.

## Project goal
This repository integrates:
- Isaac Sim (simulation)
- ROS 2 (control, perception)
- PX4 SITL (flight controller simulation)
- Docker-based development

Prefer minimal, scoped changes.
Do not introduce new tooling unless explicitly requested.

## Repository layout
Place new code in the directory matching its purpose; each entry below is both
the current contents and the intended home for that kind of code.
- `src/robotics_experiments/` -> Core Python package for reusable experiment logic.
- `isaac/isaac-sim/` -> Isaac Sim cache and config, populated automatically by container. Persists changes made inside Isaac Sim container.
- `isaac/docker/` -> Isaac Sim Docker build/init inputs.
- `isaac/runtime/extensions/` -> Isaac Kit extension workflow.
- `isaac/runtime/templates/` -> Copyable Isaac runtime templates, not loaded directly.
- `isaac/runtime/standalone/` -> Direct Isaac Sim scripts and launch helpers.
- `isaac/runtime/lab/` -> Isaac Lab configs, task definitions, and future environment code.
- `ros/ros2_ws/` -> ROS 2 workspace (colcon). Persists changes made inside ROS 2 container.
- `ros/config/` -> ROS middleware config mounted into Isaac Sim and ROS 2 containers.
- `ros/docker/` -> ROS 2 Docker build/init area.
- `ros/scripts/` -> ROS 2 host-side helper scripts.
- `px4/docker/` -> PX4 Docker build/init area.
- `px4/scripts/` -> PX4 host-side helper scripts.
- `px4/logs/` -> PX4 SITL log persistence.
- `px4/eeprom/` -> PX4 SITL simulated EEPROM mount; cleared by the default PX4 launch wrapper.
- `docs/` -> Design and usage docs.

## Python API organization
- Public top-level functions, classes, and intentional constants in Python modules must be
  listed in module-level `__all__`.
- Keep private top-level helper functions before public top-level APIs when possible, so
  declarations read from implementation helpers into public entrypoints.
- Preserve lazy imports where they prevent simulator, ROS, PX4, or Pegasus dependencies from
  loading during lightweight package imports.
- Package layering runs `config -> domain (drone / environment / simulation /
  px4_bridge) -> Isaac extensions`, and `import robotics_experiments` (plus its
  subpackages) must succeed with Isaac absent. A domain leaf module that is
  wholly Isaac-only (every function operates on a live USD stage) may keep
  top-level `carb`/`pxr`/`isaacsim` imports rather than scattering function-scope
  imports, but it must (a) carry a module docstring declaring the exception and
  (b) stay off the core import path, shielded by a lazy package `__init__`
  (re-export-only, or `__getattr__` deferral). `drone/spawn.py`,
  `environment/stage.py`, `environment/step.py`, and
  `simulation/_isaac_settings.py` follow this pattern;
  `tests/test_core_imports.py` is the regression guard (it blocks
  `carb`/`pxr`/`isaacsim` and asserts the leaf modules are not eager-imported).
- Do not rename, remove, or broaden public exports as part of cleanup unless explicitly requested.
- `VehicleState` carries pose, velocity, angular velocity, and sim time only.
  Do not add a kinematic acceleration field for the PX4 HIL path; the Pegasus
  IMU derives acceleration from velocity deltas internally.

## Isaac Sim extension design

- Keep internal logic and Isaac Sim event wiring separate.
- Treat `on_startup`, `on_shutdown`, and Isaac/Kit event callbacks as wiring.
  They should subscribe, unsubscribe, decode events, update direct runtime
  handles such as the current stage, and delegate meaningful behavior.
- Keep event callbacks small. They should route to domain actions such as
  `setup_drone`, `teardown_drone`, or equivalent names.
- Domain action methods should own complete lifecycle behavior for the feature,
  including related prims, bridge hosts, ROS nodes, callbacks, publishers, and
  subscriptions that must start and stop together.
- Prefer larger meaningful helpers over chains of tiny one-use helpers. If a
  helper is used once and only forwards to another helper, merge it into the
  caller.
- Avoid overgeneralized instance members for wiring objects. Use local variables
  for Isaac/Kit context objects unless their lifetime must be retained, and store
  direct feature state in private attributes such as `self._stage` instead of
  broad handles such as `self._usd_context`.
- Use `drone_spawner` as the extension pattern: wire Isaac/Kit events separately
  from meaningful actions, and let setup/teardown actions own related prim and
  bridge lifecycle together.
- `robotics_experiments_core` exposes the shared `src/robotics_experiments`
  package to Isaac Sim extensions. Extension code that needs the core package
  should depend on `robotics_experiments_core` instead of manipulating
  `sys.path`.
- Extension config comes from `robotics_experiments.config`, which composes the
  Hydra config lazily and caches it for the process.
- Feature extensions should depend on `robotics_experiments_core` and use a
  small local `config.py` adapter when they need feature-specific config access.
- Importable Isaac Lab task/env code lives under
  `src/robotics_experiments/workflows/isaac_lab/`; `isaac/runtime/lab/` is the
  workflow/design-note area for Isaac Lab runtime files. Keep package
  `__init__` modules lazy-import clean: expose guarded registration helpers
  there, but do not import DirectRLEnv task modules unless Isaac Lab is actually
  constructing the task.
- Isaac Lab task IDs use the Gymnasium `Isaac-...-Direct-v0` naming convention
  for direct-workflow environments. The first scaffold task is
  `Isaac-Drone-Hover-Direct-v0`; it is intentionally a no-op DirectRLEnv
  lifecycle skeleton until a later step defines real thrust dynamics/rewards.
  Isaac Lab workflow tunables live in the Hydra `isaac_lab` group with matching
  bare annotations in `src/robotics_experiments/config/types.py`.
- PX4 bridge behavior is configured through the Hydra `px4` group. The default
  experiment enables the bridge for the custom Isaac runtime; use Hydra
  overrides such as `px4.enabled=false` for Isaac sessions that should not
  connect to PX4 SITL. Lockstep tunables live in the nested `px4.lockstep`
  group (override e.g. `px4.lockstep.lockstep_timeout_s=...`), mapped to the
  library `LockstepSettings` dataclass at construction; `physics_rate_hz` is not
  a YAML knob there but injected from `sim.physics_hz`.
- The PX4 bridge's full interface contract lives in
  `isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md` -- read
  it before changing bridge code. It is the single source for the four-part
  contract (`VehicleState` -> `ActuatorCommands` -> `RotorDynamics` plus the
  `BridgeStatus` snapshot, the `LockstepSettings` object, the `send(state)` /
  `poll()` lockstep seam with `exchange(state)` as the single-vehicle
  send-then-poll convenience, and the `ArticulationIO.apply_dynamics` Isaac
  sink), the frame conventions, motor-index ordering, the lockstep algorithm, and
  the cadence/actuator-freshness guards. Load-bearing invariants you must not
  break (all detailed there): `HIL_SENSOR.time_usec` strictly increasing from sim
  time; a vehicle's `send` (sensors out, no block) strictly precedes its `poll`
  (blocking recv), and the multi-vehicle coordinator runs all sends before all
  polls; angular
  velocity is rotated world -> body-FLU before the FRD gyro conversion; sensor
  cadence must equal physics cadence (else disconnect "sensor cadence mismatch");
  prime-then-block startup plus newest-packet draining keep actuators fresh
  (backlog trips `max_sustained_actuator_lag_s`); safety gating zeros forces only,
  never the per-tick exchange; a non-finite vehicle state raises. Visual rotor
  spin is the one explicitly non-load-bearing addition: `RotorDynamics` carries an
  additive `rotor_omega_rad_s` (the clipped per-rotor omega the thrust model
  already computes) and `IsaacArticulationIO.update_rotor_spin(rotor_omega_rad_s,
  dt)` is a cosmetic-only seam called from `recv_and_apply` after the unchanged
  `apply_dynamics` (with `dt` stashed by `send_sensors`). It only writes a
  `RotateZ` xformOp on each rotor marker (scaled by `px4.rotor_spin_visual_scale`)
  and must never re-enter forces, the body, or the lockstep -- keep
  `commands_to_dynamics` pure and `apply_dynamics` the sole physics sink.
- Multi-vehicle scenes are driven by a top-level `vehicles:` list. It is the
  single source of truth and the only spawn path: one vehicle = a one-element
  list (the default config ships exactly that, reproducing the single-drone
  behavior), and an empty list `vehicles: []` is a valid empty fleet (0..N is
  handled uniformly -- the extension spawns nothing, runs an empty
  `BridgeHostGroup`, and `run-px4` starts zero containers). Each entry has an
  `id`, a `spawn_xy`, and partial `drone`/`px4` overrides. `id` is the **sole
  authored identity**: there is no `px4.vehicle_id` knob -- the MAVLink port, the
  PX4 `-i` instance index, the per-vehicle log `{vehicle}` token, and the GCS
  link offset are all *derived* from `id`. One pure enrichment step
  (`config.load.resolve_fleet`) merges each entry's `drone`/`px4` partials over
  the global `drone`/`px4` defaults (`OmegaConf.merge`) into a pure
  `VehicleConfig`, then constructs an immutable `ResolvedVehicle` carrying that
  config plus the derived runtime fields: `mavlink_port`
  (`= px4.connection_baseport + id`), `px4_instance_index` (`= id`),
  `gcs_port` (`= px4.gcs_baseport + id`, the per-drone GCS MAVLink link
  used by control/test code; the bridge itself does not use it),
  `debug_log_path` (the `{vehicle}` token rendered to `id`; `{date}`/`{time}`
  stay for runtime), and `prim_name`. Every consumer (Isaac orchestration, the
  bridge host, the host-side PX4 launch, the standalone scenario's GCS endpoint)
  reads these derived fields instead of recomputing `baseport + id`. `resolve_fleet` validates the resolved fleet
  (`_validate_fleet` -- duplicate id/endpoint/`prim_name`, bad `spawn_xy`, or
  colliding `debug_log_path` raise a `ValueError` at resolution time).
  `get_config().vehicles` is `list[ResolvedVehicle]`; the host-side launch
  scripts read `get_enabled_px4_vehicle_count()` /
  `get_enabled_px4_instance_indices()` (the latter is the launch loop's source of
  `PX4_INSTANCE`, so non-contiguous ids keep their derived ports).
  The drone-spawner extension spawns each
  vehicle at its `spawn_xy` (yaw 0) and gives every `px4.enabled` vehicle its own
  independent `BridgeHost` (constructed from the `ResolvedVehicle`), all driven by
  a single `BridgeHostGroup` that owns the
  one physics-step subscription. Each PhysX step runs in two phases -- every
  host's send leg (`send_phase`: sensors out to its PX4, no block) first, then
  every host's poll leg (`recv_phase`: block for that PX4's actuator). The PX4
  instances therefore compute concurrently, so per-tick latency is ~the max
  blocking wait across vehicles (not the sum) and the per-vehicle lockstep is
  decoupled: one stalled PX4 disconnects only its own vehicle instead of freezing
  the others. The port exposes the seam as `send(state)` / `poll()`, with
  `exchange(state)` kept as the single-vehicle send-then-poll convenience (a
  1-host group is byte-for-byte the old behavior).
  Per-vehicle isolation: the MAVLink simulator port is the resolved
  `mavlink_port` (`Px4MavlinkPort` takes a single resolved `port`, handed to the
  vendored Pegasus backend with a zero `vehicle_id` offset), the USD prim is
  `<default_prim>/<drone.prim_name>`, and bridge
  debug logs carry a `{vehicle}` token so the filenames never collide. The token
  alone is not enough: every bridge layer logs through the shared
  `robotics_experiments.px4_bridge` package logger, so the `BridgeHost` wraps each
  vehicle's construction/teardown and per-tick work in
  `px4_bridge.vehicle_log_context(id)`, and each per-vehicle file handler
  carries a `VehicleLogFilter` (see `px4_bridge/_logging.py`). Without this a
  multi-vehicle process merged every bridge's records into the last-constructed
  vehicle's file. `EnvConfig.vehicles` is annotated `List[Any]` for Hydra
  compose-time tolerance of partial entries; the runtime value is
  `list[ResolvedVehicle]`.

## Commanding (high-level command / routine layer)
- `src/robotics_experiments/commanding/` is the high-level control layer: the
  pure-function counterpart to the PX4 bridge contract. It is a domain leaf on
  the core import path -- every module imports neither Isaac nor pymavlink at
  module scope (the concrete `MavlinkCommandPort` imports `pymavlink` only inside
  its methods), so `import robotics_experiments` succeeds with both absent, and
  `commanding/__init__.py` is re-export-only. The full reference is
  `docs/COMMAND_ROUTINE_CONTRACT.md`; read it before changing this layer.
- The contract mirrors the bridge part-for-part: `RoutineObservation` (pure
  input, reusing the bridge's `VehicleState` / `BridgeStatus` plus `last_ack` /
  `failsafe` / `sim_time_us`) -> `decide(routine, state, observation)` (the pure
  stepper, analogue of `commands_to_dynamics`) -> `RoutineDecision`
  (`SEND`/`WAIT`/`ADVANCE`/`DONE`/`FAIL`, pure output) across the
  `AbstractCommandPort` seam (analogue of `AbstractFlightStackBridge`), with
  `VehicleCommander` and the pure per-drone `RoutineDriver` as the orchestrators
  (analogue of `BridgeHost`). The command layer is **per drone** like the bridge:
  one commander + one driver per drone, no fleet object, drones independent
  (cross-drone sync is deferred). `MavlinkCommand` is a thin `command_long`
  wrapper (a `MAV_CMD_*` id + seven params); `MAV_CMD_*` / `MAV_RESULT_*` /
  `PX4_FORCE_ARM_MAGIC` are named int constants so the pure modules never import
  pymavlink. `tests/test_commanding_routine.py` (the `decide` transitions + a
  `VehicleCommander`) and `tests/test_commanding_driver.py` (the `RoutineDriver`
  phase machine over fake seams) are the pure regression guards, over the fakes in
  `tests/_fakes.py`; `tests/test_core_imports.py` asserts the subpackage (incl.
  `RoutineDriver`) imports with Isaac/pymavlink absent.
- Pure stepper rule: `decide` and every completion predicate
  (`armed()`, `ack_accepted(command_id)`, `climbed_at_least(...)`) are pure
  functions -- no sockets, no pymavlink, no Isaac, unit-testable with plain
  dataclasses. The only side effect in the layer is the port
  (`send` / `connect` / `pump_heartbeat` / `poll_feedback`); `VehicleCommander`
  runs the pure `decide` and then performs the single `SEND` side effect, keeping
  decision and I/O separated. Step `timeout_s` is measured in **sim time**
  (`observation.sim_time_us`) so the stepper stays pure; the driver pairs it with
  a wall-clock backstop.
- `gcs_port` is control, not bridge: the `MavlinkCommandPort` talks the per-drone
  GCS link (`ResolvedVehicle.gcs_port`) only and never touches the bridge's
  simulator MAVLink port or its sensor/actuator lockstep -- the symmetric half of
  the bridge contract's rule that the bridge never touches `gcs_port`. The
  commander is stepped from the control driver (today the standalone scenario run
  loop), never from the bridge's physics send/recv legs.
- The routine carries its own verification, so a step's completion can require a
  physical state and a step can carry continuous flight-envelope gates. A
  `RoutineStep` has an **optional** `command` (`None` = a verification-only step
  like `hover`), a completion predicate `StepPredicate(observation, baseline)`, a
  sim-time `timeout_s`, an optional `hold_s` (the predicate must hold continuously
  for this long before advancing -- "hover stable for N seconds"), and a tuple of
  `invariants` (continuous gates). `decide` checks `invariants` first (a breach ->
  `FAIL(FailureKind.ENVELOPE)`), then completion+hold (-> `ADVANCE`), then
  ack-rejection/timeout (-> `FAIL(FailureKind.COMMAND)`). Completion helpers:
  `armed`, `ack_accepted`, `climbed_at_least(min_climb_m)` (vs the captured
  baseline), `hovering_stable(...)`; invariant helpers (breach message or `None`):
  `tilt_below`, `lateral_drift_below`, `vertical_speed_within`,
  `angular_speed_below`; `EnvelopeGates` bundles the thresholds.
- Where the executor lives: the pure per-drone `RoutineDriver`
  (`commanding/driver.py`) is the single executor -- a tick-driven phase machine
  (`WAIT_BRIDGE -> CONNECT -> HANDSHAKE -> PREARM_SETTLE -> FETCH_TARGET -> BUILD
  -> DRIVE -> DONE`) over one drone's routine, one `AbstractCommandPort`, and a
  `DriverEnvironment` seam (bridge connectivity, sim time, observation assembly,
  `log`, `on_failure`, `on_step_advanced`; wall-clock backstop via an injectable
  `monotonic`). It imports no Isaac/pymavlink and is unit-tested with fakes
  (`tests/test_commanding_driver.py`). Each workflow is a **thin Isaac adapter**
  that builds one driver per flying drone (only where the resolved
  `routine.steps` is non-empty), wires the env, and ticks the drivers; the phase
  machine and flight verification live in the driver/routine, so the adapters
  differ only in tick source and the env's `on_failure` reaction.
  - Live `drone_spawner` extension
    (`drone_spawner/DroneSpawner_python/control_driver.py`, `FleetControlDriver` +
    `_LiveDriverEnv`): the runtime path for `make run-all` / `make run-isaac`. It
    ticks each drone's `RoutineDriver` from one **Kit app-update** subscription
    (separate from the bridge's physics-step subscription, never its send/recv
    legs), reading state through a state-only `IsaacArticulationIO` and
    `BridgeHost.status` over each drone's `MavlinkCommandPort`. Always
    **forgiving**: `on_failure` (a `FAIL`, a PX4 failsafe, or a phase timeout)
    logs a warning and that drone's driver idles (never raises into the sim);
    after `DONE` it keeps the GCS link warm so PX4-console control returns. Built
    in `_setup_drone` (after `BridgeHostGroup.start()`), stopped in
    `_teardown_drone`.
  - Standalone scenario
    (`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py` +
    `_StandaloneDriverEnv`): a blocking runner that spawns the fleet, starts the
    bridge group, does the runner-level infra checks (ground settle, bridge first
    heartbeat, the pre-arm start-tilt check), builds one `RoutineDriver` per flying
    drone, and ticks them in a `while not all(is_done): simulation_app.update();
    for d: d.tick()` loop, then runs the strict bridge-log scan. The env is
    **strict** under `--strict` (the integration test, set by `STRICT=1` in the
    Make target): `on_failure` raises `ExpectedInstabilityDetected` for a
    `FailureKind.ENVELOPE` breach, `Px4FailsafeDetected` for a failsafe (so the
    negative test passes only on an envelope breach, never a failsafe flake), and
    `AssertionError` otherwise; without `--strict` it logs and the driver idles,
    like the live path. The `post_takeoff_hook` (negative-test fault injection) is
    wired to `on_step_advanced` and fires when the takeoff step completes, so the
    hover step's envelope catches the instability. It hand-rolls no MAVLink.
- The routine is config-driven and the config is the single source of truth: the
  Hydra `routine` group (`RoutineConfig`, always composed onto `EnvConfig`)
  carries `steps`, the control-flow tunables (`gcs_heartbeat_period_s`,
  `command_ack_timeout_s`, `bridge_heartbeat_timeout_s`, `handshake_timeout_s`,
  `prearm_settle_s`, `takeoff_target_timeout_s`, `ground_settle_s`,
  `takeoff_timeout_s`, `hover_settle_s`), **and the flight-envelope gates**
  (`max_tilt_deg`, `max_lateral_drift_m`, `max_vertical_speed_m_s`,
  `max_downward_speed_m_s`, `max_angular_speed_rad_s`,
  `hover_max_abs_vertical_speed_m_s`, `hover_max_abs_horizontal_speed_m_s`). There
  is **no** separate `integration_test` group: the integration test is just the
  routine driven with `--strict`. The routine is **per vehicle**: each
  `vehicles[]` entry may carry a `routine` partial, merged over the global
  `routine` exactly like `drone`/`px4` in `config.load.resolve_fleet` (the resolved
  `RoutineConfig` lands on `ResolvedVehicle.routine`). `routine.steps` is the
  **per-drone** on/off switch: a drone whose resolved `steps` is **non-empty**
  (authored as `{kind: arm}` / `{kind: takeoff, climb_m, min_climb_m}` /
  `{kind: hover, hold_s}` mappings) gets its own `RoutineDriver`; an **empty** list
  (the default `configs/experiment/routine/default.yaml`) means that drone is
  manual. Mixed fleets are allowed. There is no separate enable flag. The pure
  `commanding.build_vehicle_routine(steps, *, target_*, command_ack_timeout_s,
  gates, takeoff_timeout_s, hover_settle_s)` maps the descriptors + a per-drone
  takeoff target into a `VehicleRoutine` (unknown `kind`, or a `takeoff`/`hover`
  missing a required field, raises `ValueError`); it stays on the core import path
  and `DriverParams.from_routine_config` adapts a `RoutineConfig` for the driver. The takeoff step completes on a real
  climb (`climbed_at_least(min_climb_m)`) and the hover step on a sustained
  `hovering_stable(...)`, with the gates as continuous invariants on both (the
  historical windowed min/max-range hover gates are now sustained
  instantaneous-rate bounds held for `hold_s`).
  `configs/experiment/routine/takeoff.yaml` is the shared takeoff routine;
  `configs/runtime_arm_takeoff.yaml` runs it headless via
  `make run-all HYDRA_CONFIG=configs/runtime_arm_takeoff.yaml`, and the
  `integration_test_*` configs select the same routine and differ only in
  scene/fleet. The GCS link baseport still lives in the `px4` group
  (`px4.gcs_baseport`); the per-drone `gcs_port` the driver talks is derived from
  it.

## Pegasus vendoring policy
- The Pegasus Simulator under
  `isaac/runtime/extensions/px4_sitl_brigde/pegasus/` is a vendored, stripped,
  unpinned source snapshot (not a PyPI dependency), reduced to the modules the
  PX4 bridge transitively imports and excluded from ruff/mypy via
  `pyproject.toml`. The project's deviations from upstream are load-bearing fixes,
  each with a guarding unit test -- never "sync them away."
- `isaac/runtime/extensions/px4_sitl_brigde/pegasus/PROVENANCE.md` is the single
  source of truth: upstream repo/license and synced commit, the reachable module
  set, the dead-at-runtime modules, the full divergence index (with guarding
  tests), and the re-sync procedure. Read it before touching the vendored tree or
  re-syncing, and update it -- not just code comments -- when the snapshot or a
  divergence changes.

## Python environment
- This project uses `uv` for all host-side Python work. Do not call `pip`,
  `python -m venv`, or a system/global Python directly.
- The project virtualenv lives at `.venv` and is pinned to the Python in
  `.python-version` (3.11). Dependencies and the resolver index (including the
  NVIDIA index for Isaac Sim) are defined in `pyproject.toml`; exact versions
  are locked in `uv.lock`. Do not edit `uv.lock` by hand.
- The project uses a repo-local uv-managed Python install under `.uv-python/`
  so sandboxed tools do not need access to user-level Python installs under
  `/home/vv/.local` or `/home/vv/.pyenv`. `.uv-python/` is local machine state
  and must not be committed.
- Initialize or repair the environment with `uv sync --extra dev`. The `dev`
  extra (ruff, mypy, pre-commit, debugpy) is optional, so a plain `uv sync`
  intentionally omits it; pass `--extra dev` when you need the tooling.
- If `.venv` is missing, points outside the repository, or cannot be read from
  the sandbox, recreate it with the repo-local interpreter:
  `UV_PYTHON_INSTALL_DIR=.uv-python uv python install 3.11`, then
  `UV_PYTHON_INSTALL_DIR=.uv-python uv sync --extra dev --managed-python`.
  After that, normal `uv run ...` and `uv sync ...` commands reuse `.venv`.
- Always work inside the project venv. Either activate it with
  `source .venv/bin/activate`, or prefix commands with `uv run` (for example
  `uv run pytest`, `uv run ruff check`). Prefer `uv run` for one-off commands so
  the correct interpreter is used without relying on shell state.
- Add or change dependencies with `uv add` / `uv remove` (or by editing
  `pyproject.toml` and running `uv lock`), never with ad hoc `pip install`. Keep
  `uv.lock` committed and in sync; `uv lock --check` must pass.
- Host-side Isaac Sim, ROS 2, and PX4 work that does not run inside their Docker
  containers still goes through this venv. The throwaway `uv run --no-project`
  invocations (such as the ULog parser below) are deliberate exceptions for
  one-off tooling that must not touch the project environment.

## Common commands

Run ROS 2, Isaac Sim, and PX4 through the Docker targets below, not directly on
the host. They are meant to be used in the Dockerized environment.

### Docker
- `make build` builds Isaac Sim, ROS 2, and PX4 images.
- `make destroy` stops/removes containers and removes built images.
- `make run-isaac` starts the Isaac Sim container.
- `make run-ros` starts the ROS 2 container.
- `make run-px4` starts one PX4 SITL container per PX4-enabled vehicle.
- `make stop` stops persistent containers without removing them.
- Legacy targets such as `build-all`, `run-all`, and `clean-all` are
  compatibility aliases.

### Isaac Sim
- `make test` runs the host Python unit test suite (pytest, defined in
  `make/tests.mk`). It requires the project venv at `.venv`; create it first with
  `uv sync --extra dev` if missing. The target activates `.venv` and runs
  `python -m pytest`; the direct equivalent is `uv run pytest`. Unit tests live
  under `tests/` and do not need Docker, Isaac Sim, or PX4 running.
- Shared unit-test fakes/helpers live in `tests/_fakes.py` (pytest puts `tests/`
  on `sys.path`, so test modules import them with `from _fakes import ...`).
  Define each fake once there rather than inline per test file. The PX4 bridge
  unit tests are split by concern across `tests/test_px4_bridge_sensors.py`,
  `tests/test_px4_bridge_lockstep.py`, `tests/test_px4_bridge_actuator.py`, and
  `tests/test_px4_bridge_gating.py` (there is no single `test_px4_bridge.py`).
- pre-commit is the local quality gate; there is no CI pipeline (local-only for
  now). Install it once with `uv run pre-commit install` and run it with
  `uv run pre-commit run --all-files`. The mirror hooks (ruff, mypy) only see
  staged files, so for a full sweep run `uv run ruff check` /
  `uv run ruff format --check` / `uv run mypy src tests`. The hook list and the
  per-tool tunables (`.hadolint.yaml`, `.codespellrc`, `.yamllint.yaml`,
  `.taplo.toml`) live in `.pre-commit-config.yaml` and those dotfiles at the repo
  root.
- Keep Make recipes thin. If Makefile logic grows into a complex Bash script
  with substantial control flow, helpers, traps, or heavy `$$` escaping, extract
  it into a separate shell script and let Make prepare variables and call that
  script. Avoid `define`/`eval` target templates when they require nested
  Make/shell escaping such as `$$($(1)_VAR)`; prefer explicit targets with
  ordinary recursive Make variables so command-line overrides stay dynamic.
- `make test-integration` runs container-backed integration tests;
  `make test-integration-all` runs every target (Isaac, drone-articulation, and
  the two PX4 takeoff scenarios) with isolated transient containers and prints an
  aggregate passed/failed summary. Targets are defined in
  `make/integration-tests.mk`; their standalone Isaac scripts live under
  `isaac/runtime/standalone/tests/` (see its `README.md` for scenario notes).
- The integration harness internals -- `INTEGRATION_TEST_JOBS` concurrency
  (default 1, serial; raise it only when the host can sustain more), the
  warm-Isaac-cache pre-stage, shared cache mounts vs per-run artifact isolation
  under `.build/docker/integration/<target>.<run-id>/`, the `SCENE`
  default/override, the structured timing-line formats, and the standalone
  `_runtime.py` ENU pos/vel state logging -- are documented in
  `docs/INTEGRATION_TEST_ARCHITECTURE.md`; read it before changing the harness.
- `make test-integration-px4-takeoff` starts a temporary PX4 SITL container,
  runs the standalone Isaac PX4 arm/takeoff stability scenario, preserves
  isolated artifacts under `.build/docker/integration/<target>.<run-id>/`, and
  removes the temporary PX4 container.
- `make test-integration-px4-takeoff-instability` runs the same scenario with
  a test-local PX4 rotor mapping fault and passes only when the stability
  envelope detects the resulting instability.
- `make test-integration-px4-multi-drone-takeoff` runs the multi-drone takeoff
  scenario from the `integration_test_multi_drone_takeoff` experiment (the target
  sets `HYDRA_CONFIG` itself): it starts one PX4 SITL container per
  resolved `vehicles` entry whose `px4.enabled` is true, in one shared network
  namespace (instance 0 owns it; instance `i>0` joins via
  `--network container:` and runs PX4 `-i i` so it listens on simulator TCP
  `4560 + i` / GCS UDP `14550 + i` with a per-instance rootfs), and passes only
  when *every* drone holds its takeoff/hover envelope.
  `PX4_INSTANCE` is the per-container index threaded into `run_px4_iris_ci.sh` /
  `reset_px4_state.sh` (see `px4/docker/README.md`). The runner injects a
  `px4.debug_log_path` override carrying the `{vehicle}` token so the per-vehicle
  bridge logs never collide (config validation rejects two vehicles resolving to
  the same template). Every instance (including `0`) is launched uniformly by
  `px4_iris_common.sh::launch_px4_iris`: the prebuilt binary run with
  `-i ${PX4_INSTANCE}` and an interactive `pxh>` console (no `-d`), from its own
  per-instance rootfs subdir, with no `instance == 0` special case in the launch,
  reset, or mount logic. A divergent instance-0 launch (`make px4_sitl_default
  none_iris` for instance 0 vs. additional instances started with the binary's
  `-d` "no pxh shell" daemon flag) is exactly how the extra instances silently
  lost their console, so `commander takeoff` could only be entered for PX4[0].
  PX4's `-d` is confirmed against the live image (`px4 -h`: "daemon mode, don't
  start pxh shell"); keep the launch interactive (`-d` omitted) so every
  per-vehicle terminal accepts commands.
- The persistent Isaac container (`make run-isaac` / `make run-all`) is named
  `vv-isaac-sim-custom` (`ISAAC_CONTAINER_NAME`) and is **retained after it
  exits**, so the quickest way to inspect the last manual run is
  `docker logs vv-isaac-sim-custom`. That console stream is also the **only**
  place the shell-level scene-resolution step in `runapp_custom.sh` (the
  `get_config().scene` subprocess that picks the stage) reports its errors -- they
  never reach the Kit log file, so a blank stage with no `open_stage` exec is
  diagnosed here, not in the structured log. Kit's own log is written **inside the
  container** at
  `/root/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/<ver>/kit_<timestamp>.log`
  (container `HOME` is `/root`); it is **not** on the host, because the
  `isaac/isaac-sim/logs` host mount maps to `/isaac-sim/.nvidia-omniverse/logs`, a
  different path than `/root/...`. Copy it out with
  `docker cp 'vv-isaac-sim-custom:/root/.nvidia-omniverse/logs/Kit/' <dest>` when
  you need the structured Kit log rather than the console stream. (The
  `isaac/isaac-sim/logs/test_*.log` files there are integration-test output, which
  redirects explicitly; manual runs do not land there.)
- Every top-level config is an *experiment* (sibling of `config.yaml`), loaded
  identically; there is no system distinction between a runtime experiment and a
  test experiment, and **every** experiment is uniformly runnable both through the
  standalone workflow and the extension workflow (`make run-all`). Each experiment
  sets a top-level `scene` (the USD stage; `@`-prefixed is repo-relative). The
  flight routine *and its verification* (command-flow timeouts, takeoff target,
  flight-envelope gates) live in the Hydra `routine` group (`RoutineConfig`,
  always composed onto `EnvConfig`); there is **no** separate `integration_test`
  group -- the integration test is just a routine experiment driven in strict mode
  (the standalone runner's `--strict`, set by `STRICT=1` in the Make target). The
  GCS link baseport is in the `px4` group (`px4.gcs_baseport`). The standalone
  runner's `--scene` is optional and overrides the experiment scene; the
  multi-drone target passes the `@config` sentinel so the scene comes from the
  experiment `scene`. Experiment variants are selected with the `HYDRA_CONFIG`
  Make variable. The top-level entry configs are independent and live directly
  under `configs/` (the default runtime config is `configs/config.yaml`, the
  shared-routine demo is `configs/runtime_arm_takeoff.yaml`, and the takeoff
  experiments are `configs/integration_test_takeoff.yaml` (one drone) and
  `configs/integration_test_multi_drone_takeoff.yaml` (two drones) -- all three
  select the same `routine` `takeoff` variant and differ only in scene/fleet); the
  composable group building blocks they select (`sim`, `drone`, `switches`,
  `px4`, `isaac_lab`, `routine`) live under `configs/experiment/` and are
  referenced as `experiment/<group>@<group>`. The `routine` group is composed by
  `configs/config.yaml` itself (so every experiment that inherits `config` carries
  it; `EnvConfig.routine` is never `None`), and `routine.steps` is empty by default
  (manual operation) -- see the Commanding section for the routine switch and the
  `takeoff` variant (`configs/experiment/routine/takeoff.yaml`). The persistent
  runtime entrypoint `isaac/runtime/standalone/runapp_custom.sh` reads its stage
  from the experiment `scene` too.

### PX4
- `make run-px4` derives the number of PX4 SITL containers from the active Hydra
  `vehicles` list and opens one terminal per PX4-enabled vehicle. Each
  persistent PX4 container attaches to a `px4` tmux session with the interactive
  PX4 console selected, so manual commands such as `commander arm` and
  `commander takeoff` can be entered directly. Every persistent PX4 instance runs
  on host networking (`--network=host`, like the Isaac container), so all
  instances share the host namespace and the PX4 `-i` index keeps each one's
  simulator TCP `4560 + i` / GCS UDP `14550 + i` ports distinct. The
  shared-private-namespace `--network container:<primary>` model is specific to
  the integration harness (where instance 0 owns a private namespace that the
  other instances and the ephemeral Isaac container join); do not use it for the
  host-networked persistent path.
- Persistent and integration Make targets accept `HYDRA_CONFIG=<path>` to select
  a top-level Hydra experiment config for both host-side PX4 counting and Isaac
  runtime config composition, for example
  `make run-all HYDRA_CONFIG=configs/integration_test_multi_drone_takeoff.yaml`.

### Flight log analysis and PX4 bridge debugging
- `flight_analysis/README.md` is the single source for the repeatable flight-log
  workflow. Read it when you start investigating an unstable flight, a diverging
  takeoff, or a sim-to-real comparison; do not reconstruct the procedure from
  this file. It documents the helper tools (`list_pairs.py`, `bridge_summary.py`,
  `ulog_timeline.py`, `compare_logs.py`), the bridge-log fields, the minimum
  ULog topic set and required real-drone SDLOG profile, the NED / aiding-source
  caveats, and the step-by-step order for deciding whether PX4 commanded the bad
  motion or the bridge applied it incorrectly.
- Make targets in `make/flight-analysis.mk` run those tools through a throwaway
  `uv run --no-project --with pyulog --with numpy` environment, so pyulog/numpy
  never enter the project `.venv`: `make compare-logs A=<log_a.ulg> B=<log_b.ulg>`
  prints a per-metric divergence table, and
  `make ulog-timeline ULOG=<log.ulg> [SAMPLES=...]` prints a single-log timeline.
- Debug from the persisted logs, not `docker logs` (PX4 terminal scrollback has
  repeated starts and ANSI redraw noise; use it only for high-level warnings).
  PX4 ULogs and bridge logs are persisted under `px4/logs/<yyyy-mm-dd>/` for
  manual runs and under `.build/docker/integration/<target>.<run-id>/` for
  integration tests; pair a ULog and bridge log by date directory and `HH_MM_SS`
  stem.

## Hard rules
- Do NOT commit binary files.
- Do NOT regenerate assets unless explicitly asked.
- Do NOT hardcode absolute paths.
- Tunable thresholds, timeouts, rates, and similar runtime parameters belong in
  the Hydra `configs/` YAML as the single source of truth, not hardcoded in code.
  Add a matching field (a bare annotation with no default) to the relevant
  `*Config` dataclass in `src/robotics_experiments/config/types.py` and plumb it
  through. A library constructor may keep a neutral fallback default, but it must
  not duplicate the project value (for example `lockstep_timeout_s` is `0.1` in
  the `LockstepSettings` default but `1.0` at
  `px4.lockstep.lockstep_timeout_s` in `configs/experiment/px4/default.yaml`).
- Prefer modifying existing files over adding new entrypoints.
- Avoid using Unicode symbols.
- Invalid configuration or unexpected runtime state must not fail silently.
  Raise an error for wrong configuration; log a warning for recoverable runtime fallbacks.
- Avoid overly defensive code. Use the minimal checks needed for correctness,
  and otherwise let code fail naturally. Add explicit checks only when the code
  would otherwise fail silently, continue with an incorrect result without
  notice, or when a genuinely distinct state must be represented. Do not wrap a
  value in `Optional`/`None`, add routine `is not None` checks, or carry a
  `<= 0 disables` special case when a plain concrete default works. A numerical
  runtime parameter must carry a concrete default in its library constructor and
  a value in the Hydra YAML, and be used directly (the `*Config` dataclass field
  still stays a bare annotation per the configs rule above). Reserve
  `None`/optional handling and disable branches for states that are required for
  correctness (for example "no actuator received yet" before any timestamp
  exists), not as routine safety padding.
- Every behavior change must be reflected in this file when it changes project conventions,
  error handling policy, runtime assumptions, public behavior, or developer workflow expectations.
- Per-area README and doc files are on-demand context: read a task's relevant doc
  (for example `PX4_BRIDGE_CONTRACTS.md`, the Pegasus `PROVENANCE.md`,
  `flight_analysis/README.md`, the standalone-test README, or
  `docs/INTEGRATION_TEST_ARCHITECTURE.md`) only when the task actually touches
  that area, not
  preemptively. If a doc has diverged from the actual code, correct the doc as
  part of the same task -- a doc fix is in scope whenever you change the code it
  documents.
- After implementing a task, run `uv run pre-commit run --all-files` and fix any
  issues it reports before considering the task complete.
- Do NOT ignore pre-existing issues. When a quality gate (`pre-commit`, lint,
  type check, test) reports a failure -- even in a file the current task did not
  touch, or one that predates it -- fix it along with the current task rather
  than dismissing it as out of scope or a false positive. If a flagged line is
  genuinely correct, silence it narrowly and on purpose (a scoped inline
  suppression such as `# shellcheck disable=...` / `# noqa: <code>` with a brief
  reason), so the gate stays green for the next change.
- For intentionally aligned Python data tables, use Ruff/Black formatter pragmas
  (`# fmt: off` / `# fmt: on`) around the smallest useful block so pre-commit
  does not collapse manual column spacing.

## Docker assumptions
- Containers often run with `--network=host`.
- GPU support is required (NVIDIA runtime).
- Headless must work by default.

## When uncertain
Don't do anything, ask for instructions.
