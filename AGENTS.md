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
  `BridgeStatus` snapshot, the `LockstepSettings` object, the single-call
  `exchange(state)` lockstep seam, and the `ArticulationIO.apply_dynamics` Isaac
  sink), the frame conventions, motor-index ordering, the lockstep algorithm, and
  the cadence/actuator-freshness guards. Load-bearing invariants you must not
  break (all detailed there): `HIL_SENSOR.time_usec` strictly increasing from sim
  time; the `exchange()` send+blocking-recv is one indivisible tick; angular
  velocity is rotated world -> body-FLU before the FRD gyro conversion; sensor
  cadence must equal physics cadence (else disconnect "sensor cadence mismatch");
  prime-then-block startup plus newest-packet draining keep actuators fresh
  (backlog trips `max_sustained_actuator_lag_s`); safety gating zeros forces only,
  never the per-tick exchange; a non-finite vehicle state raises.

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
- `make run-px4` starts the PX4 SITL container.
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
  script.
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

### PX4
- `make run-px4` opens a `px4` tmux session with side-by-side panes. One pane
  starts `px4-iris`; the other remains an interactive shell.
- Inside the PX4 container, `px4-iris` clears simulated PX4 state and then runs
  `make px4_sitl_default none_iris` from the PX4 source directory on demand.

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
- Avoid overly defensive guards. Do not wrap a value in `Optional`/`None`, add
  `is not None` checks, or carry a `<= 0 disables` special case when a plain
  concrete default works. A numerical runtime parameter must carry a concrete
  default in its library constructor and a value in the Hydra YAML, and be used
  directly (the `*Config` dataclass field still stays a bare annotation per the
  configs rule above). Reserve `None`/optional handling and disable branches for
  states that are genuinely distinct and required for correctness (for example
  "no actuator received yet" before any timestamp exists), not as routine
  safety padding.
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
- For intentionally aligned Python data tables, use Ruff/Black formatter pragmas
  (`# fmt: off` / `# fmt: on`) around the smallest useful block so pre-commit
  does not collapse manual column spacing.

## Docker assumptions
- Containers often run with `--network=host`.
- GPU support is required (NVIDIA runtime).
- Headless must work by default.

## When uncertain
Don't do anything, ask for instructions.
