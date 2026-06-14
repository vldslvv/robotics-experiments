# AGENTS.md - Robotics stack (Isaac Sim + ROS 2 + PX4)

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
- Do not rename, remove, or broaden public exports as part of cleanup unless explicitly requested.

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
- PX4 bridge behavior is configured through the Hydra `px4` group. The default
  experiment enables the bridge for the custom Isaac runtime; use Hydra
  overrides such as `px4.enabled=false` for Isaac sessions that should not
  connect to PX4 SITL.
- Isaac articulation angular velocity is treated as world-frame data and must
  be rotated into body-FLU before MAVLink converts it to PX4 body-FRD gyro
  fields.
- USD rotor marker metadata identifies thrust marker prims in PX4 motor-index
  order. The bridge discovers prims with `drone:rotorMarker = true`, orders
  them by `drone:px4MotorIndex`, and derives spin from `drone:spinDir`. The
  Isaac adapter applies thrust to the drone rigid body at those marker
  positions; marker prims must not be treated as separate free rotor rigid
  bodies unless joints are explicitly authored for that topology.
- For `none_iris` / Generic Quad X, PX4 motor-index order is
  front-right CW, rear-left CW, front-left CCW, rear-right CCW. PX4 geometry is
  FRD, while Isaac marker positions are FLU, so PX4 right maps to Isaac
  negative Y when resolving USD marker positions.
- PX4 lockstep requires strictly increasing `HIL_SENSOR.time_usec` values.
  If Isaac emits duplicate or non-increasing physics timestamps, the bridge
  must not forward another HIL sensor tick or wait for a new actuator packet.
- PX4 actuator lockstep must use fresh actuator packets. Once PX4 has produced
  its first `HIL_ACTUATOR_CONTROLS` packet, the bridge drains queued actuator
  messages after each sensor tick and applies the newest packet. PX4 startup can
  produce an initial actuator timestamp lag while HIL output begins; the bridge
  may use the first actuator packet's lag as a bootstrap baseline. If later
  actuator timestamps become materially stale relative to that baseline and the
  current simulation time, the bridge must disconnect and return zero rotor
  commands instead of applying old motor outputs.

## Python environment
- This project uses `uv` for all host-side Python work. Do not call `pip`,
  `python -m venv`, or a system/global Python directly.
- The project virtualenv lives at `.venv` and is pinned to the Python in
  `.python-version` (3.11). Dependencies and the resolver index (including the
  NVIDIA index for Isaac Sim) are defined in `pyproject.toml`; exact versions
  are locked in `uv.lock`. Do not edit `uv.lock` by hand.
- Initialize or repair the environment with `uv sync --extra dev`. The `dev`
  extra (ruff, mypy, pre-commit, debugpy) is optional, so a plain `uv sync`
  intentionally omits it; pass `--extra dev` when you need the tooling.
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
- `make test-integration` runs container-backed integration tests.
- `make test-integration-all` runs every integration test target in parallel
  using isolated transient containers and prints an aggregate passed/failed
  summary after all targets finish. It defaults to two concurrent targets to
  avoid Isaac/GPU resource starvation; set `INTEGRATION_TEST_JOBS=<n>` to tune
  concurrency for the host.
- Integration test Make targets are defined in `make/integration-tests.mk`.
  Their standalone Isaac test scripts live under
  `isaac/runtime/standalone/tests/`; use that directory's `README.md` for
  scenario-specific notes.
- `make test-integration-px4` starts a temporary PX4 SITL container, runs the
  standalone Isaac PX4 bridge test, and removes the temporary PX4 container.
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

### PX4 bridge debugging
- Use `flight_analysis/README.md` for the detailed repeatable log-analysis
  workflow and helper scripts.
- PX4 ULog files and custom bridge logs are persisted under
  `px4/logs/<yyyy-mm-dd>/` for manual PX4 runs. Automated integration tests
  keep PX4 logs, bridge logs, and Isaac logs under
  `.build/docker/integration/<target>.<run-id>/` so parallel runs do not share
  PX4 state or diagnostics.
- PX4 creates the ULog when the SITL logger starts, before arm/takeoff. The
  bridge creates its log when the Isaac bridge opens the PX4 MAVLink endpoint.
  A normal paired run has matching names such as:
  - `px4/logs/2026-06-10/23_20_44.ulg`
  - `px4/logs/2026-06-10/23_20_44_bridge.log`
- Pair logs by the date directory and `HH_MM_SS` stem first. If there are
  nearby bridge logs with adjacent seconds, use file modification time and the
  bridge first line (`PX4 MAVLink port constructed`) to confirm the pair.
- Do not rely on `docker logs` alone for flight analysis. The PX4 terminal
  scrollback can contain repeated starts and ANSI redraw output. Use it only to
  identify high-level warnings, then inspect the persisted ULog and bridge log.
- For a fresh run, `px4-iris` clears PX4 non-volatile SITL state but keeps
  `px4/logs` persistent. If a warning might be stale, first confirm the latest
  `.ulg` and `_bridge.log` modification times.
- The bridge log records MAVLink connection, first heartbeat, first HIL sensor,
  periodic sensor freshness, actuator controls, rotor mapping, per-rotor forces,
  body torque/drag, pose, velocity, and stop counters. Use it to verify:
  - `HIL_SENSOR.time_usec` is strictly increasing.
  - `HIL_ACTUATOR_CONTROLS` timestamps stay close to sensor simulation time;
    stale actuator lag means the bridge is applying old control outputs.
  - `fresh={imu:True mag:True baro:True ...}` is present before send.
  - `input_reference` and `commands` match expected PX4 actuator behavior.
  - discovered rotor marker paths resolve to the expected body-frame positions
    and spin directions for PX4 motor-index order.
- Use the ULog to decide whether PX4 commanded the bad motion or the bridge
  applied the command incorrectly. The minimum topics for flight divergence are:
  `vehicle_command`, `vehicle_command_ack`, `vehicle_status`,
  `vehicle_local_position`, `vehicle_local_position_setpoint`,
  `trajectory_setpoint`, `vehicle_attitude`, `vehicle_attitude_setpoint`,
  `vehicle_rates_setpoint`, `vehicle_thrust_setpoint`,
  `vehicle_torque_setpoint`, `actuator_motors`, `actuator_outputs`,
  `estimator_status`, `estimator_aid_src_mag_heading`,
  `vehicle_magnetometer`, `vehicle_air_data`, `battery_status`,
  `failsafe_flags`, `vehicle_land_detected`, and `takeoff_status`.
- Interpret the logs in this order:
  1. Confirm the ULog and bridge log are from the same run.
  2. Check PX4 accepted `commander arm` / `commander takeoff` in
     `vehicle_command_ack`.
  3. Check `trajectory_setpoint` and `vehicle_local_position_setpoint` to see
     whether PX4 requested lateral motion.
  4. Compare `vehicle_attitude_setpoint` / `vehicle_rates_setpoint` against
     actual `vehicle_attitude`.
  5. Compare `actuator_motors` with bridge `input_reference`, `commands`, and
     `forces_n`.
  6. If PX4 commands a correction and the vehicle moves farther in the wrong
     direction, suspect rotor order, thrust axis, body frame, or sign mismatch.
  7. Treat compass/failsafe warnings that appear after the attitude diverges as
     likely consequences unless estimator topics show rejection before motion.
- If local ULog tools are unavailable, a temporary parser can be run with:
  `env UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with pyulog --with numpy python`.
  Prefer extracting a focused topic summary over exporting every topic.

## Hard rules
- Do NOT commit USD assets or binary files.
- Do NOT regenerate assets unless explicitly asked.
- Do NOT hardcode absolute paths.
- Prefer modifying existing files over adding new entrypoints.
- Avoid using Unicode symbols.
- Invalid configuration or unexpected runtime state must not fail silently.
  Raise an error for wrong configuration; log a warning for recoverable runtime fallbacks.
- Every behavior change must be reflected in this file when it changes project conventions,
  error handling policy, runtime assumptions, public behavior, or developer workflow expectations.
- For intentionally aligned Python data tables, use Ruff/Black formatter pragmas
  (`# fmt: off` / `# fmt: on`) around the smallest useful block so pre-commit
  does not collapse manual column spacing.

## Docker assumptions
- Containers often run with `--network=host`.
- GPU support is required (NVIDIA runtime).
- Headless must work by default.

## When uncertain
Don't do anything, ask for instructions.
