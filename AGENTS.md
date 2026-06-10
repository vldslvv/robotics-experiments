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
- `px4/eeprom/` -> PX4 SITL simulated EEPROM persistence.
- `docs/` -> Design and usage docs.

## Placement rules
- Keep core project code under `src/robotics_experiments`.
- Put Isaac Kit extension code under `isaac/runtime/extensions`.
- Put reusable Isaac runtime templates under `isaac/runtime/templates`, outside
  the extension discovery folder.
- Put direct Isaac Sim scripts under `isaac/runtime/standalone`.
- Put Isaac Lab-specific configs, tasks, and environments under `isaac/runtime/lab`.
- Put ROS 2 packages and colcon workspace changes under `ros/ros2_ws`.
- Put ROS 2 middleware config under `ros/config`.
- Put PX4 Docker/scripts/runtime support under `px4`.

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
- PX4 `rotor_link_paths` identify rotor thrust marker prims in motor-index
  order. The Isaac adapter applies thrust to the drone rigid body at those
  marker positions; marker prims must not be treated as separate free rotor
  rigid bodies unless joints are explicitly authored for that topology.
- PX4 lockstep requires strictly increasing `HIL_SENSOR.time_usec` values.
  If Isaac emits duplicate or non-increasing physics timestamps, the bridge
  must not forward another HIL sensor tick or wait for a new actuator packet.

## Common commands

### Docker
- `make build` builds Isaac Sim, ROS 2, and PX4 images.
- `make destroy` stops/removes containers and removes built images.
- `make run-isaac` starts the Isaac Sim container.
- `make run-ros` starts the ROS 2 container.
- `make run-px4` starts the PX4 SITL container.
- `make stop` stops persistent containers without removing them.
- Legacy targets such as `build-all`, `run-all`, and `clean-all` are
  compatibility aliases.

### ROS 2
Avoid calling ROS 2 directly on the host. It is meant to be used in the Dockerized environment.

### Isaac Sim
Avoid calling Isaac Sim directly on the host. It is meant to be used in the Dockerized environment.
- `make test` runs host Python tests.
- `make test-integration` runs container-backed integration tests.
- `make test-integration-px4` starts a temporary PX4 SITL container, runs the
  standalone Isaac PX4 bridge test, and removes the temporary PX4 container.

### PX4
Avoid calling PX4 directly on the host. It is meant to be used in the Dockerized environment.
- `make run-px4` opens a `px4` tmux session with side-by-side panes. One pane
  starts `make px4_sitl_default none_iris`; the other remains an interactive
  shell.
- Inside the PX4 container, `px4-iris` runs `make px4_sitl_default none_iris`
  from the PX4 source directory on demand.

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
