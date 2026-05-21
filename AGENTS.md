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
  direct feature state such as `self.stage` instead of broad handles such as
  `self._usd_context`.
- Use `drone_spawner` as the extension pattern: wire Isaac/Kit events separately
  from meaningful actions, and let setup/teardown actions own related prim and
  bridge lifecycle together.
- `robotics_experiments_core` exposes the shared `src/robotics_experiments`
  package to Isaac Sim extensions. Extension code that needs the core package
  should depend on `robotics_experiments_core` instead of manipulating
  `sys.path`.
- Extension config is passed through the `experiment_config` extension that uses hydra.
- Each extension connects to `experiment_config` using `config.py`, no other coupling.

## Common commands

### Docker
- `make build-all` builds Isaac Sim, ROS 2, and PX4 images.
- `make clean-all` stops/removes containers and removes built images.
- `make run-isaac` starts the Isaac Sim container.
- `make run-ros` starts the ROS 2 container.
- `make run-px4` starts the PX4 SITL container.

### ROS 2
Avoid calling ROS 2 directly on the host. It is meant to be used in the Dockerized environment.

### Isaac Sim
Avoid calling Isaac Sim directly on the host. It is meant to be used in the Dockerized environment.

### PX4
Avoid calling PX4 directly on the host. It is meant to be used in the Dockerized environment.

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
