# Isaac Sim Standalone Workflow

This directory contains scripts run directly by Isaac Sim, for example through
`/isaac-sim/python.sh` or `runapp.sh`.

`set_python_env.sh` sources Isaac Sim's generated `setup_python_env.sh`,
preserves the caller's `PYTHONPATH`, and appends the editor extension symlinks.
Keep it version-independent; do not duplicate Kit library paths or hardcode the
Kit Python minor version or extension-cache build identifiers.

Keep Isaac Kit extensions in `isaac/runtime/extensions` and Isaac Lab configs
or environments in `isaac/runtime/lab`.

## Import ordering

Standalone Isaac scripts in this directory and under `tests/` must construct
`SimulationApp` or Isaac Lab's `AppLauncher` before importing any
Kit-app-dependent module. Keep those imports function-scoped and deferred until
after the app starts.

Deferred modules include `omni.*`, `carb`, `pxr`,
`robotics_experiments.drone` when accessing its spawn/despawn exports
(loads the Isaac-only stage authoring implementation),
`robotics_experiments.runtime` (imports `omni.physx`/`carb`), and
`robotics_experiments.workflows.isaac_lab` / `isaaclab`. Put a one-line comment
on the first deferred import in each file noting why it is deferred.

Only app-independent imports belong at module top: stdlib, `numpy`, `torch`, and
the pure `robotics_experiments` packages (`config`, `simulation`, `commanding`,
`px4_bridge`; its `__init__` defers all `omni`/`carb` internally).

`build_copter_usd_standalone.py` is the USD-authoring and CLI shell. Its pure
airframe dataclasses, Hydra parsing, and mass-property calculations live in
`robotics_experiments.drone._airframe`, which can be tested on the host without
OpenUSD bindings.

These files carry `from __future__ import annotations`, and ruff lints them but
mypy does not (its scope is `src tests`), so annotate parameters/returns
precisely using `TYPE_CHECKING` imports for Kit-app-only types. Never add a
runtime import only for a hint.

Do not convert deferred imports to module top to remove redundancy; that breaks
the scripts at import time before the app boots.

The fleet runner uses the same frozen `FleetState` lifecycle as the live
extension. Rebind every lifecycle, bridge, and driver transition result. Helpers
that report simulated elapsed time receive a sim-time callable; they must not
depend on mutable resource properties.
