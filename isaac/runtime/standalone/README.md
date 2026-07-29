# Isaac Sim Standalone Workflow

This directory contains scripts run directly by Isaac Sim, for example through
`/isaac-sim/python.sh` or `runapp.sh`.

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

These files carry `from __future__ import annotations`, and both Ruff and mypy
check them. Annotate parameters and returns precisely. `TYPE_CHECKING` imports
are reserved for Kit-app-only types whose runtime import would violate startup
ordering; ordinary dependencies and pure project types remain eager.

Do not hoist the documented Kit-dependent imports; that breaks the scripts at
import time before the app boots.

The fleet runner uses the same frozen `FleetState` lifecycle as the live
extension and advances its driver from a PhysX-step subscription. Driver
creation and each due fleet cycle use one shared explicit sim/wall-clock tick.
Unexpected callback exceptions are retained and re-raised immediately after
`SimulationApp.update()`. Expected driver failures remain in immutable state
until landing recovery terminates; strict mode then raises the original typed
failure. Rebind every lifecycle, bridge, and driver transition result. Helpers
that report simulated elapsed time receive a sim-time callable; they must not
depend on mutable resource properties.

Controller alignment has two standalone backends. The PX4 capture entrypoint
under `tests/alignment_capture_reference.py` starts through the normal
`SimulationApp` fleet harness and injects only a deterministic `PolicyAdapter`.
`alignment_sweep_target.py` constructs `AppLauncher` before importing
`isaaclab`, builds one hover environment, reapplies each complete
`ControllerSettings` candidate with fresh delay and actuator state, restores
recorded PX4 initial states, and replays the captured physics-step intervals.
Keep their imports in this order; neither runner may import Kit/Isaac Lab at
module import time.

`alignment_fit.py` is host-safe and performs no simulator imports. It consumes
the two diagnostic trace files, writes the canonical `alignment.json`, and
updates relative `latest`/`best` links.
