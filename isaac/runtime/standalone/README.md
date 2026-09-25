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
`robotics_experiments.runtime` (imports `omni.physx`/`carb`), and Isaac-dependent task exports from
`robotics_experiments.workflows.isaac_lab` / `isaaclab`. The pure curriculum and
training coordinators are host-importable and start no simulator at import time. Put a one-line comment
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

Controller alignment has two standalone backends. Flight-parameters response
collection under `tests/flight_parameters_capture_response.py` starts through
the normal `SimulationApp` fleet harness and injects only a deterministic `PolicyAdapter`.
The adapter streams a neutral setpoint through a minimum handoff period and a
continuous stability window before recording each profile; its body-FLU yaw
command is sign-converted only at the PX4 NED wire boundary.
`alignment_sweep_target.py` is a host-safe coordinator. It spawns the configured
number of persistent workers, and each worker constructs `AppLauncher` before
importing `isaaclab` and builds one hover environment per local
candidate/probe pair. Scene capacity is derived from the largest beam expansion
divided across workers but capped by the Hydra execution setting. Oversized
expansions are replayed in ordered chunks, so a wide search no longer inflates
every smaller batch to its maximum. Each batch uses per-environment gains,
filters, integrals, and sub-physics-step command delays and starts with fresh
controller, delay, and
actuator state, restores every recorded PX4 initial state, then restores the
schema-1 per-profile PX4 controller boundary before advancing timestamp events
at each trace's captured physics-step intervals.
Each worker scores traces and streams them to a run-local shard. The
coordinator receives compact metric results, merges evidence by original
candidate index, and does not accumulate the full trace population in memory.
Worker count and scene capacity do not change scores or the comparison
fingerprint. Active evaluation writes GPU utilization, padding, cache, and
throughput beside the run artifacts. Worker sharding stays
configurable, but the measured default is one process because three processes
saturate the GPU while materially reducing candidate throughput through
PhysX contention.
Schema-1 traces retain position, attitude, body/world velocity, angular
velocity, yaw-rate commands, absolute PX4 simulation start time, and the
velocity/rate controller boundary at that timestamp. The PX4 alignment routine ends when its probe policy
completes; teardown stops the ephemeral simulator because landing and
auto-disarm are outside the controller target.
Keep their imports in this order; neither runner may import Kit/Isaac Lab at
module import time.

`alignment_fit.py` is host-safe and performs no simulator imports. Fit mode
consumes the two diagnostic trace files and writes `controller.json` plus
`selection.json.zst` without publishing. Finalize mode validates the complete
bundle, including hard quality and boundary diagnostics, before atomically
updating relative `latest`/`best` links. No migration mode is supported.

## Training entrypoint

`train_rl.py` is a CLI and worker startup shell. It loads a curriculum from
`configs/train/`; complete stage experiments belong in `configs/train/stage/`.
`workflows/isaac_lab/curriculum.py` sequences accepted stages and publishes them.
`training.py` owns attempt campaigns and fresh tests. Only attempt and evaluation
workers create Isaac applications. Checkpoint storage and restoration are explicit
in `rl/training/checkpoint.py`; see `docs/RL_WORKFLOW.md` for state and report fields.

Asset preparation uses `make generate-assets HYDRA_CONFIG=<experiment-or-curriculum>`.
See [asset generation](../../../docs/ASSET_GENERATION.md) for switch profiles,
immutable outputs, locking and validation.
