# Fleet Runtime Orchestrator -- Interface Contract

## Purpose

This document specifies the contract of `FleetOrchestrator`
(`src/robotics_experiments/runtime/orchestrator.py`), the single host-agnostic
pipeline that composes a runnable experiment from its parts. It is the top of the
runtime layering and the place where every high-level component is wired together:

```
config  ->  (drone / simulation / px4_bridge / commanding)  ->  FleetOrchestrator  ->  thin per-host shell
```

The orchestrator turns the resolved experiment config into a live, ticking fleet
through a fixed sequence of discrete phases. A thin per-host **shell** drives those
phases and supplies only what genuinely differs between hosts: the tick source and
the failure reaction. There are two shells today:

- **Live** -- the `experiment_runtime` Kit extension
  (`isaac/runtime/extensions/experiment_runtime/`), the path for `make run-all` /
  `make run-isaac`.
- **Standalone** -- the scenario runner
  (`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`), which the
  integration tests drive under `--strict`.

This contract is the analogue, one layer up, of the two it composes: the PX4
bridge contract
(`isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md`) and the
command/routine contract (`docs/COMMAND_ROUTINE_CONTRACT.md`). Read those for the
per-vehicle bridge tick and the per-drone routine machine; read this for how a
whole experiment is assembled and driven.

## Invariants (MUST)

The load-bearing rules. Breaking any of them re-introduces the kitchen-sink
layering this design removed.

- **One orchestrator, thin shells.** The runtime composition (open scene -> apply
  settings -> spawn fleet -> start the lockstep bridge group -> build per-drone
  routine drivers -> teardown) lives only in `FleetOrchestrator`. A shell wires
  events/loop and reaction; it owns no flight, scene, spawn, or bridge logic.
- **Host-agnostic core.** The orchestrator holds no `SimulationApp`, Kit,
  timeline, or event-subscription state. It cannot import a host loop. The shell
  owns the SimApp/Kit lifecycle and the tick source.
- **Production logic only.** Integration-test-only checks (ground-settle wait,
  bridge first-heartbeat wait, the pre-arm start-tilt check, the bridge-log scan)
  live in the standalone shell and read the per-drone `DroneRuntimeHandle`s. They
  do not enter the orchestrator. Verification that is part of *flight* (a real
  climb, a sustained stable hover, the continuous envelope gates) belongs to the
  routine/driver, not here.
- **Scene + settings are orchestrator-owned, uniformly.** `open_scene` /
  `adopt_current_stage` / `apply_settings` are the single source of truth for the
  stage and sim settings in the runtime workflow. The live boot does not use Kit's
  `--exec open_stage.py`, and there is no separate settings extension.
- **A disabled PX4 builds no bridge.** `spawn_fleet` constructs a `BridgeHost` and
  observer only for `px4.enabled` vehicles (mirrors the bridge contract). Disabled
  vehicles are spawned but carry no `host`/`observer`.
- **A driver is built only for a flying drone.** `build_drivers` builds a
  `MavlinkCommandPort` + `RoutineDriver` only where the resolved `routine.steps`
  is non-empty *and* the vehicle is `px4.enabled`. A routine on a disabled vehicle
  is left manual with a warning.
- **Control tick is separate from the bridge legs.** The drivers are ticked from
  the shell's own control tick (a Kit app-update subscription live; the blocking
  loop standalone), never from the bridge's physics-step send/recv legs (the same
  rule the command/routine contract states from the other side).
- **Reaction lives in `make_env`, not the orchestrator.** Strict (raise) vs
  forgiving (log and idle) is entirely the injected `make_env`'s `on_failure`. The
  orchestrator never branches on it.
- **Spawn is gated on PLAY (live).** The live shell opens the scene and applies
  settings at startup but spawns the fleet only on timeline PLAY, so the persisted
  stage is not mutated with drones until the user presses Play.
- **Isaac-coupled leaf, off the core import path.** `orchestrator.py` and
  `bridge_host.py` import `carb`/`omni` at module scope; the `runtime` package
  `__init__` is lazy so `import robotics_experiments.runtime` succeeds with Isaac
  absent. `tests/test_core_imports.py` is the regression guard.

## Scope and non-scope

The orchestrator's one responsibility is to assemble and tear down the runtime
pipeline for a resolved experiment config, exposing per-drone handles and progress
so a shell can drive it.

The orchestrator does NOT:

- Own the `SimulationApp` / Kit app / timeline lifecycle (the shell does).
- Decide when to tick, or own the tick loop/subscription (the shell does).
- Choose the strict-vs-forgiving failure reaction (the injected `make_env` does).
- Run any integration-test-only check (the standalone shell does).
- Implement the per-vehicle bridge tick (the `Px4Bridge` / `BridgeHost` do) or the
  per-drone routine machine (the `RoutineDriver` does).
- Apply MAVLink, thrust math, or USD authoring directly (it calls `spawn_drone`,
  `BridgeHost`, `MavlinkCommandPort`, `apply_settings`).

## The pipeline

`FleetOrchestrator(env)` then, in order:

| Phase | Method | Does |
|-------|--------|------|
| Open scene | `open_scene(scene_override=None)` | Resolve the scene (`env.scene`, or an override; `@` = repo-relative) and fire the USD `open_stage`. |
| Adopt stage | `adopt_current_stage()` | Bind `self.stage` once the stage has finished loading (after a pump in standalone, or on the `OPENED` event live). |
| Apply settings | `apply_settings()` | `simulation.apply_settings(stage, env.sim)`. |
| Spawn fleet | `spawn_fleet()` | `spawn_drone` per vehicle; a `BridgeHost` + state-only `IsaacArticulationIO` observer per `px4.enabled` vehicle; build the one `BridgeHostGroup`. |
| Start bridges | `start_bridges()` | `group.start()` (its hosts + the single physics-step subscription). |
| Build drivers | `build_drivers(make_env)` | Per flying drone: a `MavlinkCommandPort` and a `RoutineDriver` wired through `make_env(handle)`. |
| (tick) | -- | The shell ticks `orchestrator.flying[i].driver.tick()` from its own tick source until `orchestrator.all_done`. |
| Teardown | `teardown()` | `group.stop()`, disconnect ports, shut down observers, despawn every vehicle. |

Read-only views the shells use: `drones` (all handles), `bridged` (handles with a
host), `flying` (handles with a driver), `all_done` (every flying driver done;
vacuously true when none).

`DroneRuntimeHandle` is the per-vehicle seam the shells read:
`vehicle, label, articulation_path, host?, observer?, port?, driver?` (the `?`
fields are populated only when applicable per the invariants above).

## The injected reaction seam: `make_env`

`build_drivers` takes `make_env: (DroneRuntimeHandle) -> DriverEnvironment`. The
orchestrator ships one generic implementation, `FleetDriverEnv`, that provides the
production reads every host shares (`is_bridge_connected` / `sim_time_us` /
`read_observation` over the handle's `host` + `observer` + `port`) and forwards
three injected callables:

- `log(message)` -- where routine progress goes (`carb` live, the runner's
  `report` standalone).
- `on_failure(kind, message)` -- the reaction. Forgiving: log and let the driver
  idle. Strict: raise (`ExpectedInstabilityDetected` for `FailureKind.ENVELOPE`,
  `Px4FailsafeDetected` for a `"failsafe"`, `AssertionError` otherwise).
- `on_step_advanced(step_label)` -- a per-step hook (the negative test's
  post-takeoff fault injection wires here on the first drone).

A shell builds its `make_env` as a closure over these three; nothing else differs
between strict and forgiving runs.

## The two shells

**Live (`experiment_runtime` Kit extension).** `on_startup` builds the
orchestrator and calls `open_scene`; on the `OPENED` event it `adopt_current_stage`
+ `apply_settings`. On timeline PLAY it `spawn_fleet` + `start_bridges` +
(if any drone flies) `build_drivers` with a **forgiving** `make_env`, then ticks
every flying driver from one Kit app-update subscription (guarded per driver so one
bug neither stops the others nor spams Kit). STOP/PAUSE and `on_shutdown` drop the
tick and `teardown`.

**Standalone (`_px4_takeoff_scenario.py`).** A blocking runner that calls the same
phases in sequence, interleaving the integration-test-only checks between them,
builds a **strict-or-forgiving** `make_env`, and ticks the flying drivers raw
(unguarded, so a strict exception propagates and fails the test) in
`while not orchestrator.all_done: simulation_app.update(); for d: d.tick()`, then
runs the strict bridge-log scan. `--strict` selects the strict reaction.

## Lifecycle (ASCII)

```
[Shell builds FleetOrchestrator(get_config())]
        |
        v
open_scene() --> (host pumps / OPENED) --> adopt_current_stage() --> apply_settings()
        |
        v
[Play (live) | next in sequence (standalone)]
        |
        v
spawn_fleet() --> start_bridges()
        |
        v
build_drivers(make_env)     # only when a drone flies
        |
        v
[shell control tick]  for handle in flying: handle.driver.tick()
        |   (live: Kit app-update sub, guarded; standalone: blocking loop, raw)
        v
[Stop / done / shutdown]  teardown()
```

## Adding a new high-level component

This is the rule the layering exists to enforce: **a new high-level runtime
component is planned and integrated in the context of this orchestrator, not bolted
onto an extension.** Concretely:

1. Decide where it sits in the phase order and what it owns for its full lifecycle
   (construction in a phase, teardown in `teardown`). If it is per-drone state, add
   a field to `DroneRuntimeHandle` and populate it in the relevant phase; if it is
   fleet-wide, hold it on the orchestrator.
2. Express it as an orchestrator phase (or an extension of an existing one), not as
   new lifecycle logic in a shell. The shells stay thin: they only drive phases,
   choose the tick source, and inject the reaction.
3. Keep host-specific concerns (SimApp/Kit/timeline, the tick source, the reaction)
   on the shell side through a seam (a `make_env`-style injected callable or a
   handle field), so both shells reuse the component unchanged.
4. Separate production from test-only: production composition goes in the
   orchestrator; checks that only an integration test needs stay in the standalone
   shell and read the handles.
5. Preserve the import-isolation rule: anything Isaac-coupled stays in
   `runtime/` behind the lazy package `__init__`, and `tests/test_core_imports.py`
   must still pass.

If a proposed component cannot be expressed as an orchestrator phase + handle field
driven by the thin shells, that is a signal the design is putting composition in
the wrong layer -- revisit it before implementing.
