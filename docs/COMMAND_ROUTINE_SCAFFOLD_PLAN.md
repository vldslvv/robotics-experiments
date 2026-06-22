# High-level command / routine scaffolding (`VehicleRoutine`)

> Status: IMPLEMENTED. The `robotics_experiments.commanding` package, the
> standalone-scenario refactor, and the tests below are in place; the short
> contract reference lives at `docs/COMMAND_ROUTINE_CONTRACT.md`. This document is
> retained as the design rationale. The extension-workflow follow-up (driving the
> command layer from the live `drone_spawner` extension via the `FleetControlDriver`
> tick machine, the `routine` config group, and `build_vehicle_routine`) is also
> IMPLEMENTED -- see `docs/COMMAND_ROUTINE_CONTRACT.md` "Where the executor lives".
> What remains deferred: merging the standalone scenario onto the shared
> `build_vehicle_routine`/driver, a headless extension-path integration target,
> and per-vehicle routine overrides.

## Context

Today the only way a drone receives high-level commands (arm, takeoff) is an
ad-hoc 11-stage procedure buried in the integration test
`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`. It opens a pymavlink
GCS link to each vehicle's `gcs_port`, hand-sends `command_long_send`
(`MAV_CMD_COMPONENT_ARM_DISARM`, `MAV_CMD_NAV_TAKEOFF`), waits for
`COMMAND_ACK`, and sequences phases inline. There is **no reusable command/plan
abstraction anywhere** in the codebase.

We want to lay scaffolding for future autonomy (navigate-to-point, set-altitude,
and ultimately the **switch-pressing task**) as a **pure-function contract**, in
the same spirit as the PX4 bridge contract (`VehicleState -> ActuatorCommands ->
RotorDynamics` with a swappable sink) and the `resolve_fleet -> ResolvedVehicle`
config enrichment. The idea: a drone (or a scene) is handed a **command plan /
routine** -- a pure data + pure-stepper description of what to do -- and a thin
side-effecting executor turns the plan's decisions into MAVLink over the
existing `gcs_port`, paced by sim time.

The environment half of the switch task already exists: `EnvConfig.switches` /
`lightswitch_spawner` spawns switch+lamp pairs with a `press_threshold` joint and
a ROS toggle publisher. Only the **commanding** half is greenfield.

### Decisions locked with the user
1. **Execution locus:** standalone runner first. Pure core in the package; the
   executor is driven from the scenario run loop over `gcs_port`. No
   `drone_spawner` extension changes in this pass. (The bridge must never touch
   `gcs_port` -- the commander is a separate control layer by design.)
2. **Command model:** thin MAVLink wrappers (commands map ~1:1 to `command_long`
   `MAV_CMD_*`), not a PX4-agnostic intent layer.
3. **Plan structure:** ordered steps, each with a pure completion predicate over
   an observation, plus optional guard/timeout; a pure decision function. A
   degenerate decision graph, extensible to branching later.
4. **Scope:** add the contract + seams + a worked `arm -> takeoff -> hover`
   routine, **and refactor** `_px4_takeoff_scenario.py` to consume it (preserving
   `post_takeoff_hook`, the envelope gates, and multi-drone barrier behavior).

> **Extension workflow follow-up -- IMPLEMENTED.** The pure core (`commanding/`)
> is now reused by the live `drone_spawner` extension so `make run-all` /
> `make run-isaac` execute a routine headless. The
> `FleetControlDriver` (extension `control_driver.py`) is a non-blocking,
> tick-driven phase machine stepped from the Kit app-update event stream (a
> control tick separate from the bridge's physics legs); it reads each
> `ResolvedVehicle.gcs_port` via `MavlinkCommandPort`, the bridge snapshot via the
> public `BridgeHost.status`, and runs the same
> `FleetCommander(STEP_SYNCHRONIZED)`. Routine selection/tunables live in the new
> non-test `routine` Hydra group (`RoutineConfig`, always composed onto
> `EnvConfig`), with `routine.steps` as the single source of truth (non-empty ->
> the extension builds the driver on Play; empty -> manual). The pure
> `commanding.build_vehicle_routine` maps the config descriptors to a
> `VehicleRoutine`. On failure/failsafe/timeout the runtime driver logs a warning
> and stops (never raises), unlike the asserting test. The standalone and
> extension paths share the same `decide(...)` core and `AbstractCommandPort`
> seam; only the driver differs. Still deferred: merging the standalone scenario
> onto `build_vehicle_routine`/a shared driver, a headless extension-path
> integration target, and per-vehicle routine overrides.

## Design: a command contract parallel to the bridge

The bridge splits into pure data + pure transform + side-effecting seam +
orchestrator + lifecycle. The command layer mirrors it part-for-part:

| Bridge | Command layer (new) |
|---|---|
| `VehicleState` (pure input) | `RoutineObservation` (pure input) |
| thrust model `commands_to_dynamics` (pure) | `decide(...)` pure stepper |
| `RotorDynamics` (pure output) | `RoutineDecision` (pure output) |
| `AbstractFlightStackBridge` / `ArticulationIO` (seams) | `AbstractCommandPort` (seam) |
| `Px4Bridge` (orchestrator) | `VehicleCommander` (orchestrator) |
| `BridgeHost` / `BridgeHostGroup` | `VehicleCommander` / `FleetCommander` |

### New package: `src/robotics_experiments/commanding/`

A domain leaf in the existing `config -> domain -> Isaac extensions` layering.
`import robotics_experiments` must still succeed with Isaac **and** pymavlink
absent: keep `pymavlink` imports function-scoped (the current pattern in
`_px4_takeoff_scenario.py:237`), and keep `commanding/__init__.py` re-export-only
of the pure types. Each public symbol goes in module `__all__`.

- **`commands.py`** -- thin MAVLink command wrappers, pure data (no pymavlink
  import; MAV_CMD numbers as named int constants so the module stays
  dependency-free):
  - `MavlinkCommand` (frozen): `command_id: int`, `params: tuple[float, ...]`
    (7), `label: str`.
  - Named constants `MAV_CMD_COMPONENT_ARM_DISARM = 400`,
    `MAV_CMD_NAV_TAKEOFF = 22`, `MAV_CMD_DO_REPOSITION = 192`,
    `MAV_CMD_NAV_LAND = 21`, and `PX4_FORCE_ARM_MAGIC = 21196` (lifted from
    `_px4_takeoff_scenario.py:44`, an int passed as the float `command_long`
    param2).
  - Pure constructors: `arm()`, `takeoff(lat, lon, abs_alt, climb)`,
    `set_altitude(...)` / `reposition(lat, lon, alt)`, `land()`.
- **`observation.py`** -- `RoutineObservation` (frozen): `vehicle_state:
  VehicleState`, `bridge_status: BridgeStatus`, `last_ack: CommandAck | None`,
  `failsafe: bool`, `sim_time_us: int`. Plus a small `CommandAck` (frozen:
  `command_id`, `result`, `accepted: bool`). Pure; reuses the bridge's existing
  pure `VehicleState` / `BridgeStatus`.
- **`routine.py`** -- the pure plan + stepper:
  - `RoutineStep` (frozen): `command: MavlinkCommand`,
    `is_complete: Callable[[RoutineObservation], bool]`, `label: str`,
    `timeout_s: float`, optional `retry_on_temporarily_rejected: bool`.
  - `VehicleRoutine` (frozen): `steps: tuple[RoutineStep, ...]`.
  - `RoutineState` (frozen or small mutable): `cursor: int`, `sent: bool`,
    `baseline_position_enu`, `step_started_us`.
  - `RoutineDecision` (frozen union-ish): one of `SEND(command)`, `WAIT`,
    `ADVANCE`, `DONE`, `FAIL(reason)`.
  - `def decide(routine, state, observation) -> tuple[RoutineState,
    RoutineDecision]` -- the pure function. No sockets, no pymavlink, no Isaac;
    unit-testable with plain dataclasses. This is the contract centerpiece.
  - Predicate helpers (pure, over `RoutineObservation`): `armed()`,
    `climbed_at_least(min_climb_m)`, `ack_accepted()`.
- **`port.py`** -- the side-effecting seam (mirrors
  `AbstractFlightStackBridge`):
  - `AbstractCommandPort` (ABC): `connect()`, `pump_heartbeat()`,
    `send(command: MavlinkCommand)`, `poll_feedback() -> CommandAck | None`,
    `disconnect()`.
  - `MavlinkCommandPort(gcs_endpoint, ...)` concrete impl with **function-scoped**
    `from pymavlink import mavutil`. Absorbs `_connect_gcs`,
    `_send_gcs_heartbeat`, `command_long_send`, `_drain_mavlink`, and the
    `COMMAND_ACK` / failsafe / statustext parsing currently inline in
    `_px4_takeoff_scenario.py:236-360,428-502`.
- **`commander.py`** -- orchestrators:
  - `VehicleCommander` wires one `VehicleRoutine` + one `AbstractCommandPort` +
    holds `RoutineState`. `step(observation) -> RoutineDecision`: runs `decide`,
    and on `SEND` calls `port.send(...)`; tracks completion. Pure decision,
    impure side effect cleanly separated.
  - `FleetCommander` holds per-vehicle `VehicleCommander`s plus an execution
    policy: `INDEPENDENT` (each routine advances freely) or `STEP_SYNCHRONIZED`
    (barrier: no vehicle advances to step N+1 until every vehicle has completed
    step N). `STEP_SYNCHRONIZED` reproduces the current takeoff scenario
    (arm-all, then takeoff-all). Mirrors `BridgeHostGroup`'s fleet role.

### Driver + refactor of the takeoff scenario

`_px4_takeoff_scenario.py` keeps its **assertion** responsibilities (the
continuous gates in `_observe_all` / `_update_drone_observation` and the hover
envelope) and its sim-stepping (`_advance_once`, `_advance_sim_time`). It sheds
its **commanding** responsibilities to the new layer:

- Build a `VehicleRoutine` = `[arm_step, takeoff_step]` per drone from
  `IntegrationTestConfig` values already present (`takeoff_altitude_m`,
  `min_takeoff_climb_m`, `command_ack_timeout_s`, `gcs_heartbeat_period_s`).
- Construct a `FleetCommander(STEP_SYNCHRONIZED)` over `MavlinkCommandPort`s
  built from each `vehicle.gcs_port`.
- Replace the bespoke `gcs_connect` / `_settle_px4_prearm` / `_send_arm` /
  `_send_takeoff` / `_assert_command_ack` stages with: per tick, build a
  `RoutineObservation` for each drone (sample state via the existing observer,
  read `host` bridge status, drain port feedback) and call `fleet.step(...)`
  until the routine reports `DONE`, while still ticking physics with
  `_advance_once` and asserting bridge connectivity.
- Preserve `Px4TakeoffScenarioOptions.post_takeoff_hook`
  (`_px4_takeoff_scenario.py:1073`, fired at `:1014-1016`): fire it when the
  takeoff step completes,
  before the hover-observation window -- unchanged contract for the instability
  negative test.
- The arm/takeoff retry-on-`TEMPORARILY_REJECTED` and the "Armed by external
  command" statustext acceptance move into the port + the step's
  `retry_on_temporarily_rejected` flag, preserving current behavior.

### Config

Per the user, do not touch `configs/` in this pass. Source all routine tunables
from the **existing** `IntegrationTestConfig` fields (`EnvConfig.integration_test`),
which is valid here because the standalone takeoff scenario runs under the
`integration_test_*` experiments that compose that group. Do not invent new YAML
knobs now. (Note for later: `integration_test` is `None` in the default runtime
experiment, so the deferred extension path needs its own non-test config source;
genuinely new routine tunables -- and routine selection -- get bare-annotation
fields in `config/types.py` + Hydra YAML per the project's
single-source-of-truth rule.)

## Files

Create:
- `src/robotics_experiments/commanding/__init__.py` (lazy, re-export-only)
- `src/robotics_experiments/commanding/commands.py`
- `src/robotics_experiments/commanding/observation.py`
- `src/robotics_experiments/commanding/routine.py`
- `src/robotics_experiments/commanding/port.py`
- `src/robotics_experiments/commanding/commander.py`
- `tests/test_commanding_routine.py` (pure `decide` + `FleetCommander` barrier,
  with a fake `AbstractCommandPort` -- add the fake to `tests/_fakes.py`)
- `docs/COMMAND_ROUTINE_CONTRACT.md` (short contract reference once implemented:
  the contract parts, the pure/impure split, the `gcs_port`-not-bridge rule)

Modify:
- `isaac/runtime/standalone/tests/_px4_takeoff_scenario.py` (consume the new
  layer; keep gates, observation, `post_takeoff_hook`).
- `tests/_fakes.py` (shared fake command port + sample `RoutineObservation`).
- `tests/test_core_imports.py` (assert `robotics_experiments.commanding` imports
  with Isaac/pymavlink absent and that `port` does not eager-import pymavlink).
- `AGENTS.md` (new "Commanding" domain section: the contract, the
  `gcs_port`-is-control-not-bridge boundary, pure-stepper rule, where the
  executor lives, and the planned extension-workflow follow-up). `CLAUDE.md` is a
  symlink; edit `AGENTS.md` only.

Reuse (do not reinvent): `VehicleState`
(`px4_bridge/vehicle_state.py`), `BridgeStatus` (`px4_bridge/contract.py`),
`ResolvedVehicle.gcs_port` (`config/types.py:358`, derived as
`px4.gcs_baseport + id` in `config/load.py:110`), the observer/state sampling
and gate helpers already in `_px4_takeoff_scenario.py`, and the
`AbstractFlightStackBridge` send/poll seam as the structural template
(`px4_bridge/flight_stack.py`).

## Verification

1. **Pure unit tests** (no Docker/Isaac/PX4): `uv run pytest
   tests/test_commanding_routine.py` -- drive `decide(...)` through
   arm->takeoff->done with synthetic `RoutineObservation`s; assert SEND/WAIT/
   ADVANCE/DONE/FAIL transitions, timeout -> FAIL, and the `FleetCommander`
   `STEP_SYNCHRONIZED` barrier (a fast drone waits for a slow one). Confirm the
   fake port records exactly the expected `MavlinkCommand`s in order.
2. **Core import guard:** `uv run pytest tests/test_core_imports.py` -- the new
   subpackage imports clean without Isaac/pymavlink.
3. **Full host suite:** `make test` (i.e. `uv run pytest`).
4. **Integration parity** (the refactor must not regress behavior):
   - `make test-integration-px4-takeoff` (single-drone arm/takeoff/hover).
   - `make test-integration-px4-multi-drone-takeoff` (barrier across the fleet).
   - `make test-integration-px4-takeoff-instability` (confirms `post_takeoff_hook`
     still fires and instability is still detected).
   Note the ~33% host-load flake on the takeoff integration test (see
   `docs/PX4_TAKEOFF_TEST_INSTABILITY.md`); re-run on a flake rather than
   treating one failure as a regression.
5. **Quality gate:** `uv run pre-commit run --all-files` (ruff, ruff format,
   mypy, codespell) green before done.
