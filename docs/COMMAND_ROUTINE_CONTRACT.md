# Command / routine contract (`robotics_experiments.commanding`)

The commanding layer is the high-level control counterpart to the PX4 bridge
contract. Where the bridge turns `VehicleState -> ActuatorCommands ->
RotorDynamics` across a swappable sink, the command layer turns a pure plan plus
an observation into a decision across a swappable MAVLink port. It is a domain
leaf on the core import path: every module imports neither Isaac nor pymavlink at
module scope, so `import robotics_experiments` succeeds with both absent.

## The four-part contract

| Bridge | Command layer |
|---|---|
| `VehicleState` (pure input) | `RoutineObservation` (pure input) |
| `commands_to_dynamics` (pure transform) | `decide(...)` (pure stepper) |
| `RotorDynamics` (pure output) | `RoutineDecision` (pure output) |
| `AbstractFlightStackBridge` (seam) | `AbstractCommandPort` (seam) |
| `Px4Bridge` / `BridgeHost(Group)` (orchestrators) | `VehicleCommander` / `FleetCommander` |

- **`commands.py`** -- `MavlinkCommand` (frozen: a `MAV_CMD_*` id, seven float
  params, a label) plus pure constructors (`arm`, `takeoff`, `reposition`,
  `land`). `MAV_CMD_*` numbers and `PX4_FORCE_ARM_MAGIC` are named int constants
  so the module never imports pymavlink.
- **`observation.py`** -- `RoutineObservation` (frozen): the bridge's pure
  `VehicleState` and `BridgeStatus` plus the command-feedback fields (`last_ack:
  CommandAck | None`, `failsafe: bool`, `sim_time_us`). `MAV_RESULT_*` are named
  int constants; `CommandAck.from_result` derives `accepted`.
- **`routine.py`** -- the pure plan and stepper. `RoutineStep` (a command, a
  completion predicate `Callable[[RoutineObservation], bool]`, a sim-time
  `timeout_s`, and `retry_on_temporarily_rejected`), `VehicleRoutine` (ordered
  steps), `RoutineState` (cursor/sent/baseline/step-start), and `RoutineDecision`
  (`SEND` / `WAIT` / `ADVANCE` / `DONE` / `FAIL`). `decide(routine, state,
  observation) -> (state, decision)` is the contract centerpiece: no sockets, no
  pymavlink, no Isaac. Predicate helpers: `armed()`, `ack_accepted(command_id)`,
  `climbed_at_least(min_climb_m, baseline_altitude_m=...)`.
- **`port.py`** -- `AbstractCommandPort` (the seam: `connect`, `disconnect`,
  `pump_heartbeat`, `send`, `poll_feedback`) and the PX4 `MavlinkCommandPort`,
  whose `pymavlink` import is function-scoped. The port absorbs the GCS-link
  concerns (connect, throttled heartbeat, `command_long` send, draining
  `COMMAND_ACK` / `HEARTBEAT` / `STATUSTEXT` / position telemetry) and exposes
  `is_handshaked`, `failsafe`, `failsafe_system_status`, and
  `latest_global_position`.
- **`commander.py`** -- `VehicleCommander` (one routine + one port + the
  `RoutineState`: runs `decide` and performs the single `port.send` side effect)
  and `FleetCommander` with an `ExecutionPolicy`: `INDEPENDENT` advances each
  routine freely; `STEP_SYNCHRONIZED` is a barrier (no vehicle advances to step
  `N+1` until every vehicle has completed step `N`), reproducing the
  arm-all-then-takeoff-all takeoff scenario.

## Pure / impure split

`decide` and every predicate are pure functions of `(routine, state,
observation)`; they are unit-tested with plain dataclasses (no Docker, Isaac, or
pymavlink). The only side effect in the layer is `AbstractCommandPort.send`
(MAVLink out) and the port's connect/heartbeat/drain. `VehicleCommander.step`
runs the pure `decide`, then performs the one `SEND` side effect, keeping the
decision and the I/O cleanly separated. Step timeouts are measured in **sim
time** (`observation.sim_time_us`) so the stepper stays pure; the driver pairs
that with its own wall-clock backstop for a fully stalled sim.

## `gcs_port` is control, not bridge

The command port talks the per-drone GCS MAVLink link
(`ResolvedVehicle.gcs_port = px4.gcs_baseport + id`) **only**. It never touches
the bridge's simulator MAVLink port or its sensor/actuator lockstep, and
symmetrically the bridge contract forbids the bridge touching `gcs_port`. The
commander is a separate control layer by design; it is stepped from the scenario
run loop (or, later, a dedicated control subscription), never from the bridge's
physics send/recv legs.

## Where the executor lives: two drivers, one pure core

There are two drivers over the same `decide(...)` core and `AbstractCommandPort`
seam; only the driver differs.

**Standalone takeoff scenario**
(`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`): a blocking, staged
script that builds a per-drone `VehicleRoutine = [arm, takeoff]`, a
`FleetCommander(STEP_SYNCHRONIZED)` over `MavlinkCommandPort`s, and drives it to
`DONE` while ticking physics and asserting bridge connectivity; the climb/hover
envelope and `post_takeoff_hook` stay in the scenario, fired after the routine
completes.

**Live `drone_spawner` extension**
(`isaac/runtime/extensions/drone_spawner/DroneSpawner_python/control_driver.py`,
`FleetControlDriver`): the runtime path for `make run-all` / `make run-isaac`.
The same pure core re-expressed as a non-blocking, tick-driven phase machine
(`WAIT_BRIDGE -> CONNECT_GCS -> HANDSHAKE -> PREARM_SETTLE -> FETCH_TARGET ->
BUILD -> DRIVE -> DONE`) stepped from the **Kit app-update event stream** -- a
control tick separate from the bridge's physics-step subscription, never the
bridge's send/recv legs. It assembles each `RoutineObservation` from a state-only
`IsaacArticulationIO` (`read_state`, no rotor paths), the bridge snapshot via the
public `BridgeHost.status` property, and `MavlinkCommandPort.poll_feedback()` /
`failsafe`, then runs the same `FleetCommander(STEP_SYNCHRONIZED)` barrier. A
`FAIL` decision, a PX4 failsafe, or a phase timeout logs a warning and stops the
routine (it never raises into the sim -- runtime is recoverable, unlike the test
which asserts); after `DONE` it idles, keeping the GCS link warm so PX4-console
control returns to the user. The extension builds it in `_setup_drone` (after the
bridge group starts) and stops it in `_teardown_drone`.

**Config is the single source of truth.** The Hydra `routine` group
(`RoutineConfig`, always composed onto `EnvConfig`) carries `steps` plus the
control-flow tunables. `routine.steps` is the on/off switch: a non-empty list of
`{kind: arm}` / `{kind: takeoff, climb_m: ...}` mappings makes the extension build
the driver on Play; an empty list (the default) means manual operation. The pure
`build_vehicle_routine(steps, *, target_latitude_deg, target_longitude_deg,
target_altitude_m, command_ack_timeout_s)` maps the descriptors plus a per-drone
takeoff target into a `VehicleRoutine` (unknown `kind` or a `takeoff` missing
`climb_m` raises `ValueError`); it is pure (no Isaac/pymavlink) and is the shared
seam for the eventual standalone/live merge. `configs/runtime_arm_takeoff.yaml`
demos the headless routine via `make run-all HYDRA_CONFIG=...`. The GCS link
baseport stays in the `px4` group (`px4.gcs_baseport`); the per-drone `gcs_port`
is derived from it.

**Still deferred:** merging the standalone scenario onto `build_vehicle_routine` /
a shared driver so the integration-test routine and the live routine become one,
a headless extension-path integration target, and per-vehicle routine overrides
(today `routine` is fleet-global).
