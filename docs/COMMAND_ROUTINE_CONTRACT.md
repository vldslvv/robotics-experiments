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
- **`routine.py`** -- the pure plan and stepper. `RoutineStep` (an **optional**
  command -- `None` for a verification-only step such as `hover`; a completion
  predicate `StepPredicate = Callable[[RoutineObservation, baseline], bool]`; a
  sim-time `timeout_s`; `retry_on_temporarily_rejected`; an optional `hold_s`
  sustained-completion window; and a tuple of continuous `invariants`),
  `VehicleRoutine` (ordered steps), `RoutineState`
  (cursor/sent/baseline/step-start/`hold_since_us`), and `RoutineDecision`
  (`SEND` / `WAIT` / `ADVANCE` / `DONE` / `FAIL`, the `FAIL` carrying a
  `failure_kind`). `decide(routine, state, observation) -> (state, decision)` is
  the contract centerpiece: no sockets, no pymavlink, no Isaac. It checks the
  step `invariants` first (a breach -> `FAIL(ENVELOPE)`), then completion (held
  for `hold_s` -> `ADVANCE`), then ack-rejection/timeout (-> `FAIL(COMMAND)`).
  Completion predicate helpers: `armed()`, `ack_accepted(command_id)`,
  `climbed_at_least(min_climb_m)` (relative to the captured baseline),
  `hovering_stable(...)`. Invariant gate helpers (each returns a breach message
  or `None`): `tilt_below`, `lateral_drift_below`, `vertical_speed_within`,
  `angular_speed_below`. `EnvelopeGates` bundles the gate thresholds wired onto
  the flight steps.

  `FailureKind.ENVELOPE` is a flight-envelope invariant breach (flew out of
  bounds); `FailureKind.COMMAND` is a command rejection or step timeout. A driver
  reacts in kind: a strict driver maps `ENVELOPE` to its instability error and
  `COMMAND` to a plain assertion; a forgiving driver logs either and stops.
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
seam, and both build the **same** routine from `routine.steps` via
`build_vehicle_routine` (the verification lives in the routine, not the driver);
only the reaction to a `FAIL` differs.

**Standalone takeoff scenario**
(`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`): a blocking, staged
script that builds a per-drone routine from `routine.steps`
(`[arm, takeoff, hover]`), a `FleetCommander(STEP_SYNCHRONIZED)` over
`MavlinkCommandPort`s, and drives it to `DONE` while ticking physics and asserting
bridge connectivity. The flight verification (climb, sustained hover, and the
continuous envelope invariants) is inside the routine, so the driver only adds
the generic infra checks and the **strict-vs-forgiving** reaction: `--strict`
(the integration test) raises `ExpectedInstabilityDetected` on a
`FailureKind.ENVELOPE` breach and `AssertionError` otherwise; without it a regular
run logs and stops, like the live extension. The `post_takeoff_hook` (used by the
negative test to inject a fault) fires once the tick the takeoff step completes,
so the hover step's envelope then catches the resulting instability.

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
control-flow tunables **and the flight-envelope gates** (`max_*` / `hover_max_*`,
the takeoff target/timeout, and the hover settle). There is no separate
`integration_test` group: the integration test is just this routine driven with
`--strict`. `routine.steps` is the on/off switch: a non-empty list of
`{kind: arm}` / `{kind: takeoff, climb_m, min_climb_m}` / `{kind: hover, hold_s}`
mappings makes a driver fly the routine; an empty list (the default) means manual
operation. The pure `build_vehicle_routine(steps, *, target_latitude_deg,
target_longitude_deg, target_altitude_m, command_ack_timeout_s, gates,
takeoff_timeout_s, hover_settle_s)` maps the descriptors plus a per-drone takeoff
target into a `VehicleRoutine` (unknown `kind`, or a `takeoff`/`hover` missing a
required field, raises `ValueError`); it is pure (no Isaac/pymavlink) and is the
shared seam both drivers use. The takeoff step completes on a real climb
(`climbed_at_least(min_climb_m)`) and the hover step on `hovering_stable(...)` held
for `hold_s`, with the gates as continuous invariants on both.
`configs/experiment/routine/takeoff.yaml` is the shared takeoff routine;
`configs/runtime_arm_takeoff.yaml` demos it headless via
`make run-all HYDRA_CONFIG=...`, and `configs/integration_test_takeoff.yaml`
selects the same routine and only changes the scene. The GCS link baseport stays
in the `px4` group (`px4.gcs_baseport`); the per-drone `gcs_port` is derived from
it.

> Hover modeling note: the historical windowed *range* gates (min/max climb and
> horizontal range over a sampling window) are now expressed as **sustained
> instantaneous-rate** bounds (`hover_max_abs_vertical_speed_m_s` /
> `hover_max_abs_horizontal_speed_m_s`) held continuously for the hover step's
> `hold_s` -- the "hover stable for N seconds" model.

**Still deferred:** extracting the duplicated pre-routine phase sequence (wait
bridge -> handshake -> settle -> fetch target) into a shared helper, and
per-vehicle routine overrides (today `routine` is fleet-global).
