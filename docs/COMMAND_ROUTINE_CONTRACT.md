# Command / routine contract (`robotics_experiments.commanding`)

The commanding layer is the high-level control counterpart to the PX4 bridge
contract. Where the bridge turns `VehicleState -> ActuatorCommands ->
RotorDynamics` across a swappable sink, the command layer turns a pure plan plus
an observation into a decision across a swappable MAVLink port. It is a domain
leaf on the core import path: every module imports neither Isaac nor pymavlink at
module scope, so `import robotics_experiments` succeeds with both absent.

`robotics_experiments.commanding` intentionally re-exports only the production
runtime surface (`RoutineDriver`, `DriverParams`, `DriverEnvironment`,
`RoutineObservation`, `CommandAck`, `MavlinkCommandPort`, `FailureKind`, and the
deploy-time wire payloads `VelocitySetpoint` / `AttitudeRateSetpoint` consumed by
`press.deploy`). The pure stepper, command constructors, routine builder, and
lower-level seams live in non-underscored modules (`commands`, `observation`,
`routine`, `port`, `driver`), each declaring its own `__all__` and
underscore-prefixing its private members; only the `_commander` module (the
internal `VehicleCommander`, absent from the package `__init__`) stays
underscored. Production code imports from the package; unit tests import the
submodules directly as white-box regression guards.

## The four-part contract

| Bridge | Command layer |
|---|---|
| `VehicleState` (pure input) | `RoutineObservation` (pure input) |
| `commands_to_dynamics` (pure transform) | `decide(...)` (pure stepper) |
| `RotorDynamics` (pure output) | `RoutineDecision` (pure output) |
| `AbstractFlightStackBridge` (seam) | `AbstractCommandPort` (seam) |
| `Px4Bridge` / `BridgeHost` (orchestrators) | `VehicleCommander` / `RoutineDriver` |

Like the bridge, the command layer is **per drone**: one `VehicleCommander` and
one `RoutineDriver` drive one drone's routine. There is no fleet object - drones
are independent (cross-drone synchronization is deliberately deferred).

- **`commands.py`** -- `MavlinkCommand` (frozen: a `MAV_CMD_*` id, seven float
  params, a label) plus pure constructors (`arm`, `takeoff`, `reposition`,
  `land`, `set_offboard_mode`). Also `PositionSetpoint` (frozen: a geodetic
  lat/lon/alt + NED yaw), the wire payload for a streamed OFFBOARD setpoint.
  `MAV_CMD_*` numbers, the PX4 offboard main-mode id, and `_PX4_FORCE_ARM_MAGIC`
  are named int constants so the module never imports pymavlink.
- **`observation.py`** -- `RoutineObservation` (frozen): the bridge's pure
  `VehicleState` and `BridgeStatus` plus the command-feedback fields (`last_ack:
  CommandAck | None`, `failsafe: bool`, `sim_time_us`). `MAV_RESULT_*` are named
  int constants; `CommandAck.from_result` derives `accepted`.
- **`routine.py`** -- the pure plan and stepper. `RoutineStep` (an **optional**
  command -- `None` for a verification-only step such as `hover`; a completion
  predicate `_StepPredicate = Callable[[RoutineObservation, baseline], bool]`; a
  sim-time `timeout_s`; `retry_on_temporarily_rejected`; an optional `hold_s`
  sustained-completion window; and a tuple of continuous `invariants`),
  `VehicleRoutine` (ordered steps), `RoutineState`
  (cursor/sent/active-step baseline/step-start/`hold_since_us` plus the
  streaming counters `setpoints_streamed`/`offboard_requested`), and
  `RoutineDecision`
  (`SEND` / `STREAM` / `WAIT` / `ADVANCE` / `DONE` / `FAIL`; `SEND` carries a
  `command`, `STREAM` carries a `PositionSetpoint`, `FAIL` carries a
  `failure_kind`). `decide(routine, state, observation) -> (state, decision)` is
  the contract centerpiece: no sockets, no pymavlink, no Isaac. It checks the
  step `invariants` first (a breach -> `FAIL(ENVELOPE)`), then completion (held
  for `hold_s` -> `ADVANCE`), then ack-rejection/timeout (-> `FAIL(COMMAND)`).
  Completion predicate helpers: `armed()`, `ack_accepted(command_id)`,
  `climbed_at_least(min_climb_m)` (relative to the active step baseline),
  `hovering_stable(...)`, `moved_forward_at_least(distance_m)`,
  `reached_world_position(x, y, z, tol)` (the first **absolute** world-frame
  predicate), `heading_aligned(...)`, and `disarmed()`. Invariant gate helpers
  (each returns a breach message or `None`): `tilt_below`, `_lateral_drift_below`,
  `cross_track_drift_below`, `cross_track_to_segment_below` (perpendicular
  deviation from the active step baseline to target path, direction-agnostic),
  `_vertical_speed_within`, `_angular_speed_below`. `EnvelopeGates` bundles the
  gate thresholds wired onto the flight steps.

  A **streaming step** (the `goto` kind, carrying a `_StreamingSpec`) is the one
  step whose `decide` emits `STREAM` rather than a single `SEND`: it primes
  `prime_ticks` OFFBOARD setpoints, sends the one-shot offboard-mode command, then
  streams the setpoint every tick until the waypoint is reached or the step times
  out. Matching `MAV_CMD_DO_SET_MODE` ACKs are part of that pure state machine:
  accepted ACKs keep streaming, temporary rejections keep the setpoint stream
  alive and retry OFFBOARD inside the same step timeout, and hard rejections fail
  the step as `FailureKind.COMMAND`. The streamed target is the absolute
  world-ENU waypoint converted to a
  geodetic setpoint via `px4_bridge.frames.synth_gps_from_enu` against the `px4`
  home anchor -- frame-consistent **by construction** with the GPS the bridge
  feeds PX4, so flying the setpoint lands the drone at the world-ENU waypoint
  (using the global frame avoids the EKF local-NED origin ambiguity). Verification
  uses absolute Isaac world-ENU ground truth, independent of PX4's EKF. If a
  command-less verifier such as `hover` follows a `goto`, the stepper keeps
  streaming the last OFFBOARD setpoint throughout that verifier. PX4 remains in
  OFFBOARD after the waypoint is reached, so stopping the stream between `goto`
  and `hover` would trigger offboard signal-loss failsafe before the hover hold
  can complete.

  `FailureKind.ENVELOPE` is a flight-envelope invariant breach (flew out of
  bounds); `FailureKind.COMMAND` is a command rejection or step timeout. The pure
  stepper emits only those two; the driver adds `FailureKind.FAILSAFE` (a PX4
  failsafe system status on the GCS link) and `FailureKind.TIMEOUT` (a driver
  phase exceeding its wall-clock backstop) for its own terminal conditions, so
  `DriverEnvironment.on_failure(kind, message)` is fully typed. A driver
  reacts in kind: the strict driver maps `ENVELOPE` to its instability error,
  `FAILSAFE` to `Px4FailsafeDetected`, and the rest to a plain assertion; a
  forgiving driver logs any of them and stops.
- **`port.py`** -- `AbstractCommandPort` (the seam: `connect`, `disconnect`,
  `pump_heartbeat`, `send`, `send_setpoint`, `send_velocity_setpoint`,
  `send_attitude_target`, `poll_feedback`) and the PX4
  `MavlinkCommandPort`, whose `pymavlink` import is function-scoped. The port
  absorbs the GCS-link concerns (connect, throttled heartbeat, `command_long`
  send, `SET_POSITION_TARGET_GLOBAL_INT` setpoint stream, draining `COMMAND_ACK` /
  `HEARTBEAT` / `STATUSTEXT` / position telemetry) and exposes `is_handshaked`,
  `failsafe`, `failsafe_system_status`, and `latest_global_position`. The
  velocity / attitude-rate senders are the **deploy-time seam for the F1 RL
  action levels**: `send_velocity_setpoint` serializes a `VelocitySetpoint` to
  `SET_POSITION_TARGET_LOCAL_NED` (body velocity, `MAV_FRAME_BODY_NED`) for the
  `velocity` action, and `send_attitude_target` serializes an
  `AttitudeRateSetpoint` to `SET_ATTITUDE_TARGET` (collective thrust + body rates)
  for `ctbr`. The pure body-FLU action -> body-FRD payload mapping lives in
  `press.deploy.press_action_to_offboard`; the mission-level press behavior that
  streams them (a `PressDriver` / `{kind: press}` step) is F3, not this layer yet.
- **`_commander.py`** -- `VehicleCommander` (one routine + one port + the
  `RoutineState`: runs `decide` and performs the single side effect -- `port.send`
  on `SEND`, `port.send_setpoint` on `STREAM`). One commander drives one drone;
  there is no fleet object or barrier.
- **`driver.py`** -- the pure, per-drone `RoutineDriver`: a tick-driven phase
  machine (`WAIT_BRIDGE -> CONNECT -> HANDSHAKE -> PREARM_SETTLE -> FETCH_TARGET
  -> BUILD -> DRIVE -> DONE`) that takes one drone's routine plus an
  `AbstractCommandPort` and a `DriverEnvironment` (the injected seam: bridge
  connectivity, sim time, observation assembly, `log`, `on_failure`,
  `on_step_advanced`) and drives it from a fresh GCS link to a completed (or
  stopped) routine. It imports no Isaac/pymavlink; the wall-clock backstop uses an
  injectable `monotonic`, so it is unit-testable with fakes. `DriverParams`
  (`DriverParams.from_routine_config(rc, driver=..., home_*=...)`) is the
  per-drone subset of `RoutineConfig` it needs plus the global `driver`
  execution heuristics (the `goto` heading epsilon and the wall-clock backstop
  floor/multipliers from the Hydra `driver` group). The strict-vs-forgiving
  reaction lives entirely in the env's `on_failure` (raise vs log).

## Pure / impure split

`decide` and every predicate are pure functions of `(routine, state,
observation)`; they are unit-tested with plain dataclasses (no Docker, Isaac, or
pymavlink). The only side effects in the layer are `AbstractCommandPort.send` and
`send_setpoint` (MAVLink out) and the port's connect/heartbeat/drain.
`VehicleCommander.step` runs the pure `decide`, then performs the one side effect
(`SEND` -> `send`, `STREAM` -> `send_setpoint`), keeping the decision and the I/O
cleanly separated. Step timeouts are measured in **sim time**
(`observation.sim_time_us`) so the stepper stays pure; the driver pairs that with
its own wall-clock backstop for a fully stalled sim.

## `gcs_port` is control, not bridge

The command port talks the per-drone GCS MAVLink link
(`ResolvedVehicle.gcs_port = px4.gcs_baseport + id`) **only**. It never touches
the bridge's simulator MAVLink port or its sensor/actuator lockstep, and
symmetrically the bridge contract forbids the bridge touching `gcs_port`. The
commander/driver is a separate control layer by design; it is ticked from the
adapter's control loop (the standalone blocking loop or the live Kit app-update
subscription), never from the bridge's physics send/recv legs.

## Where the executor lives: one orchestrator, two thin shells

The pure `RoutineDriver` is the single executor, and the host-agnostic
`robotics_experiments.runtime.FleetOrchestrator` is the single place that *builds*
one driver per flying drone (only for drones whose resolved `routine.steps` is
non-empty). The orchestrator owns the whole runtime pipeline (open scene -> apply
settings -> spawn fleet -> start the lockstep bridge group -> `build_drivers` ->
teardown); a thin per-host **shell** drives it, supplying only what differs: the
tick source and the failure reaction. The reaction is injected as a `make_env`
callable that builds the generic `FleetDriverEnv` (the `DriverEnvironment` seam,
in `runtime/orchestrator.py`) per flying drone -- it reads connectivity/sim-time
from the drone's `BridgeHost`, state from a state-only `IsaacArticulationIO`, and
the bridge snapshot via `BridgeHost.status`, over each drone's
`MavlinkCommandPort`, and forwards `log` / `on_failure` / `on_step_advanced` to
the shell's callables. The phase machine and flight verification live in the
driver/routine; the shells differ only in tick source and reaction.

**Live `experiment_runtime` extension**
(`isaac/runtime/extensions/experiment_runtime/ExperimentRuntime_python/extension.py`):
the runtime path for `make run-all` / `make run-isaac`. On timeline PLAY it has
the orchestrator spawn the fleet, start the bridge group, and build the drivers,
then ticks every flying drone's driver from one **Kit app-update** subscription --
a control tick separate from the bridge's physics-step subscription, never the
bridge's send/recv legs. Its `make_env` is **forgiving**: `on_failure` (a `FAIL`,
a PX4 failsafe, or a phase timeout) logs a warning and the driver idles (it never
raises into the sim); after `DONE` the driver keeps the GCS link warm so
PX4-console control returns to the user. STOP/PAUSE and shutdown drop the tick and
tear down.

**Standalone scenario**
(`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`):
a blocking runner that drives the same orchestrator phases, adds the
**integration-test-only** checks the orchestrator does not own (ground settle,
bridge first-heartbeat, the pre-arm start-tilt check), and ticks every flying
driver in a `while not orchestrator.all_done: simulation_app.update(); for d:
d.tick()` loop (raw ticks so a strict exception propagates), then runs the strict
bridge-log scan. Its `make_env` is **strict** under `--strict` (the integration
test): `on_failure` raises `ExpectedInstabilityDetected` for a
`FailureKind.ENVELOPE` breach, `Px4FailsafeDetected` for a failsafe, and
`AssertionError` otherwise; without `--strict` it logs and the driver idles, like
the live path. The `post_takeoff_hook` (negative-test fault injection) is wired to
the first drone's `on_step_advanced` and fires when the takeoff step completes, so
the hover step's envelope catches the resulting instability.

**Config is the single source of truth, per drone.** The Hydra `routine` group
(`RoutineConfig`, always composed onto `EnvConfig`) carries `steps` plus the
control-flow tunables **and the flight-envelope gates** (`max_*` / `hover_max_*`,
the takeoff target/timeout, and the hover settle). There is no separate
`integration_test` group: the integration test is just this routine driven with
`--strict`. The routine is **per vehicle**: each `vehicles[]` entry may carry a
`routine` partial, merged over the global `routine` exactly like `drone`/`px4` in
`config.load._resolve_fleet` (the resolved `RoutineConfig` lands on
`ResolvedVehicle.routine`). `routine.steps` is the per-drone on/off switch: a
drone whose resolved `steps` is non-empty
(`{kind: arm}` / `{kind: takeoff, climb_m, min_climb_m}` /
`{kind: hover, hold_s}` / `{kind: forward, distance_m, timeout_s}` /
`{kind: goto, x, y, z, timeout_s[, position_tol_m, heading_tol_deg]}` /
`{kind: land, timeout_s}`)
gets a `RoutineDriver`; empty (the global default) means manual. Mixed fleets are
allowed (some drones fly, others stay manual). The pure `build_vehicle_routine(steps, *, target_latitude_deg,
target_longitude_deg, target_altitude_m, command_ack_timeout_s, gates,
takeoff_timeout_s, hover_settle_s, home_lat_deg, home_lon_deg, home_alt_m,
goto_position_tol_m, goto_heading_tol_deg, goto_cross_track_m,
goto_offboard_prime_ticks, goto_min_heading_dist_m)` maps the descriptors plus a
per-drone takeoff target into a `VehicleRoutine` (unknown `kind`, or any step missing a required
field, raises `ValueError`); it is pure (no Isaac/pymavlink) and is the shared
seam both drivers use. The takeoff step completes on a real climb
(`climbed_at_least(min_climb_m)`), the hover step on `hovering_stable(...)` held
for `hold_s`, the forward step on observed world +X/east displacement, the `goto`
step on `reached_world_position(...)` (absolute world ENU) plus heading
alignment, and the land step once PX4 disarms. Flight gates are continuous
invariants on takeoff, hover, forward, and goto. Each step captures its own
baseline when it begins: takeoff measures climb from takeoff start; hover,
forward, and every goto leg measure gates from their own step start. Takeoff and
hover use the lateral-drift threshold as radial horizontal excursion from that
active step baseline; forward
uses it as cross-track drift so the commanded +X/east displacement itself does not
breach the envelope; goto uses a direction-agnostic cross-track-to-path gate
(`goto_cross_track_m`) so an arbitrary-heading leg is allowed while a sideways
excursion off the planned path still trips. A hover after `goto` remains
verification-only from the config's perspective, but the stepper continues the
last OFFBOARD setpoint stream while the sustained hover gate is being measured.
The home anchor the `goto` builder
needs is **not** a routine knob -- it comes from the per-vehicle `px4` group
(`home_lat_deg`/`home_lon_deg`/`home_alt_m`) and is passed by the orchestrator
into `DriverParams.from_routine_config`. The routine-execution heuristics are
also not routine knobs: the Hydra `driver` group (`DriverConfig`, global, not
merged per vehicle) carries `goto_min_heading_dist_m` (below this horizontal
baseline-to-target distance a `goto` leg holds heading and the alignment
predicate passes trivially) and the driver wall-clock backstops
(`prearm_backstop_floor_s`, `prearm_backstop_multiplier`,
`drive_budget_multiplier`), passed as `driver=` into
`DriverParams.from_routine_config`.
`configs/experiment/routine/takeoff.yaml` is the shared takeoff routine;
`configs/runtime_arm_takeoff.yaml` demos it headless via
`make run-all HYDRA_CONFIG=...`, `configs/runtime_drone_navigate.yaml` demos an
absolute-waypoint `goto` routine the same way, and
`configs/integration_test_takeoff.yaml` selects the takeoff routine and only
changes the scene. The GCS link baseport stays
in the `px4` group (`px4.gcs_baseport`); the per-drone `gcs_port` is derived from
it.

> Hover modeling note: the historical windowed *range* gates (min/max climb and
> horizontal range over a sampling window) are now expressed as **sustained
> instantaneous-rate** bounds (`hover_max_abs_vertical_speed_m_s` /
> `hover_max_abs_horizontal_speed_m_s`) held continuously for the hover step's
> `hold_s` -- the "hover stable for N seconds" model.

**Still deferred:** cross-drone synchronization (today every drone's
`RoutineDriver` runs independently; a barrier/fleet-sync layer can be
re-introduced over the per-drone drivers when a use case needs it).
