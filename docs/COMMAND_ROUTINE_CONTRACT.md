# Command / routine contract (`robotics_experiments.commanding`)

The commanding layer is the high-level control counterpart to the PX4 bridge
contract. Where the bridge turns `VehicleState -> ActuatorCommands ->
RotorDynamics` across a swappable sink, the command layer turns a pure plan plus
an observation into a decision across a swappable MAVLink port. It is a domain
leaf on the core import path: every module imports neither Isaac nor pymavlink at
module scope, so `import robotics_experiments` succeeds with both absent.

`robotics_experiments.commanding` is the only public import surface for the
package. Code outside `src/robotics_experiments/commanding` imports the command
payloads, pure routine model, driver seam, and concrete PX4 GCS port from the
package root. The public-concept implementation modules (`commands`,
`observation`, `routine`, `port`, `driver`) use non-private names because they
host symbols exported through the root; they are not separate supported import
surfaces. Tests exercise the same root contract as production consumers.

## The four-part contract

| Bridge | Command layer |
|---|---|
| `VehicleState` (pure input) | `RoutineObservation` (pure input) |
| `commands_to_dynamics` (pure transform) | `decide(...)` (pure stepper) |
| `RotorDynamics` (pure output) | `RoutineDecision` (pure output) |
| `Px4MavlinkPort` lockstep seam | `MavlinkCommandPort` (GCS control seam) |
| `Px4BridgeState` / `BridgeHost` | `DriverContext` + `RoutineDriverState` |

Like the bridge, the command layer is **per drone**: one frozen `DriverContext`
holds immutable parameters and resource adapters, while one frozen
`RoutineDriverState` holds progress and is advanced by
`tick_routine_driver(context=..., state=...)`. There is no command fleet object - drones are independent
(cross-drone synchronization is deliberately deferred).

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
- **Pure routine implementation** -- `_routine_model.py`, `_routine_stepper.py`,
  `_routine_predicates.py`, and `_routine_builder.py` separately own the pure
  model, decision state machine, predicates/gates, and descriptor assembly.
  `RoutineStep` (an **optional**
  command -- `None` for a verification-only step such as `hover`; a completion
  predicate `_StepPredicate = Callable[[RoutineObservation, baseline], bool]`; a
  sim-time `timeout_s`; `retry_on_temporarily_rejected`; an optional `hold_s`
  sustained-completion window; and a tuple of continuous `invariants`),
  `VehicleRoutine` (ordered steps), `RoutineState`
  (cursor/sent/active-step baseline/step-start/`hold_since_us` plus the
  streaming counters `setpoints_streamed`/`offboard_requested`), and
  `RoutineDecision`
  (`SEND` / `STREAM` / `WAIT` / `ADVANCE` / `DONE` / `FAIL`; `SEND` carries a
  `command`, `STREAM` carries a `StreamSetpoint` -- the union
  `PositionSetpoint | VelocitySetpoint | AttitudeRateSetpoint`, dispatched by
  the commander on the concrete payload type -- and `FAIL` carries a
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

  A **streaming step** (the `goto` and `policy` kinds, carrying a
  `_StreamingSpec`) is a
  step whose `decide` emits `STREAM` rather than a single `SEND`: it primes
  `prime_ticks` OFFBOARD setpoints, sends the one-shot offboard-mode command, then
  streams the setpoint every tick until the waypoint is reached or the step times
  out. Matching `MAV_CMD_DO_SET_MODE` ACKs are part of that pure state machine:
  accepted ACKs keep streaming, temporary rejections keep the setpoint stream
  alive and retry OFFBOARD inside the same step timeout, and hard rejections fail
  the step as `FailureKind.COMMAND`. The streamed target is the absolute
  world-ENU waypoint converted to a
  geodetic setpoint via package-root `synth_gps_from_enu` against the `px4`
  home anchor -- frame-consistent **by construction** with the GPS the bridge
  feeds PX4, so flying the setpoint lands the drone at the world-ENU waypoint
  (using the global frame avoids the EKF local-NED origin ambiguity). Verification
  uses absolute Isaac world-ENU ground truth, independent of PX4's EKF. If a
  command-less verifier such as `hover` follows a `goto`, the stepper keeps
  streaming the last OFFBOARD setpoint throughout that verifier. PX4 remains in
  OFFBOARD after the waypoint is reached, so stopping the stream between `goto`
  and `hover` would trigger offboard signal-loss failsafe before the hover hold
  can complete.

  The **`policy` step** is the RL deploy plug-in over the same streaming
  lifecycle: its `_StreamingSpec` carries a `setpoint_fn` (built from the
  injected `PolicyStepFn`) instead of a fixed geodetic target, so each tick the
  streamed payload is `policy_fn(observation.vehicle_state, target_enu)` -- a
  `VelocitySetpoint` or `AttitudeRateSetpoint` per the `rl.action.kind`.
  The `PolicyStepFn` is composed at the runtime layer
  (`robotics_experiments.rl.press.build_policy_step_fn`: validated artifact load + hover
  observation packing + `press_action_to_offboard`) and injected as
  `DriverParams.policy_fn_factory`; `build_vehicle_routine` calls the factory
  per `{kind: policy, artifact, hold_s, timeout_s[, name, target]}` descriptor
  and raises `ValueError` without it (or on a missing/mismatched artifact), which
  the driver's BUILD phase routes through `on_failure` as `FailureKind.COMMAND`.
  Because the artifact path is per step, a single routine can sequence multiple
  policy steps backed by different named artifacts, such as
  `.build/training/hover/latest` followed by `.build/training/press/latest`.
  Without an authored `target` the policy holds the position captured when the
  step began. The step's completion predicate is `_always` + `hold_s`, gated by
  accepted OFFBOARD entry, so it completes after PX4 accepts OFFBOARD and then
  `hold_s` of continuous in-envelope flight. A bad policy fails as
  `FailureKind.ENVELOPE` through the unchanged flight gates. Follow a policy
  step with an explicit-command step (e.g. `land`), not a command-less verifier:
  the verifier would keep re-streaming the last policy payload.

  `FailureKind.ENVELOPE` is a flight-envelope invariant breach (flew out of
  bounds); `FailureKind.COMMAND` is a command rejection or step timeout. The pure
  stepper emits only those two; the driver adds `FailureKind.FAILSAFE` (a PX4
  failsafe system status on the GCS link) and `FailureKind.TIMEOUT` (a driver
  phase exceeding its wall-clock backstop) for its own terminal conditions, so
  `DriverEnvironment.on_failure(kind, message)` is fully typed. A driver
  reacts in kind: the strict driver maps `ENVELOPE` to its instability error,
  `FAILSAFE` to `Px4FailsafeDetected`, and the rest to a plain assertion; a
  forgiving driver logs any of them and stops.
- **`port.py`** -- the PX4-specific `MavlinkCommandPort` (the side-effect seam:
  `connect`, `disconnect`, `pump_heartbeat`, `send`, `send_setpoint`,
  `send_velocity_setpoint`, `send_attitude_target`, `poll_feedback`), whose
  `pymavlink` import is function-scoped. The port
  absorbs the GCS-link concerns (connect, throttled heartbeat, `command_long`
  send, `SET_POSITION_TARGET_GLOBAL_INT` setpoint stream, draining `COMMAND_ACK` /
  `HEARTBEAT` / `STATUSTEXT` / position telemetry). Feedback interpretation is
  a pure `_fold_feedback(status, msg)` transition over immutable
  `CommandPortStatus`; the adapter drain only rebinds that snapshot and applies
  the learned target system/component to the live connection on the first
  heartbeat. Its `read_status()` method returns the immutable snapshot containing
  `is_handshaked`, `failsafe`, `failsafe_system_status`, and
  `latest_global_position`. The
  velocity / attitude-rate senders are the **deploy-time seam for the F1 RL
  action levels**: `send_velocity_setpoint` serializes a `VelocitySetpoint` to
  `SET_POSITION_TARGET_LOCAL_NED` (body velocity, `MAV_FRAME_BODY_NED`) for the
  `velocity` action, and `send_attitude_target` serializes an
  `AttitudeRateSetpoint` to `SET_ATTITUDE_TARGET` (collective thrust + body rates)
  for `ctbr`. The pure body-FLU action -> body-FRD payload mapping lives in
  `robotics_experiments.rl.press.press_action_to_offboard`; the `{kind: policy}` routine step
  streams them in production (F2b), and the handoff-gated contact behavior
  (`{kind: press}`) is F3b on the same mechanism.
- **`driver.py`** -- the frozen, per-drone `DriverContext` (parameters, port,
  environment, label, and monotonic clock), progress-only `RoutineDriverState`,
  and stateless `create_routine_driver` / `tick_routine_driver` transition
  functions: a phase
  machine (`WAIT_BRIDGE -> CONNECT -> HANDSHAKE -> PREARM_SETTLE -> FETCH_TARGET
  -> BUILD -> DRIVE -> DONE`) that takes one drone's routine plus an
  `MavlinkCommandPort` and a `DriverEnvironment` (the injected seam: bridge
  connectivity, sim time, observation assembly, `log`, `on_failure`,
  `on_step_advanced`) and drives it from a fresh GCS link to a completed (or
  stopped) routine. It imports no Isaac/pymavlink; the wall-clock backstop uses an
  injectable `monotonic`, so it is unit-testable with fakes. `DriverParams` is
  constructed directly by the runtime from the per-drone subset of
  `RoutineConfig` it needs plus the global `driver`
  execution heuristics (the `goto` heading epsilon and the wall-clock backstop
  floor/multipliers from the Hydra `driver` group). The strict-vs-forgiving
  reaction lives entirely in the env's `on_failure` (raise vs log).

## Pure / impure split

`decide` and every predicate are pure functions of `(routine, state,
observation)`; they are unit-tested with plain dataclasses (no Docker, Isaac, or
pymavlink). The only side effects in the layer are the port's send methods
(`send`, `send_setpoint`, `send_velocity_setpoint`, `send_attitude_target` --
MAVLink out) and the port's connect/heartbeat/drain. `tick_routine_driver`
receives context and state separately, stores `RoutineState` directly in the
progress state, runs pure `decide`, then performs the one side
effect (`SEND` -> `send`, `STREAM` -> the matching send method). A `policy`
step's network forward pass runs
inside `decide` as an injected deterministic callable (numpy in/out, no I/O),
so it does not break the purity split. Step timeouts are measured in **sim time**
(`observation.sim_time_us`) so the stepper stays pure; the driver pairs that with
its own wall-clock backstop for a fully stalled sim.

## `gcs_port` is control, not bridge

The command port talks the per-drone GCS MAVLink link
(`VehicleConfig.gcs_port = bridge.gcs_baseport + id`) **only**. It never touches
the bridge's simulator MAVLink port or its sensor/actuator lockstep, and
symmetrically the bridge contract forbids the bridge touching `gcs_port`. The
driver is a separate control layer by design; it is ticked from the
adapter's control loop (the standalone blocking loop or the live Kit app-update
subscription), never from the bridge's physics send/recv legs.

## Where the executor lives: functional fleet state, two thin shells

The frozen `DriverContext` plus progress-only `RoutineDriverState` is the single
executor composition, and the host-agnostic `FleetState` lifecycle is the
single place that builds one composition per flying drone. Stateless functions own the pipeline (open scene -> apply
settings -> spawn fleet -> start the lockstep bridge group -> build drivers ->
teardown); a thin per-host **shell** explicitly rebinds every returned state and supplies
tick source and the failure reaction. The reaction is injected as a `make_env`
callable that builds the generic `FleetDriverEnv` (the `DriverEnvironment` seam,
in `runtime/orchestrator.py`) per flying drone -- it reads connectivity/sim-time
from the drone's `BridgeHost`, state from a state-only `IsaacArticulationIO`, and
the bridge snapshot via `read_bridge_host_status`, over each drone's
`MavlinkCommandPort`, and forwards `log` / `on_failure` / `on_step_advanced` to
the shell's callables. The phase machine and flight verification live in the
driver/routine; the shells differ only in tick source and reaction.

**Live `experiment_runtime` extension**
(`isaac/runtime/extensions/experiment_runtime/ExperimentRuntime_python/extension.py`):
the runtime path for `make run-all` / `make run-isaac`. On timeline PLAY it has
the fleet functions spawn the fleet, start the bridge group, and build drivers,
then calls `tick_drivers` from one **Kit app-update** subscription --
a control tick separate from the bridge's physics-step subscription, never the
bridge's send/recv legs. Its `make_env` is **forgiving**: `on_failure` (a `FAIL`,
a PX4 failsafe, or a phase timeout) logs a warning and the driver idles (it never
raises into the sim); after `DONE` the driver keeps the GCS link warm so
PX4-console control returns to the user. STOP/PAUSE and shutdown drop the tick and
tear down.

**Standalone scenario**
(`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`):
a blocking runner that drives the same lifecycle functions, adds the
**integration-test-only** checks the runtime does not own (ground settle,
bridge first-heartbeat, the pre-arm start-tilt check), and ticks every flying
drivers in a `while not fleet_is_done(state=fleet): simulation_app.update();
fleet = tick_drivers(state=fleet)` loop, then runs the strict
bridge-log scan. Its `make_env` is **strict** under `--strict` (the integration
test): `on_failure` raises `ExpectedInstabilityDetected` for a
`FailureKind.ENVELOPE` breach, `Px4FailsafeDetected` for a failsafe, and
`AssertionError` otherwise; without `--strict` it logs and the driver idles, like
the live path. The `post_takeoff_hook` (negative-test fault injection) is wired to
the first drone's `on_step_advanced` and fires when the takeoff step completes, so
the hover step's envelope catches the resulting instability.

**Config is the single source of truth, per drone.** Hydra composes a complete
`RoutineConfig` beneath every `vehicles.<label>` node. It carries typed `steps`,
control-flow tunables, and flight-envelope gates. There is no root routine or
separate integration-test group: strict mode drives the same per-vehicle routine.
Every bridge-enabled drone gets one `DriverContext` and one
`RoutineDriverState`. Non-empty `routine.steps`
(`{kind: arm}` / `{kind: takeoff, climb_m, min_climb_m}` /
`{kind: hover, hold_s}` / `{kind: forward, distance_m, timeout_s}` /
`{kind: goto, x, y, z, timeout_s[, position_tol_m, heading_tol_deg]}` /
`{kind: policy, artifact, hold_s, timeout_s[, name, target]}` /
`{kind: land, timeout_s}`)
drive the authored commands; empty steps traverse the same phase machine,
complete without commands, and retain terminal GCS servicing for manual control.
The pure `build_vehicle_routine(steps, *, target_latitude_deg,
target_longitude_deg, target_altitude_m, command_ack_timeout_s, gates,
takeoff_timeout_s, hover_settle_s, home_lat_deg, home_lon_deg, home_alt_m,
goto_position_tol_m, goto_heading_tol_deg, goto_cross_track_m,
goto_offboard_prime_ticks, goto_min_heading_dist_m, policy_fn_factory)` maps the
descriptors plus a
per-drone takeoff target into a `VehicleRoutine` (unknown `kind`, any step missing a required
field, or a `policy` step without an injected factory, raises `ValueError`); it
is pure (no Isaac/pymavlink) and is the shared
seam both drivers use. The takeoff step completes on a real climb
(`climbed_at_least(min_climb_m)`), the hover step on `hovering_stable(...)` held
for `hold_s`, the forward step on observed world +X/east displacement, the `goto`
step on `reached_world_position(...)` (absolute world ENU) plus heading
alignment, the `policy` step after accepted OFFBOARD entry plus `hold_s` of
continuous in-envelope flight, and the land step once PX4 disarms. Flight gates are continuous
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
(`home_lat_deg`/`home_lon_deg`/`home_alt_m`) and is passed by `build_drivers`
into the directly constructed `DriverParams`. The routine-execution heuristics are
also not routine knobs: the Hydra `driver` group (`DriverConfig`, global, not
merged per vehicle) carries `goto_min_heading_dist_m` (below this horizontal
baseline-to-target distance a `goto` leg holds heading and the alignment
predicate passes trivially) and the driver wall-clock backstops
(`prearm_backstop_floor_s`, `prearm_backstop_multiplier`,
`drive_budget_multiplier`), copied into that same `DriverParams` construction.
`configs/experiment/routine/takeoff.yaml` is the shared takeoff routine;
`configs/takeoff_quad.yaml` demos it headless via
`make run-all HYDRA_CONFIG=...`, `configs/navigate_quad_hex.yaml` demos an
absolute-waypoint `goto` routine the same way, and
the multi-drone takeoff config selects the same routine and only changes the
fleet. The GCS link baseport stays
in the `px4` group (`px4.gcs_baseport`); the per-drone `gcs_port` is derived from
it.

> Hover modeling note: the historical windowed *range* gates (min/max climb and
> horizontal range over a sampling window) are now expressed as **sustained
> instantaneous-rate** bounds (`hover_max_abs_vertical_speed_m_s` /
> `hover_max_abs_horizontal_speed_m_s`) held continuously for the hover step's
> `hold_s` -- the "hover stable for N seconds" model.

**Still deferred:** cross-drone synchronization (today every drone's
context/state pair runs independently; a barrier/fleet-sync layer can be
re-introduced over the per-drone drivers when a use case needs it).
