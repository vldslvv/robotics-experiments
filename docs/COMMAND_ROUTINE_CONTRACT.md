# Command / routine contract (`robotics_experiments.commanding`)

The commanding layer is the high-level control counterpart to the PX4 bridge
contract. Where the bridge turns `VehicleState -> ActuatorCommands ->
RotorDynamics` across a swappable sink, the command layer turns a pure plan plus
an observation into a decision across a swappable MAVLink port. It is a domain
leaf on the core import path and remains free of Isaac/Kit dependencies. Its
concrete port eagerly imports the ordinary `pymavlink` dependency.

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
`tick_routine_driver(context=..., state=..., tick=...)`. One immutable
`DriverTick` carries the shared logical control time and monotonic wall time for
the entire fleet cycle.

- **`commands.py`** -- `MavlinkCommand` (frozen: a `MAV_CMD_*` id, seven float
  params, a label) plus pure constructors (`arm`, `takeoff`, `reposition`,
  `land`, `set_offboard_mode`). Also `PositionSetpoint` (frozen: a geodetic
  lat/lon/alt + NED yaw), the wire payload for a streamed OFFBOARD setpoint.
  `MAV_CMD_*` numbers, the PX4 offboard main-mode id, and `_PX4_FORCE_ARM_MAGIC`
  are named int constants so the module never imports pymavlink.
- **`observation.py`** -- `RoutineObservation` (frozen): the bridge's pure
  `VehicleState` and `BridgeStatus` plus the command-feedback fields (`last_ack:
  CommandAck | None`, `failsafe: bool`, `sim_time_us`) and an opaque
  `task_observation`. The runtime may inject a task-owned frozen observation
  there; commanding passes it to the selected adapter without importing or
  interpreting its type. `MAV_RESULT_*` are named int constants;
  `CommandAck.from_result` derives `accepted`.
- **Pure routine implementation** -- `_routine_model.py`, `_routine_stepper.py`,
  `_routine_predicates.py`, and `_routine_builder.py` separately own the pure
  model, decision state machine, predicates/gates, and descriptor assembly.
  `RoutineStep` (an **optional**
  command -- `None` for a verification-only step such as `hover`; a completion
  predicate `_StepPredicate = Callable[[RoutineObservation, baseline], bool]`; a
  sim-time `timeout_s`; `retry_on_temporarily_rejected`; an optional `hold_s`
  sustained-completion window; and a tuple of continuous `invariants`),
  `VehicleRoutine` (ordered steps), `RoutineState`
  (cursor/sent/active-step baseline/step-start/`hold_since_us` plus nested
  frozen `offboard` progress containing policy cadence, transmitter cadence,
  the cached payload, actual transmission count, and mode request/acceptance),
  and `RoutineDecision`.
  Its outcome is `CONTINUE` / `ADVANCE` / `DONE` / `FAIL`, independent of its
  optional `command` and `setpoint` effects; both effects may be present, and
  the driver applies the setpoint first. `StreamSetpoint` is the union
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

  A **streaming step** (a speedless `goto` or any `policy`, carrying a
  `_StreamingSpec`) is a step whose `decide` may emit independent setpoint and
  command effects. It
  primes `prime_ticks` actual OFFBOARD transmissions, sends the one-shot
  offboard-mode command, then streams at `driver.offboard_setpoint_rate_hz`
  until the step completes. The last priming transmission and mode command may
  share a cycle. Matching `MAV_CMD_DO_SET_MODE` ACKs are part of that pure state machine:
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
  can complete. The transmitter phase and cached payload also remain continuous
  across consecutive OFFBOARD steps. A fixed `goto` payload is calculated once
  on step entry and replaces the cache, then is sent on the next scheduled
  emission; routine boundaries never create an extra transmission.

  A `goto` with an authored finite positive `ground_speed_m_s` instead sends
  one acknowledged `MAV_CMD_DO_REPOSITION`, with the requested ground speed in
  param1 and the same ENU-to-geodetic target in params 5-7. PX4 owns trajectory
  generation and braking; the value is a guided target speed, not a hard
  velocity governor. At step entry, commanding derives param4 yaw from the
  captured world-ENU baseline with the same ENU-to-NED heading conversion used
  by a speedless goto; temporary rejection retries that same derived command.
  After the first accepted ACK, commanding reissues the target exactly once so
  PX4 Navigator sees a fresh reposition triplet after its Commander-driven
  loiter-mode transition (PX4 only consumes that triplet while it is younger
  than 500 ms). Position, heading, cross-track, flight-envelope, and timeout
  predicates are identical to the speedless form. Omitting the field preserves
  the streaming OFFBOARD lifecycle exactly. Guided goto entry clears any prior
  OFFBOARD cache, and a later policy therefore performs fresh OFFBOARD priming.

  The **`policy` step** is the task-neutral policy plug-in over the same streaming
  lifecycle. Its `_StreamingSpec` carries an injected `PolicyAdapter` instead
  of a fixed geodetic target. It evaluates
  when a new OFFBOARD stream starts and then only when its adapter's continuous
  exact-average artifact-owned cadence is due; its latest payload is held
  without interpolation while the independent setpoint streamer runs. A
  compatible OFFBOARD-step boundary does not reset either cadence. The payload is a
  `VelocitySetpoint` or `AttitudeRateSetpoint` per the `rl.action.kind`.
  Commanding constructs one frozen `PolicyInput(vehicle_state, sim_time_us,
  task_observation, reference_position_enu)` per evaluation. The reference is
  the authored `target_enu`, or the immutable step-entry position when omitted.
  Each adapter owns opaque frozen, non-`None` state and returns a
  `PolicyResult(state, setpoint, complete, failure_reason)`. Stateless adapters
  use `()`. Commanding stores the successor state but never interprets the
  policy's task, observation schema, contact phase, or network architecture.
  Adapters are composed at the runtime layer and injected as
  `DriverParams.policy_factory`; `build_vehicle_routine` calls the narrowed
  `policy_factory(*, artifact, hold_s)` per
  `{kind: policy, artifact, timeout_s[, hold_s, label, target_enu, switch_index]}`
  descriptor
  and raises `ValueError` without it (or on a missing/mismatched artifact), which
  the driver's BUILD phase routes through `on_failure` as `FailureKind.COMMAND`.
  Artifact metadata contains the typed `PolicyInterface`, which selects
  `hover` or `press-approach` in the private runtime policy module without adding
  a task-specific routine kind. The fleet orchestrator only preflights
  configured artifacts, prepares interface-required resources, and injects the
  module's adapter factory.
  Controller alignment reuses this lifecycle through the narrow optional
  `build_drivers(..., policy_factory=...)` seam in its standalone PX4 shell.
  That shell injects a deterministic immutable velocity-probe adapter for an
  ordinary `{kind: policy}` descriptor and calls
  `spawn_fleet(..., preflight_policy_artifacts=False)` because the probe is not
  a production policy artifact. Production callers retain the default
  preflight and omit the factory argument, continuing to use the private
  runtime policy factory. Alignment adds no routine kind, artifact interface,
  or task branch to commanding.
  Because the artifact path is per step, a single routine can sequence multiple
  policy steps backed by different named artifacts, such as
  `.build/training/hover/latest` followed by
  `.build/training/press-approach/latest`.
  `PolicyAdapter.policy_rate_hz` carries each artifact's immutable cadence, so those
  steps may run at different rates; neither `DriverParams` nor
  `VehicleRoutine` repeats a routine-wide policy rate.
  `hover` uses the authored `target_enu`, or the entry position when it is
  omitted, and requires `hold_s`; its adapter completes after that duration.
  `press-approach` consumes the injected exact 34-value `PressObservation`;
  `curriculum_stage` selects the stage-specific adapter. The implemented
  `approach` adapter waits for the configured handoff gate, fails on contact or
  invalid observation, and completes with a zero terminal setpoint after the
  configured 600 continuous samples inside the complete hover envelope embedded
  in the policy artifact from its pinned flight-parameters run. The adapter
  reconstructs world-ENU position error and contact-frame attitude error from
  each live observation and evaluates the same measured bounds as training and
  held-out evaluation. Task-owned range and velocity-reference fields still
  define the target and control diagnostics, but cannot replace or relax the
  embedded completion envelope. Contact remains a failure. It does not require
  `hold_s`.
  Stage-less and unknown-stage press artifacts fail preflight. Policy
  completion never advances before PX4 accepts OFFBOARD. A policy timeout
  reports the adapter's final opaque state rather than goto-specific text. The
  retained approach state exposes the exact first policy-step observation,
  handoff, valid actor-evaluation count, stable-goal count, initial/closest
  range, completion, and contact failure while commanding remains unaware of
  those fields. Generic OFFBOARD progress separately numbers cached policy
  generations and the latest generation actually transmitted, allowing an
  invocation smoke test to distinguish policy entry or `handoff_wait` from a
  real actor result that reached the command port.
  An adapter failure becomes `FailureKind.COMMAND`; a bad flight still fails as
  `FailureKind.ENVELOPE` through the unchanged flight gates. Follow a policy
  step with an explicit-command step (e.g. `land`), not a command-less verifier:
  the verifier would keep re-streaming the last policy payload.

  `FailureKind.ENVELOPE` is a flight-envelope invariant breach (flew out of
  bounds); `FailureKind.COMMAND` is a command rejection or step timeout. The pure
  stepper emits only those two; the driver adds `FailureKind.FAILSAFE` (a PX4
  failsafe system status on the GCS link) and `FailureKind.TIMEOUT` (a driver
  phase exceeding its wall-clock backstop) for its own terminal conditions.
  `DriverEnvironment.on_failure(kind, message)` is a typed, non-throwing
  notification: an armed failure must retain control long enough to recover.
- **`port.py`** -- the PX4-specific `MavlinkCommandPort` (the side-effect seam:
  `connect`, `disconnect`, `pump_heartbeat`, `send`, `send_setpoint`,
  `send_velocity_setpoint`, `send_attitude_target`, `poll_feedback`), which
  imports its ordinary `pymavlink` dependency eagerly. The port
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
  `robotics_experiments.rl.policy_action_to_offboard`; every learned task,
  including handoff-gated switch pressing, remains a `{kind: policy}` routine
  step. There is intentionally no `{kind: press}` branch in commanding.
- **`driver.py`** -- the frozen, per-drone `DriverContext` (parameters, port,
  environment, and label), progress-only `RoutineDriverState`,
  and stateless `create_routine_driver` / `tick_routine_driver` transition
  functions: a phase
  machine (`WAIT_BRIDGE -> CONNECT -> HANDSHAKE -> PREARM_SETTLE -> FETCH_TARGET
  -> BUILD -> DRIVE -> SUCCEEDED`, or `RECOVERING -> FAILED`) that takes one
  drone's routine plus an
  `MavlinkCommandPort` and a `DriverEnvironment` (the injected seam: bridge
  connectivity, sim time, observation assembly, `log`, `on_failure`,
  `on_step_advanced`) and drives it from a fresh GCS link to a completed (or
  stopped) routine. It imports no Isaac modules. Creation requires
  `create_routine_driver(context=..., tick=...)`, and ticking requires the same
  explicit `DriverTick` contract, so the runtime can share one sim/wall-clock
  snapshot across the whole fleet and tests can supply deterministic time.
  `DriverParams` is
  constructed directly by the runtime from the per-drone subset of
  `RoutineConfig` it needs plus the global `driver`
  execution heuristics (the `goto` heading epsilon and the wall-clock backstop
  floor/multipliers from the Hydra `driver` group). `RoutineDriverState.failure`
  preserves the original kind/message and recovery outcome.

On an armed routine failure, the driver stops policy setpoints and enters
`RECOVERING`. It sends `MAV_CMD_NAV_LAND`, retries temporary rejection, and
waits for disarm after acceptance. A hard rejection or missing acknowledgement
stops the OFFBOARD stream and waits for PX4's independently configured
offboard-loss action; an already active PX4 failsafe is monitored without a
competing command. Recovery ends in terminal `FAILED`, preserving the original
failure even when landing succeeds. Unarmed failures need no recovery and become
`FAILED` immediately. `routine.failure_recovery_timeout_s` bounds the entire
attempt.

## Pure / impure split

`decide` and every predicate are pure functions of `(routine, state,
observation)`; they are unit-tested with plain dataclasses (no Docker, Isaac, or
pymavlink). The only side effects in the layer are the port's send methods
(`send`, `send_setpoint`, `send_velocity_setpoint`, `send_attitude_target` --
MAVLink out) and the port's connect/heartbeat/drain. `tick_routine_driver`
receives context, state, and the shared tick separately, stores `RoutineState`
directly in the progress state, runs pure `decide`, then performs its optional
effects (setpoint first, then command). A `policy` step's due network forward
pass runs inside `decide` as an injected deterministic callable (numpy in/out, no I/O),
so it does not break the purity split. Step timeouts are measured in **sim time**
(`observation.sim_time_us`) so the stepper stays pure; the driver pairs that with
its own wall-clock backstop for a fully stalled sim.

`make benchmark-commanding` is the isolated driver performance gate. It measures
fixed pure-stepper and inert-adapter `tick_routine_driver` snapshots through this
public package-root contract on the host. It does not replace the full flight
benchmark, which remains authoritative for system-level Isaac/PX4 behavior and
timing. The workload, statistical, and acceptance contracts are documented in
`docs/COMMANDING_BENCHMARK.md`.

## `gcs_port` is control, not bridge

The command port talks the per-drone GCS MAVLink link
(`VehicleConfig.gcs_port = bridge.gcs_baseport + id`) **only**. It never touches
the bridge's simulator MAVLink port or its sensor/actuator lockstep, and
symmetrically the bridge contract forbids the bridge touching `gcs_port`. The
driver is a separate control layer by design; it is ticked from the adapter's
50 Hz physics-derived cyclic executive, never from the bridge's physics
send/recv legs.

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
then calls `tick_drivers` from one separate **PhysX-step** subscription. The
fleet cadence emits at `driver.control_rate_hz` (50 Hz by default), synchronized
across vehicles and independent of render/app-update frequency. Its `make_env`
is **forgiving**: `on_failure` (a `FAIL`,
a PX4 failsafe, or a phase timeout) logs a warning while the driver performs
landing recovery (it never raises into the sim); after terminal `SUCCEEDED` or
`FAILED` the driver keeps the GCS link warm. STOP/PAUSE and shutdown drop the tick and
tear down. Setup errors are caught at this lifecycle boundary, recorded in the
public failure field, and cleaned before a later Play retry; a failed setup never
installs the PhysX control subscription.

The live shell also exposes an immutable status snapshot for dependent Kit
plugins. It reports each vehicle label, simulated time, completed-step count,
driver phase, failure/recovery outcome, bridge status, active step, latest
already-consumed motion/task observation, navigation progress, and policy
progress. OFFBOARD state retains both the newest actor generation and the last
setpoint predicted to have reached the port; because the imperative driver only
commits the successor after the send succeeds, live snapshots never present a
pending actor output as transmitted. Snapshot reads do not expose or mutate
`DriverContext`, `RoutineDriverState`, or `FleetState`, perform another sensor
read, invoke an actor, or send a command. Observation therefore does not create
a second executor path. The existing runtime failure field is the authoritative
signal that a forgiving driver stopped early.

Every routine step emits machine-readable `ROUTINE_STAGE` JSON records for its
`start` and `complete` boundaries (or `failed` instead of `complete`), carrying
the configured `kind`, built label, zero-based index, and exact
`sim_time_us`, plus `ground_speed_m_s` when the descriptor authors it. The fleet
environment sends the same record to the normal shell log and the
vehicle-filtered bridge log. Flight analysis must select a policy interval from
these boundaries rather than treating takeoff, waypoint travel, or
return/landing as learned-policy behavior.

An adapter may also return immutable diagnostic strings in `PolicyResult`.
The pure stepper carries them to `RoutineDecision`, and the imperative driver
logs them only after applying that cycle's wire effects. Press deployment uses
this path for `PRESS_POLICY` actor-evaluation records. The driver separately
emits `POLICY_SETPOINT` JSON after each policy setpoint is successfully handed
to the concrete MAVLink port, so actor cadence and actual transmission cadence
remain distinguishable.

**Standalone scenario**
(`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`):
a blocking runner that drives the same lifecycle functions, adds the
**integration-test-only** checks the runtime does not own (ground settle,
bridge first-heartbeat, the pre-arm start-tilt check), and ticks every flying
drivers from the same PhysX subscription while its host loop calls
`SimulationApp.update()`. Callback failures are captured and re-raised after
the update returns so strict exceptions keep their types. It then runs the strict
bridge-log scan. Its `on_failure` notification never raises. Under `--strict`,
after recovery reaches terminal `FAILED`, the shell raises
`ExpectedInstabilityDetected` for a `FailureKind.ENVELOPE` breach,
`Px4FailsafeDetected` for a failsafe, and `AssertionError` otherwise; without
`--strict` it only reports the terminal failure. The `post_takeoff_hook`
(negative-test fault injection) is wired to
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
`{kind: policy, artifact, timeout_s[, hold_s, label, target_enu]}` /
`{kind: land, timeout_s}`)
drive the authored commands; empty steps traverse the same phase machine,
complete as `SUCCEEDED` without commands, and retain terminal GCS servicing.
The pure `build_vehicle_routine(*, steps, target_latitude_deg,
target_longitude_deg, target_altitude_m, command_ack_timeout_s, gates,
takeoff_timeout_s, hover_settle_s, home_lat_deg, home_lon_deg, home_alt_m,
goto_position_tol_m, goto_heading_tol_deg, goto_cross_track_m,
goto_offboard_prime_ticks, offboard_setpoint_rate_hz,
goto_min_heading_dist_m, policy_factory)` maps the
descriptors plus a
per-drone takeoff target into a `VehicleRoutine` (unknown `kind`, any step missing a required
field, or a `policy` step without an injected factory, raises `ValueError`); it
is pure (no Isaac/pymavlink) and is the shared
seam both drivers use. The takeoff step completes on a real climb
(`climbed_at_least(min_climb_m)`), the hover step on `hovering_stable(...)` held
for `hold_s`, the forward step on observed world +X/east displacement, the `goto`
step on `reached_world_position(...)` (absolute world ENU) plus heading
alignment, the `policy` step when its adapter reports completion after accepted
OFFBOARD entry, and the land step once PX4 disarms. Flight gates are continuous
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
Its transmission phase is unchanged, as it is across consecutive OFFBOARD
steps; explicit commands such as `land` clear the stream before they begin.
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
