# Functional style review and improvement plan

Assessment of how well the codebase follows its functional paradigm (frozen
dataclasses for project-owned evolving state, stateless transition functions,
mutable classes only as external resource adapters read through immutable
snapshots), followed by improvement steps. Each numbered step is a
self-contained plan that can be picked up independently and in any order,
except where a step notes otherwise.

Out of scope, by design:

- The vendored Pegasus tree
  (`isaac/runtime/extensions/px4_sitl_brigde/pegasus/`): an unpinned upstream
  snapshot with stateful sensor models; not project style.
- Isaac/Kit/PhysX/ROS adapter classes themselves (`IsaacArticulationIO`,
  `BridgeHost`/`BridgeHostGroup`, `LightSwitchNode`, Isaac Lab envs): mutation
  there is the documented adapter allowance, and Isaac Lab's batched
  tensor-in-place style is a deliberate hot-path exception.

## Assessment

The paradigm is followed strongly on the main runtime path, and the two entry
points are model citizens:

- `src/robotics_experiments/runtime/orchestrator.py`: frozen `FleetState` /
  `SpawnedDrone` / `FlyingDrone`, every phase a `f(state=...) -> FleetState`
  transition; the live extension
  (`isaac/runtime/extensions/experiment_runtime/.../extension.py`) and the
  standalone scenario (`isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`)
  are thin imperative shells that only rebind `self._fleet` / `fleet` at event
  boundaries.
- `src/robotics_experiments/commanding/`: `decide` in `_routine_stepper.py` is
  a genuinely pure kernel `(routine, state, observation) -> (state, decision)`;
  `RoutineDecision` reifies the side effect as data and `_apply_decision` in
  `driver.py` is the single interpreter. `driver.py` threads frozen
  `RoutineDriverState` through stateless transitions, with even the wall clock
  injected (`DriverContext.monotonic`). Predicates and invariants
  (`_routine_predicates.py`) and the handoff gate (`rl/press/handoff.py`) are
  pure closures over frozen threshold dataclasses.
- `src/robotics_experiments/px4_bridge/`: frozen `Px4BridgeState` and
  `SimClockState` with stateless transitions (`send_sensors`, `recv_and_apply`,
  `advance_sim_clock`); `frames.py` is pure math; `thrust_model.py` even makes
  its coefficient arrays read-only (`_readonly_array`). Effects are confined to
  the two adapter seams (`Px4MavlinkPort`, `_ArticulationIO`) and read back via
  immutable `BridgeStatus` snapshots, with `is_connected()` as the documented
  narrow hot-path probe.
- `src/robotics_experiments/runtime/bridge_host.py`: even where PhysX forces a
  mutable holder (callbacks ignore return values), the state is a frozen
  `_BridgeHostState` replaced by module-level transitions; the class is a
  minimal cell.
- `src/robotics_experiments/rl/actuation.py`: frozen params + pure
  `f(params, inputs) -> outputs` pipeline, explicitly kept as the numpy oracle
  for the torch twin.
- `src/robotics_experiments/config/`: the whole resolved config graph is
  deep-frozen after loading (`_immutability.py`), so the paradigm's inputs are
  actually immutable, not just conventionally so.

The review findings and their outcomes, in rough order of value, are:

1. `environment/switches.py`: resolved in step 1 with frozen state and pure
   latch transitions.
2. `lightswitch_spawner` extension: resolved in step 2 with a frozen runtime
   state and a thin event-wiring shell.
3. `commanding/port.py`: resolved in step 3 by folding feedback through an
   immutable `CommandPortStatus` snapshot.
4. Writable `np.ndarray` payloads: resolved at the production construction
   seams in step 4 with copied or ownership-frozen arrays.
5. `environment/stage.py`: resolved in step 5 with a per-spawn seeded RNG from
   experiment configuration.
6. Incidentally mutable value-object dataclasses: resolved in step 6 by
   declaring the airframe graph and scenario flight samples frozen.
7. `px4_bridge/mavlink.py` lockstep progress: reviewed in step 7 and retained
   as the documented mutable adapter exception until a concrete extraction
   trigger appears.

## Step 1: Freeze `SwitchState` and make the button latch a pure transition

- [x] Complete

Goal: `environment` follows the same frozen-state + transition-function rule
as `runtime` / `commanding` / `px4_bridge`, and the articulation resource stops
living inside project-owned progress state.

Current behavior:

- `SwitchState` is frozen and contains only switch identity and latch progress;
  PhysX articulation handles are held separately by the live adapter.
- `apply_button_sample` is a pure transition returning the successor state and
  a toggle flag while leaving its input unchanged.
- `init_articulations` accepts prim paths and returns the complete handle list,
  or `None` so initialization can be retried after a failure.
- `step_switches` consumes parallel immutable switch snapshots and articulation
  handles, performs lamp and notification effects only for confirmed toggles,
  and returns the successor switch tuple.
- The stage spawner returns a tuple of frozen switch snapshots, and the live
  extension rebinds that tuple after every step while retaining articulation
  resources separately pending the runtime restructuring in step 2.
- Unit coverage threads transition results, verifies the original state remains
  unchanged, and exercises the extension's returned-state flow. The core-import
  regression continues to keep the pure switch module importable without Isaac.

Risk: low. The latch logic is small and fully covered by unit tests.

## Step 2: Rebuild `lightswitch_spawner` on the `experiment_runtime` pattern

- [x] Complete

Goal: the second live extension follows the same shape as the first -- a thin
Kit shell rebinding one frozen state through host-agnostic transitions --
instead of an imperative flag machine (`_switches_spawned`,
`_articulations_ready`, `_simulation_running`) with feature logic inline.

Current behavior:

- The Isaac-only `environment._runtime` module owns a frozen
  `SwitchesRuntimeState` and explicit idle, arming, running, and paused phases.
  Stateless transitions create the runtime, adopt a new stage, play, pause,
  retry articulation initialization, step switch interaction, and tear down.
- Play spawns and validates once, repeated play is inert, pause preserves the
  spawned prims and articulation resources, and resume returns to arming or
  running as appropriate. Failed validation tears down partial prims and leaves
  the runtime idle; failed articulation setup remains in arming for a later
  retry.
- The extension contains only Kit event routing, ROS node lifecycle and
  pumping, publisher adaptation, subscription handles, and one `self._state`
  rebinding. It carries no domain phase flags or switch composition logic.
- ROS QoS normalization and profile construction live beside the ROS node in
  `LightSwitch_python.qos`, outside the extension wiring module.
- Runtime-transition tests execute without Kit through injected stage and
  articulation adapters. Extension tests verify event delegation, successor
  rebinding, ROS publication, shutdown, and configured QoS behavior.

Risk: medium. This is a restructuring of a working extension; verify manually
with `make run-isaac` (switch spawn on Play, lamp toggle, ROS message) since
there is no integration test for switches.

## Step 3: Extract the pure feedback fold from `MavlinkCommandPort`

- [x] Complete

Goal: the GCS message-interpretation logic (handshake, failsafe detection, arm
statustext, global-position tracking) becomes a pure function over a frozen
state, leaving `MavlinkCommandPort` as transport only.

Current behavior:

- `CommandPortStatus` serves as both the adapter's immutable feedback state and
  its public read snapshot, avoiding a duplicate private model.
- `_fold_feedback(state, msg)` is a pure transition containing all message-type
  branching for heartbeat handshake and failsafe state, command acknowledgments,
  arm acceptance statustext, and valid global-position tracking.
- `poll_feedback` drains transport messages and rebinds `self._feedback`. When
  the fold first changes `is_handshaked` to true, the adapter applies the source
  system/component to the live MAVLink connection; no other protocol decisions
  remain in the mutable adapter.
- Direct fold tests cover handshake, failsafe entry and recovery, rejected and
  statustext acknowledgments, zero-fix rejection, and valid GPS capture. Existing
  port and driver tests continue to exercise the transport-facing behavior.

Risk: low. Pure refactor with existing coverage; no timing or socket behavior
changes.

## Step 4: Make numpy payloads in frozen dataclasses read-only

- [x] Complete

Goal: production frozen-state construction seams expose immutable NumPy
payloads instead of writable arrays behind a frozen facade.

Current behavior:

- `VehicleState.hover_at` owns and freezes its vectors. `IsaacArticulationIO`
  copies arrays returned by the external simulator adapter, converts angular
  velocity, and marks the complete sampled state read-only before publication.
- `Px4MavlinkPort` copies backend actuator references and freezes them; its zero
  command output is frozen in place. Both thrust-model implementations freeze
  their owned `RotorDynamics` arrays, and the tilt-cutoff path does the same for
  its replacement zero dynamics without extra copies.
- The routine stepper copies and freezes the active-step baseline when it is
  captured, so later observation aliases cannot change progress state.
- `build_actuation_params` copies every caller-owned mixer array into a frozen
  `MixerGeometry`, precomputes a read-only allocation pseudoinverse, and freezes
  all controller gain arrays.
- Focused tests attempt writes at every producer seam and receive `ValueError`;
  they also verify that mutating the caller's mixer geometry cannot affect the
  built parameter graph. Existing bridge sensor tests confirm that the Pegasus
  boundary consumes the read-only state without an in-place writer.

Risk: medium-low. The failure mode is loud (`ValueError` on write), which is
exactly the point; any that fire reveal real aliasing bugs.

## Step 5: Replace the hidden global RNG in switch spawning with an injected seeded RNG

- [x] Complete

Goal: `spawn_switches` becomes a deterministic function of its inputs --
referential transparency and reproducible experiment layouts.

Current behavior:

- `LightSwitchConfig.spawn_seed` is a required typed field, with the project
  value authored in `configs/experiment/switches/default.yaml`.
- Each `spawn_switches` call creates one local `random.Random` from that seed
  and threads it through position sampling and lamp-height sampling. No helper
  reads the module-global random generator.
- The complete switch/lamp layout is reproducible for a resolved experiment:
  repeated calls with the same seed produce identical positions and heights,
  while a different seed selects a different layout.
- Config composition asserts the shipped seed, and an Isaac-free stage test
  stubs the USD types to exercise deterministic layout sampling directly.

Risk: low. Behavior change is the (desired) determinism; docs
(`docs/EXPERIMENT_CONFIG.md`) mention the new key.

## Step 6: Freeze incidentally-mutable value-object dataclasses

- [x] Complete

Goal: value objects that are never mutated are declared immutable, so the
default in the codebase is `frozen=True` and mutability is a visible,
deliberate exception.

Current behavior:

- The complete USD-builder value graph (`BoxPart`, `CylinderPart`, `BoomPart`,
  `ArmGeometry`, `ArmPlacement`, `AirframeConfig`, `PusherDriveConfig`,
  `PusherConfig`, and `BuilderConfig`) is declared with
  `@dataclass(frozen=True)`. Derived placement updates continue to use
  `dataclasses.replace` rather than field assignment.
- Standalone PX4 scenario `_FlightSample` snapshots are frozen after sampling.
- The remaining plain dataclasses under the project package are Hydra schema
  inputs. They deliberately remain mutable during composition and are
  deep-frozen at the public config boundary, as described in the assessment.
- Existing airframe calculation, standalone import, and USD-builder config
  tests cover all construction and replacement paths.

Risk: minimal.

## Step 7: Retain lockstep progress state inside `Px4MavlinkPort`

- [x] Complete

Goal: make an explicit, evidence-based decision about the largest remaining
mutable block of project logic rather than extracting it opportunistically.

Current behavior and decision:

- `Px4MavlinkPort` retains send/receive counters, timestamp bookkeeping,
  heartbeat pacing, prime-then-block bootstrap, actuator freshness, and
  disconnect progress as mutable adapter state. This remains within the
  project's documented exception for external resource adapters.
- Consumers see one immutable `BridgeStatus` through `read_status()`;
  `is_connected()` remains the deliberate narrow hot-path probe that avoids a
  snapshot allocation on every physics tick.
- The heartbeat, sensor, actuator, freshness, and backlog-draining phases stay
  private to the adapter. Focused lockstep, actuator, gating, and sensor tests
  drive them through fake backends, while the full integration matrix exercises
  the load-bearing prime-then-block, all-sends-before-polls, cadence, and
  multi-vehicle behavior.
- The status quo is retained because no new protocol behavior or testability
  bottleneck supplies the documented trigger for a frozen `LockstepProgress`
  extraction. Such an extraction would add transition/action machinery and
  per-tick allocation risk without changing behavior today.
- Reconsider a full extraction only if the port gains protocol behavior or its
  adapter-level tests become a bottleneck. Any future attempt must preserve the
  PX4 bridge contract exactly, benchmark the 250 Hz path, and pass the full PX4
  integration suite.
