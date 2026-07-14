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
- `src/robotics_experiments/rl/press/actuation.py`: frozen params + pure
  `f(params, inputs) -> outputs` pipeline, explicitly kept as the numpy oracle
  for the torch twin.
- `src/robotics_experiments/config/`: the whole resolved config graph is
  deep-frozen after loading (`_immutability.py`), so the paradigm's inputs are
  actually immutable, not just conventionally so.

The gaps cluster in the areas the convention does not yet name (the
`environment` package and the older `lightswitch_spawner` extension) plus a few
impurity leaks inside otherwise-frozen values. In rough order of value:

1. `environment/switches.py`: resolved in step 1 with frozen state and pure
   latch transitions.
2. `lightswitch_spawner` extension: resolved in step 2 with a frozen runtime
   state and a thin event-wiring shell.
3. `commanding/port.py`: resolved in step 3 by folding feedback through an
   immutable `CommandPortStatus` snapshot.
4. Writable `np.ndarray` payloads inside frozen dataclasses -- aliasable
   mutable state behind a frozen facade (step 4).
5. `environment/stage.py`: hidden module-global RNG, so spawns are impure and
   unreproducible (step 5).
6. Value-object dataclasses that are incidentally mutable (`drone/_airframe.py`,
   scenario `_FlightSample`) (step 6).
7. `px4_bridge/mavlink.py`: a large body of project-owned lockstep protocol
   state inside the adapter -- allowed by the current convention, listed as a
   deferred decision (step 7).

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

Goal: frozen dataclasses are actually immutable. Today `frozen=True` values
carry writable, aliasable `np.ndarray` fields, so "immutable" state can be
mutated through a leaked reference -- the one hole in the paradigm's
foundation. `thrust_model.py` already solved this locally with
`_readonly_array` (copy + `setflags(write=False)`).

Where to apply (producer seams, not every consumer):

- `px4_bridge/vehicle_state.py` `VehicleState`: freeze arrays in
  `IsaacArticulationIO.read_state` (`isaac_articulation.py`) and in
  `VehicleState.hover_at`.
- `px4_bridge/contract.py` `ActuatorCommands`: freeze in
  `Px4MavlinkPort._references` / `_zeros` (`mavlink.py`).
- `px4_bridge/thrust_model.py` `RotorDynamics`: freeze the arrays built in
  `commands_to_dynamics` (both implementations) and the zero-dynamics built in
  `bridge.py` `recv_and_apply`'s cutoff branch.
- `commanding/_routine_model.py` `RoutineState.baseline_position_enu`: freeze
  at capture in `_routine_stepper.decide` (the `np.asarray(...)` on step
  entry).
- `rl/press/actuation.py` `MixerGeometry` / `ActuationParams`: freeze in
  `build_actuation_params` (and wherever `MixerGeometry` is constructed).

Mechanics:

- Generalize the existing `_readonly_array` idea: a tiny private helper per
  owning module, or promote the thrust-model helper to a shared private
  function inside `px4_bridge` (no new public API; keep `__all__` unchanged).
- Copy only when the source array aliases external memory (e.g. Isaac handle
  outputs); `read_state` already builds fresh arrays, so `setflags` alone is
  enough there -- keep per-tick allocations flat (this is the 250 Hz path; see
  the hot-path note in `AGENTS.md`'s functional-style balance).
- Watch the one known writer risk: `_pegasus_sensors._to_pegasus_state` does
  `np.asarray(state.position_enu)` (no copy) and hands arrays to vendored
  Pegasus sensor code. Run the bridge sensor tests first; if Pegasus mutates
  its `State` fields, copy at that boundary (in `_to_pegasus_state`, not by
  unfreezing `VehicleState`).

Tests: add a small assertion per touched constructor test (e.g. in
`test_thrust_model.py`, `test_px4_bridge_sensors.py`,
`test_commanding_routine.py`, `test_press_actuation.py`) that writing to a
field raises `ValueError` (`array is read-only`). Full existing suites must
stay green -- they are the detector for any hidden in-place writer.

Done when: every `np.ndarray` field on a `frozen=True` dataclass in
`commanding`, `px4_bridge`, and `rl/press` is non-writable at construction;
integration smoke (`make test-integration`) passes.

Risk: medium-low. The failure mode is loud (`ValueError` on write), which is
exactly the point; any that fire reveal real aliasing bugs.

## Step 5: Replace the hidden global RNG in switch spawning with an injected seeded RNG

Goal: `spawn_switches` becomes a deterministic function of its inputs --
referential transparency and reproducible experiment layouts.

Problem today (`src/robotics_experiments/environment/stage.py`):
`_sample_positions` uses module-global `random.choice` / `random.uniform`, and
`_spawn_lamps` draws lamp heights the same way. Layouts differ every run and
cannot be reproduced from config.

Changes:

- Config: add `spawn_seed: int` as a bare-annotation field to
  `LightSwitchConfig` in the owning module under
  `src/robotics_experiments/config/` and set the project value in the Hydra
  `configs/` YAML (tunables live in YAML; raise on invalid config as usual).
- `stage.py`: `spawn_switches` builds `rng = random.Random(config.spawn_seed)`
  and threads it into `_sample_positions` and `_spawn_lamps` (parameter, not a
  global). No other call-site changes: both helpers are private to the module.

Tests: extend
`tests/src/robotics_experiments/environment/test_environment_switches.py` (or
the config tests if stage.py stays untestable off-Isaac -- `stage.py` imports
`pxr`, so put the determinism test wherever the existing spawner extension
test stubs Isaac): same seed -> identical positions; different seed ->
different positions. `tests/src/robotics_experiments/config/test_config.py`
gains the new field.

Done when: `grep -rn "random\." src/robotics_experiments/environment/` shows
only the injected `Random` usage; two runs with the same config produce the
same prim positions.

Risk: low. Behavior change is the (desired) determinism; docs
(`docs/EXPERIMENT_CONFIG.md`) mention the new key.

## Step 6: Freeze incidentally-mutable value-object dataclasses

Goal: value objects that are never mutated are declared immutable, so the
default in the codebase is `frozen=True` and mutability is a visible,
deliberate exception.

Where:

- `src/robotics_experiments/drone/_airframe.py`: `BoxPart`, `CylinderPart`,
  `BoomPart`, `ArmGeometry`, `ArmPlacement`, `AirframeConfig`,
  `PusherDriveConfig`, `PusherConfig`, `BuilderConfig` -- all plain
  `@dataclass`. `_replace_pos` already uses `dataclasses.replace`, which works
  on frozen dataclasses, and no in-place field assignment exists in the
  module.
- `isaac/runtime/standalone/tests/_px4_takeoff_scenario.py`: `_FlightSample`.

Changes: add `frozen=True` to each; fix any construction path that turns out
to assign post-init (none expected).

Tests: existing `tests/src/robotics_experiments/drone/test_airframe.py`,
`tests/isaac/runtime/standalone/test_standalone_px4_scripts.py`, and
`tests/isaac/runtime/standalone/test_usd_builder_config.py` cover the touched
code; no new tests needed.

Done when: `grep -n "@dataclass$" src/robotics_experiments isaac/runtime -r`
(excluding pegasus) returns only classes with a stated reason to stay mutable;
pre-commit clean.

Risk: minimal.

## Step 7 (deferred decision): lockstep progress state inside `Px4MavlinkPort`

`px4_bridge/mavlink.py` keeps ~15 mutable fields of project-owned lockstep
protocol progress (send/receive counters, last-timestamp bookkeeping,
heartbeat pacing, prime-then-block bootstrap, freshness tripwire, disconnect
reason) inside the MAVLink adapter. This is permitted by the current
convention ("mutable classes are limited to external resource adapters"), and
`read_status()` already exposes it immutably -- but by weight it is the largest
block of non-functional project logic left, and the phase logic
(`_heartbeat_phase`, `_sensor_phase`, `_needs_to_block_for_actuator`,
`_actuator_is_fresh`) is pure decision-making testable without a socket.

Options:

- A (status quo, recommended until a concrete need appears): keep the port as
  the documented adapter exception. It is well covered
  (`test_px4_bridge_lockstep.py`, `_actuator.py`, `_gating.py`, `_sensors.py`
  drive it through fakes), and the bridge contract doc treats its sequencing
  as load-bearing. Cost of change > benefit today.
- B (full extraction): frozen `LockstepProgress` + pure transitions returning
  `(actions, next_state)` interpreted against the Pegasus backend, mirroring
  the `bridge_host.py` pattern. Prerequisites if attempted: read
  `isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md`
  first; preserve exact sequencing (prime-then-block, drain-newest, cadence
  guard); benchmark per-tick allocation on the 250 Hz path before and after;
  land behind the full px4 integration suite
  (`make test-integration-all`).

Decision needed only if the port grows new protocol behavior or its tests
become the bottleneck; do not do this as opportunistic cleanup.
