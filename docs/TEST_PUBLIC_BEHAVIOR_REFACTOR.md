# Test public-behavior refactor

A resumable workflow for reworking the unit tests under `tests/` to assert on
**public behavior** instead of internals. `tests/test_bridge_host.py` was already
reworked this way; this document applies the same treatment to the rest of the
suite, one file at a time.

The two smells we remove are (1) external private-member access (`obj._foo`, read
*or* write) and (2) asserting on a mock/interaction-log when a public observable
exists. The capstone enables Ruff `SLF001` repo-wide so the win cannot regress
(this guard was deferred until the backlog was cleared).

## How to run this workflow

1. Pick the first unchecked step below and do **only** that step.
2. Run that step's verification.
3. Tick its checkbox in the Progress checklist (and commit).
4. **Clear context**, then start the next step fresh.

Steps are ordered easy -> hard so the pattern is established before the large
`test_lightswitch_spawner_extension.py` rework. Each step body is self-contained;
read the Shared conventions once and you should not need to re-derive anything.

## Reference exemplar

`tests/test_bridge_host.py` is the gold standard:

- Lifecycle tests assert observable public state (`host.is_running`,
  `host.is_connected`, `host.status`, `host.sim_time_us`) rather than a `calls`
  interaction-log.
- `test_bridge_host_uses_discovered_rotor_mapping` and
  `test_bridge_host_group_runs_all_sends_before_all_polls` are *wiring-contract*
  tests: they assert on what a collaborator received (construction kwargs) or the
  order of public calls the coordinator makes -- the adapter/coordinator's
  observable output.
- The fake `Px4Bridge` carries a comment explaining when an interaction assertion
  is legitimate and why it stopped recording send/recv.

## Shared conventions

**Decision rule -- KEEP vs FIX:**

- **FIX**: reading a unit's private state (`port._sensor_send_count`,
  `bridge._tilt_cutoff_count`, `extension._switches_spawned`) when a public
  property / return value / raised error exposes the same fact.
- **KEEP** (legitimate -- document inline): asserting on a fake's recorded
  interaction that mirrors a **documented contract seam** -- constructor params a
  collaborator received, commands sent to a port, callbacks invoked on an env,
  forces/dynamics applied to the articulation IO. These are an adapter's
  observable output. Prefer to *supplement* the seam assertion with the public
  observable rather than delete it.

**Two SLF-friendly techniques** (the capstone needs zero `obj._x` access):

- Convert `module._private_fn(...)` attribute calls to
  `from module import _private_fn; _private_fn(...)`. SLF001 only flags attribute
  access, not bare imported names -- no behavior change.
- Drive Kit-extension events by firing the **fake event stream's registered
  callbacks** (`for cb in stream.callbacks: cb(event)`) instead of calling the
  extension's private `_on_*` handler. This also exercises the real subscription
  wiring.

**Per-step verification:** `uv run pytest tests/<file> -q` passes with the same
test count (assertions change basis, not coverage), then
`uv run pre-commit run --all-files` is green. The behavior each test verifies must
be identical to before.

**Fakes:** `tests/_fakes.py` needs **no** changes -- its recorded seams are
intentional. Test-local fakes may be adjusted (e.g. rename a fake's `_drones` to a
public attribute).

## Progress checklist

- [x] 1. `test_fleet_orchestrator.py`
- [x] 2. `test_isaac_lab_workflow.py`
- [x] 3. `test_commanding_port.py`
- [x] 4. `test_commanding_routine.py`
- [ ] 5. `test_drone_spawn.py`
- [ ] 6. `test_experiment_runtime_extension.py`
- [ ] 7. `test_px4_bridge_gating.py`
- [ ] 8. `test_px4_bridge_sensors.py`
- [ ] 9. `test_px4_bridge_actuator.py`
- [ ] 10. `test_px4_bridge_lockstep.py`
- [ ] 11. `test_lightswitch_spawner_extension.py`
- [ ] 12. Audit the already-clean files
- [ ] 13. Capstone: enable Ruff `SLF001`

---

## Step 1 -- `test_fleet_orchestrator.py` (Small)

- **Unit / public surface:** `FleetOrchestrator`; public properties include
  `drones`, `bridged`, `flying`, `all_done`, `stage`.
- **Smell:** `orch._group.hosts` count read.
- **Reframing:** assert `len(orch.bridged) == 1` via the public property instead
  of reaching into `_group.hosts`. Where `group = orch._group` is captured before
  `teardown()` to later check `group.stopped`, reach the group through the public
  surface or the test-local fake instead of `orch._group`.
- **KEEP:** the orchestrator phase-sequence assertions on the fake's `calls` list
  (wiring seam).
- **Verify:** `uv run pytest tests/test_fleet_orchestrator.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 2 -- `test_isaac_lab_workflow.py` (Minimal)

- **Unit / public surface:** Isaac Lab registration --
  `register_isaac_lab_tasks() -> bool`, the registered task id, and
  `DroneHoverEnvCfg`. `action_space` / `observation_space` are public Gymnasium
  attributes -> KEEP those assertions.
- **Smell:** `registration._DRONE_HOVER_TASK_ID` attribute access.
- **Reframing:** assert against the literal task-id string (the
  `Isaac-Drone-Hover-Direct-v0`-style id) or import the constant by name
  (`from ... import _DRONE_HOVER_TASK_ID`) so there is no `module._x` access.
- **Verify:** `uv run pytest tests/test_isaac_lab_workflow.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 3 -- `test_commanding_port.py` (Small)

- **Unit / public surface:** `MavlinkCommandPort.poll_feedback`; public
  observables `failsafe`, `failsafe_system_status`, `latest_global_position`, and
  the returned `CommandAck`.
- **Smell:** `port._mav = _FakeConn(...)` private write to inject the connection.
- **Reframing:** make the connection an **injectable seam**. The
  `MavlinkCommandPort` module is under `src/robotics_experiments/commanding/` and
  dials its pymavlink connection inside `connect()`; add an optional constructor
  parameter (e.g. `connection=None`) that, when provided, is used instead of
  dialing pymavlink, and have the test pass the fake there. (Alternative:
  monkeypatch the pymavlink `mavutil.mavlink_connection` factory.) This is the one
  small production change in the whole workflow; it is covered by the existing
  commanding tests.
- **KEEP:** all other assertions (already public).
- **Verify:** `uv run pytest tests/test_commanding_port.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 4 -- `test_commanding_routine.py` (Small)

- **Unit / public surface:** pure `_decide` + `_VehicleCommander` + gate
  predicates. Mostly clean already: asserting `decision.kind` / `decision.command`
  and the commands captured in `FakeCommandPort.sent` are legitimate seams.
- **Smell:** a few tests call `forward_step.invariants` / `takeoff_step.invariants`
  predicates directly in isolation (white-box predicate poking).
- **Reframing:** drive the gate breach/pass through `_decide()` with an
  observation that violates / satisfies the gate and assert
  `decision.kind is _DecisionKind.FAIL` (or `is not`). Not SLF-blocking
  (`invariants` is a public field on the step), but it removes the isolated
  predicate calls.
- **Verify:** `uv run pytest tests/test_commanding_routine.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 5 -- `test_drone_spawn.py` (Small)

- **Unit / public surface:** `spawn_drone` / `despawn_drone`; the observable is
  the USD prims authored on the stage.
- **Smell:** the `_last_translate(stage)` helper iterates the fake prim's internal
  `xform_ops` list.
- **Reframing:** compute the drone's world position through the public USD API --
  `UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())`,
  then transform the origin -- the same way the reference articulation test reads
  pose.
- **KEEP:** the `random.uniform` monkeypatch (deterministic-placement seam) and
  the `carb` log-fallback assertion (documented warning path).
- **Verify:** `uv run pytest tests/test_drone_spawn.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 6 -- `test_experiment_runtime_extension.py` (Small)

- **Unit / public surface:** the `experiment_runtime` `Extension` wiring to a fake
  orchestrator; the observable is the orchestrator phase sequence (`orch.calls`),
  which is a legitimate wiring seam -> KEEP.
- **Smells:** callback-identity comparison against private handlers
  (`app.updates.callbacks == [extension._open_scene_when_ready]`); reads of
  `extension._scene_open_sub` / `extension._update_sub`; `orch._drones = [...]`
  injection.
- **Reframing:**
  - assert a callback is registered by non-empty / count
    (`assert app.updates.callbacks`) instead of identity to a private method;
  - drive events by firing the fake stream's registered callbacks
    (`for cb in stream.callbacks: cb(event)`) rather than calling
    `extension._on_*`;
  - rename the **test-local** fake orchestrator's `_drones` to a public attribute
    (the real orchestrator exposes `.drones` / `.flying`) and configure the
    branches through it.
- **Verify:** `uv run pytest tests/test_experiment_runtime_extension.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 7 -- `test_px4_bridge_gating.py` (Small)

- **Unit / public surface:** `Px4Bridge` gating; public `status`, `is_connected`,
  `is_armed`. The forces/spin applied to `_FakeArticulationIO.applied_forces` /
  `spin_updates` are the bridge's downstream output -> KEEP.
- **Smell:** `bridge._tilt_cutoff_count` counter reads (about 5).
- **Reframing:** assert the gating *effect* -- forces zeroed on the gated ticks
  via `io.applied_forces` -- instead of the internal counter.
- **KEEP:** `port.states` / `port.sim_times` are already supplemented with
  force/spin assertions; leave them as seam.
- **Verify:** `uv run pytest tests/test_px4_bridge_gating.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 8 -- `test_px4_bridge_sensors.py` (Small)

- **Unit / public surface:** `Px4MavlinkPort` sensor contract; public `status`
  (with `sensor_send_count` etc.).
- **Smells:** `port._sensor_noise_enabled` and `port._physics_tick_us` config
  peeks; reads of backend privates (`backend._sensor_data.new_imu_data`).
- **Reframing:** drop/replace the config peeks with observable cadence behavior
  (sensors sent at the expected sim times).
- **KEEP:** `_FakeConnection.hil_sensor_sends` as the documented HIL_SENSOR wiring
  seam, supplemented with `status.sensor_send_count`.
- **Verify:** `uv run pytest tests/test_px4_bridge_sensors.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 9 -- `test_px4_bridge_actuator.py` (Small-Medium)

- **Unit / public surface:** `Px4MavlinkPort` actuator contract; public `status`,
  `poll()`, `exchange()`.
- **Smells:** `port._actuator_receive_count`, `port._last_actuator_receive_us`,
  `port._last_disconnect_reason` reads; a direct `port._drain_actuator_backlog()`
  call.
- **Reframing:** route the counts/reason through `port.status.*`
  (`actuator_receive_count`, `last_actuator_receive_us`, `last_disconnect_reason`).
  Drive `_drain_actuator_backlog()` through the public `exchange()` / `poll()` path
  -- the backlog-drain behavior is observable via the returned commands plus
  `status`. Only if a focused unit test of the helper is truly needed, keep it and
  grandfather it in the capstone.
- **KEEP:** `_FakeConnection.recv_calls` (blocking/timeout history) as seam,
  supplemented with `status` checks.
- **Verify:** `uv run pytest tests/test_px4_bridge_actuator.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 10 -- `test_px4_bridge_lockstep.py` (Medium -- biggest of the status family)

- **Unit / public surface:** `Px4MavlinkPort` lockstep; public `status`,
  `is_connected`, `is_armed`, `start`, `send`, `poll`, `exchange`.
- **Smells (~18):** `port._sensor_send_count`, `port._actuator_receive_count`,
  `port._last_disconnect_reason`, `port._last_actuator_receive_us`, and
  `port._backend...` reaches.
- **Reframing:** replace every one with the matching public `port.status.*` field
  (`sensor_send_count`, `actuator_receive_count`, `last_disconnect_reason`,
  `last_actuator_receive_us`). The file already uses `port.status` in places --
  make all assertions consistent.
- **KEEP:** the `_FakeConnection.hil_sensor_sends` seam, supplemented by `status`
  assertions.
- **Verify:** `uv run pytest tests/test_px4_bridge_lockstep.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 11 -- `test_lightswitch_spawner_extension.py` (Large -- do last)

- **Unit / public surface:** `LightSwitchExtension` / `LightSwitchNode`; public
  `on_startup`, `on_shutdown`, `publish_toggle`.
- **Smells (~73):** assertions on 9 private state flags (`_switches_spawned`,
  `_simulation_running`, `_articulations_ready`, `_switch_state`, `_stage`,
  `_ros_node`) and direct private event-handler calls (`_on_timeline_event`,
  `_on_stage_event`, `_on_update`, `_publish_toggle`).
- **Reframing:**
  - (a) drive lifecycle through `on_startup` / `on_shutdown` and fire events via
    the fake streams' registered callbacks (not `_on_*`);
  - (b) infer state transitions from **observable side effects** -- the
    monkeypatched `spawn_switches` / `step_switches` / `despawn_switches` call
    records (already in the `calls` list) and published messages on the fake
    `_Publisher.messages` -- instead of reading the private flags;
  - (c) assert subscriptions exist by non-empty stream `callbacks`, and
    post-shutdown idempotency by calling `on_shutdown` twice without raising.
- This file most needs care -- budget the most time, and lean on the pattern built
  in the earlier steps.
- **Verify:** `uv run pytest tests/test_lightswitch_spawner_extension.py -q`; then
  `uv run pre-commit run --all-files`.

## Step 12 -- Audit the already-clean files

No behavior change; confirm each is already public-behavior and note the
SLF-only follow-ups (handled in the capstone, not here). Files:
`test_config.py`, `test_vehicles_config.py`, `test_usd_builder_config.py`,
`test_environment_switches.py`, `test_commanding_driver.py`,
`test_standalone_px4_scripts.py`, `test_core_imports.py`, `test_thrust_model.py`,
`test_rotor_discovery.py`, `test_frames_pegasus_equivalence.py`,
`test_px4_asset_diagnostics.py`, `test_isaac_articulation.py`,
`test_isaac_settings.py`, `test_sim_clock.py`.

Notes to carry into the capstone:

- `test_isaac_articulation.py`: private `io._body = ...` etc. are **fixture
  injection** (there is no real Isaac stage to call `initialize()`), not
  assertions -> acceptable; grandfather for SLF.
- `test_isaac_settings.py`: direct unit tests of private module helpers
  (`_apply_gravity`, `_apply_physx`, `_apply_render_rate`) are acceptable; in the
  capstone convert `module._apply_*` attribute calls to
  `from module import _apply_*` to clear SLF without changing intent.
- `test_sim_clock.py`: the single `clock._total_us = 1000` write is a deliberate
  corruption-guard regression test -> grandfather for SLF.
- `test_core_imports.py`: `sys.modules` manipulation is the lazy-import contract,
  not a private-member smell.

- **Verify:** `uv run pytest -q` (whole suite) is green.

## Step 13 -- Capstone: enable Ruff `SLF001`

1. Confirm steps 1-11 are done and green.
2. Run `uv run ruff check --select SLF` to get the residual list.
3. For each residual, either clear it cheaply (convert `module._fn` attribute
   access to a direct import per the technique above) or, where the external
   private access is the legitimate intended pattern, add a `per-file-ignores`
   entry with a one-line justification. Expected grandfather set (verify against
   the actual residual): `tests/test_isaac_articulation.py` (fake-state fixture
   injection), `tests/test_sim_clock.py` (deliberate time-base corruption guard),
   and any focused private-helper unit test deliberately kept.
4. Add `"SLF"` to `[tool.ruff.lint] select` in `pyproject.toml` (currently
   `["E", "F", "I", "B"]`) with a short comment, plus the `per-file-ignores` block
   from step 3.
5. Verify `uv run ruff check` and `uv run pre-commit run --all-files` are green.
6. Note in `AGENTS.md` (Python API organization / testing area) that external
   private-member access is now lint-enforced off, with the documented
   grandfathered exceptions.

- **Verify:** `uv run ruff check` (with `SLF` enabled) and
  `uv run pre-commit run --all-files` are green.
