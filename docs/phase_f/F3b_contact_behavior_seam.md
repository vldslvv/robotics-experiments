# F3b -- Contact-behavior seam + scripted press baseline

Goal served: the project goal is the contact task family, not one learned policy.
The durable asset is the pipeline (handoff gate, pose provider, compliant pusher,
deploy path); the press behavior itself must be a replaceable module so a scripted
model-based press and the learned policy are interchangeable implementations, and
future contact tasks reuse the same seam.

Depends on: F3 exit gate (the first trained policy exists). Does NOT block F4/F5;
it can run in parallel with F4.

Sequencing rule (deliberate): do not start this rework before the F3 first policy
is trained. The seam must be extracted from two concrete working consumers (the
F3 policy and the scripted press), not speculated up front while the press env is
still moving.

Why this exists: F2b already ships the generic deploy mechanism -- the
`{kind: policy}` routine step, the widened `STREAM` setpoint payload, the
validated artifact loader, all flowing through the unchanged `FleetOrchestrator`
(the orchestrator is not touched here either). What F2b does NOT cover is
contact: phases, the handoff gate, and a scripted baseline. This plan adds the
explicit contact-behavior contract on top of that mechanism, symmetric with the
command-routine contract (`docs/COMMAND_ROUTINE_CONTRACT.md`): a pure decision
surface over an injected side-effect seam. It also delivers the scripted press
as the first production behavior -- it closes the Phase D loop end-to-end,
exercises the handoff gate, pusher, latch, and deploy path ahead of the policy,
and becomes the metric floor the learned policy must beat.

---

## Step 1 -- Pure contact-behavior contract

- Action: create `src/robotics_experiments/press/_behavior.py` (pure; numpy +
  stdlib only, like `_observation.py`).
  - `class ContactBehavior(Protocol)`:
    - `reset() -> None` -- new attempt; clears internal phase/state.
    - `act(obs: PressObservation) -> ContactDecision`.
  - `@dataclass(frozen=True) class ContactDecision`: the normalized F1 action
    (`action: np.ndarray`, `press.action.action_dim`, same semantics as the env
    action for the configured `press.action.kind`) plus a
    `phase: ContactPhase` enum (`APPROACH`, `PRESS`, `CONFIRM`, `RETREAT`,
    `DONE`, `ABORT`) so the caller (routine step, eval harness) can observe
    progress without reaching into behavior internals.
  - INVARIANT (docstring + test): a behavior consumes only `PressObservation`
    and emits only the normalized action -- never simulator internals, never a
    raw sensor, never a PX4 message. Swapping behaviors changes no caller code.
- `__all__`: `ContactBehavior`, `ContactDecision`, `ContactPhase`.
- Test: `tests/test_press_behavior.py` (pure): the protocol is satisfied by a
  trivial fake; `ContactDecision.action` shape matches `action_dim`.
- Done when: `import robotics_experiments.press` still succeeds without
  torch/Isaac and exposes the contract.

## Step 2 -- Policy behavior (wrap the F3 policy)

- Action: in `press/_behavior.py` (or `press/policy.py` if the import weight
  fits better there), implement `PolicyContactBehavior` wrapping F2b's
  validated `load_policy(path, action_cfg)` (the artifact's metadata --
  obs schema version, action kind, dims, scales -- is checked at load, so a
  stale press artifact is refused, never silently misread): `act` runs
  `to_vector(obs)` through the loaded policy and returns the action; `phase` is
  derived from the observation (contact/latch fields), not from the network.
- Test: extend `tests/test_press_policy.py`: the behavior over the tiny exported
  test module produces a `ContactDecision` with the right shape.
- Done when: the F3-trained policy runs behind the seam with no change to its
  export format.

## Step 3 -- Scripted press behavior (model-based baseline)

- Action: implement `ScriptedPressBehavior` in `press/_behavior.py` (pure state
  machine over `PressObservation`):
  - `APPROACH`: velocity command along the contact-frame wall normal toward the
    stand-off point, lateral/vertical PD to hold alignment.
  - `PRESS`: ramp a bounded wall-normal velocity command (tilt-into-wall press),
    monitoring `pusher_compression_m` as the force proxy; cap commanded press by
    compression, not time.
  - `CONFIRM`: hold press until `lamp_state` toggles via the latch or the press
    window expires.
  - `RETREAT`: back off along the wall normal to the retreat stand-off; then
    `DONE`.
  - Any envelope-relevant anomaly (compression overshoot, misalignment beyond
    threshold, `detection_valid` lost) -> `RETREAT` then `ABORT` after bounded
    retries.
- Config: add `PressScriptedConfig` (stand-off distances, approach/press speed
  caps, alignment thresholds, compression cap, confirm window, retry bound) to
  `config/types.py` (bare annotations) and a `scripted:` block to
  `configs/experiment/press/default.yaml`. No literal thresholds in Python.
- Test: `tests/test_press_scripted.py` (pure): a synthetic observation sequence
  (approach -> contact -> compression -> lamp toggle) walks the phases in order;
  a lost detection triggers retreat; the toggle is required before `RETREAT`
  ends in `DONE`; retries are bounded.
- Done when: the scripted behavior toggles the latch in a scripted Lab rollout
  (the F3 env, ground-truth provider, no policy).

## Step 4 -- Behavior selection + routine integration

- Action: behavior selection is config: `press.behavior: scripted | policy`
  (+ `press.policy_path` for `policy`). Add the field to `PressConfig`; unknown
  kind raises `ValueError` at resolution time.
- Action: add the `{kind: press}` routine step in the commanding layer,
  building on F2b's `{kind: policy}` step mechanism (the per-tick
  observation -> action -> `press/deploy.py` mapping -> `STREAM` setpoint
  chain): the press step differs by entering only when F2's `handoff_ready`
  gate passes and by ticking the configured `ContactBehavior` in place of the
  direct policy call, completing on `DONE` (-> ADVANCE) or `ABORT` (-> FAIL).
  Keep `decide` pure, exactly as in F2b Step 6. The orchestrator remains
  untouched: `FleetOrchestrator.build_drivers` already runs any routine step
  through the per-drone `RoutineDriver`.
- Test: extend `tests/test_commanding_routine.py` /
  `tests/test_commanding_driver.py` with the press step over fakes: gate blocks
  entry, `DONE` advances, `ABORT` fails with `FailureKind.COMMAND` semantics.
- Done when: a routine `[... goto, press, land]` runs end-to-end in the
  standalone PX4-bridge scenario with `press.behavior: scripted` (this is the
  sim-to-sim deploy check for the seam).

## Step 5 -- Baseline evaluation (the metric floor)

- Action: run the F3 eval entrypoint over both behaviors on the same env/config
  and emit the same metric table (press success, first-contact stability,
  retreat rate, time-to-toggle, compression overshoot).
- Rule: the scripted behavior's numbers are the published floor; the learned
  policy earns its place in a config by beating the floor on the same table.
  Record both in the eval output, not prose.
- Done when: one command prints the side-by-side table for
  `scripted` vs `policy`.

## Files touched by F3b

- add `src/robotics_experiments/press/_behavior.py`; edit `press/__init__.py`
  (re-exports)
- edit `config/types.py` (`PressScriptedConfig`, behavior selector),
  `configs/experiment/press/default.yaml`
- edit the commanding layer for the `{kind: press}` step (+ the routine builder
  and `docs/COMMAND_ROUTINE_CONTRACT.md`)
- add `tests/test_press_behavior.py`, `tests/test_press_scripted.py`; extend
  `tests/test_press_policy.py`, the commanding tests, `tests/test_core_imports.py`
- record an ADR for the contact-behavior contract (load-bearing seam)

## Exit gate

The scripted press and the F3 policy run behind the identical `ContactBehavior`
seam; the routine `press` step drives either by config; the scripted baseline's
metric table exists as the floor. Future contact tasks add behaviors, not
plumbing.
