# F3b -- Contact-behavior seam + scripted press baseline

Status: PLANNED; BLOCKED BY F3. No contact-behavior contract,
`{kind: press}` routine step, scripted baseline, or comparative evaluation is
implemented. The generic F2b `{kind: policy}` step is the only implemented
policy deployment mechanism.

Goal served: the project goal is the contact task family, not one learned policy.
The durable asset is the pipeline (handoff gate, pose provider, compliant pusher,
deploy path); the press behavior itself must follow a replaceable contract so a
scripted model-based press and the learned policy are interchangeable
implementations, and future contact tasks reuse the same seam.

Depends on: F3 exit gate (the first trained policy exists). Does NOT block F4/F5;
it can run in parallel with F4. One design decision remains open: how the live
Isaac shell injects a complete immutable `PressObservation` into the pure
routine/behavior transition without adding an Isaac or RL dependency to the
commanding core. Record that boundary in the F3b ADR before Step 4.

Sequencing rule (deliberate): do not start this rework before the F3 first policy
is trained. The seam must be extracted from two concrete working consumers (the
F3 policy and the scripted press), not speculated up front while the press env is
still moving.

Why this exists: F2b already ships the generic deploy mechanism -- the
`{kind: policy}` routine step, the widened `STREAM` setpoint payload, the
validated artifact loader, all flowing through the frozen `FleetState`
lifecycle. What F2b does NOT cover is contact: phases, the handoff gate, and a
scripted baseline.
This plan adds the explicit contact-behavior contract on top of that mechanism,
symmetric with the command-routine contract (`docs/COMMAND_ROUTINE_CONTRACT.md`):
a pure decision surface over an injected side-effect seam. It also delivers the
scripted press as the first production behavior -- it closes the Phase D loop
end-to-end, exercises the handoff gate, pusher, latch, and deploy path ahead of
the policy, and becomes the metric floor the learned policy must beat.

Current limitation (do not paper over it): `RoutineObservation` contains only
`VehicleState`, `BridgeStatus`, command acknowledgement, and sim time. It cannot
assemble pusher compression, contact, lamp/latch, or provider confidence/age.
Those values also cannot be read from inside `decide`, because that would make
the pure stepper perform Isaac side effects. Step 4 therefore needs a scoped
imperative-boundary observation assembler and may require a small runtime/shell
composition change; the generic `{kind: policy}` step alone is insufficient.

---

## Step 1 -- Pure contact-behavior contract

- Action: create `src/robotics_experiments/rl/press/_behavior.py` (pure; numpy +
  stdlib only, like `observation.py`). Model project-owned progress as a frozen
  `ContactBehaviorState` and advance it with a stateless transition such as
  `step_contact_behavior(*, params, state, observation) -> (state, decision)`.
  Behavior selection resolves immutable params and the injected policy
  callable once; it does not create a mutable project-owned behavior object.
  - `@dataclass(frozen=True) class ContactDecision`: the normalized F1 action
    (`action: np.ndarray`, `rl.action.action_dim`, same semantics as the env
    action for the configured `rl.action.kind`) plus a
    `phase: ContactPhase` enum (`APPROACH`, `PRESS`, `CONFIRM`, `RETREAT`,
    `DONE`, `ABORT`) so the caller (routine step, eval harness) can observe
    progress without reaching into behavior internals.
  - INVARIANT (docstring + test): a behavior transition consumes only immutable
    params/state plus `PressObservation`
    and emits only the normalized action -- never simulator internals, never a
    raw sensor, never a PX4 message. Swapping behaviors changes no caller code.
- Export only the production contract needed by callers: the frozen params,
  state/decision values, phase enum, and transition/factory functions. Test-only
  fakes do not justify public exports.
- Test: `tests/src/robotics_experiments/rl/press/test_press_behavior.py`
  (pure): a trivial injected decision function advances frozen state and
  `ContactDecision.action` has the configured dimension.
- Done when: `import robotics_experiments.rl.press` still succeeds without
  torch/Isaac and exposes the contract.

## Step 2 -- Policy behavior (wrap the F3 policy)

- Action: add a pure policy-backed transition using the validated vector-policy
  loader generalized in F3 (the artifact's metadata --
  obs schema version, action kind, dims, scales -- is checked at load, so a
  stale press artifact is refused, never silently misread): the policy
  transition runs `to_vector(obs)` through the loaded policy and returns the
  action; `phase` is derived from the observation (contact/latch fields), not
  from the network.
- Test: extend
  `tests/src/robotics_experiments/rl/press/test_press_policy.py`: the behavior
  over the tiny exported test module produces a `ContactDecision` with the
  right shape.
- Done when: the F3-trained policy runs behind the seam with no change to its
  export format.

## Step 3 -- Scripted press behavior (model-based baseline)

- Action: implement the scripted variant in `press/_behavior.py` as frozen
  params/state plus the shared pure transition over `PressObservation`:
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
  the owning private RL config module (bare annotations) and a `scripted:` block
  to `configs/experiment/rl/press/default.yaml`. No literal thresholds in
  Python.
- Test: `tests/src/robotics_experiments/rl/press/test_press_scripted.py` (pure):
  a synthetic observation sequence (approach -> contact -> compression -> lamp
  toggle) walks the phases in order; a lost detection triggers retreat; the
  toggle is required before `RETREAT` ends in `DONE`; retries are bounded.
- Done when: the scripted behavior toggles the latch in a scripted Lab rollout
  (the F3 env, ground-truth provider, no policy).

## Step 4 -- Behavior selection + routine integration

- Action: behavior selection is config: `press.behavior: scripted | policy`
  (+ `rl.press.policy_path` for `policy`). Add the fields to `PressTaskConfig`;
  an unknown kind raises `ValueError` during config validation.
- Action: first record the live-observation injection boundary in the
  contact-behavior ADR. The selected design must assemble a complete frozen
  `PressObservation` at an imperative Isaac boundary, pass it into the pure
  behavior/routine transition, and keep `commanding` import-clean from Isaac and
  torch. Do not hide stage reads in a policy callable or `decide`.
- Action: add the `{kind: press}` routine step on top of F2b's OFFBOARD
  prime/policy/transmit cadence machinery. Its per-tick chain is injected
  `PressObservation` -> F2 handoff gate -> frozen contact transition ->
  normalized action ->
  `robotics_experiments.rl.press.press_action_to_offboard` -> `STREAM` payload.
  `DONE` advances and `ABORT` fails. Any runtime/orchestrator or shell change is
  limited to assembling/injecting the observation and owning its external
  resources; contact phase/progress remains in frozen routine/behavior state.
- Test: extend the routine/driver tests under
  `tests/src/robotics_experiments/commanding/` with the press step over fakes:
  gate blocks entry, `DONE` advances, `ABORT` fails with
  `FailureKind.COMMAND` semantics.
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

- add `src/robotics_experiments/rl/press/_behavior.py`; edit
  `rl/press/__init__.py` only for the minimal production exports
- edit the owning private RL config module (`PressScriptedConfig`, behavior
  selector) and `configs/experiment/rl/press/default.yaml`
- edit the commanding layer for the `{kind: press}` step (+ the routine builder
  and `docs/COMMAND_ROUTINE_CONTRACT.md`)
- edit the scoped runtime/live/standalone composition selected by the ADR to
  assemble and inject `PressObservation`; do not add Isaac reads to commanding
- add press tests under `tests/src/robotics_experiments/rl/press/`; extend the
  commanding tests and `tests/src/robotics_experiments/test_core_imports.py`
- record an ADR for the contact-behavior and live-observation injection
  contracts (load-bearing seams)

## Exit gate

The scripted press and the F3 policy run behind the identical immutable
contact-behavior transition; the routine `press` step drives either by config;
the scripted baseline's metric table exists as the floor. Future contact tasks
add behavior params/transitions, not plumbing.
