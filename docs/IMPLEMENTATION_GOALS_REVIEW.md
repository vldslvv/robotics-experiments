# Implementation and Goals Review

Date: 2026-07-09

## Scope

This review evaluates how the project goals and implementation stages are
defined, what the current code and executable gates actually prove, and which
implementation improvements would most directly advance the light-switch
mission. It is not an architecture review.

The review started from `AGENTS.md` and followed its context map into the
canonical roadmap, experiment configuration, Phase F plans, runtime workflows,
tests, generated airframes, and current implementation.

## Overall assessment

The project is technically healthy, but implementation progress is communicated
less clearly than it is executed. The code and recorded gates strongly validate
the enabling flight capabilities: config-generated airframes, PX4 lockstep,
takeoff, waypoint navigation, multi-vehicle execution, failure detection, Isaac
Lab hover, policy export, and policy deployment through PX4.

The central mission outcome is not yet validated. There is no executable gate
that proves the complete pusher-contact-button-latch-lamp chain under real Isaac
physics, nor a complete approach -> handoff -> press -> confirm -> retreat
experiment. The highest-value next milestone is therefore one deterministic
known-switch interaction, not additional fleet, X8, sensor, or autonomous
discovery capability.

## Goal definition

`AGENTS.md` currently describes the project goal as integrating Isaac Sim, ROS
2, PX4 SITL, and Docker. That describes the technology stack, not the result the
project is intended to produce. The canonical roadmap supplies the actual
mission: a PX4-driven multirotor navigates to a wall switch, presses it with a
compliant pusher, confirms the lamp state, retreats, and eventually discovers
switches autonomously.

A clearer top-level outcome would be:

> Build and validate a PX4-controlled X500 that can approach a known wall
> switch, hand off safely, press it with the compliant pusher, confirm the lamp
> state, retreat, and land; then replace ground-truth task state with
> sensor-derived state.

This outcome should appear in `AGENTS.md` and `README.md`, followed by the
current milestone and its measurable exit gate. The stack remains an important
constraint, but it should not be mistaken for project success.

## Evidence-based stage status

| Area | Implementation status | Evidence and limitation |
|---|---|---|
| Runtime and PX4 foundation | Validated | Strict takeoff, lockstep, multi-drone, actuator-fault detection, logging, and recent successful integration artifacts exist. |
| Navigation routines | Validated for demonstrations | Arm, takeoff, hover, forward, absolute `goto`, circular waypoint sequences, and land are implemented. Strict navigation exercises quad and placeholder hexa flights, but not a switch approach. |
| Stage 1 X4 airframe | Substantially implemented | One YAML drives generated USD geometry and mass properties, PX4 control allocation, and bridge thrust values. Some physical components remain approximate or unverified. |
| Hexa and Stage 2 X8 | Placeholder or absent | The S550 hexa is explicitly a placeholder. The hardware-driven X8 geometry, mapping, and tuning are not implemented. |
| Switch and lamp environment | Component-level implementation | Spawning, articulation polling, scalar latch behavior, lamp state changes, and ROS publication exist. No live gate validates the real pusher-to-switch contact chain. |
| RL Phase F foundation | F1, F2, and F2b complete | The Lab vehicle flies; observation, provider, and handoff contracts exist; training, export, and policy-hover deployment through PX4 have passed. |
| RL switch-press task | Not implemented | F3 has a concrete plan and metrics, but no `Isaac-Switch-Press-Direct-v0` task, contact reward, press evaluation, or trained press policy exists. |
| ROS sensor and fusion work | Mostly future | Runtime middleware and a switch-state publisher exist. There is no implemented perception/fusion stack or automated cross-container ROS gate. |
| Full mission and discovery | Not implemented | There is no complete known-switch mission, retry/abort mission flow, wall scan, switch detector, or discovery queue. |

## Stage-definition findings

### The canonical roadmap is stale

`docs/LIGHT_SWITCH_MULTIROTOR_ROADMAP.md` still describes the Isaac Lab hover
environment as a no-op scaffold and the action level as a deferred decision.
The implementation and the Phase F status documents show that F1, F2, and F2b
are complete, `velocity` is the default action level, `ctbr` is available, a
hover policy was trained, and its strict PX4 deployment gate passed.

Status summaries should be updated whenever a stage gate passes. Otherwise the
roadmap understates real progress and makes the next blocking task harder to
identify.

### Two roadmaps compete for authority

`docs/LIGHT_SWITCH_MULTIROTOR_ROADMAP.md` declares itself canonical and defines
phases A-H. `docs/LIGHT_SWITCH_MULTIROTOR_ROADMAP1.md` also calls itself a
canonical recommendation and defines phases A-L. Phase F documents still refer
to ROADMAP1 for thresholds and evaluation metrics.

Keep one current roadmap. Move any still-authoritative thresholds and metrics
from ROADMAP1 into the canonical roadmap, Hydra configuration, or the relevant
Phase F document, then archive ROADMAP1 as historical source material.

### Airframe phases overlap

Canonical phases B and E both track Stage 1 X4, the placeholder hexa, and Stage
2 X8. Their completion states are therefore difficult to interpret. Treat the
following as distinct implementation outcomes:

1. Generated airframe consistency: config -> USD, PX4 airframe, and bridge
   values agree.
2. Stage 1 physical credibility: component values and geometry are verified,
   and simulation behavior is correlated with bench or flight evidence.
3. Stage 2 X8 upgrade: conditional work started only after the documented
   empirical triggers are met.

### Stage states need executable exit gates

Use a single status table in which each stage records:

- its concrete deliverable;
- the public command that validates it;
- quantitative acceptance criteria;
- the last verified revision and date;
- one of `planned`, `implemented`, `validated`, or `blocked`.

This distinguishes code presence from a demonstrated outcome. Phase F already
uses this pattern better than the top-level roadmap.

## Prioritized implementation improvements

### 1. Add a deterministic contact-chain integration gate

Current integration documentation explicitly states that the tests do not
validate warehouse geometry, lamps, pusher mechanisms, or obstacle navigation.
The `test-integration-press-hover-goto` gate creates a referenced switch prim and
validates relative pose plus the handoff predicate, but supplies placeholder
values for pusher compression, contact, and lamp state.

Add a real Isaac scenario that proves:

1. The wall, switch, lamp, and pusher-enabled drone assets load.
2. The pusher reaches the button and produces controlled compression/contact.
3. Button displacement crosses the configured threshold.
4. The N-step latch toggles exactly once for one sustained press.
5. The paired lamp changes state.
6. A scripted vehicle or contact fixture retreats without instability.

The first gate can use a deterministic scripted contact behavior or fixture. It
does not need to introduce the future production `ContactBehavior` seam before
F3. Its purpose is to establish that the environment physics work before policy
training, so policy failure cannot be confused with broken contact geometry.

### 2. Make switch spawning fail closed

The current switch stage can report a successful simulation start with no valid
switch/lamp pairs:

- missing switch or lamp assets log an error and return an empty list;
- switch and lamp paths are joined with `zip(..., strict=False)`;
- validation checks only the switch prims represented in the resulting list;
- an empty list therefore passes validation;
- lamp prims are not validated.

Make spawning atomic and strict:

- require exactly `num_switches` switch/lamp pairs;
- raise on missing assets rather than returning an empty successful result;
- use strict pair-count validation;
- validate both switch and lamp prims;
- tear down partial results after any failure;
- add semantic config validation for positive `num_switches`, positive
  `press_confirm_steps`, non-empty bounds, valid min/max bounds, and a positive
  retry budget.

### 3. Single-source RL contact physics

`workflows/isaac_lab/tasks/_drone_cfg.py` duplicates pusher stiffness and damping
already present in `configs/airframes/holybro_x500_v2.yaml`, and hardcodes a
rough effort limit, velocity limit, spawn pose, and rigid-body velocity caps.
The module describes these values as authoritative for training, so they can
silently diverge from the generated USD and runtime contact behavior.

Move genuine tunables into Hydra-backed dataclasses and pass the resolved values
into the Lab articulation config. Reuse the airframe pusher leaves where they
describe the same physical property. Also use the exported
`HOVER_OBSERVATION_DIM` constant instead of a second literal `13` in the hover
environment.

### 4. Define one goal-driven experiment

The shipped experiments demonstrate takeoff, forward movement, navigation,
circles, multi-vehicle operation, and policy hover. None represents the target
known-switch task.

After the deterministic contact gate exists, add a normal top-level experiment,
for example `press_known_switch_quad.yaml`, with:

- a deterministic switch pose or explicit experiment seed;
- one Stage 1 X4;
- approach and handoff conditions;
- a scripted or learned contact behavior;
- lamp confirmation;
- retreat and land;
- strict evaluation metrics.

It should be runnable both as a regular experiment and as a strict integration
gate, following the existing routine-config convention.

### 5. Make Stage 1 physical claims precise

The X4 config is unusually traceable, but it also records known limitations:

- the exact battery SKU, dimensions, and mass remain unverified;
- pusher hardware has no selected spring, guide, pad, or mount SKU;
- prop-guard mass is folded into a residual lump while prop-guard geometry is
  not generated;
- several dynamics values remain engineering estimates rather than measured
  values.

Either close these gaps or describe the current result as a
hardware-oriented/traceable simulation model rather than fully
hardware-faithful. Add an explicit Stage 1 validation gate based on bench or
flight-log comparison before prioritizing X8. The roadmap already states that
the X8 upgrade must be triggered by evidence, not availability.

### 6. Clarify the current ROS goal

ROS 2 is named as a top-level project integration, but the tracked project
currently supplies middleware setup and the Isaac-side light-switch publisher;
`test-integration-all` does not start a ROS container or validate a topic across
the container boundary. The persisted ROS workspace is intentionally ignored
and is not a reproducible project implementation in a fresh clone.

Choose and document one honest state:

- ROS sensor/perception is Phase G and not part of the current validated
  milestone; or
- ROS is a current delivered capability, in which case add a small tracked
  project package or manifest and a cross-container topic/QoS smoke gate.

### 7. Improve reproducibility and documentation hygiene

- Add a configured experiment seed or fixed placement mode for switch/lamp
  generation. The current global `random` use makes identical experiments
  produce different stages.
- Remove developer-specific absolute source paths from generated USDA metadata.
- Reconcile the binary-asset rule with the tracked binary USD crates by either
  documenting a narrow asset exception or replacing the crates with an allowed
  distribution workflow.
- The former `TODO.txt` has been groomed: completed observations were removed
  and durable drone-platform ideas now live in
  `docs/backlog/drone_platform_followups.md`.
- Verify a fresh-clone bootstrap and document the shortest sequence for the host
  environment, image build, asset generation, host tests, and one integration
  test.
- Align packaging with the import-clean implementation. The package can import
  without Isaac, torch, or PX4, but Isaac Sim packages and pytest are mandatory
  runtime dependencies in `pyproject.toml`. Simulator and test dependencies
  should be optional unless installation without them is explicitly unsupported.

## Recommended implementation sequence

1. Fix fail-closed switch/lamp spawning and semantic config validation.
2. Add the deterministic real-asset contact-chain gate.
3. Implement F3's ground-truth switch-press environment and evaluation.
4. Add the goal-driven known-switch experiment through retreat and land.
5. Apply F4 curriculum/domain randomization and F5 synthetic pose degradation.
6. Add the Phase G ROS/sensor provider and prove the same policy-facing contract.
7. Build the complete mission and discovery flow.
8. Start Stage 2 X8 work only if Stage 1 measurements meet an upgrade trigger.

## Suggested outcome stages

The following outcome sequence is easier to assess than parallel technology
phases:

### Outcome 1: Deterministic environment contact

A scripted fixed scenario presses the real switch asset, toggles its lamp exactly
once, retreats, and repeats reliably. This validates the physical task
implementation.

### Outcome 2: Ground-truth learned contact

The F3 policy solves the easy ground-truth task using the existing targets:
press success at least 80 percent, first-contact stability at least 85 percent,
and retreat at least 95 percent.

### Outcome 3: Robust contact

The same task passes under the F4 physical randomization and F5 noisy, delayed,
and dropping pose provider, with bounded retry/abort behavior.

### Outcome 4: Sensor-driven complete mission

The observation provider uses simulated ROS sensor/fusion inputs, and one known
switch mission completes takeoff -> approach -> handoff -> press -> confirm ->
retreat -> land. Discovery and real-hardware transfer follow this gate.

The conditional X8 hardware upgrade and multi-vehicle expansion should remain
separate capability tracks rather than blockers for these single-vehicle mission
outcomes.

## Verification observed during this review

The following host-side checks passed on 2026-07-09:

- `uv run pytest`: 285 passed, 1 skipped;
- `uv run ruff check`;
- `uv run mypy src tests`;
- `uv lock --check`.

Existing local integration artifacts from 2026-07-06 and 2026-07-07 also record
successful runs for Isaac settings, drone articulation, Lab hover/goto and
handoff, PX4 takeoff, quad/hexa waypoint navigation, intentional instability
detection, multi-drone takeoff, and trained-policy hover through PX4. Those
artifacts support the strong foundation status but do not replace the missing
contact-chain and complete-mission gates identified above.
