# Safety Harness Plan -- architecture changes for harnessed hardware flights

Scope: the real Stage 1 X500 V2 flies indoor contact experiments under a physical
safety harness (tether/catch system). This plan records how the harness changes
the software architecture and the sim plans, so the accommodations land before
the first harnessed contact flight rather than being discovered on it.

Sequencing: Steps 1-3 are software/contract work that must exist before any real
contact attempt; Step 4 feeds F4 training/validation; Step 5 is a physical
design rule. Nothing here blocks Phase F sim work.

The harness changes three things at once:

1. It is a physics change, not only a safety device: a tether adds a
   state-dependent force/moment disturbance and a snag hazard near the wall.
2. It changes validation economics: a failed contact attempt costs a reset, not
   an airframe, so real-contact experiments can move earlier and rely on the sim
   gates to protect training time rather than hardware.
3. It must appear in software as a hardware safety profile, not only as rope.

---

## Step 1 -- Qualify the shared landing recovery on hardware

- The shared driver now owns failure recovery for every shell. Any armed
  routine `FAIL`, PX4 failsafe, or driver timeout enters `RECOVERING`, commands
  Land at the current position, and falls back to PX4's configured offboard-loss
  action if Land is rejected or unacknowledged. `on_failure` is notification-only
  and cannot interrupt recovery.
- Action: the future hardware shell reuses this state machine and reports the
  preserved terminal `FAILED` result. It must not introduce a third side-effecting
  `on_failure` profile or select RTL as contact-failure recovery.
- Invariant: the RC kill switch and PX4's own failsafes stay above everything in
  this repo; the abort profile is the software layer's reaction, never the only
  line of defense.
- Done when: the hardware shell's `on_failure` provably lands (bench test
  against SITL: inject a `FailureKind.ENVELOPE` breach, observe the land
  command on the GCS link).

## Step 2 -- Geofence sized to the harness volume

- Action: add a geofence invariant to the routine layer, in the existing
  envelope-gate style (`commanding/_routine_predicates.py` invariant helpers): a
  `within_bounds(...)` continuous gate over world-ENU position, with bounds in
  the Hydra `routine` group (a `geofence` block: x/y/z min/max). Sim default is
  a box comfortably larger than every existing scene, so behavior is unchanged;
  the hardware experiment config authors the harness operating volume.
- Action: mirror it on the PX4 side for hardware (`GF_*` params in the real
  airframe config) so the fence holds even if the companion link drops.
- Test: pure invariant test in the routine suite (inside passes, each face
  breach reports its message); config test that the default bounds resolve.
- Done when: a routine step breaching the configured box fails with an
  `ENVELOPE` failure (and, on hardware, PX4's fence acts independently).

## Step 3 -- Contact entry/exit contract (minimal Phase-H subset, pulled forward)

- Rule: the contact behavior is only *entered* through the F2 `handoff_ready`
  gate, and always *exits* to a retreat hover -- `DONE` and `ABORT` both end at
  the retreat stand-off, never in contact. Retries are bounded by config;
  exhaustion aborts (Step 1 profile). F3b's planned `{kind: press}` step must
  encode this; this plan makes it a hardware precondition: no real contact
  flight before the bounded-retry/abort path is exercised in the PX4-bridge
  sim-to-sim scenario.
- Done when: the standalone scenario demonstrates gate-blocked entry, a bounded
  retry, and an abort-to-land, before the first harnessed contact flight.

## Step 4 -- Tether as a modeled disturbance (F4 randomization)

- A policy or controller validated only in free-flight sim meets an unmodeled
  disturbance on its first real contact attempt. Mitigation: a crude tether
  model in the F4 `EventCfg` randomization -- configured anchor point and slack
  length; zero force while slack, a spring-damper pull toward the anchor when
  taut. Behaviors are trained/validated tether-tolerant; the nominal (slack)
  case stays identical to free flight.
- Action: `PressRandomizationConfig` gains the tether record (anchor position
  range, slack length range, taut stiffness/damping), `stage_introduced` at the
  contact stage. See `docs/phase_f/F4_curriculum_and_randomization.md` Step 3.
- Done when: F4 eval includes taut-catch samples and the policy/scripted
  behavior recovers to retreat hover from a taut event.

## Step 5 -- Harness geometry rules (physical design)

- Prefer a geometry that is slack in the whole operating volume (overhead
  gantry or high anchor with managed slack), so "no tether force" is the
  nominal case and taut only occurs on a catch.
- Anchor placement must respect the press geometry: the drone pitches *toward*
  the wall to press, so a catch must not be able to swing the drone, the pusher
  boom, or the tether itself into the wall or the prop discs. Verify statically
  like the Stage 2 pusher/prop clearance check: worst-case pitch at the press
  stand-off, taut tether direction, prop-disc clearance.
- The tether attachment point on the frame sits above the CoM (pendulum-stable
  catch attitude) and clear of the GPS mast and props.
- Done when: a written clearance check (angles/lengths, like the roadmap's
  guard sizing note) exists before the first harnessed flight.

## Risks this plan mitigates

| Risk | Mitigation here |
|------|-----------------|
| Tether snag/swing into wall or props during press | Step 5 geometry rules + Step 4 taut modeling |
| Unmodeled taut-tether disturbance destabilizes contact | Step 4 randomization |
| Software keeps flying after a breach on hardware | Step 1 abort profile + Step 2 geofence |
| Contact behavior entered/exited in a bad state | Step 3 entry/exit contract |
