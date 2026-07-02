# F1b -- In-Lab velocity-controller tuning (make the hover+goto gate pass)

Goal served: unblocks the F1 Step-7 gate so the in-Lab inner loop is trustworthy.
Depends on: F1 (flyable env + actuation seam + the `isaac-lab` Docker image).
Blocks: F3 (the gate is F3's precondition -- do not train on an inner loop that
limit-cycles). Runs in parallel with F2.
Deferred decisions (options surveyed in the last section): whether to add an
integral term, whether to hold yaw explicitly, and whether to seed the env for a
deterministic gate.

Why this exists: F1 shipped a flyable `DroneHoverEnv` and the
`test-integration-press-hover-goto` gate, and F1b's precondition is now in place --
the `isaac-lab` Docker image (`FROM isaac-sim`, IsaacLab v2.3.2; see the
`AGENTS.md` Docker section) makes `isaaclab` importable so the gate runs to
completion (no `ModuleNotFoundError`). But the gate does not yet pass: hover holds
tightly while the 2 m goto fails to settle. The failure is a genuine
limit cycle in the env's cascaded velocity controller, not infra and not the test
harness alone. This plan diagnoses and tunes it to green.

---

## Findings (measured in-container on the `isaac-lab` image)

The gate output:

```
hover max position error: 0.030 m      (passes, < 0.1 m)
goto  max position error: 1.399 m      (fails,  > 0.2 m)
```

Diagnostics (scripted drives of `DroneHoverEnv` under the analytic velocity
controller, 16 envs, `press.action.kind=velocity`):

- **Actuation authority is fine.** A constant full-forward body-velocity action
  reaches ~3.3 m/s and covers 4 m in ~2 s; the out-of-bounds reset
  (`_MAX_POSITION_ERROR_M = 5.0`) is what clips a constant-forward probe, not weak
  control.
- **Yaw is not drifting.** During a forward goto, `max|yaw|` stays ~1 deg, so the
  test's "spawn yaw is 0, body command == world command" assumption holds; the bug
  is not a frame/heading drift.
- **The 2 m goto is a sustained limit cycle, not slow convergence.** For the *same*
  gentle outer config (outer `kp=1.0`, velocity command capped at 1.0 m/s, ~6 s
  settle) the final error was 0.075 m in one run but 1.8-2.5 m in others. The error
  oscillates (e.g. 1.97 -> 1.84 -> 2.50 -> 1.53 -> 0.31 -> 1.20 -> 1.77 over one
  4 s window) with a period of a few seconds and does not reliably decay; the
  instant the gate samples catches a ~random phase over the reset randomization.
- **Small-signal stable, large-signal marginal.** Hover (small errors) settles to
  ~0.03 m; the 2 m maneuver excites the limit cycle. This is the signature of an
  under-damped cascade with too little inner-loop authority and/or saturation
  (velocity cap + `max_tilt_cmd_deg`) interaction.

Leading hypothesis: the inner rate loop is under-tuned. `press.action` ships
`controller_kp_rate = [0.3, 0.3, 0.1]` (low), so the body-rate loop tracks the
attitude-loop's rate setpoint with lag; the resulting phase lag around the
velocity -> attitude -> rate -> allocation cascade produces the limit cycle once
the maneuver is large enough to leave the linear regime. The tilt/velocity
saturations then sustain it (a describing-function limit cycle).

The relevant code (all config-driven, single source of truth):

- gains: `configs/experiment/press/default.yaml` -> `action.controller_kp_vel`,
  `controller_kd_vel`, `controller_kp_att`, `controller_kp_rate`,
  `max_tilt_cmd_deg`, `max_body_rate_radps`, `max_body_velocity_mps`
  (typed in `src/robotics_experiments/config/_types.py::PressActionConfig`).
- batched controller the env runs:
  `src/robotics_experiments/workflows/isaac_lab/tasks/_flight.py::VectorizedActuation`
  (`_velocity` -> `_rate_to_omega` -> `_wrench_to_omega`).
- pure numpy oracle that must stay in lockstep:
  `src/robotics_experiments/press/_actuation.py::VelocitySetpointActuation`.
- the gate harness:
  `isaac/runtime/standalone/tests/test_press_hover_goto.py` (analytic outer P loop,
  `_POSITION_KP`, `_GOTO_*`).

---

## Step 1 -- Deterministic reproduction harness

- Action: add a short, seeded diagnostic (scratch or a `--diag` path in the
  standalone test) that drives one env from hover to a 2 m goto and logs the step
  response: per-step position error, body velocity, tilt, and commanded vs actual
  body rate. Seed the env (`DirectRLEnvCfg.seed` / `torch.manual_seed`) so the
  reset box is fixed and runs are comparable.
- Metric: report settling time to <0.2 m and peak overshoot, and the max error over
  the *last* 1-2 s (a limit cycle shows a high last-window max even when a single
  final sample looks good).
- Done when: the limit cycle reproduces deterministically and you can compare two
  gain sets on identical initial conditions.

## Step 2 -- Retune the inner rate loop, then the outer velocity loop

- Action: tune in `configs/experiment/press/default.yaml` (never hardcode), keeping
  the numpy oracle in sync (Step 3). Tune inside-out:
  1. Raise `controller_kp_rate` until the body-rate loop tracks its setpoint with
     little lag (the rate loop should be several times faster than the attitude
     loop). Watch the allocation head-room (`_wrench_to_omega` clips negative
     thrust and to `[min,max]_rotor_velocity`); back off if omegas saturate at
     hover+maneuver.
  2. Set `controller_kp_att` for a well-damped attitude response under the faster
     rate loop.
  3. Set `controller_kp_vel` / `controller_kd_vel` for a critically-damped velocity
     response; `kd_vel` currently biases steady-state velocity to
     `kp_vel/(kp_vel+kd_vel)` of setpoint -- keep that in mind or move damping into
     the rate loop.
- Constraint: `velocity` and `ctbr` share this gain block, and `ctbr` is the inner
  loop of `velocity`; verify `ctbr` still behaves after retuning (a quick body-rate
  step check).
- Done when: on the Step-1 harness the 2 m goto settles < 0.2 m with < ~10%
  overshoot and no residual oscillation, across at least 3 seeds.

## Step 3 -- Keep the numpy oracle and its tests in lockstep

- Action: mirror every gain-structural change (not just the config values, which
  both already read) into
  `src/robotics_experiments/press/_actuation.py::VelocitySetpointActuation` /
  `BodyRateActuation`, since `_flight.py` must reproduce it term-for-term.
- Action: `tests/test_press_actuation.py` asserts on the realized wrench via the
  `PegasusQuadraticThrustModel` oracle (hover cancels weight; +x/+y tilt). Pure
  gain changes should not break the hover/tilt-direction assertions; if a
  structural change (e.g. an added integral term) does, update the oracle test
  deliberately.
- Done when: `uv run pytest tests/test_press_actuation.py` is green and the numpy
  and torch models agree on a hover and a small velocity step.

## Step 4 -- Make the gate assert settling, not a lucky sample

- Action: in `isaac/runtime/standalone/tests/test_press_hover_goto.py`, after the
  goto drive, require the max position error to stay under `_GOTO_TOLERANCE_M`
  for a sustained window (e.g. the last ~1 s of steps), not just the final sample,
  so a limit cycle cannot pass on a favorable phase. Match `_GOTO_SETTLE_S` to the
  tuned settling time. Keep the tolerances (`0.1 m` hover, `0.2 m` goto) -- do NOT
  loosen them to hide the limit cycle. Consider seeding the env for a deterministic
  gate.
- Optionally soften the outer analytic loop to a gentle capped approach (it is
  validation-only, "not a config surface"), but the env must be the thing that
  settles -- the harness change alone is not the fix.
- Done when: the gate expresses "held within tolerance", not "was within tolerance
  at one instant".

## Step 5 -- Verification

- `make test-integration-press-hover-goto` passes, run 3x back-to-back (the limit
  cycle was run-to-run variable, so a single green run is insufficient).
- Hover is unregressed (< 0.1 m) and `ctbr` still stabilizes.
- `uv run pytest` (oracle/config/core-import suites) and
  `uv run pre-commit run --all-files` are clean.
- Update this file's status and the F1 Step-7 note if the gate contract changes.

## Files touched by F1b

- edit `configs/experiment/press/default.yaml` (gain retune)
- edit `src/robotics_experiments/workflows/isaac_lab/tasks/_flight.py` and
  `src/robotics_experiments/press/_actuation.py` only if the change is structural
  (e.g. an integral term / explicit yaw hold), else config-only
- edit `isaac/runtime/standalone/tests/test_press_hover_goto.py` (sustained-settle
  assertion; optional seed / gentler outer loop)
- edit `tests/test_press_actuation.py` only if the oracle structure changes

## Deferred decisions (options)

1. **Integral term on velocity/position.** Option A (recommended first): pure
   PD/cascade retune, no integrator -- simplest, no steady-state offset expected for
   a hover setpoint. Option B: add an integral term if a residual bias survives
   retuning (adds anti-windup complexity and an oracle change).
2. **Explicit yaw hold.** Option A (recommended): leave yaw uncontrolled -- measured
   drift is ~1 deg and not implicated. Option B: command yaw rate to hold yaw = 0
   in `_velocity` (adds a heading reference; only if a future maneuver shows drift).
3. **Deterministic gate.** Option A (recommended): seed the env so the gate is
   reproducible and the limit cycle cannot hide behind randomization. Option B: keep
   it unseeded but assert sustained settling over many envs (Step 4) -- weaker but
   avoids a fixed seed masking a rare bad initial condition.
