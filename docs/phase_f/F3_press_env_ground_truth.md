# F3 -- Switch-press RL env on ground truth (easy/large switch) + first policy

Status: PLANNED; NEXT PHASE F BLOCKER. None of the F3 task, reward,
termination, evaluation, or trained press-policy deliverables exist yet. The
implemented baseline is F1/F2/F2b: a flyable hover task, the pure press
observation/provider/handoff contracts, and the validated hover-policy
train/export/deploy pipeline.

Goal served: goal 1 (easier task on ground-truth pose). Depends on: F1 (flyable
env + actuation seam), F2 (observation + provider + handoff), F2b (the proven
hover task->policy pipeline), and the deterministic real-asset contact-chain
gate from `docs/IMPLEMENTATION_GOALS_REVIEW.md`. Blocks: F4.
No deferred decisions (inherits F1's action-level decision).

Why this exists: with a flyable env (F1), a stable observation (F2), and a
working train->export->deploy pipeline (F2b), build the contact task itself --
spawn a switch per env, wire the pusher/contact/lamp, define the reward, and
train the first policy that approaches, presses, confirms the toggle, and
retreats. F3 primarily adds a task: the training adapter and artifact format
exist from F2b. It also removes the remaining hover-specific schema assumptions
from the shared training entrypoint/validated loader; it does not add a second
pipeline. The task is deliberately easy: oversized/soft switch, ground-truth
pose, no domain randomization (F4 adds those).

---

## Precondition -- deterministic contact-chain gate

Before policy training, add the narrow Isaac gate that proves the configured
pusher can contact the real switch asset, cross the configured button threshold,
toggle the N-step latch and paired lamp exactly once, and retreat without
instability. This gate may use a deterministic fixture or scripted command; it
does not require F3b's production behavior seam. Its purpose is to keep broken
asset/contact physics distinguishable from a learning failure.

Also remove the duplicated pusher contact values from the task-local
`_drone_cfg.py` path before they become task authority: stiffness/damping shared
with the generated airframe come from the resolved airframe config, while
genuine Lab-only limits/tunables get Hydra fields. F3 must not train against
hardcoded contact physics that can diverge from the runtime asset.

## Step 1 -- New task: `Isaac-Switch-Press-Direct-v0`

- Action: create
  `src/robotics_experiments/workflows/isaac_lab/tasks/switch_press_env.py`
  (torch/isaaclab leaf) with `SwitchPressEnvCfg(DirectRLEnvCfg)` and
  `SwitchPressEnv(DirectRLEnv)`, reusing F1's `flight.py` actuation/force sink and
  F2's observation assembly. Do not fork the hover env; import the shared pieces.
- Action: register it in `workflows/isaac_lab/registration.py`
  (`_SWITCH_PRESS_TASK_ID = "Isaac-Switch-Press-Direct-v0"`, guarded like the hover
  task).
- Config: `SwitchPressEnvCfg` receives the resolved `ExperimentConfig` from the
  standalone root, like the hover task. Shared sizes/spacing/episode and action
  values come from `cfg.rl`; contact-task values come from `cfg.rl.press`. The
  env must not reload global config.
- Done when: `gym.make("Isaac-Switch-Press-Direct-v0")` builds under Isaac Lab and
  steps N envs.

## Step 2 -- Spawn one switch per env + a wall

- Action: in `_setup_scene`, spawn a wall + one light switch per env from the switch
  USD (`configs/experiment/switches/default.yaml switch_usd_path`). The switch is an
  articulation with a button DOF (the existing env reads
  `articulation.get_joint_positions()[0]`; reuse that indexing).
- For the "easy" stage, place the switch directly in front of the drone at a short
  stand-off, upright wall, no lateral offset. Its pose is a per-env reset parameter
  (F4 randomizes it).
- The implemented `GroundTruthPoseProvider` reads one prim and was sufficient
  for F2's env-0 side channel. Add a thin batched Isaac adapter (or vectorized
  stage/articulation query in the env leaf) for the cloned switches while
  preserving F2's per-sample `PoseSample` semantics and the unchanged
  `PressObservation` layout.
- Done when: each env has a drone + wall + switch and the ground-truth provider
  returns the correct per-env relative pose.

## Step 3 -- Wire pusher compression, contact, and the lamp latch

- Action: read the pusher joint (`pusher_joint`, prismatic, from the drone USD) per
  env for `pusher_compression_m` and its rate for `pusher_compression_rate_mps`.
- Action: add a contact signal for `contact_flag` (Isaac Lab contact sensor on the
  pad, or threshold on pusher compression as a first cut) and `contact_valid`.
- Action: vectorize the N-step press latch. The pure single-switch logic is
  exported as `robotics_experiments.environment.apply_button_sample` (press_threshold +
  press_confirm_steps -> single toggle). Add a pure batched version in `press/`
  (e.g. `press/_latch.py`, numpy/torch-agnostic core with a torch wrapper in the
  leaf) that maintains `press_steps` / `toggle_latched` / `state` per env, so
  `lamp_state` in the observation comes from the same latch semantics the runtime
  environment uses. Do not re-implement the threshold logic differently.
- Config: reuse `switches.press_threshold` / `switches.press_confirm_steps`; the
  "easy" stage may loosen them via a `press.reset`/curriculum override (F4).
- Test: `tests/src/robotics_experiments/rl/press/test_press_latch.py` (pure):
  batched latch matches the scalar
  switch latch for the same input sequences; a sustained press toggles
  once; a tap does not.
- Done when: pressing the switch in a scripted rollout flips `lamp_state` exactly
  once per sustained press across envs.

## Step 4 -- Reward, termination, reset

- Action: create `src/robotics_experiments/rl/press/_reward.py` (pure) with
  config-weighted terms returning per-env scalars from a `PressObservation`
  plus a frozen minimal input value carrying facts such as distance improvement
  and toggled-this-step:
  - alive/stability, approach-improvement (pre-contact), gentle-contact (post first
    touch), compression-progress (while aligned), switch-travel/actuation,
    one-time toggle reward (gated by the latch, not by generic wall contact),
    retreat-after-success, smoothness/energy penalty, penalties for oscillation,
    scraping, sustained misalignment, timeout.
  - Anti-pattern to avoid (encode as a comment + test): reward is tied to actuation
    + lamp confirmation, never to generic plate contact.
- Action: create `press/_termination.py` (pure): success (toggle confirmed +
  retreat), timeout, crash/instability (tilt/altitude), out-of-bounds.
- Action: reset logic in the env: randomize drone start within the "easy" range
  from `press.reset` (close to the switch, calm). Curriculum ranges come in F4.
- Config: add `PressRewardConfig` (all term weights) and `PressResetConfig`
  (start-pose ranges, easy switch scale/stiffness/threshold overrides) to the
  owning private config module and to
  `configs/experiment/rl/press/default.yaml`; add both to `PressTaskConfig`.
  Production consumers import the resolved types through
  `robotics_experiments.config`.
- Test: `tests/src/robotics_experiments/rl/press/test_press_reward.py` and
  `test_press_termination.py` (pure):
  each term fires only in its phase; toggle reward is one-shot; timeout terminates;
  generic contact without toggle earns no toggle reward.
- Done when: reward/termination unit tests pass and a scripted successful press
  yields a positive shaped return with the one-shot toggle bonus.

## Step 5 -- Generalize task metadata; keep the training adapter

- The rsl_rl adapter (`rl/training/rsl_rl.py`), `RlTrainingConfig`, and the
  training dependency profile exist from F2b Step 3 and remain unchanged.
  The standalone entrypoint is not task-generic yet: it hardcodes
  `HOVER_OBSERVATION_SCHEMA_VERSION` when exporting metadata. Add a task-owned
  observation schema field/contract to each env config and make `train_rl.py`
  read it, so hover and press artifacts record the correct version and
  dimension without a task-id conditional.
- Action: add a press-training RL config selection rather than changing the
  hover default in place. It selects the press task id and artifact namespace
  and authors any press-specific hyperparameters; no hyperparameter lives in
  Python.
- Done when: a 5-iteration smoke `train` runs against
  `Isaac-Switch-Press-Direct-v0` on the Isaac Lab image through the unchanged
  adapter.

## Step 6 -- Generalize the validated inference contract

- The validated artifact format and generic `policy_metadata(...)` composer
  exist from F2b, but public `build_policy_step_fn(...)` is hover-specific: it
  validates and assembles the 13-value hover layout from `VehicleState`.
  Expose one minimal validated observation-vector policy loader from
  `rl.press.policy` for press evaluation and F3b's policy-backed contact
  transition. It accepts the caller's press schema version/dimension, returns
  `observation_vector -> normalized action`, and imports neither rsl_rl nor the
  training stack. Keep `build_policy_step_fn(...)` as the hover routine
  composer over that lower-level loader.

## Step 7 -- Point the F2b entrypoint at the press task + eval

- Action: select `Isaac-Switch-Press-Direct-v0` through the press-training
  config passed to the F2b `train_rl.py` entrypoint (`make train-rl` and its
  rsl_rl adapter remain shared after Step 5 removes the hover-specific metadata
  assumption, running on `ISAAC_LAB_IMAGE`); add an eval script that loads an
  exported artifact and reports metrics, with a thin `make eval-press` target
  on the same image.
- Metrics to log (using the historical research roadmap's eval definitions):
  press success rate, first-contact
  stability, post-success retreat rate, median time-to-toggle, peak compression
  overshoot -- logged with per-term reward breakdown.
- Done when: a documented run trains a press policy and eval prints the metric
  table.

## Step 8 -- Deployment/runtime handoff note (scope boundary)

- The trained policy runs at deploy time as a local contact behavior ABOVE PX4
  OFFBOARD: the exported policy loader consumes a `PressObservation`
  assembled from live bridge state plus the pose provider, pusher/contact
  state, and lamp latch, and emits the F1 action, which
  maps to a PX4 OFFBOARD setpoint stream (velocity or attitude/rate per the F1
  choice). The deploy mechanism (the routine-step plug-in over the widened
  `STREAM` payload, through the unchanged frozen `FleetState` lifecycle) exists
  from F2b Step 6; the press-specific integration (handoff-gated `{kind: press}`
  step, scripted press + policy behind one contact-behavior seam, the baseline
  eval) is F3b (`F3b_contact_behavior_seam.md`), which starts only after this
  plan's exit gate. F3b must first resolve the live observation injection
  boundary recorded in that plan. Full mission sequencing, retry, and abort are
  Phase H (mission manager). Keep F3 focused on training: the generic OFFBOARD
  policy mechanism is already proven, while contact observation/routine
  composition belongs to F3b.
- Done when: this boundary is documented; no mission-manager code is added in F3.

## Files touched by F3

- add `.../tasks/switch_press_env.py`; edit `.../isaac_lab/registration.py`
- add `press/_reward.py`, `press/_termination.py`, `press/_latch.py`
- edit the owning private RL config module (`PressRewardConfig`,
  `PressResetConfig`) and `configs/experiment/rl/press/default.yaml`; add a
  separate press-training RL config selection
- edit `isaac/runtime/standalone/train_rl.py` and both env config contracts for
  task-owned observation metadata; extend `rl/press/policy.py` with the minimal
  validated vector-policy loader
- add the eval script + `make eval-press` (the rsl_rl adapter and dependency
  profile come from F2b)
- add tests under `tests/src/robotics_experiments/rl/press/` for the latch,
  reward, termination, and press-artifact contracts; extend the standalone
  training entrypoint tests

## Exit gate

On the easy/large switch with ground-truth pose and no DR, a trained policy
reliably approaches, presses (lamp toggles via the latch), and retreats -- meeting
the initial targets recorded here (press success >= 80%,
first-contact stability >= 85%, retreat >= 95%). This is the "easier task on
ground-truth pose" milestone and the precondition for F4.
