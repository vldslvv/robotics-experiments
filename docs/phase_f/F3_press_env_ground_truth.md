# F3 -- Switch-press RL env on ground truth (easy/large switch) + first policy

Goal served: goal 1 (easier task on ground-truth pose). Depends on: F1 (flyable
env + actuation seam), F2 (observation + provider + handoff), and F2b (the
proven task->policy pipeline: training adapter, validated artifact,
`{kind: policy}` deploy step). Blocks: F4.
No deferred decisions (inherits F1's action-level decision).

Why this exists: with a flyable env (F1), a stable observation (F2), and a
working train->export->deploy pipeline (F2b), build the contact task itself --
spawn a switch per env, wire the pusher/contact/lamp, define the reward, and
train the first policy that approaches, presses, confirms the toggle, and
retreats. F3 adds a task, not plumbing: the training adapter, artifact format,
loader, and entrypoint all exist from F2b and are reused unchanged. Deliberately
easy: oversized/soft switch, ground-truth pose, no domain randomization (F4 adds
those).

---

## Step 1 -- New task: `Isaac-Switch-Press-Direct-v0`

- Action: create
  `src/robotics_experiments/workflows/isaac_lab/tasks/switch_press_env.py`
  (torch/isaaclab leaf) with `SwitchPressEnvCfg(DirectRLEnvCfg)` and
  `SwitchPressEnv(DirectRLEnv)`, reusing F1's `flight.py` actuation/force sink and
  F2's observation assembly. Do not fork the hover env; import the shared pieces.
- Action: register it in `workflows/isaac_lab/registration.py`
  (`_SWITCH_PRESS_TASK_ID = "Isaac-Switch-Press-Direct-v0"`, guarded like the hover
  task).
- Config: `SwitchPressEnvCfg` reads sizes/spacing/episode from `cfg.rl`.
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
- Wire `GroundTruthPoseProvider` (F2 Step 4) to this per-env switch prim.
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
- Test: `tests/test_press_latch.py` (pure): batched latch matches the scalar
  switch latch for the same input sequences; a sustained press toggles
  once; a tap does not.
- Done when: pressing the switch in a scripted rollout flips `lamp_state` exactly
  once per sustained press across envs.

## Step 4 -- Reward, termination, reset

- Action: create `src/robotics_experiments/rl/press/_reward.py` (pure) with
  composable, config-weighted terms returning per-env scalars from a
  `PressObservation` (+ minimal extra state like distance-improved,
  toggled-this-step):
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
  (start-pose ranges, easy switch scale/stiffness/threshold overrides) to
  `config/types.py` + `configs/experiment/rl/default.yaml`; add to `RlConfig`.
- Test: `tests/test_press_reward.py`, `tests/test_press_termination.py` (pure):
  each term fires only in its phase; toggle reward is one-shot; timeout terminates;
  generic contact without toggle earns no toggle reward.
- Done when: reward/termination unit tests pass and a scripted successful press
  yields a positive shaped return with the one-shot toggle bonus.

## Step 5 -- Training adapter + config (exists from F2b; press specifics only)

- The rsl_rl adapter (`rl/training/rsl_rl.py`), `RlTrainingConfig`, and
  the `training` pyproject extra exist from F2b Step 3. F3 adds NO training or
  loader code.
- Action: author the press-task hyperparameter values in the `training:` block
  (they may differ from the hover run; still no hyperparameter in Python).
- Done when: a 5-iteration smoke `train` runs against
  `Isaac-Switch-Press-Direct-v0` on the Isaac Lab image through the unchanged
  adapter.

## Step 6 -- Inference contract (exists from F2b)

- The validated artifact format (TorchScript + metadata sidecar) and
  `robotics_experiments.rl.build_policy_step_fn` exists from F2b Step 4; the runtime press
  behavior depends only on that seam, never on the training library. Nothing to
  add here; the F2b metadata validation (obs schema version, action kind, dims,
  scales) applies to press artifacts identically -- the press observation's
  `OBSERVATION_SCHEMA_VERSION` (F2) is what lands in the sidecar.

## Step 7 -- Point the F2b entrypoint at the press task + eval

- Action: select `Isaac-Switch-Press-Direct-v0` as the config-selected task id
  for the F2b `train_rl.py` entrypoint (`make train-rl` reused unchanged,
  running on `ISAAC_LAB_IMAGE`); add an eval script that loads an exported
  artifact and reports metrics, with a thin `make eval-press` target on the
  same image.
- Metrics to log (per ROADMAP1 eval table): press success rate, first-contact
  stability, post-success retreat rate, median time-to-toggle, peak compression
  overshoot -- logged with per-term reward breakdown.
- Done when: a documented run trains a press policy and eval prints the metric
  table.

## Step 8 -- Deployment/runtime handoff note (scope boundary)

- The trained policy runs at deploy time as a local contact behavior ABOVE PX4
  OFFBOARD: the exported policy loader consumes a `PressObservation`
  assembled from the live bridge state + provider and emits the F1 action, which
  maps to a PX4 OFFBOARD setpoint stream (velocity or attitude/rate per the F1
  choice). The deploy mechanism (the routine-step plug-in over the widened
  `STREAM` payload, through the unchanged `FleetOrchestrator`) exists from F2b
  Step 6; the press-specific integration (handoff-gated `{kind: press}` step,
  scripted press + policy behind one contact-behavior seam, the baseline eval)
  is F3b (`F3b_contact_behavior_seam.md`), which starts only after this plan's
  exit gate. Full mission sequencing, retry, and abort are Phase H (mission
  manager). Keep F3 focused on training; the deploy seam needs no stubbing --
  it is already proven on the hover task.
- Done when: this boundary is documented; no mission-manager code is added in F3.

## Files touched by F3

- add `.../tasks/switch_press_env.py`; edit `.../isaac_lab/registration.py`
- add `press/_reward.py`, `press/_termination.py`, `press/_latch.py`
- edit `config/types.py` (`PressRewardConfig`, `PressResetConfig`),
  `configs/experiment/rl/default.yaml` (press values in the F2b `training:`
  block)
- add the eval script + `make eval-press` (the training entrypoint, adapter,
  artifact loader, and `pyproject.toml` extra come from F2b)
- add `tests/test_press_latch.py`, `tests/test_press_reward.py`,
  `tests/test_press_termination.py`

## Exit gate

On the easy/large switch with ground-truth pose and no DR, a trained policy
reliably approaches, presses (lamp toggles via the latch), and retreats -- meeting
the initial-target column of the ROADMAP1 metrics (press success >= 80%,
first-contact stability >= 85%, retreat >= 95%). This is the "easier task on
ground-truth pose" milestone and the precondition for F4.
