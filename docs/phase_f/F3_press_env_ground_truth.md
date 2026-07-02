# F3 -- Switch-press RL env on ground truth (easy/large switch) + first policy

Goal served: goal 1 (easier task on ground-truth pose). Depends on: F1 (flyable
env + actuation seam) and F2 (observation + provider + handoff). Blocks: F4.
No deferred decisions (inherits F1's action-level decision).

Why this exists: with a flyable env (F1) and a stable observation (F2), build the
contact task itself -- spawn a switch per env, wire the pusher/contact/lamp, define
the reward, and train the first policy that approaches, presses, confirms the
toggle, and retreats. Deliberately easy: oversized/soft switch, ground-truth pose,
no domain randomization (F4 adds those).

---

## Step 1 -- New task: `Isaac-Switch-Press-Direct-v0`

- Action: create
  `src/robotics_experiments/workflows/isaac_lab/tasks/switch_press_env.py`
  (torch/isaaclab leaf) with `SwitchPressEnvCfg(DirectRLEnvCfg)` and
  `SwitchPressEnv(DirectRLEnv)`, reusing F1's `_flight.py` actuation/force sink and
  F2's observation assembly. Do not fork the hover env; import the shared pieces.
- Action: register it in `workflows/isaac_lab/_registration.py`
  (`_SWITCH_PRESS_TASK_ID = "Isaac-Switch-Press-Direct-v0"`, guarded like the hover
  task).
- Config: `SwitchPressEnvCfg` reads sizes/spacing/episode from `cfg.press`.
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
- Action: vectorize the N-step press latch. The pure single-switch logic already
  exists: `environment/_switches.py::_apply_button_sample` (press_threshold +
  press_confirm_steps -> single toggle). Add a pure batched version in `press/`
  (e.g. `press/_latch.py`, numpy/torch-agnostic core with a torch wrapper in the
  leaf) that maintains `press_steps` / `toggle_latched` / `state` per env, so
  `lamp_state` in the observation comes from the same latch semantics the runtime
  environment uses. Do not re-implement the threshold logic differently.
- Config: reuse `switches.press_threshold` / `switches.press_confirm_steps`; the
  "easy" stage may loosen them via a `press.reset`/curriculum override (F4).
- Test: `tests/test_press_latch.py` (pure): batched latch matches the scalar
  `_apply_button_sample` for the same input sequences; a sustained press toggles
  once; a tap does not.
- Done when: pressing the switch in a scripted rollout flips `lamp_state` exactly
  once per sustained press across envs.

## Step 4 -- Reward, termination, reset

- Action: create `src/robotics_experiments/press/_reward.py` (pure) with
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
  `config/_types.py` + `configs/experiment/press/default.yaml`; add to `PressConfig`.
- Test: `tests/test_press_reward.py`, `tests/test_press_termination.py` (pure):
  each term fires only in its phase; toggle reward is one-shot; timeout terminates;
  generic contact without toggle earns no toggle reward.
- Done when: reward/termination unit tests pass and a scripted successful press
  yields a positive shaped return with the one-shot toggle bonus.

## Step 5 -- rsl_rl adapter (module-level, swappable) + config

- Action: create `src/robotics_experiments/press/training/__init__.py` and
  `src/robotics_experiments/press/training/_rsl_rl.py` -- the ONLY module importing
  `rsl_rl`. Expose the functional contract:
  - `build_runner(env, runner_cfg) -> Runner`
  - `train(runner, iterations) -> TrainResult`
  - `export_policy(runner, out_path) -> None` (TorchScript or ONNX)
  - a `PolicyRunner` `Protocol` typing the surface so `_skrl.py` / `_rl_games.py`
    are drop-in later.
- `press/training/__init__.py` stays lazy (rsl_rl is a training-only dep). Keep the
  pure `press` modules free of any training import.
- Config: add `PressTrainingConfig` (PPO hyperparameters: lr, num_steps_per_env,
  max_iterations, gamma, lam, entropy_coef, actor/critic hidden dims, seed) to
  `config/_types.py` + YAML. No hyperparameter hardcoded in Python.
- Dependency: add `rsl_rl` to an optional `training` extra in `pyproject.toml`
  (installed in the Isaac/training environment only), not to the core deps. Update
  `uv.lock` via `uv add --optional training rsl-rl` (do not hand-edit the lock).
- Done when: `build_runner` + a 5-iteration smoke `train` runs in the Isaac
  container without importing rsl_rl anywhere in `press/` pure modules.

## Step 6 -- Framework-agnostic inference contract (for deployment)

- Action: create `src/robotics_experiments/press/_policy.py` (pure-ish; imports
  torch only inside functions, like `MavlinkCommandPort` imports pymavlink inside
  methods). Provide `load_policy(path) -> Callable[[np.ndarray], np.ndarray]` that
  loads the exported TorchScript/ONNX policy and runs a forward pass, importing
  neither rsl_rl nor the training stack. This is the deploy-time seam: the runtime
  press behavior depends only on this, never on the training library.
- Test: `tests/test_press_policy.py`: a tiny exported identity/linear TorchScript
  module loads and produces the right-shaped action from a sample observation
  vector (guarded on torch availability).
- Done when: an exported policy loads and infers with the training stack uninstalled.

## Step 7 -- Train/eval entrypoint + make targets (thin)

- Action: add a training entrypoint script under
  `isaac/runtime/standalone/train_press.py` (or extend the standalone runner) that
  composes config, builds `Isaac-Switch-Press-Direct-v0`, and calls the training
  adapter; and an eval script that loads an exported policy and reports metrics.
- Action: add thin make targets (`make train-press`, `make eval-press`) that run in
  the Isaac container, following the existing standalone/integration target style
  (keep recipes thin per the Makefile rule).
- Metrics to log (per ROADMAP1 eval table): press success rate, first-contact
  stability, post-success retreat rate, median time-to-toggle, peak compression
  overshoot -- logged with per-term reward breakdown.
- Done when: a documented run trains a policy and eval prints the metric table.

## Step 8 -- Deployment/runtime handoff note (scope boundary)

- The trained policy runs at deploy time as a local contact behavior ABOVE PX4
  OFFBOARD: the exported policy (`load_policy`) consumes a `PressObservation`
  assembled from the live bridge state + provider and emits the F1 action, which
  maps to a PX4 OFFBOARD setpoint stream (velocity or attitude/rate per the F1
  choice). A thin press behavior in the commanding layer (a new routine step
  `{kind: press}` / a `PressDriver` over `MavlinkCommandPort`) is the integration
  point -- but full mission sequencing, retry, and abort are Phase H (mission
  manager). Keep F3 focused on training; only stub the deploy seam.
- Done when: this boundary is documented; no mission-manager code is added in F3.

## Files touched by F3

- add `.../tasks/switch_press_env.py`; edit `.../isaac_lab/_registration.py`
- add `press/_reward.py`, `press/_termination.py`, `press/_latch.py`,
  `press/_policy.py`, `press/training/__init__.py`, `press/training/_rsl_rl.py`
- edit `config/_types.py` (`PressRewardConfig`, `PressResetConfig`,
  `PressTrainingConfig`), `configs/experiment/press/default.yaml`
- add `isaac/runtime/standalone/train_press.py` + eval; make targets
- add `tests/test_press_latch.py`, `tests/test_press_reward.py`,
  `tests/test_press_termination.py`, `tests/test_press_policy.py`
- `pyproject.toml` training extra + `uv.lock`

## Exit gate

On the easy/large switch with ground-truth pose and no DR, a trained policy
reliably approaches, presses (lamp toggles via the latch), and retreats -- meeting
the initial-target column of the ROADMAP1 metrics (press success >= 80%,
first-contact stability >= 85%, retreat >= 95%). This is the "easier task on
ground-truth pose" milestone and the precondition for F4.
