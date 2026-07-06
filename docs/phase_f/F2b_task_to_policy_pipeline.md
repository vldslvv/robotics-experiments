# F2b -- Task -> policy pipeline proof on the hover task

Status: COMPLETE (2026-07-05). `make train-rl` (1500 iterations, ~9 min on
the Isaac Lab image) converged the hover task from mean reward 1.15 to 92.12
with mean episode length 249/250, and `make test-integration-press-policy`
passed strict on the first attempt: arm -> takeoff -> the exported artifact
holding the hover for 5 s of continuous in-envelope OFFBOARD velocity
streaming over PX4 SITL -> land, through the unchanged `FleetOrchestrator`.
The sim-to-PX4 fidelity caveat (Step 7) did not bite for the hover hold; it
remains the watch item for aggressive maneuvers.

Goal served: the durable RL contract itself -- `task definition (Hydra YAML) ->
train -> exported policy artifact -> {kind: policy} routine step ->
FleetOrchestrator`. Depends on: F1 (done). Parallel with F2 (the hover
observation is assemblable from `VehicleState` alone, so this plan does not need
`PressObservation`; conversely the hover env keeps THIS plan's 13-dim layout as
its training observation -- F2's `PressObservation` is validated as a gate side
channel and first becomes an env input in F3's press env). Blocks: F3.
No deferred decisions (artifact metadata validation and hover-first sequencing
were decided 2026-07-05).

Why this exists: F3/F3b would otherwise be the first place the training adapter,
policy artifact, and deploy integration ever run -- bundled with a new contact
env, a new observation, and a new reward, so a pipeline failure would be
indistinguishable from a task failure. This plan proves the whole pipeline
end-to-end on the simplest task that already exists and flies
(`Isaac-Drone-Hover-Direct-v0`, F1). After F2b, "define a task" means authoring
YAML, "train" is one make target, and "deploy" is one routine step; F3 only adds
the press task on top.

Orchestrator plug point (load-bearing, state it once): `runtime/orchestrator.py`
requires NO code change. `FleetOrchestrator.build_drivers` already constructs a
`RoutineDriver` per drone with non-empty `routine.steps`; the policy plugs in as
a new routine step kind executed by that driver, reusing the existing OFFBOARD
streaming, envelope invariants, timeouts, and `on_failure` machinery. The seam
is the routine step, not the orchestrator.

---

## Step 1 -- Hover task-as-config

- Action: move the placeholder task constants out of
  `workflows/isaac_lab/tasks/drone_hover_env.py` into config:
  `_REWARD_POS_WEIGHT` / `_REWARD_LINVEL_WEIGHT` / `_REWARD_ANGVEL_WEIGHT` /
  `_REWARD_ALIVE`, `_CRASH_TILT_COS` / `_MIN_ALTITUDE_M` /
  `_MAX_POSITION_ERROR_M`, `_RESET_XY_HALF_M` / `_RESET_Z_HALF_M`.
- Action: add `@dataclass class PressHoverTaskConfig` (bare annotations, per the
  configs rule) to `config/types.py`; add `hover_task: PressHoverTaskConfig` to
  `RlConfig`; author a `hover_task:` block in
  `configs/experiment/rl/default.yaml` with the current values.
- The env reads `cfg.rl.press.hover_task`; no reward/termination/reset literal
  remains in the env module. This is the "task definition = one YAML block"
  half of the contract, established before the press task repeats the pattern
  (F3's `PressRewardConfig` etc. follow it).
- Test: extend the config resolution test to assert
  `get_config().rl.press.hover_task` resolves with the expected fields.
- Done when: the F1 hover+goto gate still passes with the constants sourced
  from YAML and the env module contains no task-tuning literals.

## Step 2 -- Versioned hover observation layout (train/deploy parity)

- Action: create a pure packing module in `press/` (numpy + stdlib only, on the
  core import path) defining `HOVER_OBSERVATION_SCHEMA_VERSION` and a pure
  packing function for the documented, versioned field order
  `[pos_err(3), quat_wxyz(4), lin_vel_w(3), ang_vel_b(3)]` (13 floats).
- PARITY INVARIANT (docstring + test): the same packing function is fed from
  Lab tensors in training and from `VehicleState` (world-ENU pose/velocity,
  body angular velocity) plus a target position at deploy. This is what makes a
  trained artifact deployable at all; any field a deploy-side caller cannot
  assemble from `VehicleState` is forbidden in this layout.
- Action: fix the F1 frame placeholder now, on both sides: the env's
  `_hover_observation` switches from `root_ang_vel_w` to the body-frame angular
  velocity (`root_ang_vel_b`), matching what `VehicleState` carries and what
  the actuation controller consumes. (F1's doc deferred this reconciliation to
  F2; it lands here instead.)
- Convention note: F2's `PressObservation.to_vector` /
  `OBSERVATION_SCHEMA_VERSION` adopts this same versioned-layout mechanism --
  one implementation style, two layouts.
- Test: pure layout test -- vector length equals the declared layout; field
  order is stable; changing an input field changes the expected slot;
  `HOVER_OBSERVATION_SCHEMA_VERSION` bumps are required by the test when the
  layout changes.
- Done when: env and a pure deploy-side assembly produce identical vectors for
  the same synthetic state.

## Step 3 -- rsl_rl training adapter (module-level, swappable)

- Action: create `src/robotics_experiments/rl/training/__init__.py` and
  `src/robotics_experiments/rl/training/rsl_rl.py` -- the ONLY module
  importing `rsl_rl`. Expose the functional contract:
  - `build_runner(env, runner_cfg) -> Runner`
  - `train(runner, iterations) -> training result`
  - `export_policy(runner, out_dir) -> None` (Step-4 artifact format)
  - a private runner `Protocol` typing the surface so `_skrl.py` /
    `_rl_games.py` are drop-in later.
- `rl/training/__init__.py` stays lazy (rsl_rl is a training-only dep). Keep
  the pure `press` modules free of any training import;
  `tests/test_core_imports.py` guards it.
- Config: add `RlTrainingConfig` (PPO hyperparameters: lr,
  num_steps_per_env, max_iterations, gamma, lam, entropy_coef, actor/critic
  hidden dims, seed) to `config/types.py` + a `training:` block in
  `configs/experiment/rl/default.yaml`. No hyperparameter hardcoded in
  Python.
- Dependency: add `rsl_rl` to an optional `training` extra in `pyproject.toml`
  (installed in the Isaac Lab training environment only), not to the core deps.
  Update `uv.lock` via `uv add --optional training rsl-rl` (do not hand-edit
  the lock).
- Trained against `Isaac-Drone-Hover-Direct-v0` (the Step-1 config-defined
  task); F3 points the identical adapter at the press task.
- Done when: `build_runner` + a 5-iteration smoke `train` runs on the Isaac Lab
  image without importing rsl_rl anywhere in `press/` pure modules.

## Step 4 -- Policy artifact + validated loader

- Action: `export_policy` writes a TorchScript module PLUS a JSON metadata
  sidecar next to it carrying the contract fields:
  `obs_schema_version`, `action.kind`, `action_dim`, `obs_dim`, and the action
  scale values (`max_body_velocity_mps` / `max_yaw_rate_radps` /
  `max_body_rate_radps` / `hover_thrust_fraction` as applicable to the kind).
  The artifact = module + metadata; a bare checkpoint is not a valid artifact.
- Action: create the press policy-artifact loader behind the `robotics_experiments.rl` facade
  (implemented in `src/robotics_experiments/rl/press/policy.py`; imports
  torch only inside functions, like `MavlinkCommandPort` imports pymavlink
  inside methods). Provide
  `build_policy_step_fn(path, action_cfg) -> Callable[[VehicleState, np.ndarray], setpoint]`
  that loads the artifact and raises `ValueError` on ANY metadata mismatch
  against the active config (schema version, kind, dims, scales) -- fail loudly
  at load, never a shape error deep in the driver. It imports neither rsl_rl nor
  the training stack; the runtime depends only on this seam, never on the
  training library.
- Test: `tests/test_press_policy.py`: a tiny exported identity/linear
  TorchScript module + metadata round-trips through `build_policy_step_fn`
  (guarded on torch availability); each metadata field mismatch (version, kind,
  dims, a scale) rejects with `ValueError`.
- Done when: an exported artifact loads and infers with the training stack
  uninstalled, and a mismatched artifact is refused at load time.

## Step 5 -- Train entrypoint + make target (thin)

- Action: add `isaac/runtime/standalone/train_rl.py`: composes config,
  builds the task from a config-selected task id (so F3 reuses this entrypoint
  unchanged for `Isaac-Switch-Press-Direct-v0`), calls the Step-3 adapter, and
  writes the Step-4 artifact. Follow the standalone import-ordering rule
  (AppLauncher first, Kit-dependent imports deferred).
- Action: add a thin `make train-rl` target that runs on **`ISAAC_LAB_IMAGE`**
  (the Isaac Lab superset image, like `test-integration-press-hover-goto` --
  NOT the base `ISAAC_IMAGE`), keeping the recipe thin per the Makefile rule.
- Artifacts land under a gitignored per-run directory
  (`.build/training/<policy-name>/<run-id>/`), with the deploy-facing symlink at
  `.build/training/<policy-name>/latest`, mirroring the integration-artifact
  layout; no binaries are ever committed (hard rule). `policy_name` is required
  in `rl.training` so hover and future press artifacts do not overwrite each
  other.
- Done when: `make train-rl` trains the hover task and leaves a loadable
  artifact + metadata under the run directory.

## Step 6 -- `{kind: policy}` routine step (the orchestrator plug-in)

- Action: add a `policy` routine step in `commanding/routine.py`. `decide`
  stays pure: the loaded policy is an injected pure callable
  (`Callable[[np.ndarray], np.ndarray]`, deterministic, no I/O) carried by the
  step, so the per-tick chain inside `decide` is pure end-to-end:
  pack the Step-2 layout from `RoutineObservation.vehicle_state` (+ the step's
  target position) -> `policy(obs)` -> `robotics_experiments.rl.press_action_to_offboard`
  -> a `STREAM` decision carrying the setpoint payload.
- Action: extend the `STREAM` decision payload from `PositionSetpoint`-only to
  a union also carrying `VelocitySetpoint` / `AttitudeRateSetpoint` (the wire
  payloads and port methods `send_velocity_setpoint` / `send_attitude_target`
  already exist); the private commander dispatches by payload type in its single
  side-effect position. OFFBOARD entry reuses the existing
  `goto_offboard_prime_ticks` priming semantics.
- Config: a `{kind: policy, artifact: <path>, hold_s, ...}` step descriptor in
  the pure routine builder (missing/unloadable artifact raises `ValueError` at
  build time via Step-4 validation). Each policy step carries its own artifact
  path, so one routine may use multiple named policies. The step completes after
  `hold_s` of
  in-envelope flight; the existing envelope invariants (`tilt_below`, drift,
  vertical/angular speed) apply unchanged as continuous gates, so a bad policy
  fails as `FailureKind.ENVELOPE` like any other step.
- Action: update `docs/COMMAND_ROUTINE_CONTRACT.md` for the new step kind and
  the widened `STREAM` payload as part of the same change.
- Test: extend `tests/test_commanding_routine.py` /
  `tests/test_commanding_driver.py` over fakes: a fake policy's action is
  packed/streamed each tick, envelope breach fails, `hold_s` completion
  advances; no pymavlink/torch at module scope.
- Done when: the commanding unit tests drive a fake policy through the step
  and the contract doc describes it.

## Step 7 -- End-to-end exit gate

- Action: run the routine `[arm, takeoff, policy (hover hold), land]` with a
  Step-5-trained hover artifact over PX4 SITL through the standalone scenario
  in strict mode -- the same orchestrator phases and `RoutineDriver` tick loop
  as every other routine, proving a Lab-trained artifact flies through the
  unchanged `FleetOrchestrator`.
- Measured (2026-07-05): first strict run passed -- takeoff at ~2.5 m climb,
  the policy streamed body-velocity OFFBOARD setpoints for the full 5 s hold
  with no envelope breach, then landed and disarmed inside the 30 s budget
  (routine wall time ~38 s end to end, bridge log validation clean).
- Known-risk caveat (record measured behavior here when the gate runs): in Lab
  the action drives the in-sim actuation model directly; at deploy PX4's own
  controllers track the setpoint, so the closed-loop dynamics differ. The
  `velocity` action level was chosen to minimize this gap, but the F1b
  limit-cycle boundary is exactly where sim-to-PX4 fidelity issues will
  surface first. An envelope failure here with a Lab-stable policy is a
  fidelity finding, not automatically a pipeline bug.
- Done when: the strict scenario passes with the trained hover policy holding
  the envelope through the policy step.

## Files touched by F2b

- edit `workflows/isaac_lab/tasks/drone_hover_env.py` (task-as-config, body
  ang vel observation)
- add the pure hover-observation layout module under `press/`;
  add `rl/press/policy.py`, `rl/training/__init__.py`,
  `rl/training/rsl_rl.py`; edit `rl/__init__.py` (re-exports)
- edit `config/types.py` (`PressHoverTaskConfig`, `RlTrainingConfig`),
  `configs/experiment/rl/default.yaml`
- add `isaac/runtime/standalone/train_rl.py`; add the `train-rl` make
  target (Isaac Lab image)
- edit `commanding/routine.py` (+ commander/driver for the widened `STREAM`
  payload), `docs/COMMAND_ROUTINE_CONTRACT.md`
- add `tests/test_press_policy.py` + the layout test; extend the commanding
  tests, the config test, `tests/test_core_imports.py`
- `pyproject.toml` `training` extra + `uv.lock`

## Exit gate

`make train-rl` produces a validated artifact from the YAML-defined hover
task; the policy-step loader enforces the metadata contract; the `{kind: policy}` step
flies that artifact over PX4 SITL through the unchanged orchestrator in the
strict standalone scenario. The task->policy pipeline is proven end-to-end;
F3 adds the press task, not plumbing.
