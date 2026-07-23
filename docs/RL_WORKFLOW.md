# RL workflow (Isaac Lab tasks, actuation, training, deployment)

This is the single source for the reinforcement-learning workflow detail:
the Isaac Lab image, the in-Lab task/actuation seam, and the task->policy
pipeline (training, artifact, deploy-time routine step, gates). `AGENTS.md`
keeps only the short summary and points here. The deploy-time `{kind: policy}`
routine step's execution semantics live in `docs/COMMAND_ROUTINE_CONTRACT.md`;
the phase F design notes live under `docs/phase_f/` and the ADRs under
`docs/adr/` (notably 0013 and 0014).

## Isaac Lab image

The Isaac Lab image (`make build-isaac-lab`, `ISAAC_LAB_IMAGE` =
`vv-isaac-lab-custom:<tag>`) is the Dockerfile `isaac-lab` stage layered
`FROM isaac-sim`: Isaac Lab shares the same kit python (`/isaac-sim/python.sh`),
USD, and PhysX and cannot run standalone, so it is a superset image, not a
separate container. It is built from `isaac-sim/IsaacLab` pinned by the
`ISAACLAB_REF` build ARG / Make variable (the immutable `v2.3.2` commit for
Isaac Sim 5.1). The image installs only the Isaac Lab base package, its rsl_rl
adapter, and the pinned closure in
`isaac/docker/lab-runtime-requirements.txt`; unused RL frameworks and device
retargeting stacks are excluded. Only the RL workflow needs it -- training,
checkpoint selection, independent press evaluation, and the press/Lab targets
defined in `make/rl.mk` run on `ISAAC_LAB_IMAGE`; every other integration target
and the persistent containers stay on the leaner `isaac-sim` image
(`ISAAC_IMAGE`). Rebuild it after changing `ISAACLAB_REF` or when moving Isaac
Sim versions.

## In-Lab task and actuation seam

Importable Isaac Lab task/env code lives under
`src/robotics_experiments/workflows/isaac_lab/`; `isaac/runtime/lab/` is the
workflow/design-note area for Isaac Lab runtime files. Code outside that package
imports only from `robotics_experiments.workflows.isaac_lab`; its lazy package
root is the public contract and exposes guarded task registration, both task
IDs/env/config factories, the generic `make_isaac_lab_env(...)`, and the named
batched rotor-omega diagnostic seam without eager-importing Isaac Lab task
modules during host-side package imports.

Isaac Lab task IDs use the Gymnasium `Isaac-...-Direct-v0` naming convention
for direct-workflow environments. The registered tasks are
`Isaac-Drone-Hover-Direct-v0` and `Isaac-Switch-Press-Direct-v0`; registration
is independently idempotent for each. Both fly N parallel drones through the
pluggable in-Lab actuation seam (`robotics_experiments.rl.press`), which maps a
normalized RL action to per-rotor angular-velocity references (and thence to
forces) without a PX4 lockstep per env. The action level is a config switch
(`rl.action.kind`): `velocity` (default; cascaded PD tracking body-frame
velocity + yaw rate) or `ctbr` (collective thrust + body-rate setpoint). The
actuation is written functionally -- state in frozen params dataclasses built
once (`build_actuation_params` / `build_flight_actuation_params`, both
precomputing the allocation pseudoinverse), behavior in pure functions
(`action_to_rotor_omega`, `hover_action`, `rotor_omega_to_body_wrench`) -- so
a future pipeline composes it as plain functions with the RL policy as one
stage. The public single-env reference contract is exported only from
`robotics_experiments.rl.press` (implemented in `rl/press/actuation.py`; numpy, the
test oracle) and the Lab package exposes the env diagnostic entry point as
`robotics_experiments.workflows.isaac_lab.compute_lab_rotor_omega`; both mirror
the PX4 bridge's `PegasusQuadraticThrustModel` allocation so an action
reproduces the commanded wrench.
The NumPy reference builder copies the caller's mixer geometry and marks every
geometry, gain, and derived-allocation array read-only, so its frozen parameter
graph cannot be mutated through an external alias.

The action contract for a new training run lives in the Hydra `rl` group, with
matching bare annotations in `src/robotics_experiments/config/_rl.py`.
Hydra owns the action kind, dimensions, physical scales, and cadence until the
run captures them in `policy_meta.json`; the exported artifact owns those
values for evaluation and deployment. Controller gains, tilt limits, vehicle
dynamics, and physical task settings remain YAML-owned.
`rl.training.vehicle_id` selects the experiment vehicle whose spawn asset and
airframe-enriched bridge dynamics configure the Lab task; selection fails when
the task is constructed if that ID is absent. `bridge.enabled` is irrelevant
because Lab deliberately bypasses PX4.
The standalone training root passes the resolved `ExperimentConfig` into the
generic registry-owned factory; the Isaac Lab leaf never reloads global config. The
physics decimation is selected per policy step by the shared exact-average
cadence, not authored. At 250 Hz physics and the default 20 Hz policy rate it
alternates 13 and 12 physics steps (52/48 ms), while `step_dt` remains the
nominal 50 ms. The source period is derived from `cfg.sim.dt`; there is no
redundant physics-rate field in the Lab config. `max_episode_length` is fixed to
`ceil(episode_length_s * policy_rate_hz)`, so the alternating decimation cannot
shorten an episode (20 steps per simulated second and exactly 100 steps for the
default five-second episode). The fixed render interval uses the ceiling
decimation. The hover task's
reward/termination/reset values are the `rl.press.hover_task` block (task
definition = one YAML block; no task-tuning literal in the env module). The RL
action also maps to a PX4 OFFBOARD setpoint at deploy time
(`press_action_to_offboard` + the commanding-layer velocity / attitude-rate
setpoint seams).

The private flight base owns cadence, action buffering, rotor force application,
pusher hold, and drone reset. Task leaves own scene and task behavior. The press
task clones one real switch articulation and static wall per environment, reads
batched switch-root poses as ground truth, measures pusher-pad contact with an
Isaac Lab contact sensor, and reads pusher/button joints directly. Its easy-task
geometry and initial pose ranges, contact threshold, reward, reset, termination,
and evaluation values live under `rl.press`; Lab-only vehicle limits live under
`rl.lab_vehicle`. Pusher stiffness and damping are inherited from the generated
USD rather than duplicated in Lab config. The generated pusher is a complete
tool: a fixed bracket and 265 mm body-mounted boom, a visible moving plunger,
and a flat forward-facing pad. The length keeps the compressed pad face ahead
of the 13-inch forward rotor sweep. Arm capsules remain collidable and each
propeller has an invisible swept-disc collider authored as a stationary sibling
of its animated visual marker. The compliant body and sensor paths remain
`/pusher_joint` and `/pusher/pad/geom`. Fixed/moving masses are explicit; the
fixed body is 2.3372 kg and the 0.010 kg moving pad/plunger makes the complete
X500 articulation 2.3472 kg.

Press progression is the backend-neutral frozen `PressTaskState` advanced by
`step_press_task(...)`. Its vector latch matches the production scalar
`apply_button_sample` semantics. DirectRLEnv advances it once from `_get_dones`
and caches the result for rewards and observations. Success is a confirmed
toggle plus stable retreat; generic contact alone never earns toggle reward.
Confirmation progress is signed: every uninterrupted above-threshold sample
earns part of the latch-progress reward, while releasing before confirmation
removes the accumulated progress credit. Approach and retreat progress are also
signed. Retreat returns toward at least the episode's initial wall-normal
range, stops earning distance reward there, penalizes alignment error beyond
the acceptance tolerance and excess speed, and completes only while
contact-free, aligned, and stable. Precontact motion within 0.42 m of the switch
penalizes linear speed above 0.20 m/s and angular speed above 0.50 rad/s.
After the first toggle, continued contact and every step before stable retreat
carry small costs; another confirmed toggle carries a one-shot failure-scale
penalty. The terminal success bonus is paid only when the first contact was
stable.
Crash/altitude/envelope termination has an explicit terminal failure cost,
preventing timeout-hover and early-exit reward shortcuts. The configured
terminal failure cost is larger than the timeout cost so an uncontrolled early
exit cannot outscore a controlled but unsuccessful full attempt.

During training, the press environment publishes cumulative completed-episode
contact, confirmed-toggle, full-success, and failure percentages under the
`Press/` TensorBoard namespace. The same namespace reports stable first contact
among contacted episodes, retreat among toggled episodes, and repeated toggles
among toggled episodes. It also publishes the active-episode contact and toggle
percentages and the principal signed reward terms, including every
fine-tuning-specific term, under `Reward/`.
These are diagnostic training signals; deterministic checkpoint evaluation
remains the source of selection and acceptance metrics.

## Task->policy pipeline (F2b)

The pipeline is `task YAML -> make train-rl -> make select-rl-checkpoint ->
policy artifact -> make eval-press -> {kind: policy} routine step ->
FleetState lifecycle`. The runtime injects the
artifact-selected adapter and, when its interface requires one, the task-owned
observation assembled from live resources. Commanding remains task-neutral.
Pieces:

- **Observation layouts** (exported from `robotics_experiments.rl.press`, pure,
  versioned): the
  13-dim hover layout (`HOVER_OBSERVATION_SCHEMA_VERSION`,
  `pack_hover_observation` / `hover_observation_from_state`; body-FLU angular
  velocity) and the press-task `PressObservation`/`to_vector`
  (`OBSERVATION_SCHEMA_VERSION`) with the contact frame
  (`world_switch_to_contact_frame`; canonical switch `+X` = outward plate
  normal). PARITY INVARIANT: the same packing feeds Lab tensors in training
  and the live runtime at deploy. The pose-provider seam is
  `AbstractPoseProvider` / `PoseSample`; ground truth is the lazy Isaac-only
  `GroundTruthPoseProvider`, and the nav->contact handoff is the pure
  `handoff_ready` gate (thresholds in `rl.press.handoff`). See ADR 0014.
- **Training and selection** (`rl/training/`; `rsl_rl.py` is the only rsl_rl
  importer): import `build_runner`/`train`/`load_checkpoint`/
  `initialize_actor_from_checkpoint`/`deterministic_actor`/`export_policy`/
  `TrainingResult` from
  `robotics_experiments.rl.training.rsl_rl`. The contract runs
  over `RlTrainingConfig` (the `rl.training` block, incl. explicit `task_id`
  and `policy_name`).
  `make train-rl` (thin target in `make/rl.mk`, runs
  `isaac/runtime/standalone/train_rl.py` on `ISAAC_LAB_IMAGE`, never the base
  image) writes the stored policy contract, Git provenance, and all checkpoints
  under `.build/training/<policy-name>/<run-id>/`; it does not export
  `policy.pt` or advance `latest`.
  `TRAIN_ARGS='--max-iterations 5 --num-envs 2'` is a small smoke run. Each
  task config owns its observation schema version and dimension, so stored
  metadata contains no task-ID conditional. Before PPO begins, the run writes
  the complete policy contract and Git provenance to its run directory.
  `--initialize-checkpoint PATH` creates the same fresh run but loads only the
  compatible `actor.*` tensors before PPO starts. The critic, optimizer,
  exploration standard deviation, and iteration counter remain freshly
  initialized. The source observation/action dimensions and exact actor tensor
  layout must match. Stored strict-JSON provenance records `method:
  actor-only`, source run, checkpoint iteration/path, and source commit.
  `--initialize-checkpoint` and `--export-checkpoint` are mutually exclusive;
  export-only loading retains its existing no-optimizer behavior.
  rsl_rl comes from the pinned Lab image requirements; the host `training`
  extra (`rsl-rl-lib`) exists for tooling only.
  `make select-rl-checkpoint RL_RUN_ID=<run-id>` resumes selection without
  retraining. The task-neutral selector enumerates every numeric checkpoint,
  builds one compatible environment and runner, and dispatches by stored
  `policy_interface`. `press-v1` evaluates every checkpoint with the full
  workload shared with `eval-press`; an interface without a scorer warns and
  selects its final checkpoint with `selection_method: final-no-scorer`.
  Selection reloads the winner without optimizer state, stages the TorchScript
  actor, selected metadata, and `checkpoint_selection.json`, then atomically
  advances the relative `.build/training/<policy-name>/latest` symlink.
  A scorer always selects a checkpoint: it prefers acceptance-passing
  candidates and otherwise publishes the highest-ranked candidate as an
  explicitly rejected fallback. The independent evaluator remains the
  acceptance gate.
  Explicit export remains available for manual checkpoint diagnostics:

  ```console
  make train-rl HYDRA_CONFIG=configs/train_press_quad.yaml \
    TRAIN_ARGS='--num-envs 2 --export-checkpoint /isaac-sim/proj/.build/training/press/<run-id>/logs/model_<iteration>.pt'
  ```

  Manual export loads policy weights without optimizer state, reuses the
  contract and provenance captured in that run before training, writes the
  artifact back into the checkpoint's existing run directory, and deliberately
  does not publish `latest`. It rejects a run missing stored metadata. The
  checkpoint must be below that namespace's run-local `logs/` directory.
- **Artifact** = TorchScript `policy.pt` + `policy_meta.json`
  (`policy_interface`, obs schema version/dim, policy rate, action kind/dim,
  action scales, source Git commit, and dirty-worktree flag). The interface is
  the runtime adapter selector (`hover-v1` or `press-v1`), not a commanding
  routine kind. Train accepted policies from a clean revision;
  `source_dirty: true` is diagnostic provenance, not a claim that the commit
  alone reproduces the artifact. The frozen `PolicyContract` is the typed
  runtime authority for the interface, observation layout, cadence, action
  shape, and physical action scales. `load_vector_policy(...)` validates that
  the runtime supplies the artifact's supported observation contract and
  returns a `LoadedVectorPolicy` carrying both the contract and inference
  callable. It never compares against deployment `rl.action` or
  `rl.policy_rate_hz`. `validate_policy_artifact(...)` performs the same
  contract validation without importing Torch and backs the pre-launch Make
  preflight. Malformed, missing, non-finite, structurally invalid, and
  unsupported contracts fail before simulator startup. Exact schema-v1 hover
  and press artifacts made before `policy_interface` are recognized only from
  their complete contract fields; legacy press inference retains its exact
  34-value, 20 Hz, four-value velocity tuple and warns.
  A scored selected artifact also records `selected_checkpoint_iteration`,
  `selection_method`, and `checkpoint_evaluation`. The last field copies the
  selected checkpoint's acceptance result, evaluation seeds and episode count,
  thresholds, metrics, normalized margins, ranking tuple, and reward-term
  means. These additional fields are strict-JSON diagnostic provenance and do
  not change the policy runtime contract.
- **Gate**: `make test-integration-press-policy` flies
  `configs/policy_hover_quad.yaml` (routine `policy_hover`: arm ->
  takeoff -> policy hover hold -> land) strict over PX4 SITL on
  `ISAAC_LAB_IMAGE`. It needs a published artifact from `make train-rl`
  followed by `make select-rl-checkpoint` (the routine
  references `.build/training/hover/latest`), so it is deliberately NOT part of
  `test-integration-all`. The hover+goto gate additionally runs the F2
  side-channel validation (ground-truth provider + `PressObservation` +
  handoff flip) without touching the env's training observation.

## Ground-truth press training and evaluation (F3)

`configs/train_press_quad.yaml` selects `Isaac-Switch-Press-Direct-v0`, the
`press` artifact namespace, ten-second episodes, and the conservative
fine-tuning PPO settings: fixed `1e-4` learning rate, 0.20 initial action noise,
25-iteration checkpoints, and at most 150 iterations. The network, rollout,
clipping, entropy, GAE, and batch settings remain the established press-task
settings. Actor-only fine-tuning from the protected checkpoint-50 fallback is:

```console
make train-rl HYDRA_CONFIG=configs/train_press_quad.yaml RL_RUN_ID=<run-id> \
  TRAIN_ARGS='--num-envs 49152 --initialize-checkpoint /isaac-sim/proj/.build/training/press/20260723T215635Z.61663/logs/model_50.pt'
make select-rl-checkpoint HYDRA_CONFIG=configs/train_press_quad.yaml RL_RUN_ID=<run-id>
make eval-press
```

For `press-v1`, selection creates one 100-environment scene and compatible
runner, then loads every numeric `model_<iteration>.pt` once, including
`model_0`, periodic checkpoints, and the final checkpoint. Each candidate runs
100 deterministic episodes for seeds 101, 202, and 303. There is no screening,
TensorBoard-reward shortcut, or reduced episode count. The shared evaluator
records every reward-term mean, confirmed toggle rate, stable first contact
among contacted episodes, retreat among toggled episodes, full completion,
median toggle time, and peak compression overshoot.

A press candidate is accepted only at the configured 80% confirmed-toggle,
85% stable-first-contact, and 95% post-toggle-retreat thresholds. Accepted
candidates are always preferred. Within that set, candidates are ranked by the
lexicographically largest sorted tuple of each acceptance rate divided by its
threshold, maximizing the weakest normalized gate first. Exact ties prefer
higher full completion, lower peak compression overshoot, lower median toggle
time, then earlier iteration. If no candidate is accepted, the same ranking
and tie-breakers select the best fallback from all evaluated checkpoints; PPO
return is never a selection input.
Schema-2 `checkpoint_selection.json` uses
`selection_method: press-gated-maximin-v2` and retains the run/interface
identity, seeds, thresholds, all candidates and metrics, normalized margins,
ranking values, reward means, selected iteration, and `selected_accepted`.
Selection publishes either the accepted winner or the explicitly marked
fallback and advances `latest`.

`make eval-press` independently evaluates the selected artifact with the
identical rollout and aggregation logic. With `RL_RUN_ID` it reads that exact
published run; otherwise it resolves `.build/training/press/latest`. The Lab
environment uses the artifact's action kind, dimension, physical scales, and
cadence, while controller gains and physical task settings continue to come
from the training YAML. Artifacts remain ignored under `.build/training`; no
policy binary is committed.

For the canonical logged train-select-evaluate workflow, use the wrapper:

```console
make/scripts/run_training_evaluation_then_shutdown.sh \
  --hydra-config configs/train_press_quad.yaml \
  --train-target train-rl \
  --select-target select-rl-checkpoint \
  --eval-target eval-press \
  --train-args '--num-envs 49152 --initialize-checkpoint /isaac-sim/proj/.build/training/press/20260723T215635Z.61663/logs/model_50.pt' \
  --label press-finetune
```

Shutdown is disabled by default, so this form never invokes `sudo`, refreshes
sudo authorization, calls `sync`, or requests power-off. Add the valueless
`--shutdown` flag for an unattended run that powers off after a successful
evaluation, a failed stage, or a handled interrupt. In that mode the wrapper
validates sudo access before training and refreshes the authorization while the
workload runs. The wrapper creates one artifact run ID and passes it as
`RL_RUN_ID` to all three stages. Training failure skips selection and
evaluation; an operational selection/export failure skips evaluation and
cannot change the previous `latest`. A gate-rejected fallback is a successful
selection, so independent evaluation runs against that exact published run and
fails the overall workflow when the artifact still misses a threshold.
The wrapper preserves the terminal stage status unless the explicitly
requested shutdown command fails. Its target names, Hydra config, per-stage
arguments, label, and log directory are command-line options. The combined
console log and final three-stage exit status, including target names, artifact
run ID, and shutdown mode, default to
`.build/overnight/<label>.<UTC-time>.<pid>.console.log` and `.status`. The
ordinary integration and policy artifacts remain in `.build/docker/integration`
and `.build/training`.
The wrapper also starts `tensorboard-rl` in the pinned Isaac Lab image. It
resolves `rl.training.policy_name` from the selected Hydra config, begins
training, waits for that run's non-empty event file, and then serves only
`.build/training/<policy-name>/<artifact-run-id>/logs` on
`http://127.0.0.1:6006/`. Starting after the event file exists avoids a
TensorBoard process whose initially absent log root never activates any
dashboard, while exact scoping keeps unrelated runs out of the UI. The wrapper
prints the URL and `<policy-name>/<artifact-run-id>` label to the terminal and
combined console log only after the scalar-tags endpoint reports loaded data.
It stops the named TensorBoard container on every wrapper exit path. If the
server exits or does not load scalar data, the error is logged and a
successfully completed training stage does not proceed to selection.

For explicit unattended shutdown, add the flag:

```console
make/scripts/run_training_evaluation_then_shutdown.sh \
  --hydra-config configs/train_press_quad.yaml \
  --eval-target eval-press \
  --train-args '--num-envs 49152 --initialize-checkpoint /isaac-sim/proj/.build/training/press/20260723T215635Z.61663/logs/model_50.pt' \
  --label press-finetune \
  --shutdown
```

The zero-manual-update deployment sequence is:

1. Run the wrapper with `--hydra-config configs/train_press_quad.yaml`, adding
   `--shutdown` only when power-off is wanted.
2. Training writes an unpublished checkpoint run under the wrapper's
   `RL_RUN_ID`.
3. Selection publishes the best checkpoint and advances
   `.build/training/press/latest`; metadata marks whether it passed acceptance
   or is the best ranked fallback.
4. Independent evaluation reads the exact selected run and remains the
   acceptance gate. `latest` means the newest selected artifact, which may be
   an explicitly rejected fallback rather than an accepted policy.
5. Run `make run-all-autorun HYDRA_CONFIG=configs/policy_press_quad.yaml`; the
   composed `policy_press` routine reads the same symlink, while action kind,
   dimensions, scales, and cadence come from `policy_meta.json`.

From another terminal, follow the newest wrapper log with:

```console
tail -F "$(ls -1t .build/overnight/*.console.log | head -n 1)"
```

## Full press-policy mission gate

`configs/policy_press_quad.yaml` is the deployment sanity routine in the same
top-level Hydra format as `configs/takeoff_quad_quad.yaml`. It runs one X500 in
the warehouse through arm -> takeoff -> hover -> goto handoff pose -> hover ->
generic `policy` -> return goto -> land. The policy descriptor carries
`artifact: .build/training/press/latest` and `switch_index: 0`; the artifact's
`policy_interface: press-v1` selects the runtime adapter. The deployment config
does not repeat action scales or policy cadence: OFFBOARD conversion, press
observation action shape, adapter scheduling, and the production switch latch
use artifact metadata. A fleet with press artifacts at conflicting cadences is
rejected before spawn; when no press policy is present, the switch latch uses
the configured training cadence.

The private Isaac Sim 5.1 adapter authors the easy-task switch/pusher settings
before physics starts, creates a pad `ContactSensor`, and reads canonical
switch pose, vehicle state, pusher compression/rate, contact, and lamp state.
The non-policy contact-chain gate also requires the fixed mount/boom, visible
plunger, a +X-facing collidable pad, full initial joint deployment, enabled arm
and swept-propeller collision, wall collision, and rotor-sweep clearance before
attempting a press. The Lab task owns a cloned static wall per environment; the
live warehouse adapter adds an invisible collision proxy behind the selected
switch so deployment cannot pass through decorative warehouse geometry. It
packs the exact 34-value `PressObservation`. The pure frozen deployment
transition waits for the handoff gate, supplies the preceding normalized action
to the next observation, rejects an initially-on lamp or repeated lamp toggle,
records first-contact stability plus diagnostic approach/compression/action
peaks, and completes only after one toggle plus the configured stable retreat.
Its normalized output is mapped through the same
`press_action_to_offboard` seam as training.

After an artifact passes `make eval-press`, run:

```console
make test-integration-press-routine
```

This artifact-dependent PX4/Isaac gate is intentionally outside
`test-integration-all`. It requires exactly one off-to-on event, stable first
contact, one toggle, stable retreat, return waypoint, landing, and disarm. The
current legacy artifact can be run only as a diagnostic at its recorded 0.5 m/s
action scale. It was trained against the obsolete pad-only geometry, so it is
not a parity-valid candidate after the complete pusher tool was introduced. Its
measured 0% toggles and 0% retreat already blocked this gate; the next accepted
export must be trained on the complete tool and carry `press-v1` explicitly. No
new training was started while adding the deployment path or correcting the
pusher geometry.
