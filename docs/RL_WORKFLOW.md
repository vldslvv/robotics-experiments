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
retargeting stacks are excluded. Only the RL workflow needs it -- the
`test-integration-press-hover-goto` and `test-integration-press-policy` gates
and `make train-rl` run on `ISAAC_LAB_IMAGE`; every other integration target
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
Approach/retreat progress is signed, and crash/altitude/envelope termination
has an explicit terminal failure cost, preventing timeout-hover and early-exit
reward shortcuts.

## Task->policy pipeline (F2b)

The pipeline is `task YAML -> make train-rl -> policy artifact ->
{kind: policy} routine step -> FleetState lifecycle`. The runtime injects the
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
- **Training** (`rl/training/`; `rsl_rl.py` is the only rsl_rl importer): import
  `build_runner`/`train`/`export_policy`/`TrainingResult` from
  `robotics_experiments.rl.training.rsl_rl`. The contract runs
  over `RlTrainingConfig` (the `rl.training` block, incl. explicit `task_id`
  and `policy_name`).
  `make train-rl` (thin target in `make/training.mk`, runs
  `isaac/runtime/standalone/train_rl.py` on `ISAAC_LAB_IMAGE`, never the base
  image) writes the artifact under `.build/training/<policy-name>/<run-id>/`
  and updates the named `.build/training/<policy-name>/latest` symlink;
  `TRAIN_ARGS='--max-iterations 5 --num-envs 2'` is a small smoke run. Each
  task config owns its observation schema version and dimension, so exported
  metadata contains no task-ID conditional. Before PPO begins, the run writes
  the complete policy contract and Git provenance to its run directory.
  Normal export rereads that stored metadata, rather than regenerating it from
  the current config. rsl_rl comes from the
  pinned Lab image requirements; the host `training` extra (`rsl-rl-lib`)
  exists for tooling only.
  If training finishes but artifact export is interrupted, recover the
  checkpoint without another optimization run:

  ```console
  make train-rl HYDRA_CONFIG=configs/train_press_quad.yaml \
    TRAIN_ARGS='--num-envs 2 --export-checkpoint /isaac-sim/proj/.build/training/press/<run-id>/logs/model_<iteration>.pt'
  ```

  Export recovery loads policy weights without optimizer state, reuses the
  contract and provenance captured in that run before training, writes the
  artifact back into the checkpoint's existing run directory, and updates the
  policy namespace's `latest` symlink. Recovery rejects a run missing its
  stored metadata. The checkpoint must be below that namespace's
  `.build/training/<policy-name>/<run-id>/logs/` directory.
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
- **Gate**: `make test-integration-press-policy` flies
  `configs/policy_hover_quad.yaml` (routine `policy_hover`: arm ->
  takeoff -> policy hover hold -> land) strict over PX4 SITL on
  `ISAAC_LAB_IMAGE`. It needs a prior `make train-rl` (the routine
  references `.build/training/hover/latest`), so it is deliberately NOT part of
  `test-integration-all`. The hover+goto gate additionally runs the F2
  side-channel validation (ground-truth provider + `PressObservation` +
  handoff flip) without touching the env's training observation.

## Ground-truth press training and evaluation (F3)

`configs/train_press_quad.yaml` selects `Isaac-Switch-Press-Direct-v0`, the
`press` artifact namespace, ten-second episodes, and press PPO settings. Run:

```console
make train-rl HYDRA_CONFIG=configs/train_press_quad.yaml
make eval-press
```

`make eval-press` loads `.build/training/press/latest` and evaluates 100
deterministic episodes for each of three non-training seeds. It reports reward
term means, latch-confirmed press success, first-contact stability among
contacted episodes, retreat among toggled episodes, median time-to-toggle, and
peak compression overshoot. The Lab environment uses the artifact's action
kind, dimension, physical scales, and cadence, while controller gains and
physical task settings continue to come from the training YAML. The command
fails below the configured 80% press,
85% stable-contact, or 95% retreat thresholds. Artifacts remain ignored under
`.build/training`; no policy binary is committed.

For an unattended train-then-evaluate run that powers off after either success
or failure, use the reusable wrapper:

```console
make/scripts/run_training_evaluation_then_shutdown.sh \
  --hydra-config configs/train_press_quad.yaml \
  --train-target train-rl \
  --eval-target eval-press \
  --train-args '--num-envs 49152 --max-iterations 300' \
  --label press-overnight
```

The wrapper validates sudo access before starting, refreshes that authorization
while the workload runs, skips evaluation when training fails, and requests
power-off after a successful evaluation, a failed stage, or a handled
interrupt. Its target names, Hydra config, stage arguments, label, and log
directory are command-line options so the same script can run later RL
workflows. The combined console log and final per-stage exit status default to
`.build/overnight/<label>.<UTC-time>.<pid>.console.log` and `.status`. The
ordinary integration and policy artifacts remain in `.build/docker/integration`
and `.build/training`.

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
