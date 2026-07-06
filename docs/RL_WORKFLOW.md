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
`ISAACLAB_REF` build ARG / Make variable (default `v2.3.2`, the Isaac Lab
release built on Isaac Sim 5.1) via `./isaaclab.sh --install` (all RL
frameworks). Only the RL workflow needs it -- the
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
root is the public contract and exposes guarded task registration, the hover
task id/env/config, and the named batched rotor-omega diagnostic seam without
eager-importing Isaac Lab task modules during host-side package imports.

Isaac Lab task IDs use the Gymnasium `Isaac-...-Direct-v0` naming convention
for direct-workflow environments. The first task is
`Isaac-Drone-Hover-Direct-v0`; it flies N parallel drones through the
pluggable in-Lab actuation seam (`robotics_experiments.rl`), which maps a
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
`robotics_experiments.rl` (implemented in `rl/press/actuation.py`; numpy, the
test oracle) and the Lab package exposes the env diagnostic entry point as
`robotics_experiments.workflows.isaac_lab.compute_lab_rotor_omega`; both mirror
the PX4 bridge's `PegasusQuadraticThrustModel` allocation so an action
reproduces the commanded wrench.

Shared RL training scale and the RL action level live in the Hydra `rl` group
with matching bare annotations in `src/robotics_experiments/config/types.py`.
The physics decimation is derived as
`round(sim.physics_hz / rl.policy_rate_hz)`, not authored, and the hover task's
reward/termination/reset values are the `rl.press.hover_task` block (task
definition = one YAML block; no task-tuning literal in the env module). The RL
action also maps to a PX4 OFFBOARD setpoint at deploy time
(`press_action_to_offboard` + the commanding-layer velocity / attitude-rate
setpoint seams).

## Task->policy pipeline (F2b)

The pipeline is `task YAML -> make train-rl -> policy artifact ->
{kind: policy} routine step -> FleetOrchestrator` (the orchestrator itself is
unchanged apart from injecting the artifact-loader factory into
`DriverParams`). Pieces:

- **Observation layouts** (exported from `robotics_experiments.rl`, pure,
  versioned): the
  13-dim hover layout (`HOVER_OBSERVATION_SCHEMA_VERSION`,
  `pack_hover_observation` / `hover_observation_from_state`; body-FLU angular
  velocity) and the press-task `PressObservation`/`to_vector`
  (`OBSERVATION_SCHEMA_VERSION`) with the contact frame
  (`world_switch_to_contact_frame`; switch local `+X` = outward plate
  normal). PARITY INVARIANT: the same packing feeds Lab tensors in training
  and `VehicleState` at deploy; a field the deploy side cannot assemble is
  forbidden in a layout. The pose-provider seam is
  `AbstractPoseProvider` / `PoseSample`; ground truth is the lazy Isaac-only
  `GroundTruthPoseProvider`, and the nav->contact handoff is the pure
  `handoff_ready` gate (thresholds in `rl.press.handoff`). See ADR 0014.
- **Training** (`rl/training/`, lazy; `rsl_rl.py` is the only rsl_rl
  importer): import `build_runner`/`train`/`export_policy`/`TrainingResult`
  from the `robotics_experiments.rl` package root. The contract runs
  over `RlTrainingConfig` (the `rl.training` block, incl. explicit `task_id`
  and `policy_name`).
  `make train-rl` (thin target in `make/training.mk`, runs
  `isaac/runtime/standalone/train_rl.py` on `ISAAC_LAB_IMAGE`, never the base
  image) writes the artifact under `.build/training/<policy-name>/<run-id>/`
  and updates the named `.build/training/<policy-name>/latest` symlink;
  `TRAIN_ARGS='--max-iterations 5'` is the smoke run. rsl_rl comes from the Lab
  image (`isaaclab.sh --install`); the host `training` extra (`rsl-rl-lib`)
  exists for tooling only.
- **Artifact** = TorchScript `policy.pt` + `policy_meta.json` (obs schema
  version/dim, action kind/dim, action scales). `build_policy_step_fn` raises
  `ValueError` on ANY metadata/config mismatch at load (a bare checkpoint is
  not a valid artifact) and composes artifact loading + hover-observation
  packing + `press_action_to_offboard` into the `PolicyStepFn` the
  `{kind: policy}` routine step streams.
- **Gate**: `make test-integration-press-policy` flies
  `configs/policy_hover_quad.yaml` (routine `policy_hover`: arm ->
  takeoff -> policy hover hold -> land) strict over PX4 SITL on
  `ISAAC_LAB_IMAGE`. It needs a prior `make train-rl` (the routine
  references `.build/training/hover/latest`), so it is deliberately NOT part of
  `test-integration-all`. The hover+goto gate additionally runs the F2
  side-channel validation (ground-truth provider + `PressObservation` +
  handoff flip) without touching the env's training observation.
