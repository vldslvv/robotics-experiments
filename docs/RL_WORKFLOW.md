# Reinforcement-learning workflow

This document defines the Isaac Lab tasks, scratch PPO workflow, policy
artifacts, in-loop checkpoint evaluation/selection, and deployment contract. Routine execution is
defined in `docs/COMMAND_ROUTINE_CONTRACT.md`; controller fitting is defined in
`docs/SIM_TO_SIM_ALIGNMENT.md`. Chronological press-policy training decisions
and measurements are recorded in
`docs/phase_f/PRESS_POLICY_TRAINING_DECISION_LOG.md`.

## Dependency boundary

`robotics_experiments.controller` owns controller settings, strict controller
artifacts, normalized action-to-rotor behavior, and the NumPy oracle.
`robotics_experiments.rl` owns task-neutral observations, action interpretation,
policy artifacts, and normalized-action-to-OFFBOARD behavior. Press-specific
observation, handoff, reward, task, and progress contracts live under
`robotics_experiments.rl.press`. Isaac-only vector environments live under the
lazy `robotics_experiments.workflows.isaac_lab` package.

RL never imports alignment. Before Isaac Lab is constructed, training loads a
quality-passing schema-1 controller artifact and validates its vehicle, airframe,
cadence, certified control mode, and four-dimensional action contract. Policy metadata embeds the complete
controller record and its digest. Evaluation and deployment reuse that embedded
record; they never resolve a newer alignment implicitly.

The registered tasks are `Isaac-Drone-Hover-Direct-v1` and
`Isaac-Switch-Press-Direct-v1`. Both use the normalized controller seam. The
Torch implementation is required to match the NumPy controller sequentially.
The pinned Isaac Lab `Thruster` realizes per-rotor response while the project
owns allocation, yaw reaction, drag, the pusher-equipped articulation, and the
sole body-wrench sink. Lab rigid-body gyroscopic forces are Hydra-owned and
disabled to match the standalone PX4 plant's generated-USD/default PhysX
contract; yaw reaction remains explicitly modeled by the rotor wrench. The
reduced Lab rotor wrench is translated from the body's local-frame origin to
PhysX's center-of-mass force path before submission.

## Action and cadence

`rl.action` contains only normalized action kind, dimension, and scales:

- `velocity`: normalized body-FLU `[vx, vy, vz, yaw_rate]`;
- `ctbr`: hover-centered normalized collective plus body rates.

Controller gains, hover thrust, filters, limits, delay, allocation, and actuator
response come only from the embedded controller settings. At 250 Hz physics and
20 Hz policy rate, the exact-average cadence alternates 13 and 12 physics ticks
per policy step. The retained velocity action passes through its configured
transport delay and is converted from body-FLU to a world-frame setpoint on the
independent 50 Hz OFFBOARD stream, using the attitude at each transmission. The
world command is held across the faster controller and physics substeps,
matching deployment through PX4's body-NED MAVLink receiver.

## Press approach task

The `press-approach` policy consumes the schema-1 34-value observation and
emits four normalized velocity actions. The observation contains relative switch
pose, vehicle motion and attitude, pusher state, contact and lamp state, previous
action, handoff mode, estimator health, and availability bits. Training and
runtime use the same packing function. Its scratch experiment maps a unit
translation action to 0.5 m/s. Normalized exploration starts at 0.12 standard
deviation, retaining a 0.06 m/s physical standard deviation while resolving the
task's 0.10 m/s-or-less velocity references with useful action-head range.

The current stage learns a contact-free approach and stable hover within
0.40-0.42 m of the switch, not switch activation. Success requires five
consecutive 20 Hz samples satisfying:

- combined lateral/vertical error <= 0.03 m;
- linear speed <= 0.10 m/s;
- angular speed <= 0.40 rad/s;
- body-Z alignment >= 0.965926;
- no pusher or body collision.

The reset distribution is the deployable handoff envelope: range 0.45-0.50 m,
combined lateral/vertical error <= 0.08 m, low translational/angular speed,
upright attitude, fresh valid pose, no contact, and healthy estimation. The
task also exposes a configured difficult-region mixture for the later phase of
the same scratch run. Evaluation always uses the complete held-out handoff
distribution.

Collision is privileged training/evaluation state and is not added to the actor
observation. Success, collision, other flight failure, and timeout remain
mutually exclusive terminal classifications.

## Scratch training

Every `make train-rl` invocation starts a new runner, actor, critic, optimizer,
and environment. There is no checkpoint initialization, resume mode, actor-only
copy, or post-run fine-tuning workflow.

Hydra `rl.training.phases` defines an ordered schedule inside that one optimizer
lineage. Each phase declares maximum iterations, complete held-out evaluation
cadence, learning rate, schedule, entropy coefficient, and difficult-reset
fraction. A phase boundary updates those values in place; it does not reload a
checkpoint or recreate the optimizer.

The evaluator and phase driver have separate contracts. Evaluation receives an
exact exported actor and returns raw named metrics plus
`held_out_task_cost`. The plateau detector reads only the configured metric,
direction, minimum delta, and patience. It never reads the PPO reward or any
acceptance gate. The current uniform rule minimizes raw task cost, requires a
`0.003` reduction from the phase reference, and declares a plateau after 30
consecutive stale evaluations. This intentionally relaxed initial value can be
tightened from recorded curves without changing evaluator code.

The press configuration currently runs:

1. 241 iterations at `1e-4`, fixed schedule, `1e-4` entropy, full handoff reset;
2. 11 iterations at `3e-5`, fixed schedule, zero entropy, 20% difficult resets;
3. 31 iterations at `1e-5`, fixed schedule, zero entropy, 40% difficult resets.

The broad phase evaluates every ten updates and both refinement phases evaluate
every five. Model zero and every phase boundary are also evaluated. Training
advances after a phase reaches its maximum or the same plateau rule fires.

Run training with:

```bash
make train-rl HYDRA_CONFIG=configs/train_press_approach_quad.yaml
```

Use the run-scoped TensorBoard log directory under
`.build/training/press-approach/<run-id>/logs`. The standard monitor is:

```bash
make tensorboard-rl RL_TENSORBOARD_LOGDIR=.build/training/press-approach/<run-id>/logs
```

and is served at `http://127.0.0.1:6006` by default.

The press-approach reward has eleven named signed terms, including the
current-range barrier, three-axis state-velocity tracking, and physical
wall-normal braking-command supervision. Both velocity terms use the same
task-owned wall-normal target; the command term receives the action only after
the Lab actuation seam maps it to a physical contact-frame velocity command.
It is active only when the predictive target requests retreat and penalizes a
command above that retreat target, so ordinary approach commands remain
unconstrained by direct imitation.
The task projects positive
wall-normal speed through the configured controller braking horizon before
computing that target. A projected approach past the neutral range therefore
requests retreat directly, while retreating motion is never projected farther
inward. The horizon and reward weights remain task-owned Hydra data.

## Evaluation, artifact, and acceptance

`make train-rl` trains and evaluates in one process-level workflow. On the first
cache miss, a persistent worker starts exactly 100 evaluation environments and
evaluates three configured seeds; the same worker handles later misses after
each training chunk. A fully cached run never starts the second simulator. The
trainer retains one runner and optimizer throughout. It exports the
deterministic actor first, so the evaluation digest identifies the exact
`policy.pt` bytes rather than an in-memory approximation.

The raw held-out task cost is the untransformed weighted sum of collision,
flight failure, timeout, non-success, terminal range, trajectory alignment,
velocity tracking, upright error, action change, and completion time costs.
Lower is better. This cost is independent of the per-step PPO reward. After all
phases end, the lowest-cost evaluated checkpoint is selected, with the earlier
checkpoint winning an exact tie. Only then are the unchanged press acceptance
gates applied to its already-recorded metrics; no duplicate automatic
evaluation stage follows training.

Evaluations are cached by executable TorchScript content digest, comparison
fingerprint, controller digest, and evaluator revision. The content digest
excludes only PyTorch's generated archive serialization ID; the raw SHA-256 of
the exact exported `policy.pt` remains in every evaluation and published
artifact record. `training_report.json` is updated after every evaluation and
records phase progress, raw metrics, cache source, plateau state, phase
termination, final selection, and final acceptance.

A policy artifact is exactly `policy.pt` plus `policy_meta.json`. Metadata uses
the strict schema-1 `robotics-experiments-policy` contract and records policy
interface, observation/action contracts, cadence, controller artifact and
digest, comparison fingerprint, source commit, dirty state, and training phase
provenance. Older policy schemas and policies without complete controller
provenance are unsupported.

`latest` advances after each completed run. `best` advances only for a strictly
lower raw task cost within the same comparison fingerprint. Optional independent
diagnostic reproduction remains available with:

```bash
make eval-press HYDRA_CONFIG=configs/train_press_approach_quad.yaml
```

Final integration uses:

```bash
make test-integration-press-policy-transfer
make test-integration-press-approach-routine
```

The transfer gate is downstream evidence. It must never mutate alignment,
controller settings, policy weights, or the held-out evaluator.

## Deployment

Runtime policy composition is private to `runtime/_policy.py`. A `{kind:
policy}` routine step resolves the strict policy artifact, builds the matching
pose provider and adapter, waits for the configured handoff predicate, and then
runs the actor at its embedded cadence. The controller artifact embedded during
training supplies Lab actuation behavior; PX4 deployment receives the actor's
normalized velocity intent through the OFFBOARD adapter.

For the approach policy, correct behavior is a controlled, contact-free move to
the 0.40-0.42 m range followed by a stable hover. It should not touch the switch,
climb away, or perform a learned retreat. The routine's later `goto` step owns
any return motion.
