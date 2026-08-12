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

## Training campaign

Every `make train-rl` invocation owns one campaign ID and tries the configured
`rl.training.attempt_seeds` in order. Each attempt constructs a fresh training
Isaac process, environment, actor, critic, runner, and optimizer. The campaign
coordinator never owns a training environment: a completed worker exits before
the next seed starts, making simulator and GPU cleanup an operating-system
process boundary. The default seeds are 42, 43, and 44; the campaign stops after
the first completed attempt with a validation-passing checkpoint. External
checkpoint initialization, process resume, actor-only copy, and post-run
fine-tuning remain unsupported.

Hydra `rl.training.phases` defines an ordered schedule within each attempt. Each
phase declares maximum iterations, validation cadence, learning rate, schedule,
entropy coefficient, and difficult-reset fraction. The first phase starts from
the new scratch actor. Before every later phase, the campaign restores the
attempt-wide best checkpoint's actor, critic, optimizer, and learner iteration,
then applies the new phase settings and resets every training environment. This
is a validation-selected fine-tuning branch with reset environment state, not an
exact process resume. An earlier checkpoint remains selectable if refinement
regresses.

The evaluator and phase driver have separate contracts. Validation receives an
exact exported actor and returns raw named metrics plus `held_out_task_cost`.
The plateau detector reads only the configured metric, direction, minimum delta,
and patience. It never reads the PPO reward or any gate. The uniform rule
minimizes raw task cost, requires a `0.003` reduction from the phase reference,
and declares a plateau after 10 consecutive stale evaluations.

The press configuration runs:

1. 180 iterations at `1e-4`, fixed schedule, `1e-4` entropy, full handoff reset;
2. 100 iterations at `3e-5`, fixed schedule, zero entropy, 20% difficult resets;
3. 100 iterations at `1e-5`, fixed schedule, zero entropy, 40% difficult resets.

The broad phase evaluates every ten updates and both refinement phases evaluate
every five. Model zero and every cadence-aligned phase boundary are evaluated;
there is no extra boundary update. The broad phase stops before the observed
late collapse region, while each refinement phase provides twenty evaluation
windows. Training advances after a phase reaches its maximum or the same
plateau rule fires.

Run training with:

```bash
make train-rl HYDRA_CONFIG=configs/train_press_approach_quad.yaml
```

The GPU procedural gate keeps every phase/restore boundary, performs one update
per phase on every configured attempt seed, and deliberately stops before
reserved test seeds are used:

```bash
make train-rl HYDRA_CONFIG=configs/train_press_approach_quad.yaml \
  TRAIN_ARGS=--procedural-smoke
```

The procedural smoke executes one update in every phase of every configured
attempt, including full attempt teardown/recreation, and stops without using
the reserved test seeds.

Use the campaign-scoped TensorBoard log directory under
`.build/training/press-approach/<run-id>/logs`. Each attempt is exactly one
TensorBoard run containing its `train/`, `val/`, and `system/` metrics; the
selected attempt also receives `test/`. The standard monitor is:

```bash
make tensorboard-rl RL_TENSORBOARD_LOGDIR=.build/training/press-approach/<run-id>/logs
```

and is served at `http://127.0.0.1:6006` by default. The training wrapper prints
a run-scoped URL with `smoothing=0` and a filter for the primary validation
charts: all-gates pass, approach success, other flight failure, collision,
minimum range, finite traces, and held-out task cost. Set
`RL_TENSORBOARD_PORT=<port>` when that address is occupied. The Make target
uses the stable loader for the double-precision scalar tensors, passes
`--reload_interval 5`, and retains up to one million scalar samples per tag.
TensorBoard 2.21 also stores one smoothing value in browser-local global
settings, so keep that global setting at zero. The exact selected-step values
are recorded under each attempt's `evaluations` list in `training_report.json`.
The wrapper keeps the live server under supervision while it waits for the
first validation scalars; readiness is not bounded by a fixed wall-clock delay.
All rate metrics end in `_rate` and are fractions in `[0, 1]`.

TensorBoard uses only the lowercase `train/`, `val/`, `test/`, and `system/`
namespaces. Validation and test share one ordered registry: gate status,
success/failure/collision rates, minimum range, trace finiteness, task cost,
secondary terminal rates, range/completion measurements, and mean cost
components. Undefined values such as successful completion time when there are
no successes remain `null` in the report and are not emitted as a scalar.
Acceptance is report metadata, not a TensorBoard namespace. Total attempted PPO
updates are the X coordinate and never rewind when a refinement phase restores
an earlier learner checkpoint; `train/03_checkpoint_iteration` records that
learner position separately.

`system/` is diagnostic-only and cannot influence PPO, plateau progress,
selection, retry choice, validation/test gates, or acceptance. At the
Hydra-configured one-second cadence, every training chunk and fresh validation
or test evaluation samples NVIDIA utilization, memory, power, temperature, and
the owning process's resident memory and CPU use. NVIDIA percentages are stored
as fractional rates. Per-update rollout/PPO timing and Torch allocator peaks are
also emitted during training. Raw samples live in each attempt's
`telemetry.json`; interval summaries live in the strict campaign report. A
preflight probe fails before rollout when telemetry is unavailable, and a
sampler failure terminates at the interval boundary with evidence preserved.
The configured `0.95` GPU-utilization target is diagnostic: a miss is recorded
as the measured bottleneck and never changes policy qualification.

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

## Validation, final test, artifact, and acceptance

Every checkpoint in every phase and attempt uses the same validation workload:
100 episodes for each of seeds 101, 202, and 303. A persistent worker serves
validation cache misses; cache identity includes the `validation` role, exact
seeds, and episode count. The actor is exported before evaluation, so every
record identifies exact `policy.pt` bytes.

The raw held-out task cost is the untransformed weighted sum of collision,
flight failure, timeout, non-success, terminal range, trajectory alignment,
velocity tracking, upright error, action change, and completion time costs.
Lower is better. This cost is independent of the per-step PPO reward. After all
phases end, validation-passing checkpoints rank ahead of failures. Within each
group the lower task cost wins, with the earlier checkpoint winning an exact
tie. A validation pass only makes the candidate eligible for the final test.

Validation evaluations are cached by workload identity, executable TorchScript
content digest, comparison fingerprint, controller digest, and evaluator
revision. The content digest
excludes only PyTorch's generated archive serialization ID; the raw SHA-256 of
the exact exported `policy.pt` remains in every evaluation and published
artifact record. The campaign parent is the only writer of
`training_report.json`. Before each worker launch it records the active attempt
identity and its report path. The worker atomically writes
`attempt_report.json` below its attempt directory; the parent validates and
incorporates that report after the worker exits. An interruption preserves
completed attempts, the active pointer, and the partial attempt report, marks
the campaign interrupted, exits nonzero, and publishes nothing. Process resume
remains unsupported.

`training_report.json` schema 3 records every attempt, phase parent, raw
validation metrics, exact TensorBoard step, telemetry summaries, cache source,
plateau state, phase termination, selection, final test, and gate-by-gate
acceptance. The evaluator, cache, metric registry, and comparison fingerprint
carry matching strict revisions; older metric/report shapes are unsupported.

After selection, the exported actor is immutable. A new uncached worker runs the
disjoint final workload: 100 episodes for each of seeds 404, 505, and 606. The
resulting raw `PolicyEvaluation` is the sole input to
`evaluate_press_acceptance`. Every gate is independently binding: finite traces,
at least 0.95 success, at most 0.01 collision, at least 0.40 m minimum closest
range, and exactly zero other flight failures. Test data cannot affect training,
phase advancement, checkpoint selection, or retries. A test failure ends the
command immediately and preserves evidence without publishing.

A policy artifact is exactly `policy.pt` plus `policy_meta.json`. Metadata uses
the strict schema-1 `robotics-experiments-policy` contract and records policy
interface, observation/action contracts, cadence, controller artifact and
digest, comparison fingerprint, source commit, dirty state, and training phase
provenance. Older policy schemas and policies without complete controller
provenance are unsupported.

On acceptance, publication copies the already-tested actor bytes and verifies
their digest. Metadata and the report record both workloads, selected and tested
digests, raw metrics, gate thresholds, individual results, overall status, and
failure reasons. Only accepted campaigns advance `latest`; `best` advances only
for a strictly lower validation cost within the same comparison fingerprint.
Independent audit remains available with:

```bash
make eval-press HYDRA_CONFIG=configs/train_press_approach_quad.yaml
```

`make eval-press` reloads the published digest and reruns the same 300 test
episodes without participating in selection. Final integration uses:

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
