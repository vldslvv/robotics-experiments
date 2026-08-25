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

RL never imports alignment. It consumes the accepted schema-1 controller
artifact produced by alignment and the independently published matching
schema-1 flight parameters. It does not reopen raw alignment traces.

## Policy workflow

`robotics_experiments.rl.workflow` defines the procedure used by every policy
campaign:

```text
inputs = prepare_policy_inputs(...)
criteria = create_press_acceptance_criteria(...)
candidates = train_and_validate(inputs=inputs, ...)
selected = select_candidate(candidates=candidates, ...)
evaluation = run_final_test(inputs=inputs, policy=selected, ...)
acceptance = evaluate_press_acceptance(criteria=criteria, evaluation=evaluation)
if acceptance.passed:
    publish_policy(policy=selected, evaluation=evaluation, ...)
```

Preparation validates each producer artifact once, confirms the controller was
qualified with that exact flight-parameter run and fingerprint, and pins both
resolved paths and digests. The coordinator then creates the press acceptance
criteria once from `rl.press.acceptance` and the measured hover limits. The
campaign report records those criteria before training starts.

Workers receive the pinned paths and digests and confirm the files have not
changed before constructing Isaac Lab. Training reads the producer files in
place and never copies or mutates them. Invalid input creates no run directory
or attempt process. A worker that exits without its report is recorded as a
failed attempt.

The registered tasks are `Isaac-Drone-Hover-Direct-v1` and
`Isaac-Switch-Press-Direct-v1`. Both use the normalized controller interface. The
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

## Press policy curriculum

The press policy consumes the schema-1 34-value observation and
emits four normalized velocity actions. The observation contains relative switch
pose, vehicle motion and attitude, pusher state, contact and lamp state, previous
action, handoff mode, estimator health, and availability bits. Training and
runtime use the same packing function. Its scratch experiment maps a unit
translation action to 0.5 m/s. Normalized scratch exploration starts at 0.03
standard deviation, or 0.015 m/s for a
translation component at the task scale.

`rl.press.curriculum.stages` is an ordered list. A training config selects one
entry by name; there are no parent or child links. The first entry starts from
scratch. A later entry restores the complete accepted checkpoint published as
`best` by the entry immediately before it. Each entry owns its reset
mixture, hold schedule, episode duration, PPO phases, and plateau rule. The
current order is `hover`, then `approach`.

The hover stage starts around the common 0.48 m press target using only poses
sampled from the pinned flight-parameter artifact. Its training phases require
5, 15, then 30 continuous seconds inside that measured envelope. The approach
stage adds measured navigation-handoff resets while retaining target-local
envelope resets. Both remain parts of the press task; neither activates the
switch yet.

Success uses only the measured position and pose bounds: asymmetric ENU axis
bounds, horizontal and 3D position bounds, roll/pitch/yaw bounds, tilt, and full
orientation error. Speed is observable and may contribute diagnostic cost, but
it is not part of the hover acceptance predicate. Contact or another flight
failure still terminates the episode. Training resolves the envelope bundle
once and policy metadata pins its exact fingerprint and digest, so a later
`latest` measurement cannot change an active campaign.

The accepted 2026-08-12 schema-2 production campaign recorded ten successful
arrivals: switch range 0.464594-0.498991 m, maximum combined lateral/vertical
error 0.033746 m, and maximum full relative-attitude error 0.016339 rad
(0.93616 degree). With the explicit 0.005 m position and 0.25 degree rotation
margins, that evidence requires 0.44-0.52 m range, a 0.08 m alignment disk, and
a 0.020702 rad (1.18616 degree) rotation ball. The current handoff gate extends
the range maximum to 0.67 m for the nominal 0.60 m waypoint route; its accepted
measurement must replace the older route evidence before approach training.
The accepted prior-route sample summary is
`benchmarks/press_navigation_handoff/waypoint_settle_20260812.json`.

Refresh the navigation evidence after changes to the GPS/IMU sensor models,
PX4 estimator/navigation behavior, vehicle controller, waypoint routine, or
warehouse geometry:

```bash
make measure-press-navigation-handoff
```

The target runs at least ten independent executions of the exact production
config through the final waypoint and two-second settle, then records the
switch-relative entry-pose extrema and the 0.005 m position / 0.25 degree
rotation margins under
`.build/benchmarks/press-navigation-handoff-campaigns/`. Accepting a new
measurement means committing its summary, updating `rl.press.navigation_noise`,
and expanding (never narrowing past that evidence) the shared handoff bounds.
It does not qualify an existing actor; retrain to qualify against the expanded
reset distribution.

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
the first completed attempt with a validation-passing checkpoint. Apart from
the preceding ordered press stage, external checkpoint initialization, process
resume, actor-only copy, and post-run fine-tuning remain unsupported.

The selected `rl.press.curriculum.stages[].phases` entry defines the ordered
schedule within each press attempt; task-neutral training continues to use
`rl.training.phases`. Each
phase declares maximum iterations, validation cadence, learning rate, schedule,
entropy coefficient, and the training-only recovery boundary. The first phase
starts from the stage-initial learner. Before every later phase, the campaign restores the
attempt-wide best checkpoint's actor, critic, optimizer, and learner iteration,
then applies the new phase settings and resets every training environment. This
is a validation-selected fine-tuning branch with reset environment state, not an
exact process resume. An earlier checkpoint remains selectable if refinement
regresses.

The evaluator and phase driver have separate contracts. Validation receives an
exact exported actor and returns raw named metrics plus `held_out_task_cost`.
The plateau detector reads only the configured metric, direction, minimum delta,
and patience. It never reads the PPO reward or any acceptance check. The active
curriculum stage owns the rule. Hover maximizes mean completed hold fraction;
approach minimizes held-out mean normalized pose error. Both require a `0.003`
material change from the phase reference and declare a plateau after 10
consecutive stale evaluations. Raw aggregate task cost remains diagnostic and
cannot select a timeout-heavy checkpoint over demonstrated task progress.

Hover training uses a 40-second episode horizon; approach uses 45 seconds. Both
press stages run:

1. 50 iterations at `1e-4`, fixed schedule and zero entropy, with recovery
   at 0.38 m for hover and 0.40 m for approach;
2. 30 iterations at `3e-5`, fixed schedule and zero entropy, with recovery
   at 0.39 m for hover and 0.40 m for approach;
3. 100 iterations at `1e-5`, fixed schedule and zero entropy, with the strict
   0.40 m undershoot boundary.

The broad phase evaluates every ten updates and both refinement phases evaluate
every five. Model zero and every cadence-aligned phase boundary are evaluated;
there is no extra boundary update. Hover expands the continuous measured-envelope
hold from 5 to 15 to 30 seconds. Approach uses the complete 30-second hold in
every phase. The campaign totals 180 attempted PPO updates. Training advances
after a phase reaches its maximum or the same plateau rule fires.

Run training with:

```bash
make train-rl HYDRA_CONFIG=configs/train_press_hover_quad.yaml
```

Environment count, attempt seeds, phase lengths, evaluation cadence, PPO
settings, artifact selectors, and plateau rules are Hydra-owned. The launcher
does not provide command-line overrides for them. `--procedural-smoke` remains
an operational workflow check rather than a training parameter.

The GPU procedural check keeps every phase/restore boundary, performs one update
per phase on every configured attempt seed, and deliberately stops before
reserved test seeds are used:

```bash
make train-rl HYDRA_CONFIG=configs/train_press_hover_quad.yaml \
  TRAIN_ARGS=--procedural-smoke
```

The procedural smoke executes one update in every phase of every configured
attempt, including full attempt teardown/recreation, and stops without using
the reserved test seeds.

Use the campaign-scoped TensorBoard log directory under
`.build/training/press-hover/<run-id>/logs`. Each attempt is exactly one
TensorBoard run containing its `train/`, `val/`, and `system/` metrics; the
selected attempt also receives `test/`. The standard monitor is:

```bash
make tensorboard-rl RL_TENSORBOARD_LOGDIR=.build/training/press-hover/<run-id>/logs
```

and is served at `http://127.0.0.1:6006` by default. The training wrapper prints
a run-scoped URL with `smoothing=0` and a filter for the primary learning and
validation charts: training return, signed per-sample reward, all-checks pass,
approach success, dwell progress, other flight failure, collision, minimum
range, finite traces, and held-out task cost. Set
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
namespaces. Validation and test share one ordered registry: acceptance-check status,
success/failure/collision rates, minimum range, trace finiteness, maximum dwell
progress, task cost,
secondary terminal rates, every strict-predicate pass rate, mean and maximum
dwell, normalized pose error, range/completion measurements, and every mean
cost component. Undefined values such as successful completion time when there are
no successes remain `null` in the report and are not emitted as a scalar.
Acceptance is report metadata, not a TensorBoard namespace. Total attempted PPO
updates are the X coordinate and never rewind when a refinement phase restores
an earlier learner checkpoint; `train/schedule/checkpoint_update` records that
learner position separately.

`system/` is diagnostic-only and cannot influence PPO, plateau progress,
selection, retry choice, validation results, or acceptance. At the
Hydra-configured one-second cadence, every training chunk and fresh validation
or test evaluation samples NVIDIA utilization, memory, power, temperature, and
the owning process's resident memory and CPU use. NVIDIA percentages are stored
as fractional rates. Per-update rollout/PPO timing and Torch allocator peaks are
also emitted during training. Raw samples live in each attempt's
`telemetry.json`; interval summaries live in the strict campaign report. A
startup probe fails before rollout when telemetry is unavailable, and a
sampler failure terminates at the interval boundary with evidence preserved.
The configured `0.95` GPU-utilization target is diagnostic: a miss is recorded
as the measured bottleneck and never changes policy qualification.

The press PPO reward has five signed contributions. Hover `position_progress`
rewards each reduction in measured pose utilization and the bounded
stabilization cost is `-utilization / (1 + utilization)`, alongside the strict
hold and terminal outcome. Utilization is the larger of exact position and
attitude envelope utilization from the pinned flight parameters. Hover sets
only `predictive_barrier` to zero. In the approach stage, `position_progress`
rewards reduction in target error until the measured position envelope is
reached, `stabilization` is `1 / (1 + utilization)` inside that position
envelope, and the predictive range barrier remains active.
`strict_hold` is one on every complete envelope-passing sample.
`terminal_outcome` is +150 on the selected stage's hold success and -300 on every unsuccessful
termination, including timeout. The first monitored scratch campaign showed
range undershoot as the dominant early flight failure, so the one added factor,
`predictive_barrier`, penalizes squared projected undershoot of the strict range
minimum. It projects positive wall-normal closing speed over the existing
velocity-reference horizon and normalizes the deficit by the strict range
width. Its weight is 10; weight 20 delayed the first dwell without improving
its maximum, while a weight-50 isolation caused immediate attitude instability.
Contact remains a binding predicate without a separate PPO shaping term.

Training metrics use semantic groups without numeric prefixes:
`train/reward/`, `train/outcome/`, `train/goal/`, `train/state/`,
`train/optimization/`, `train/performance/`, and `train/schedule/`.
`train/schedule/recovery_boundary_m` records the stage's configured safety
boundary. Envelope pass rate, utilization, and dwell fractions refer to the
pinned measured envelope. Validation and test use matching `outcome/`, `goal/`, `state/`,
`trace/`, and `cost/` subgroups. Hardware telemetry is under `system/`.

## Validation, final test, artifact, and acceptance

Every checkpoint in every phase and attempt uses the same validation workload:
100 episodes for each of seeds 101, 202, and 303. A persistent worker evaluates
each exported actor on that workload, so every record identifies exact
`policy.pt` bytes. Validation is not cached.

The raw held-out task cost is the untransformed weighted sum of collision,
flight failure, timeout, non-success, missing maximum dwell progress, terminal
range, trajectory alignment,
relative-attitude error, velocity tracking, angular velocity, upright error,
action change, and completion time costs. Relative-attitude and angular-velocity
costs each carry weight 0.10. Missing dwell progress carries weight 2.0, so
checkpoint progress distinguishes partial settling from a zero-dwell timeout.
Lower is better. This cost is independent of the per-step PPO reward. After all
phases end, validation-passing checkpoints rank ahead of failures. Within each
group the lower task cost wins, with the earlier checkpoint winning an exact
tie. A validation pass only makes the candidate eligible for the final test.

The raw SHA-256 of the exact exported `policy.pt` remains in every evaluation
and published artifact record. The campaign parent is the only writer of
`training_report.json`. Before each worker launch it records the active attempt
identity and its report path. The worker atomically writes
`attempt_report.json` below its attempt directory; the parent validates and
incorporates that report after the worker exits. An interruption preserves
completed attempts, the active pointer, and the partial attempt report, marks
the campaign interrupted, exits nonzero, and publishes nothing. Process resume
remains unsupported.

`training_report.json` schema 20 records the prepared acceptance criteria and every attempt, phase parent, raw
validation metrics, exact TensorBoard step, telemetry summaries,
plateau state, phase termination, selection, final test, and individual
acceptance. The metric registry and comparison fingerprint carry
metric-registry revision 16 and comparison revision 22; older report shapes are
unsupported. Flight failure and timeout each cost 3.0 in checkpoint ranking, so
the selector cannot improve merely by converting failed attempts into waits.

After selection, the exported actor is immutable. A new uncached worker runs the
disjoint final workload: 100 episodes for each of seeds 404, 505, and 606. The
resulting raw `PolicyEvaluation` is the sole measured input to
`evaluate_press_acceptance`. Every check is independently binding: finite traces,
at least 0.95 success, zero collision, at least 0.40 m minimum closest
range, exactly zero other flight failures, and the flight-parameters maximum,
RMSE, p95, and p99 limit for every hover error signal. Hover statistics use the
final 24 seconds of the successful 30-second stable hold; the first six seconds
are the artifact-configured policy warmup. The evaluator records every raw
statistic in the `PolicyEvaluation` and `training_report.json`; acceptance adds
no consumer-owned tolerance. Test data cannot affect training,
phase advancement, checkpoint selection, or retries. A test failure ends the
command immediately and preserves evidence without publishing.

A policy artifact is exactly `policy.pt` plus `policy_meta.json`. Metadata uses
the strict schema-1 `robotics-experiments-policy` contract and records policy
interface, observation/action contracts, cadence, controller artifact and
digest, flight-parameters artifact and digest, comparison fingerprint, source
commit, dirty state, and training phase provenance. Older policy schemas and
policies without complete controller and flight-parameters provenance are
unsupported.

On acceptance, publication copies the already-tested actor bytes and verifies
their digest. Metadata and the report record both workloads, selected and tested
digests, raw metrics, required values, individual results, overall status, and
failure reasons. An accepted campaign automatically advances `latest`; `best`
advances only for a strictly lower validation cost within the same comparison
fingerprint.
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

The transfer check is downstream evidence. It must never mutate alignment,
controller settings, policy weights, or the held-out evaluator.
The production-routine target is an invocation smoke test: it ends after the
final waypoint and settle reach the policy step, the unchanged handoff check
latches, one valid real-artifact actor result is produced, and that result's
first setpoint is transmitted. It intentionally does not claim task success;
held-out evaluation and the transfer check remain the policy-quality and
physical-behavior evidence.

## Deployment

`latest` is the active operator-selected deployment label, not an acceptance
claim. Automatic campaign publication assigns it only to an accepted, exactly
tested actor. An operator may explicitly assign it to another strict policy
artifact, including a validation-selected diagnostic candidate. That assignment
must preserve the campaign report and failed or unavailable validation/test
evidence, must not advance `best`, and does not make downstream execution an
acceptance or qualification result. Runtime startup validates the stored policy
and controller contracts but does not reinterpret an operator deployment as a
passed acceptance result.

Runtime policy composition is private to `runtime/_policy.py`. A `{kind:
policy}` routine step resolves the strict policy artifact, builds the matching
pose provider and adapter, waits for the configured handoff predicate, and then
runs the actor at its embedded cadence. The controller artifact embedded during
training supplies Lab actuation behavior; PX4 deployment receives the actor's
normalized velocity intent through the OFFBOARD adapter.

For a press policy, correct behavior is a controlled, contact-free move to the
task-owned range center followed by a stable hover inside the complete envelope
embedded from the pinned flight-parameters artifact. Deploy-time completion
uses that exact envelope for the configured 600 continuous samples, matching
training and evaluation. It should not touch the switch, climb away, or perform
a learned retreat. The routine's later `goto` step owns any return motion.
