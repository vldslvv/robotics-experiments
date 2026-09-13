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

Each task's `policy.normalize_observations` Hydra setting controls RSL-RL's
actor and critic empirical observation normalization. The current press-hover
configuration enables it. Statistics update from training observations only
and are part of the complete learner checkpoint restored between phases.
Each phase's `update_observation_normalization` setting controls further updates
to both actor and critic statistics. The press-hover broad phase updates them;
refinement freezes the statistics from its restored validation-selected learner.
The adapter uses the normalizer's sample limit so repeated training-mode entry
cannot restart updates. Later phases apply their declared setting after restoring
the complete learner; actor and critic weights and optimizer state continue training.
Before restoring a checkpoint, the adapter makes normalization buffers created
during inference-mode rollouts writable for checkpoint loading. The deterministic
actor includes the actor normalizer followed by the actor network. Export deep-copies both into the same TorchScript artifact, freezing
the input mapping for validation, final testing, and deployment. Evaluation
does not update these statistics or modify producer artifacts. Observation
schema, action limits, sensor noise, reset sampling, and acceptance criteria
are unchanged.

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

The press policy consumes the schema-2 34-value observation and emits four
normalized velocity actions. Its first three values are world-ENU hover position
error divided by the measured flight-parameter envelope on each axis. The rest
contains relative switch attitude, vehicle motion and attitude, pusher state,
contact and lamp state, previous action, handoff mode, estimator health, and
availability bits. Training and runtime calculate the same target error and use
the same packing function. Its scratch experiment maps a unit
translation action to 0.5 m/s. Normalized scratch exploration starts at 0.10
standard deviation, or 0.05 m/s for a
translation component at the task scale.

`rl.press.curriculum.stages` is an ordered list. A training config selects one
entry by name; there are no parent or child links. The first entry starts from
scratch. A later entry restores the complete accepted checkpoint published as
`best` by the entry immediately before it. Each entry owns its reset
mixture, hold duration, episode duration, PPO phases, and plateau rule. The
current order is `hover`, then `approach`.

Author press episode duration only in the selected
`rl.press.curriculum.stages[].episode_length_s` entry. Both the environment and
comparison record read that value. `rl.episode_length_s`, `rl.training.phases`,
and `rl.training.plateau` belong to the separate standalone drone-hover task;
press experiments use their stage's duration, phases, and plateau rule.
Shared PPO parameters, rollout length, and attempt seeds remain in `rl.training`.

The hover stage starts around the common 0.48 m press target using paired poses
drawn directly from every final policy window in the pinned flight-parameter
traces. Position and attitude from one measured sample always stay together;
the reset distribution adds no synthetic noise. Validation and acceptance
require 30 continuous seconds inside that measured envelope. Hover training
minimizes pose error throughout collection, including after a qualifying hold.
The approach stage adds measured navigation-handoff resets while retaining
target-local envelope resets. Both remain parts of the press task;
neither activates the switch yet.

Success uses only the measured position and pose bounds: asymmetric ENU axis
bounds, horizontal and 3D position bounds, roll/pitch/yaw bounds, tilt, and full
orientation error. Speed is observable but is not part of the hover acceptance
predicate. Contact or another flight
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
observation. Evaluation success, collision, other flight failure, and timeout
remain mutually exclusive completion classifications. Hover training terminates on
physical failure and truncates at its collection limit; completing a hold does
not end collection.

## Training campaign

Every `make train-rl` invocation owns one campaign ID and tries the configured
`rl.training.attempt_seeds` in order. Each attempt constructs a fresh training
Isaac process, environment, actor, critic, runner, and optimizer. The campaign
coordinator never owns a training environment: a completed worker exits before
the next seed starts, making simulator and GPU cleanup an operating-system
process boundary. Shipped training configs use one attempt seed, `[42]`, authored
in Hydra. Checkpoints are selected on the fixed validation workload with seeds
101, 202, and 303. The selected actor is frozen and tested once on separate
acceptance seeds 404, 505, and 606, with 100 episodes per seed. Qualification
applies to that actor, not to training repeatability across initializations.
Failure requires diagnosis before another campaign; final-test results never
select a replacement checkpoint. A campaign stops after the first completed
attempt with a validation-passing checkpoint. Apart from
the preceding ordered press stage, external checkpoint initialization, process
resume, actor-only copy, and post-run fine-tuning remain unsupported.

The selected `rl.press.curriculum.stages[].phases` entry defines the ordered
schedule within each press attempt; task-neutral training continues to use
`rl.training.phases`. Each
phase declares maximum iterations, validation cadence, learning rate, schedule,
and entropy coefficient. The first phase
starts from the stage-initial learner. Before every later phase, the campaign
restores the attempt-wide best checkpoint's actor, critic, optimizer, and
learner iteration, then applies the new phase settings and resets every training
environment. This is a validation-selected fine-tuning branch with reset
environment state, not an exact process resume. An earlier checkpoint remains
selectable if refinement regresses.

The evaluator and phase driver have separate contracts. Validation receives an
exact exported actor and returns raw named metrics.
The plateau detector reads only the configured metric, direction, minimum delta,
and patience. It never reads the PPO reward or any acceptance check. The active
curriculum stage owns the rule. Hover maximizes mean longest continuous-hover
duration, requires a `0.01` second improvement, and declares a plateau after
five stale evaluations. Approach minimizes mean measured pose-envelope
utilization, requires a `0.003` improvement, and declares a plateau after 10
stale evaluations.

Hover training uses a 40-second episode horizon; approach uses 45 seconds. The
hover schedule trains up to 50 iterations at `3e-4`, then restores the best complete
checkpoint and refines it for up to 100 iterations at `1e-5`. Both phases are
evaluated every five updates. Five stale evaluations stop either phase. Active
monitoring may interrupt a campaign without a credible path to acceptance.
The configuration records the maximum budget; experiment decisions and measured
results are retained in `FLIGHT_TO_POLICY_IMPROVEMENTS.md`.

Hover uses 4,096 environments with 256 policy steps per environment per update:
12.8 simulated seconds per rollout and 1,048,576 transitions per PPO update.
The broad phase allows 640 simulated seconds per environment and refinement
allows 1,280, including resets. PPO uses four epochs and eight minibatches,
discount gamma 0.995, and GAE lambda 0.99. GPU utilization and rollout/optimizer timing must be measured
for each campaign and retained alongside its training results.

Every hover training reset starts its episode clock at zero. The runner does
not randomize initial episode lengths for hover. At 20 Hz, collection truncates
after 800 steps (40 seconds), even when the 600-step hold has been reached.
Physical failures remain terminal and use the same 0.40 m minimum range as
acceptance. The explicit training/evaluation choice is passed through
environment construction; evaluation still terminates on hold success.
Actor and critic observations do not contain a clock or hold counters.

The press environment sets `is_finite_horizon=False`. The
[pinned Isaac Lab adapter](https://github.com/isaac-sim/IsaacLab/blob/37ddf626871758333d6ed89cf64ad702aef127d0/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py)
passes truncations as `time_outs`;
[RSL-RL 3.1.2](https://github.com/leggedrobotics/rsl_rl/blob/v3.1.2/rsl_rl/algorithms/ppo.py)
adds the discounted critic value to timeout rewards before storing the transition. Failures do not receive that
bootstrap. Isaac Lab clears each reset episode clock before collection, and
increments it before checking completion.

Approach runs 50 iterations at `1e-4`, 30 at `3e-5`, and 100 at `1e-5`. Its
broad phase evaluates every
ten updates and both refinement phases evaluate every five. Model zero and every
cadence-aligned phase boundary are evaluated; there is no extra boundary update.
Both stages qualify against the complete 30-second hold, and every
hover run uses the same measured reset population. Hover permits 150 attempted
PPO updates and approach
permits 180. Training advances after a phase reaches its maximum or its plateau
rule fires. Validation and test always sample the full pinned envelope.

Each measured hover reset is one complete state from one trace index: ENU
position error, relative attitude, ENU linear velocity, and ENU angular
velocity. Both velocities are recorded directly from the simulator snapshot
alongside pose, then selected using the same final-window sample indices.
Only fitting flights populate resets; recorded validation and acceptance flights
remain held out. Policy validation and final test continue to use their separate
configured evaluation seeds. The environment restores
all four values together; it does not replace measured motion with zero. The
trace does not contain controller filters, integrators, or command-delay state,
so those start uninitialized and initialize from the restored physical state on
the first controller step. Rotor thrust starts at the vehicle's hover value.

Before PPO, evaluate deterministic position/yaw feedback in the actual press-hover
environment:

```bash
make eval-press HYDRA_CONFIG=configs/train_press_hover_quad.yaml \
  EVAL_ARGS=--hover-feasibility
```

This mode loads the same accepted controller, fitting-flight reset samples,
selected stage duration, action interface, and production evaluation logic as
training validation. It uses validation seeds 101/202/303 with 100 episodes per
seed and applies the existing acceptance criteria. It does not load a trained
actor or use test seeds. Feedback reads the schema-2 observed ENU position error,
body velocity, body-to-world attitude, and contact-relative yaw. It converts the
corrective velocity into body FLU, normalizes by policy action scales, and clips
to the production action limits. Sensor noise and reset mixtures are unchanged.

The three `rl.press.evaluation.feasibility_*` gains are Hydra-owned and excluded
from the policy comparison fingerprint because they affect only this diagnostic.
Reports live under `.build/hover-feasibility/<run-id>/feasibility.json`; an optional
`--output-dir` selects a new directory. Reports retain pinned inputs, gains,
workload, raw results, acceptance checks, and errors. Resolve a failed feasibility
check before starting PPO; a pass is diagnostic evidence, not a policy artifact.

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

Use the campaign-scoped Trackio database under
`.build/training/press-hover/<run-id>/monitoring`. Each attempt is one Trackio
run containing its `training/`, `validation/`, and `system/` metrics; the selected attempt
also receives `test/`. The standard monitor is:

```bash
make trackio-rl \
  RL_TRACKIO_DIR=.build/training/press-hover/<run-id>/monitoring \
  RL_TRACKIO_PROJECT=press-hover
```

and is served at `http://127.0.0.1:7860` by default. The training wrapper prints
a run-scoped URL that selects the attempt, sets smoothing to zero, shows complete
metric names, and filters for the main learning and validation charts. Set
`RL_TRACKIO_PORT=<port>` when that address is occupied. The exact selected-step
values are recorded under each attempt's `evaluations` list in
`training_report.json`. The wrapper keeps the live server under supervision and
waits until the expected attempt contains its first reset metric before reporting
it ready. Displayed rates explicitly identify fractions in `[0, 1]`; raw report keys
retain their semantic names.

Trackio uses the `training/`, `validation/`, `test/`, and `system/`
namespaces. Chart names spell out the quantity and units, with no numeric group
codes or internal variable names. Presentation registry revision 19 changes
labels only; historical dashboards retain their recorded labels. Validation and
test share one ordered registry: acceptance status, success, collision, other
flight failure, minimum range, trace finiteness, measured-envelope occupancy and
utilization, mean and maximum episode-longest hover duration, range undershoot,
timeout, completed episodes, and all 40 flight-parameter statistics. Acceptance
is report metadata, not a Trackio namespace. Total attempted PPO updates are the
X coordinate and never rewind when a refinement phase restores an earlier
learner checkpoint; `training/Updates in current learner checkpoint` records that learner
position separately.

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

The press PPO reward reports six signed contributions. Hover adds position
and attitude utilization so both remain visible to the optimizer independently.
Its `hover_stabilization` config selects `uniform` (weight 0.01 everywhere),
`boundary_weighted` (0.01 inside the full measured envelope, 0.03 outside), or
`outside_only` (zero inside, 0.03 outside). The current hover setup selects
`boundary_weighted`: the cost is the sum of position and attitude utilization,
multiplied by -0.01 inside the complete envelope and -0.03 outside. This cost
remains active for nonzero error inside the envelope and reaches zero at the
desired pose. It never chooses the larger of position and attitude.

Hover retains the squared normalized-action cost and a -30.0 physical-failure
penalty. Its `position_progress`, `predictive_barrier`, and `hold_progress`
contributions are zero. Crossing the 30-second hold or reaching the 40-second
collection limit adds no reward or penalty. Hold counters remain qualification
and reporting state, without changing actor or critic observations.

Approach rewards are unchanged: `position_progress` rewards reduction in
target error until the measured position envelope is reached; `stabilization`
is `0.01 / (1 + utilization)` inside that position envelope; the predictive
range barrier remains active. `hold_progress` is 0.01 only when the current
continuous hold exceeds the episode's previous longest hold. Its terminal
outcome is +1.5 on success and -3.0 on unsuccessful completion, including
timeout. Position progress and the predictive barrier have weight 0.1.
Contact remains a binding failure predicate.

Training uses `training/reset/` for the fixed reset population and
`training/reward/` for the six PPO reward contributions. Other progress,
optimizer, schedule, and performance charts use descriptive names under
`training/`. Reset charts include position error, linear and angular speed,
and position and attitude error relative to their hover limits. A value of 1
is the measured limit; all fraction charts use `[0, 1]`. Hardware telemetry
uses `system/training/`, `system/validation/`, and `system/test/`.

Use **`validation/Mean longest continuous hover (seconds)`** to compare each
checkpoint: it averages each evaluated episode's longest uninterrupted hold.
`validation/Longest continuous hover in any episode (seconds)` shows the
maximum. `training/Mean longest hover so far in active episodes (seconds)`
reports ongoing training episodes and has a different averaging population.
Successful-hold flight statistics spell out maximum absolute error, root mean
square error, and the 95th and 99th percentiles with meters or radians.

PPO minimizes the clipped policy surrogate plus the configured value-loss
coefficient times the clipped critic return error, minus the entropy
coefficient times action-distribution entropy. Monitoring names these
`ppo_clipped_surrogate_loss`, `critic_return_mse`, and
`action_distribution_entropy`. Actor and critic relative parameter changes are recorded after
every update because RSL-RL clips their combined gradient norm.

## Validation, final test, artifact, and acceptance

Every checkpoint in every phase and attempt uses the same validation workload:
100 episodes for each of seeds 101, 202, and 303. A persistent worker evaluates
each exported actor on that workload, so every record identifies exact
`policy.pt` bytes. Validation is not cached.

After all phases end, validation-passing checkpoints rank ahead of failures.
Within each group the configured stage progress metric ranks checkpoints, with
the earlier checkpoint winning an exact tie. A validation pass only makes the
candidate eligible for the final test.

If the final refinement phase cannot improve the selected actor from an earlier
phase, the campaign preserves the attempt and stops before starting another
seed. Repeating the same curriculum is not treated as a credible path to
acceptance; the reset, reward, or phase strategy must be diagnosed first.

The raw SHA-256 of the exact exported `policy.pt` remains in every evaluation
and published artifact record. The campaign parent is the only writer of
`training_report.json`. Before each worker launch it records the active attempt
identity and its report path. The worker atomically writes
`attempt_report.json` below its attempt directory; the parent validates and
incorporates that report after the worker exits. An interruption preserves
completed attempts, the active pointer, and the partial attempt report, marks
the campaign interrupted, exits nonzero, and publishes nothing. Process resume
remains unsupported.

`training_report.json` schema 26 records the prepared acceptance criteria, the
exact reward configuration, resolved rollout settings and derived maximum
simulated durations per phase, trace-backed reset distribution, and every attempt,
phase parent, raw validation metrics, exact monitoring step, telemetry summaries,
plateau state, phase termination, selection, final test, and individual
acceptance. The metric registry and comparison fingerprint carry
metric-registry revision 19 and comparison revision 36; older report shapes are
unsupported.

The report's `provenance` contains the complete resolved experiment config,
config path, procedural-smoke flag, canonical controller and flight-parameter
paths, producer run IDs, digests and fingerprints. It also records Python,
Torch, NumPy, RSL-RL and CUDA versions. The coordinator captures this before
creating the campaign; each attempt captures its own provenance before
environment construction and retains it in its report and campaign attempt record.
Attempts also record their learner initialization, including the preceding
stage's checkpoint lineage when applicable. Existing evaluation records retain
checkpoint paths, iterations, actor digests, seeds and results.

The experiment record supports comparing a small number of consecutive policy
runs. Training does not collect Git commits, working-tree state or container
identity. Keep the checkout and configuration fixed during a campaign. See
[flight-to-policy improvements](FLIGHT_TO_POLICY_IMPROVEMENTS.md) for the changes and
expected impact.

After selection, the exported actor is immutable. A new uncached worker runs the
disjoint final workload: 100 episodes for each of seeds 404, 505, and 606. The
resulting raw `PolicyEvaluation` is the sole measured input to
`evaluate_press_acceptance`. Every check is independently binding: finite traces,
at least 0.95 success, zero collision, at least 0.40 m minimum closest
range, exactly zero other flight failures, and the flight-parameters maximum,
RMSE, p95, and p99 limit for every hover error signal. Hover statistics use only
episodes that complete the 30-second hold. They cover the final 24 seconds of
each episode's actual successful hold, ending at its completion time. Earlier recovery, the first six seconds of
the hold (the artifact-configured policy warmup), failed episodes, and
post-completion samples are excluded. The evaluator retains pose history on the
GPU and transfers it once per seed; seeds, reset order, episode counts, and
aggregation are unchanged. Comparison revision 35 and earlier used an
episode-start window for these statistics, so their delayed-success statistics
are not comparable to revision 36. Historical artifacts remain immutable.
Until an episode succeeds, those statistics are unavailable rather than being
calculated from a failed partial hold. The evaluator records every raw
statistic in the `PolicyEvaluation` and `training_report.json`; acceptance adds
no consumer-owned tolerance. Test data cannot affect training,
phase advancement, checkpoint selection, or retries. A test failure ends the
command immediately and preserves evidence without publishing.

Acceptance pools the retained successful-hold samples across all episodes and
seeds. For each signal it computes maximum absolute error, `sqrt(mean(x^2))`,
and the 95th and 99th percentiles of `abs(x)` using NumPy's default linear
interpolation. These are sample-level statistics, not averages of episode
RMSEs or percentiles. Each successful hold contributes the same number of
samples. Reference limits are the largest corresponding **per-flight**
policy-window statistic plus the artifact's authored margin. They are
conservative limits applied to the pooled policy statistics; no per-episode
RMSE or percentile acceptance requirement is added.

Success, collision, failure and timeout rates count outcomes across all
evaluated episodes, including unsuccessful ones. Occupancy and utilization
diagnostics average each episode's active-step mean equally; reward diagnostics
average all active steps. Minimum closest range is the minimum over all
episodes. Missing successful-hold statistics remain `null` and fail their
acceptance checks.

Evaluation raises `ValueError` on NaN or infinity in an active episode's policy
observation, raw actor output before clipping, or numeric transition values
used for hover, range, duration or reward metrics. The error identifies the
field, seed, zero-based episode and step. The transition that ends an episode
is checked; later auto-reset samples from that slot are excluded. Non-finite
aggregate metrics, reward means and available hover statistics also raise
before a `PolicyEvaluation` is returned. Errors propagate through the existing
worker error handling and stop selection or acceptance without publication.
Intentional accumulator initialization and unavailable hover statistics are
not invalid trajectory samples. The existing `all_traces_finite` metric is 1
for a completed valid evaluation.

A policy artifact is exactly `policy.pt` plus `policy_meta.json`. Metadata uses
the strict schema-1 `robotics-experiments-policy` contract and records policy
interface, observation/action contracts, cadence, controller artifact and
digest, flight-parameters artifact and digest, comparison fingerprint,
and training phase provenance. Older policy schemas and
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
artifact, including a validation-selected candidate. That assignment
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
evaluation and training qualification counters. It should not touch the
switch, climb away, or perform a learned retreat. The routine's later `goto` step owns any return motion.
