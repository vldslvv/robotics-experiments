# Reinforcement-learning workflow

This document defines the Isaac Lab tasks, curriculum and stage training workflow, policy
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

## Press tasks

Both policy interfaces use shared NumPy/Torch body-relative calculations.
Hover observation schema 2 has 12 values: desired-minus-actual position in body
FLU (metres, 0:3), body linear velocity (m/s, 3:6), body angular velocity (rad/s,
6:9), and unit gravity direction in body coordinates (9:12). Standalone hover
retains its position-only objective and has no desired-heading input.

The task-neutral `pack_pose_observation` appends columns 0 and 1 of the
goal-to-body rotation matrix (12:18), concatenated column by column. It receives
actual and desired world orientations from callers and reads no scene or switch.
Quaternion signs do not affect the packed values. Consistent translation and
global yaw rotation of the drone, target, and motion leave these flight features
unchanged; rotating only the drone changes them. Gravity retains roll/pitch
information even when the goal is tilted.

Press observation schema 3 has 35 values. It starts with this 18-value pose block,
then contains pose confidence/age/validity (18:21), pusher compression/rate
(21:23), contact validity/state (23:25), lamp state/confidence/age (25:28), previous
action (28:32), and handoff mode/estimator health/availability mask (32:35).
The press task supplies the contact frame as the desired drone orientation.
Raw `PressObservation` measurements remain available for physical handoff and
completion checks; their dataclass order is not the actor's vector layout.

Position inputs are metres, without per-axis flight-envelope division. The
configured empirical normalizer remains part of training and the exported actor.
Physical reset samples, rewards, sensor noise, and asymmetric ENU acceptance
bounds are independent of this input representation and retain their original
frames and values. Actor schemas are strict: hover schema 1 and press schema 2
actors require retraining, and their bytes and qualification records are never
migrated. The outer policy artifact schema remains version 1.

The press policy emits four normalized velocity actions. Its scratch experiment
maps a unit translation action to 0.5 m/s. Normalized scratch exploration starts
at 0.10 standard deviation, or 0.05 m/s for a translation component.

`rl.press.task` explicitly selects `hover` or `approach`. Task code owns rewards
and resets; task identity never comes from a curriculum label or position.
`rl.press.reset.mix` owns the measured reset mixture. `rl.episode_length_s`
owns episode duration, and `rl.press.goal.stable_steps` is the common training,
evaluation, and deployment hold requirement. `rl.training.phases` and
`rl.training.plateau` own every experiment's training schedule and stopping rule.
The shipped numerical settings and reward formulas are unchanged.
The schema-1 policy metadata field `curriculum_stage` continues to record this
task identity; it never stores curriculum labels. Existing artifacts are immutable.

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

Hover deployment derives its handoff bounds from the same fitting-flight
policy-window states returned by `load_hover_reset_samples` for PPO resets.
The runtime validates the selected flight bundle against the policy's embedded
run ID and digest before reading these states. This work runs during startup
validation and is cached by canonical bundle path, run ID, and digest so routine
construction cannot stall live PX4 lockstep by rereading the traces.
It uses their asymmetric ENU
position extrema, horizontal/total position maxima, relative roll/pitch/yaw
extrema, tilt/orientation maxima, and ENU linear/angular velocity extrema.
No margins or validation/acceptance flights expand this region. These are
bounds on a paired empirical population, not a claim that every combination
within the bounds was sampled. Approach deployment retains the navigation
reset bounds from `rl.press.handoff`.

While waiting, both press stages apply position and yaw feedback toward the
common target through the normal normalized velocity action interface. The
`rl.press.handoff.feedback_*` gains, damping, and speed cap are deployment-only
Hydra settings and do not alter the training comparison fingerprint. Invalid
or stale pose observations never drive this feedback. Handoff is checked on
each policy tick and latches before the first actor call. The previous-policy
action remains zero until that call, matching PPO episode starts; preparation
commands do not become actor history. Completion and acceptance bounds remain
those embedded in the actor artifact.

The accepted 2026-08-12 schema-2 production campaign recorded ten successful
arrivals: switch range 0.464594-0.498991 m, maximum combined lateral/vertical
error 0.033746 m, and maximum full relative-attitude error 0.016339 rad
(0.93616 degree). With the explicit 0.005 m position and 0.25 degree rotation
margins, that evidence requires 0.44-0.52 m range, a 0.08 m alignment disk, and
a 0.020702 rad (1.18616 degree) rotation ball. The approach handoff gate extends
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

## Curriculum and stage training

`make train-rl` and `isaac/runtime/standalone/train_rl.py` load an explicit
curriculum through `load_curriculum_config`. Entry configurations live in
`configs/train/`; their complete stage experiments live in
`configs/train/stage/` and use `load_config`. A single stage uses the same flow
as several stages. Empty curricula and nested curricula are invalid.

```yaml
# configs/train/press_hover_curriculum_quad.yaml
# @package _global_
defaults:
  - /base_curriculum
  - _self_
policy_name: press-hover
initial_learner: null
stages:
  - name: stabilize
    config_path: configs/train/stage/press_hover_quad.yaml
  - name: reduce_corrections
    config_path: configs/train/stage/press_hover_outside_only_quad.yaml
```

Each entry contains only a unique nonempty label and an experiment path relative
to the repository. Labels never select task behavior, rewards, resets, checkpoint
compatibility, or namespaces. `policy_name` is the one artifact namespace for the
whole curriculum. Each stage owns its task, rewards, reset mixture, architecture,
normalization, PPO settings, phases, evaluation workload and acceptance criteria.
Shared values use ordinary Hydra defaults. Stage experiments have no initializer,
pipeline list, or policy namespace.

`workflows/isaac_lab/curriculum.py` pins all stage input evidence, then calls
`training.run_training`. That function owns a campaign of attempt seeds. Each
attempt uses a distinct Isaac process with fresh environments and rollout buffers.
The CLI builds an `AppLauncher` only for attempt workers; evaluation workers build
their own. The curriculum coordinator never owns an environment.

The immutable `TrainingResult` contains `selected_checkpoint`, `accepted`,
`completed_updates`, and `report_path`. A selected learner references canonical
checkpoint, candidate policy and metadata, and producer report paths, with four
content digests. Producer identity comes from the canonical report directory.
A failed qualification retains the selected diagnostic candidate; absence means
no candidate was evaluated. The coordinator reads only `accepted` to advance.
It publishes each accepted stage before forwarding that exact checkpoint, without
resolving deployment labels again. Rejection, runtime error, interruption, or
publication failure stops later stages. There are no curriculum retries,
branches, checkpoint merges, or process resume. Within a stage the existing
attempt campaign and refinement stopping rules remain unchanged.

`initial_learner` optionally names a producer `training_report.json`, including a
compatible failed or interrupted run. Acceptance of this input is not required.
Before training, checkpoint tensors must match the exact exported actor and its
normalizer. Controller and flight evidence, observation/action contract, network
dimensions, activation and normalization must be compatible. Hover and approach
retain their different normalization settings, so they are not silently made
compatible by curriculum ordering. Producer files remain immutable.

### Complete learner checkpoints

`rl/training/checkpoint.py` owns `save_checkpoint`, `load_checkpoint`, and
`restore_checkpoint`. Schema 1 has exactly these fields:

| Field | Stored state and restore behavior |
| --- | --- |
| `schema_version` | Strict checkpoint format; unsupported formats fail. |
| `model_state_dict` | Actor, critic, learned exploration, actor/critic observation means, variances, standard deviations and counts; all restored. |
| `optimizer_state_dict` | Adam moments, per-parameter steps, parameter groups and current LR; all restored. |
| `progress` | Local iteration, selected-branch `learner_updates`, and invocation-wide `curriculum_updates` at capture. |
| `rng_state` | Python, NumPy, Torch CPU and CUDA snapshots, stored as restricted-load-safe primitives and tensors. |

There is no persistent reward/return normalizer or separate LR scheduler object
in the current PPO setup. Rollout advantage normalization is transient. Adding
persistent algorithm state requires an explicit checkpoint contract extension.
No placeholder state or partially restored learner is supported.

New attempts reseed from their stage's `attempt_seeds`; saved RNG state is retained
as evidence, not applied over that seed. Actor, critic, optimizer, exploration and
observation statistics continue from the selected checkpoint. New stages restart
local iteration at zero. Later phases restore the selected local iteration and
learner ancestry, apply their configured LR, schedule, entropy coefficient and
normalization-update setting, then reset environments. LR changes do not discard
learned exploration. Simulator state, episode returns and rollout buffers never
cross stage boundaries.

`learner_updates` follows the selected branch and may rewind at phase selection.
`curriculum_updates` counts every completed update in this invocation, including
discarded branches and attempts, and never rewinds. If a stage executes 100
updates but selects update 60, the next stage inherits learner count 60 and starts
at curriculum offset 100. A new curriculum invocation starts its total at zero
even when it initializes from an older learner. These counts are independent of
optimizer step counters.

### Reports and publication

```text
.build/training/<policy_name>/
  <run-id>/curriculum_report.json
  <run-id>/monitoring/
  <run-id>.stage_01/training_report.json
  <run-id>.stage_01/attempt_01_seed_42/attempt_report.json
  <run-id>.stage_01/attempt_01_seed_42/checkpoints/...
  <run-id>.stage_02/...
  latest -> <accepted-stage-run-id>
  best -> <best-comparable-stage-run-id>
```

`RL_RUN_ID` selects the curriculum run ID once. Stage suffixes are uniform even
for one-stage curricula. Curriculum report schema 1 records configuration and
initial lineage, ordered stage report references, `accepted`, `published`, completed
updates and status. These two booleans distinguish acceptance from a publication
error. Detailed measurements remain in stage and attempt reports.

Training and attempt reports use strict schema 28. Old reports are not migrated.
The stage records resolved configuration and pinned evidence once, prepared
acceptance criteria, effective phases (including smoke reductions), attempt report
references, progress, selected candidate identity/paths/digests/comparison metric,
fresh-test evaluation and acceptance checks. Attempts record seed, process ID,
evaluations, phase results, progress, selected evaluation index and telemetry
references. Acceptance reasons derive from failed checks. Raw metrics and hardware
samples are not copied between reports. Each level owns its terminal report;
publication never rewrites a completed training report.

Publication preserves the tested actor bytes and policy schema 1. `latest` advances
for every accepted stage; `best` retains the existing fingerprint comparison and
earlier exact ties. Failure at a later stage leaves the earlier published actor
available. Supplemental policy metadata records the accepted curriculum prefix as
stage labels and exact report/policy digests. This is historical evidence for those
actors, not a claim that the latest actor was retested on previous stages.
Initializer lineage is separate. The existing `curriculum_stage` artifact field
continues to mean task identity (`hover` or `approach`).

The evaluator and phase driver have separate contracts. Validation receives an
exact exported actor and returns raw named metrics.
The plateau detector reads only the configured metric, direction, minimum delta,
and patience. It never reads the PPO reward or any acceptance check. The experiment owns the rule. Hover minimizes `hover_envelope_utilization_mean`,
requires a `0.003` improvement, and declares a plateau after five stale
evaluations. Approach uses the same metric, direction, and improvement with
10 stale evaluations. Checkpoint selection prioritizes validation-passing actors,
then the lowest configured error, preserving the earlier actor on exact ties.
Plateau console records identify the configured metric, direction, current
value, and material-progress reference.

Hover training uses a 40-second episode horizon; approach uses 45 seconds. The
hover schedule trains up to 50 iterations at `3e-4`, then restores the best complete
checkpoint and refines it for up to 100 iterations at `1e-5`. Both phases are
evaluated every five updates. Five stale evaluations stop either phase. Active
monitoring may interrupt a campaign without a credible path to acceptance.
Early failures in scratch training alone do not establish that condition;
assess later validation and the configured phase schedule before declaring
that learning will not recover.
The configuration records the maximum budget; experiment decisions and measured
results are retained in `FLIGHT_TO_POLICY_IMPROVEMENTS.md`.

Hover uses 4,096 environments with 256 policy steps per environment per update:
12.8 simulated seconds per rollout and 1,048,576 transitions per PPO update.
The broad phase allows 640 simulated seconds per environment and refinement
allows 1,280, including resets. PPO uses four epochs and eight minibatches,
discount gamma 0.999, and GAE lambda 0.995 for the longer-horizon trial.
GPU utilization and rollout/optimizer timing must be measured
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
make eval-press HYDRA_CONFIG=configs/train/stage/press_hover_quad.yaml \
  EVAL_ARGS=--hover-feasibility
```

This mode loads the same accepted controller, fitting-flight reset samples,
selected stage duration, action interface, and production evaluation logic as
training validation. It uses validation seeds 101/202/303 with 100 episodes per
seed and applies the existing acceptance criteria. It does not load a trained
actor or use test seeds. Feedback reads the schema-3 body goal offset and body velocity directly. It
recovers drone yaw in the goal frame from the relative rotation columns,
normalizes commands by policy action scales, and clips to production action
limits. Sensor noise and reset mixtures are unchanged.

The three `rl.press.evaluation.feasibility_*` gains are Hydra-owned and excluded
from the policy comparison fingerprint because they affect only this diagnostic.
Reports live under `.build/hover-feasibility/<run-id>/feasibility.json`; an optional
`--output-dir` selects a new directory. Reports retain pinned inputs, gains,
workload, raw results, acceptance checks, and errors. Resolve a failed feasibility
check before starting PPO; a pass is diagnostic evidence, not a policy artifact.

Run training with:

```bash
make train-rl HYDRA_CONFIG=configs/train/press_hover_quad.yaml
```

Environment count, attempt seeds, phase lengths, evaluation cadence, PPO
settings, artifact selectors, and plateau rules are Hydra-owned. The launcher
does not provide command-line overrides for them. `--procedural-smoke` remains
an operational workflow check rather than a training parameter.

The GPU procedural check keeps every phase/restore boundary, performs one update
per phase on every configured attempt seed, and deliberately stops before
reserved test seeds are used:

```bash
make train-rl HYDRA_CONFIG=configs/train/press_hover_quad.yaml \
  TRAIN_ARGS=--procedural-smoke
```

The procedural smoke executes one update in every phase of every configured
attempt, including full attempt teardown/recreation, and stops without using
the reserved test seeds.

To check restoration, set `initial_learner` in a new curriculum to that smoke
stage's `training_report.json` and run with `--procedural-smoke`. An unaccepted
smoke learner is a valid compatible initializer. Neither check uses reserved test
seeds or updates deployment labels. With existing assets use
`make -o generate-drone-usd train-rl ...` to skip asset generation.

Use the curriculum-scoped Trackio database under
`.build/training/press-hover/<run-id>/monitoring`. Each stage/attempt is a unique
Trackio run, for example `<run-id>.stage_01.attempt_01`. The standard monitor is:

```bash
make trackio-rl \
  RL_TRACKIO_DIR=.build/training/press-hover/<run-id>/monitoring \
  RL_TRACKIO_PROJECT=press-hover
```

The default URL is http://127.0.0.1:7860. The wrapper selects the first stage's
attempt, sets zero smoothing and shows complete metric names. All stages share
the same database. Exact validation values live in the attempt reports referenced
by `training_report.json`. The wrapper waits for the first reset metric before
reporting the monitor ready. It skips automatic post-training evaluation when
training fails. Explicit evaluation can still select an earlier published stage.
Normal `eval-press` resolves the published artifact once, uses its producing stage
configuration and pinned evidence, and rejects a changed stage configuration.
`--hover-feasibility` instead requires an explicit stage experiment.

Trackio uses the `training/`, `validation/`, `test/`, and `system/`
namespaces. Chart names spell out the quantity and units, with no numeric group
codes or internal variable names. Presentation registry revision 19 changes
labels only; historical dashboards retain their recorded labels. Validation and
test share one ordered registry: acceptance status, success, collision, other
flight failure, minimum range, trace finiteness, measured-envelope occupancy and
utilization, mean and maximum episode-longest hover duration, range undershoot,
timeout, completed episodes, and all 40 flight-parameter statistics. Acceptance
is report metadata, not a Trackio namespace. Completed curriculum PPO updates are the
X coordinate and never rewind when a refinement phase restores an earlier
learner checkpoint; `training/Updates in current learner checkpoint` records selected-branch ancestry separately;
`training/Completed curriculum updates` records the invocation total.

`system/` is diagnostic-only and cannot influence PPO, plateau progress,
selection, retry choice, validation results, or acceptance. At the
Hydra-configured one-second cadence, every training chunk and fresh validation
or test evaluation samples NVIDIA utilization, memory, power, temperature, and
the owning process's resident memory and CPU use. NVIDIA percentages are stored
as fractional rates. Per-update rollout/PPO timing and Torch allocator peaks are
also emitted during training. Raw samples and interval summaries live in telemetry sidecars referenced by
the training and attempt reports. A
startup probe fails before rollout when telemetry is unavailable, and a
sampler failure terminates at the interval boundary with evidence preserved.
The configured `0.95` GPU-utilization target is diagnostic: a miss is recorded
as the measured bottleneck and never changes policy qualification.

The press PPO reward reports six signed contributions. Hover adds position
and attitude utilization so both remain visible to the optimizer independently.
Its `hover_stabilization` config selects `uniform` (the configured stabilization
weight everywhere), `boundary_weighted` (the configured inside/outside weights),
or `outside_only` (zero inside, the configured outside weight elsewhere). The current hover setup selects
`boundary_weighted`: `configs/train/stage/press_hover_quad.yaml` overrides the cost
weights to 0.03 inside the complete envelope and 0.09 outside for the first
correction campaign. The cost is the negative weighted sum of position and
attitude utilization. Shared approach reward weights remain unchanged. This cost
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

Use `hover_envelope_utilization_mean` in `training_report.json` to audit
checkpoint selection. The validation chart
`Mean worst pose error relative to hover limit (1 = limit)` shows this error;
lower is better.
**`validation/Mean longest continuous hover (seconds)`** averages each evaluated
episode's longest uninterrupted hold and remains a qualification diagnostic.
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

The report schema and ownership are described above. Metric-registry revision 19
and comparison revision 38 remain unchanged. Task reset mixtures affect comparison;
curriculum labels and ordering do not. Complete resolved configuration, runtime
versions and pinned input lineage appear once in the stage's `provenance`.
Workers verify this configuration and the pinned producer digests before training.

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
make eval-press HYDRA_CONFIG=configs/train/press_approach_quad.yaml
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

For hover correction qualification, retain Lab publication semantics and record
PX4 results separately. A candidate must pass the existing fresh Lab acceptance
workload, then three consecutive complete warehouse transfer executions using
the same canonical policy run and actor digest. Each execution must complete all
three policy segments without contact, lamp transitions, timeout, or flight
failure. The 45-second policy budget includes handoff, settling, and any restarted
30-second continuous hold; it does not guarantee completion. A failed execution
ends that candidate's transfer qualification. On failure to qualify, restore the
baseline deployment through `latest`; retain candidates and Lab-ranked `best`.

The correction experiment retains the 0.03/0.09 reward override, gamma 0.999,
and lambda 0.995 for reproducibility. Both completed stronger-reward trials
failed validation, including the longer-horizon restart, which ended normally
after 50 broad and 85 refinement updates. A fresh control using the previous
complete experiment settings reproduced its Lab acceptance exactly. Recorded-input
command comparisons show persistent excessive upward commands near the goal;
increasing reward strength further is not justified by insufficient correction.
No tighter-hover candidate qualified for PX4 testing. All task-owned workers and
monitors are stopped, and `latest` remains baseline `20260913T222441Z`.
See `FLIGHT_TO_POLICY_IMPROVEMENTS.md` for exact inputs and results.
