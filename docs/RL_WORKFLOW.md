# RL workflow (Isaac Lab, training, artifacts, and deployment)

This is the single source for the reinforcement-learning workflow: the Isaac
Lab image, task and actuation seams, training and checkpoint selection, policy
artifacts, deterministic evaluation, and deploy-time `{kind: policy}` behavior.
The generic routine executor contract lives in
`docs/COMMAND_ROUTINE_CONTRACT.md`.

## Isaac Lab image and task boundary

RL targets run on `ISAAC_LAB_IMAGE` (`vv-isaac-lab-custom:<tag>`), built from
the `isaac-lab` Dockerfile stage layered on the project Isaac Sim image. Isaac
Lab and Isaac Sim therefore share Kit Python, USD, and PhysX. The image uses
the `ISAACLAB_REF` pin and installs the base Lab and rsl_rl packages. Ordinary
integration and persistent runtime containers remain on the leaner Isaac Sim
image.

Importable Lab workflow code lives under
`src/robotics_experiments/workflows/isaac_lab/`. Its lazy package root exports
task registration, environment factories, and host-safe evaluation records
without importing Kit or Isaac Lab on the host. The registered direct tasks
are `Isaac-Drone-Hover-Direct-v0` and `Isaac-Switch-Press-Direct-v0`.

Both tasks use the task-neutral actuation seam from
`robotics_experiments.rl`. The configured `rl.action.kind` selects either the
body-velocity/yaw-rate cascade (`velocity`) or collective-thrust/body-rate
control (`ctbr`). NumPy is the pure reference; the batched torch implementation
is its Lab twin. The allocation geometry and derived arrays are immutable.
PX4 is bypassed in Lab, but the schema-6 aligned controller is a stateful
PX4-like velocity, attitude, and rate cascade with filtered acceleration
feedback, integrators, a smooth horizontal large-error gain schedule, thrust
saturation/anti-windup, and rate
PID/feed-forward. The velocity and rate controllers emit PX4-normalized thrust
and torque into the pinned multirotor pseudo-inverse and sequential-desaturation
allocator. The composed bridge offset, scaling, armed zero, and quadratic rotor
curve then produce physical per-rotor thrust. Schema-4 artifacts retain their
hover-scaled physical-force path; schema-2 and schema-3 artifacts retain their
original physical collective feed-forward, and schema 2 retains its unit
rate-output scale. Schemas 2 through 5 retain a neutral large-error gain
multiplier. Attitude tracking uses PX4's thrust-prioritized
reduced-quaternion law, and zero velocity-filter cutoff has PX4's
disabled/pass-through meaning. The pinned Isaac Lab
`isaaclab_contrib.actuators.Thruster` realizes the resulting per-rotor thrust
references. Controller gains, filters, integral and thrust limits, command
delay, tilt, and thruster response come from an immutable alignment artifact
rather than `rl.action`.
The project still owns allocation, per-rotor yaw reaction, drag, and the sole
body-wrench sink because the pusher-equipped articulation is not a pure
multirotor asset.
Both Lab flight environments pass the authored enhanced-determinism flag
through their `SimulationCfg`. Rotor forces use PhysX's default one-application-
per-step behavior, matching the standalone bridge's one complete wrench per
physics tick.

The shared exact-average cadence derives policy stepping from `sim.physics_hz`
and `rl.policy_rate_hz`. At 250 Hz physics and 20 Hz policy rate, the scheduler
alternates 13 and 12 physics samples while `step_dt` remains 0.05 s. Exported
artifacts own evaluation and deployment cadence and action scaling.

PX4 SITL is the response reference for the separate, policy-independent
alignment subsystem. Run `make align-controllers`, then
`make test-integration-alignment-target`; see
`docs/SIM_TO_SIM_ALIGNMENT.md`. Training resolves the accepted
`.build/alignment/x500-velocity-px4-v6/best` by default and may select an exact
run with `ALIGNMENT_ARTIFACT`. The command
buffer, velocity/rate controller state, and thruster state are cleared on every
environment reset.

`rl.action` now contains only normalized action kind, dimension, and scales.
The selected alignment's complete `ControllerSettings` is supplied explicitly
to both the NumPy reference and torch twin. Changing the controller requires a
new alignment artifact and new training; an existing actor is never silently
upgraded to current `best`.

After Lab evaluation, `make test-integration-press-policy-transfer` runs the
trained actor at three PX4 entry ranges in the production warehouse with
physical wall and switch contact enabled. It requires every approach to finish
without body contact, pusher contact, or a lamp transition. This is downstream
transfer acceptance, not a fitting input. The 95% success, 1% collision, and
0.40 m minimum-range thresholds are advisory publication diagnostics; a
failing independent evaluation still blocks this PX4 acceptance workflow.

## Press observation and approach stage

The press actor contract remains the schema-v1 34-value observation and
four-value normalized action. `PressObservation` contains switch-relative pose,
vehicle velocity and attitude, pusher compression and rate, contact, lamp,
previous action, handoff mode, estimator health, and availability bits.
Training and deployment use the same packing function. The contact frame is
fixed by `world_switch_to_contact_frame`; switch-relative +X points from the
drone toward the surface. `contact_flag` means pusher/pad contact only; it does
not represent contact by the frame, arms, propellers, or wall.

The pose-provider seam is `AbstractPoseProvider` / `PoseSample`.
`GroundTruthPoseProvider` is the lazy Isaac-only implementation.
`handoff_ready` is the pure navigation-to-policy gate configured by
`rl.press.handoff`. That config is the shared actor-entry contract for Lab
resets and deployment: 0.45-0.50 m wall-normal range, at most 0.08 m combined
lateral/vertical error, at most 0.02 m/s wall-normal and lateral speed, at most
0.05 rad/s angular speed, upright attitude, valid pose/contact samples, no
contact, zero previous action, pre-contact mode, and a healthy estimator.

The curriculum is intentionally staged:

1. Stage 1, the implemented `approach` stage, learns a controlled contact-free
   approach, alignment, and stable stop.
2. Stage 2 will learn contact and switch activation.
3. Stage 3 will learn controlled post-contact retreat and may initialize from a
   compatible earlier stage.

There is no automatic promotion and no fixed multi-stage enum. Each future
stage adds its own Hydra option, policy namespace, transition, reward record,
evaluator, and runtime completion predicate. The routine's ordinary return
`goto` remains downstream of Stage 1 and is not learned retreat behavior.

Stage 1 is the frozen `PressApproachState` advanced by
`step_press_approach(...)`. State contains previous range, previous action,
consecutive stable-goal steps, and closest range. The full observation remains
packed for later actor compatibility; compression and contact are measured,
the lamp remains off, and `handoff_mode` remains pre-contact.

Collision is privileged training state, not policy state. During each 20 Hz
policy interval, Isaac advances 12 or 13 physics substeps while separate PhysX
contact sensors latch collisions on the drone body and pusher pad. The
environment supplies the two latched flags only to `PressApproachInput` after
the interval. They are absent from `PressApproachState`, the schema-v1
34-value actor observation, and the critic input. PPO receives only the
resulting reward and terminal flag. Collision terminates and resets the
episode, so no collision value persists into successor policy state.

Success requires five consecutive 20 Hz policy samples satisfying all of:

- Wall-normal range from 0.40 through 0.42 m.
- Combined lateral/vertical error no greater than 0.03 m.
- Total linear speed no greater than 0.10 m/s.
- Angular speed no greater than 0.40 rad/s.
- Body-Z/world-Z alignment at least 0.965926.
- No pusher collision during the policy interval.

Body collision, pusher collision, crash, altitude, or relative-envelope failure
terminates the episode.
Timeout is a truncation and is mutually exclusive with terminal success or
failure. Reset samples range from 0.45 through 0.50 m and samples lateral and
vertical offsets uniformly inside the shared 0.08 m radial bound. Initial
linear velocity is sampled in the contact frame inside the shared wall-normal
and lateral speed limits, initial angular velocity is sampled inside the shared
0.05 rad/s ball, attitude is upright, previous action is zero, and mode is
pre-contact. Reset owns no independent entry-velocity tunables.

`PressApproachRewardTerms` has exactly nine signed contributions:

- `range_progress = 50 * (previous_range - current_range)`.
- `alignment_error = -50 * ||relative_yz||^2`.
- `velocity_tracking = -100 * ||measured_velocity - target_velocity||^2`.
- `upright_error = -5 * (1 - clamped_body_z_alignment)^2`.
- `action_delta = -0.004 * mean((action - previous_action)^2)`.
- `success = 150` on the successful terminal step.
- `collision = -300` on a body or pusher collision.
- `flight_failure = -100` on an ordinary flight-envelope failure.
- `timeout = -25` on truncation.

The contact-frame velocity target is dense in all three axes. Its wall-normal
component is `clip(1.25 * (range - 0.41), -0.04, 0.08)` m/s. Its tangential
components are twice the lateral and vertical switch errors, radially limited
to 0.10 m/s. This target teaches braking, small controlled retreat after range
undershoot, and simultaneous alignment without encoding a task-specific actor.
Collision overrides success and the ordinary flight-failure penalty;
simultaneous body and pusher collision produces one collision reward penalty.
`PressApproachTerminationFlags` still retains both collision sources for
diagnostics. `Reward/` TensorBoard scalars enumerate every contribution plus
`total`. `Press/` reports completed episodes, success, combined collision,
body-collision, pusher-collision, other-flight-failure, and timeout percentages.

The actor action drives training physics directly. No deterministic
near-wall governor changes the forward command in training or deployment.
Collision reward and termination must therefore produce the learned braking
behavior; a flat zero-collision run from the initial checkpoint is not evidence
of learning unless the unguided actor already satisfies the independent gates.

The physical switch, wall, pusher, arm, and propeller-collision assets remain
present. Pusher stiffness and damping remain authored in the generated USD.
The policy observation's pad contact sensor remains on `/pusher/pad`, while
training additionally monitors `/body`; collisions from child frame, arm, and
swept-propeller colliders resolve through that body. The pusher joint remains
`/pusher_joint`. `make test-integration-press-contact-chain` exercises physical
contact, button travel, production latch/lamp behavior, and retreat through a
test-local no-dones environment. Contact in that gate does not invoke or reset
the approach transition.

## Training, initialization, and export

`configs/train_press_approach_quad.yaml` selects the press task, the
`approach` Hydra option, ten-second episodes, and the `press-approach` policy
namespace. The actor and critic remain independent `[256, 256, 128]` MLPs with
ELU activation; the actor still maps 34 values to four action means.
`rl.training` owns the explicit training mode, PPO settings, rollout length,
checkpoints, seed, task ID, vehicle ID, and policy namespace. Scratch training
uses 32,768 parallel environments, 300 iterations, 32 steps per environment,
an adaptive `1e-4` learning rate, `0.0001` entropy coefficient, five epochs,
eight minibatches, and checkpoints every ten iterations. This is approximately
315 million transitions and about 35 minutes of PPO on the measured GPU. A
49,152-environment run remains an explicit capacity probe rather than the
production default.

A new run writes checkpoints and `policy_meta.json` below
`.build/training/<policy-name>/<run-id>/`. Before it constructs the
environment, training loads
`rl.training.alignment_artifact` (defaulting to the controller alignment
namespace's `best` link), validates vehicle, airframe, cadence, interface, and
action dimension, and stores the alignment digest, run identity, fingerprints,
quality status, and complete settings in `policy_meta.json`. Normalized action
scales remain task-owned and may differ from the scale used by the
policy-independent alignment probe experiment. This happens before PPO begins.
An explicit `ALIGNMENT_ARTIFACT` or
`--alignment-artifact` overrides the configured link for reproduction.
After PPO, `make train-rl` automatically evaluates every saved checkpoint,
exports the highest-scoring checkpoint, always advances the relative `latest`
link, and advances `best` only when the score strictly improves within the
same comparison fingerprint.

`training_mode: scratch` forbids `--initialize-checkpoint`. After rsl_rl builds
the actor and critic from random weights, the task-neutral
`scratch-neutral-output` method orthogonally initializes the final actor layer
with gain `0.01` and zero bias. The initial action mean is therefore near zero,
while Gaussian exploration starts at standard deviation `0.03`. Metadata
records this initialization method before PPO begins.

`training_mode: actor-only` requires `--initialize-checkpoint` and copies only
compatible `actor.*` tensors into a fresh run. Critic, optimizer, exploration
standard deviation, and iteration counter remain fresh. The separate
`configs/train_press_approach_quad_finetune.yaml` retains the former
50-iteration, fixed `1e-6`, 0.01-noise, zero-entropy recipe:

```console
make train-rl HYDRA_CONFIG=configs/train_press_approach_quad_finetune.yaml \
  RL_RUN_ID=<run-id> \
  TRAIN_ARGS='--initialize-checkpoint /isaac-sim/proj/.build/training/press-approach/<source-run>/logs/model_<iteration>.pt'
```

The source may be any validated
`.build/training/<source-policy>/<run>/logs/model_N.pt`; observation/action
dimensions and actor tensor layout must match. Initialization provenance
records method, source policy, source stage, source run, iteration, path, and
source commit.

Manual export remains restricted to the configured namespace that owns the
checkpoint:

```console
make train-rl HYDRA_CONFIG=configs/train_press_approach_quad.yaml \
  TRAIN_ARGS='--export-checkpoint /isaac-sim/proj/.build/training/press-approach/<run-id>/logs/model_<iteration>.pt'
```

Manual export does not publish either label.

## Artifact and curriculum contract

An artifact is `policy.pt` plus `policy_meta.json`. The metadata owns
`policy_interface`, observation schema/dimension, policy rate, action
kind/dimension/scales, aligned controller settings and provenance, source
commit, and dirty-worktree status. A `press-v1`
artifact also requires `curriculum_stage: approach`. `press-v1` describes the
shared 34-by-4 family; `curriculum_stage` selects stage-specific evaluation and
runtime completion.

Malformed, missing, non-finite, stage-less press, unknown-stage press, and
stage-mismatched artifacts fail during loading, selection, evaluation, and
runtime preflight. Historical press or hover artifacts without alignment
provenance are unsupported and are never upgraded into the current comparison
generation.
Checkpoint selection, manual export, and independent evaluation reconstruct
the exact controller settings embedded before PPO. Runtime deployment validates
that record, while its OFFBOARD mapper continues to consume only
`PolicyContract`.

## Automatic checkpoint selection and independent evaluation

`make train-rl` performs checkpoint selection automatically. To repeat
selection for an existing run:

```console
make select-rl-checkpoint \
  HYDRA_CONFIG=configs/train_press_approach_quad.yaml \
  RL_RUN_ID=<run-id>
make eval-press RL_RUN_ID=<run-id>
```

Selection evaluates every numeric checkpoint, including iteration zero and the
final checkpoint, on 100 deterministic episodes for each seed 101, 202, and
303. It records:

- Approach success rate.
- Combined simulated collision rate.
- Body-collision and pusher-collision source rates.
- Other-flight-failure rate.
- Timeout rate.
- Mean closest range.
- Minimum closest range.
- Median successful completion time.
- Normalized terminal range, trajectory alignment, target-velocity tracking,
  upright, action-change, and completion-time costs.
- Every reward-term mean.

`held-out-press-task-quality-v2` subtracts configured collision, other-flight-
failure, timeout, and non-success penalties plus the six normalized continuous
costs. Higher scores are better; PPO return is never a selection input. The
95% success, 1% collision, 0.40 m minimum-range, finite completed trace, and
zero flight-envelope-failure gates are recorded as `advisory_accepted` and
never block a structurally valid publication.

```text
score = -(4.00 * collision_rate
        + 3.00 * other_flight_failure_rate
        + 2.00 * timeout_rate
        + 1.00 * (1 - success_rate)
        + 0.25 * terminal_range_cost
        + 0.20 * trajectory_alignment_cost
        + 0.20 * velocity_tracking_cost
        + 0.10 * upright_error_cost
        + 0.10 * action_delta_cost
        + 0.15 * completion_time_fraction)
```

Schema-7 `checkpoint_selection.json` records policy/stage identity, metric and
comparison fingerprint, seeds, advisory thresholds, every candidate and score
component, reward means, and the selected checkpoint. `latest` always points
to the newest completed selection. `best` changes only for a strictly higher
score under the same fingerprint; the first run under a new metric or
evaluation fingerprint initializes a new `best`.

`make eval-press` independently repeats the same workload and gates. With
`RL_RUN_ID`, it evaluates that published run; otherwise it resolves
`.build/training/press-approach/latest`.

For a logged training and evaluation run, use:

```console
make/scripts/run_training_evaluation_then_shutdown.sh \
  --hydra-config configs/train_press_approach_quad.yaml \
  --train-target train-rl \
  --eval-target eval-press \
  --label press-approach
```

The wrapper uses one run ID, serves only that run in TensorBoard, preserves the
terminal stage status, and does not shut down unless `--shutdown` is supplied.
When an agent launches training directly instead of through the wrapper, it
must likewise provide run-scoped monitoring, report the TensorBoard URL and log
directory to the user, and leave monitoring available while the run remains
active unless asked to stop it.

## Approach deployment mission

`configs/policy_press_approach_quad.yaml` runs arm -> takeoff -> hover -> goto
a safe staging point approximately 0.59 m from the switch -> hover -> make a
short goto into the approximately 0.48 m handoff range -> hover -> generic
`policy` -> return goto -> land. Splitting the approach keeps braking from the
long navigation leg outside the pusher-to-wall clearance. The staging and
return gotos request PX4-guided 0.5 m/s motion; the final 0.11 m approach
requests 0.08 m/s and has a 10 s timeout. The policy descriptor reads
`.build/training/press-approach/latest` and selects switch index zero.

The runtime preflights metadata before simulator mutation. The `approach`
adapter enters the policy stage with a zero velocity command and continues
streaming zero until the complete shared `handoff_ready` gate passes. It then
applies inference output directly, supplies the preceding action to the next
observation, tracks initial and closest range, and uses the same stable-goal
predicate as training. Completion cannot accrue before handoff and returns a
zero terminal setpoint. Contact or an invalid observation returns a policy
failure. The generic routine driver remains task-neutral and retains the
opaque final policy state. The preceding 0.48 m `goto` uses a 0.02 m completion
tolerance so its settled position remains inside the entry range.

Deployment logs machine-readable `PRESS_POLICY` JSON at every policy
evaluation, including the first-handoff marker, exact 34-value observation,
raw and clipped normalized actions, derived setpoint, stable-goal count,
contact state, and completion/failure value. `POLICY_SETPOINT` records every
setpoint successfully handed to the MAVLink port, while `ROUTINE_STAGE` records
the policy-stage start and terminal boundary. All timestamps use the same
post-startup-settle simulation clock sent to PX4 HIL.

Deployment continues to use deployable providers for pusher compression and
contact plus lamp state. No any-body collision detector or new real-drone
sensor is part of `press-v1`. The real policy may react indirectly to measured
linear or angular-rate disturbances, but that is not reliable collision
detection. If Lab evaluation passes while the live PX4 approach still visibly
contacts the wall, record the dynamics mismatch and stop; do not add collision
to the observation or artifact contract.

After `make eval-press` and the physical warehouse
`make test-integration-press-policy-transfer` acceptance pass, run:

```console
make test-integration-press-approach-routine
```

The strict gate requires stable contact-free approach completion, no lamp
transition, return flight, landing, and disarm. It is artifact-dependent and
therefore excluded from `test-integration-all`.
