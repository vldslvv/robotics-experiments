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
PX4 is bypassed in Lab.

The shared exact-average cadence derives policy stepping from `sim.physics_hz`
and `rl.policy_rate_hz`. At 250 Hz physics and 20 Hz policy rate, the scheduler
alternates 13 and 12 physics samples while `step_dt` remains 0.05 s. Exported
artifacts own evaluation and deployment cadence and action scaling.

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
`rl.press.handoff`.

The only implemented curriculum stage is the named `approach` stage. There is
no automatic promotion and no fixed multi-stage enum. Future stages add their
own Hydra option, policy namespace, transition, reward record, evaluator, and
runtime completion predicate. They may initialize a compatible actor from an
earlier checkpoint.

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
vertical offsets independently from -0.08 through 0.08 m.

`PressApproachRewardTerms` has exactly nine signed contributions:

- `range_progress = 50 * (previous_range - current_range)`.
- `alignment_error = -50 * ||relative_yz||^2`.
- `closing_overspeed = -100 * max(closing_speed - safe_target, 0)^2`.
- `upright_error = -5 * (1 - clamped_body_z_alignment)^2`.
- `action_delta = -0.004 * mean((action - previous_action)^2)`.
- `success = 150` on the successful terminal step.
- `collision = -300` on a body or pusher collision.
- `flight_failure = -100` on an ordinary flight-envelope failure.
- `timeout = -25` on truncation.

The safe wall-normal closing speed is 0.08 m/s at and beyond 0.50 m, tapers
linearly to zero at 0.44 m, and stays zero closer than that. Slower motion is
not penalized. Collision overrides success and the ordinary flight-failure
penalty; simultaneous body and pusher collision produces one collision reward
penalty. `PressApproachTerminationFlags` still retains both collision sources
for diagnostics. `Reward/` TensorBoard scalars enumerate every contribution
plus `total`. `Press/` reports completed episodes, success, combined collision,
body-collision, pusher-collision, other-flight-failure, and timeout
percentages.

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
`rl.training` owns PPO settings, rollout length, checkpoints, seed, task ID,
vehicle ID, and policy namespace.

A new run writes checkpoints and `policy_meta.json` below
`.build/training/<policy-name>/<run-id>/`. It does not export `policy.pt` or
advance `latest`. The collision-aware Stage 1 procedure actor-only initializes
from accepted run `approach-20260726d`, checkpoint 24:

```console
make train-rl HYDRA_CONFIG=configs/train_press_approach_quad.yaml \
  RL_RUN_ID=<run-id> \
  TRAIN_ARGS='--initialize-checkpoint /isaac-sim/proj/.build/training/press-approach/approach-20260726d/logs/model_24.pt'
```

`--initialize-checkpoint` starts a fresh run and copies only compatible
`actor.*` tensors. Critic, optimizer, exploration standard deviation, and
iteration counter remain fresh. A source may be any validated
`.build/training/<source-policy>/<run>/logs/model_N.pt`; observation/action
dimensions and actor tensor layout must match. Initialization provenance
records method, source policy, source stage, source run, iteration, path, and
source commit. This run uses 50 iterations, a fixed `1e-6` learning rate, 0.01
initial action noise, zero entropy bonus, and checkpoints every five
iterations.

Manual export remains restricted to the configured namespace that owns the
checkpoint:

```console
make train-rl HYDRA_CONFIG=configs/train_press_approach_quad.yaml \
  TRAIN_ARGS='--export-checkpoint /isaac-sim/proj/.build/training/press-approach/<run-id>/logs/model_<iteration>.pt'
```

Manual export does not publish `latest`. Automatic publication belongs only to
checkpoint selection.

## Artifact and curriculum contract

An artifact is `policy.pt` plus `policy_meta.json`. The metadata owns
`policy_interface`, observation schema/dimension, policy rate, action
kind/dimension/scales, source commit, and dirty-worktree status. A `press-v1`
artifact also requires `curriculum_stage: approach`. `press-v1` describes the
shared 34-by-4 family; `curriculum_stage` selects stage-specific evaluation and
runtime completion.

Malformed, missing, non-finite, stage-less press, unknown-stage press, and
stage-mismatched artifacts fail during loading, selection, evaluation, and
runtime preflight. Historical press artifacts without stage metadata are not
inferred. Legacy hover metadata can still infer `hover-v1` from its complete
contract.

## Checkpoint selection and independent evaluation

Run selection after training:

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
- Median successful completion time.
- Every reward-term mean.

Publication requires approach success of at least 80% and simulated collisions
of at most 5%. If no checkpoint passes, selection writes the diagnostic report,
fails, and does not export or advance `latest`. Ranking compares the worst
normalized acceptance margin first, then the other acceptance margin, fewer
other flight failures, fewer timeouts, faster successful completion, and
earlier iteration. If no candidate passes, the same ranking publishes the best
explicitly rejected fallback. PPO return is never a selection input.

Schema-4 `checkpoint_selection.json` uses
`selection_method: press-approach-gated-maximin-v2` and records policy/stage
identity, seeds, thresholds, every candidate, normalized margins, ranking
values, reward means, selected iteration, and `selected_accepted`. Selection
atomically advances `.build/training/press-approach/latest`.

`make eval-press` independently repeats the same workload and gates. With
`RL_RUN_ID`, it evaluates that published run; otherwise it resolves
`.build/training/press-approach/latest`.

For a logged three-stage run, use:

```console
make/scripts/run_training_evaluation_then_shutdown.sh \
  --hydra-config configs/train_press_approach_quad.yaml \
  --train-target train-rl \
  --select-target select-rl-checkpoint \
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
approximately 0.48 m from the switch -> hover -> generic `policy` -> return
goto -> land. The policy descriptor reads
`.build/training/press-approach/latest` and selects switch index zero.

The runtime preflights metadata before simulator mutation. The `approach`
adapter holds zero action until `handoff_ready` passes, then runs inference.
It supplies the preceding action to the next observation, tracks initial and
closest range, and uses the same stable-goal predicate as training. Completion
returns a zero terminal setpoint. Contact or an invalid observation returns a
policy failure. The generic routine driver remains task-neutral and retains
the opaque final policy state.

Deployment continues to use deployable providers for pusher compression and
contact plus lamp state. No any-body collision detector or new real-drone
sensor is part of `press-v1`. The real policy may react indirectly to measured
linear or angular-rate disturbances, but that is not reliable collision
detection. If Lab evaluation passes while the live PX4 approach still visibly
contacts the wall, record the dynamics mismatch and stop; do not add collision
to the observation or artifact contract.

After `make eval-press` passes, run:

```console
make test-integration-press-approach-routine
```

The strict gate requires stable contact-free approach completion, no lamp
transition, return flight, landing, and disarm. It is artifact-dependent and
therefore excluded from `test-integration-all`.
