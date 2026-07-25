# F3 training and parameter-tuning decision log

Status: TRAINING COMPLETE; ACCEPTANCE EVALUATION PAUSED (2026-07-21).

This is a chronological record of the decisions made while training the first
press policy for `Isaac-Switch-Press-Direct-v0`. Each entry first states the intuition for someone learning RL,
then gives the implementation-level reasoning and measurements.

### 1. Establish a conservative baseline

Plain-language view: Training first ran with 256 simulated drones. That made it
easy to confirm that the new task worked, but the GPU spent too much time
waiting for work: utilization was about 50%, with only 8 of 24 GB of VRAM in
use. The policy also learned that hovering until timeout was safer than trying
the switch.

Technical view: The original 256-environment, 3,000-update run produced about
6,400 environment steps/s and had an estimated duration of 75 minutes. At
iteration 269 its mean reward was about 76 and mean episode length was about
198 of 200 steps. Those values identified a long-episode local optimum rather
than task completion. The run was stopped because additional updates under the
same reward and sampling regime were unlikely to solve either the hardware
under-utilization or the behavioral optimum.

### 2. Sweep environment count to saturate the GPU

Plain-language view: Isaac Lab can simulate many independent copies of the task
at once. Increasing that number gives the GPU a larger batch of physics and
neural-network work. We tried progressively larger batches and stopped just
below the VRAM limit so training would be fast without risking an
out-of-memory failure.

Technical view: Short probes measured 768 environments at about 63% GPU,
7.6 GB VRAM, and 14,900 steps/s; 2,048 at 64-67%, 8 GB, and 37,000 steps/s;
8,192 at 72-80%, 10.1 GB, and 120,000 steps/s; 32,768 at 86-89%, 17.5 GB, and
179,000 steps/s; and 49,152 at 87-96%, 22.5 GB, and roughly 173,000 steps/s in
the probe. The final batch is 49,152 environments. It uses about 91% of the
24.56 GB device memory and normally 88-95% compute during the full run. A
larger batch was rejected because the remaining roughly 2 GB is needed for
allocator and simulator transients. This is a memory-safe saturation point,
not an attempt to reserve every byte of VRAM.

### 3. Preserve enough policy updates after increasing the batch

Plain-language view: A bigger simulation batch gathers experience much faster,
but a policy still needs repeated chances to learn from that experience. Very
short high-throughput probes would finish quickly but provide too few learning
updates.

Technical view: Equivalent-timestep candidates included 768 environments for
1,000 updates, 2,048 for 375, and 8,192 for only 94. The 32,768 and 49,152
probes used ten updates solely to measure capacity. The production choice is
49,152 environments for 300 PPO updates: it deliberately collects more total
experience than a strict timestep-preserving conversion while retaining
enough optimizer iterations to change the policy after new behaviors appear.

### 4. Remove reward farming from unsigned progress

Plain-language view: The drone should earn progress reward only when it moves
in the desired direction. If every movement near the switch pays, it can
wiggle back and forth instead of completing the task.

Technical view: Approach, compression, button, and retreat progress were made
signed differences from their previous values. Reversing a useful change now
returns the earlier reward instead of creating another positive payment. The
passive survival terms were reduced relative to task progress, and contact,
toggle, retreat, and success rewards were strengthened so the optimal return
requires interacting with the switch rather than hovering for 10 seconds.

### 5. Penalize unsafe early termination explicitly

Plain-language view: After the progress change, some policies discovered that
ending an episode quickly avoided accumulating timeout and control costs. A
failed episode therefore needed an immediate cost of its own.

Technical view: The first tuned run developed short episodes with strongly
negative returns, revealing an early-termination exploit. A configurable
`failure_weight` was added for crash, altitude, and relative-envelope failures.
Contact penalties were also kept phase-specific so sustained contact needed to
confirm the latch is not treated as arbitrary scraping. This changed failure
from an inexpensive way to truncate future costs into a dominated outcome.

### 6. Balance survival credit against the timeout objective

Plain-language view: The failure penalty initially made the drone cautious but
did not pay it enough to remain controlled while learning the longer sequence.
Small stability payments were restored, while timeout stayed costly so merely
hovering was still not the best result.

Technical view: With the first failure penalty, mean episode length remained
about 45-60 steps and return about -93 on the then-current scale. Increasing
alive and stability terms, with a proportional timeout penalty, raised episode
length toward 174. The final normalized weights are 0.025 alive, 0.025
stability, 10.0 failure, and 12.5 timeout. The terminal task payments remain
large enough that successful toggle and retreat exceed passive survival.

### 7. Correct the retreat geometry

Plain-language view: The original retreat distance overlapped the starting
contact range. That could label a drone successful without visibly backing
away. The target was moved farther from the wall so success means a real,
stable retreat.

Technical view: The initial wall-normal reset range is 0.20-0.30 m, so the
original 0.25 m retreat threshold was not beyond every starting pose. The
threshold was changed to 0.35 m and still requires five stable steps after a
confirmed toggle. Success is gated against crash, altitude, and envelope
failures in the same transition, preventing simultaneous success and unsafe
termination.

### 8. Make first-contact quality learnable

Plain-language view: Touching the button gently should be rewarded, but hitting
it hard should not receive the same bonus. The first contact now provides a
clear positive or negative teaching signal.

Technical view: At first contact, translational speed and angular-rate
thresholds classify the event. A stable first contact earns the configurable
gentle-contact reward; an unstable one receives the symmetric negative value.
This signal is emitted only on first contact and directly aligns with the 85%
first-contact-stability acceptance metric.

### 9. Keep the 32-step rollout horizon

Plain-language view: Shorter batches make each training update arrive sooner,
but they also make it harder to understand a multi-step action such as
approach, press, confirm, and retreat. A trial with shorter rollouts was faster
but learned much worse behavior.

Technical view: Reducing `num_steps_per_env` from 32 to 16 lowered the projected
wall-clock duration to about 23 minutes. Mean episode length nevertheless
collapsed to roughly 33 steps by iteration 27, compared with more than 160 in
the longer-horizon run. The 32-step rollout was restored. With 49,152
environments it still forms a very large on-policy batch, while exposing PPO
to longer temporal fragments of the task sequence.

### 10. Reduce exploration after observing noise growth

Plain-language view: Exploration noise helps a policy discover new actions,
but too much makes precise contact unreliable. The original policy became more
random during training instead of settling into controlled presses.

Technical view: With entropy coefficient 0.01 and initial action-noise standard
deviation 0.7, learned noise rose from about 0.70 to 0.77. The entropy
coefficient was reduced to 0.001 and initial noise to 0.3. This retains
stochastic exploration across 49,152 diverse rollouts while reducing violent
contact and making the deterministic exported mean policy more representative
of training behavior.

### 11. Stabilize PPO value learning

Plain-language view: One run improved quickly, then a single oversized update
destroyed the behavior. Rewards were rescaled and the learning rate lowered so
each update would be easier for the neural network to absorb.

Technical view: The lower-noise run suffered a catastrophic update around
iteration 14: surrogate loss reached about 0.43, critic loss exceeded 4,800,
and mean episode length fell to about 12. Adaptive KL control did not recover
by iteration 24. All reward weights were divided uniformly by ten, preserving
their relative optimization preferences while reducing value targets and
critic gradients under the shared gradient clip. Learning rate was reduced
from 5e-4 to 1e-4. The next run held critic loss near 6-16 without the collapse;
by iteration 51 it had positive mean return and near-full episode length.

### 12. Enforce a genuinely one-shot toggle reward

Plain-language view: The switch can physically toggle again after release and
re-press. The policy learned to repeat that motion for bonus points instead of
retreating after its first success. Only the first confirmed toggle in an
episode should count as the task achievement.

Technical view: The physical latch continues matching the production scalar
latch, including later release/re-press transitions, but task state separately
records whether a toggle has ever occurred. Toggle reward is now emitted only
when `toggled` is true and `toggle_time` was previously unset. Phase remains
post-toggle thereafter, even if the physical lamp toggles again. A regression
test covers release and re-press and verifies no second bonus. Retreat progress
and terminal success weights were increased to 30 and 50 respectively so the
post-toggle optimizer incentive aligns with the 95% retreat acceptance gate.

Why not pay continuously for the latched state: a one-shot event reward marks
the causal achievement without making return depend on how long the episode
continues. A large per-step `lamp_state` reward could teach the policy to toggle
once and loiter instead of terminating through retreat; paying a smaller value
after regression could still make toggle cycles or episode-boundary behavior
profitable. Increasing the one-shot latch weight remains a valid future
experiment if evaluation shows frequent stable contacts but few confirmed
toggles. It is not the first response to a retreat failure, and an excessively
large latch bonus could encourage unsafe impacts or recreate large critic
targets. The phase-conditioned metric breakdown determines which weight is
responsible.

### 13. Run the final high-utilization configuration

Plain-language view: The final run combines the fastest memory-safe simulation
batch with the stable learning settings and corrected task incentives. It will
only be accepted after deterministic evaluation, because training reward alone
does not prove reliable task completion.

Technical view: The final command is:

```console
make train-rl HYDRA_CONFIG=configs/train_press_quad.yaml \
  TRAIN_ARGS='--num-envs 49152 --max-iterations 300'
```

The run uses 32 steps/environment/update, learning rate 1e-4, entropy
coefficient 0.001, and initial action-noise standard deviation 0.3. Its policy
artifact is evaluated with deterministic actions over three non-training seeds
and 100 episodes per seed. Final acceptance measurements and any subsequent
rerun decision are recorded below when the gate completes.

## Why the acceptance evaluator is separate from training

Plain-language view: Training reward is a teaching signal, not a task score.
It combines small payments and costs for staying level, approaching, touching,
compressing, using smooth actions, timing out, and failing. Training also adds
random action noise to help the policy explore. A drone can therefore improve
its average reward by approaching the switch or hovering safely without
reliably completing the whole sequence. The acceptance evaluator tests the
exported policy without exploration noise and counts the outcomes people
actually care about.

Technical view: `make eval-press` loads the exported deterministic policy mean
from `.build/training/press/latest`. It runs 100 episodes for each of seeds 101,
202, and 303; none is the training seed 42. Separating evaluation seeds checks
that the policy generalizes across the configured reset distribution instead
of exploiting one pseudorandom trajectory. Deterministic inference matches the
artifact that downstream control will execute and removes stochastic training
noise from the measurement. The evaluator reports confirmed-latch rate across
all 300 episodes, first-contact stability conditioned on episodes that made
contact, and stable-retreat rate conditioned on episodes that toggled. It also
reports median time-to-toggle, peak compression overshoot, and mean contribution
of every reward term for diagnosis. The gate passes only at 80% latch success,
85% stable first contact, and 95% post-toggle retreat. These conditional
denominators distinguish an approach failure from a contact-quality failure or
a post-toggle control failure, which one aggregate PPO return cannot do.

Evaluator performance note: the Lab environment is already on `cuda:0`, but
the first evaluator attempt spent 1,514 seconds without completing seed 101
and showed an idle GPU. Inspection found 32 forced CUDA synchronizations per
policy step: reward sums and active counts called `.item()` separately for all
16 reward terms. The attempt was stopped without changing the policy. Reward
vectors and their common sample count now accumulate as CUDA tensors and
transfer to the host once per seed; tests preserve the same aggregate contract.
A second attempt still did not complete seed 101 in 309 seconds because the
generic deployment loader moved observations through NumPy, ran TorchScript on
CPU, copied actions back to CUDA, and the Python loop converted a CUDA active
flag to `bool` every step. That attempt was also stopped without modifying the
policy. The final evaluator path validates the artifact through the public
loader once, loads the same TorchScript actor on CUDA for rollout inference,
keeps observation/action tensors on-device, uses the fixed configured episode
horizon, and performs one completion check afterward. If evaluation remains a
material bottleneck, a later optimization could preserve three seeded
partitions in one 300-environment scene or run seed processes concurrently
subject to their combined VRAM cost.

Fabric-cloning context: USD remains the flexible authored scene graph, while
Omniverse Fabric is an optimized runtime representation. With
`clone_in_fabric=True`, Isaac Lab constructs replicated PhysX environments
through Fabric; ordinary USD inspection of the clones is then limited and
USDRT is required. This is distinct from `replicate_physics=True`, which lets
the environments share one physics schema/template. The press task keeps
physics replication enabled but Fabric cloning disabled because setup modifies
the source stage through USD and the real cloned pad contact-reporting path was
validated under regular USD cloning. Fabric cloning primarily reduces scene
creation cost; it is not a general switch that moves per-step contact physics
from CPU to GPU.

How to detect this class of bottleneck: tensor device inspection is only the
static check. The preferred runtime evidence is an NVIDIA Nsight Systems CUDA
timeline with NVTX ranges around `env.step`, observation packing, inference,
metric aggregation, and termination checks. Host-to-device/device-to-host copy
events, CUDA synchronization APIs, or a host thread blocked between short GPU
kernels identify round trips and forced waits. PyTorch Profiler with both CPU
and CUDA activities provides a narrower code-level trace; relevant operators
include `aten::item`, `aten::to`, `aten::_to_copy`, and `aten::copy_` alongside
CUDA memcpy/synchronization events. `torch.cuda.set_sync_debug_mode("warn")` is
a useful quick detector. Wall-clock time should also be compared with CUDA-event
time over the same named region: large wall time with little device time points
to host work or synchronization. `nvidia-smi` utilization alone is only a
symptom and cannot attribute the responsible source operation.

## Artifact provenance and storage decision

Plain-language view: A model file is useful only if we can identify the code
that created it. Exported metadata now records the Git commit. It also says
whether the working tree had uncommitted changes, because a commit hash by
itself would otherwise give false confidence that the exact model architecture
can be reconstructed. Model binaries remain outside Git.

Technical view: Future `policy_meta.json` sidecars contain `source_commit` from
`git rev-parse --verify HEAD` and `source_dirty` from porcelain Git status.
Accepted policies should be trained from a clean revision so the commit is a
complete source identifier. This run records base commit
`c6ef12cadcedaf28c50bebdf3535c0f927b34ab9` with `source_dirty: true`; it is
therefore a diagnostic artifact, and the base hash alone is not an exact source
reconstruction. The current local convention remains an immutable timestamped
directory plus the `latest` symlink under `.build/training/press/`. For durable
or multi-machine use, preserve the deployable policy, metadata, final PPO
checkpoint, resolved config, evaluation report, container image digest, and
checksums in object storage or a backed-up NAS. DVC is a reasonable later layer
when Git-linked push/pull and shared artifact versioning are needed; a model
registry is preferable when aliases, approvals, promotion stages, and searchable
lineage become requirements. No new storage tool is introduced by F3.

## Pause checkpoint and exact continuation

Plain-language view: The high-utilization training run finished and the model
is safely stored. Evaluation was stopped before it finished its first group of
100 episodes, so there are no honest success, contact-stability, or retreat
percentages yet. No further training will start automatically. Continuing later
means rerunning evaluation from the beginning; the trained model does not need
to be regenerated.

Technical view: Training completed 300 PPO updates with 49,152 environments in
3,242.884 seconds. The artifact directory is
`.build/training/press/20260721T231700Z.19`; its deployable files are
`policy.pt` and `policy_meta.json`, its final resumable checkpoint is
`logs/model_299.pt`, and `.build/training/press/latest` resolves to this run.
The final printed update had mean reward 2.15, episode length 199/200, action
noise standard deviation 0.18, critic loss 6.56, and surrogate loss 0.0083.
During training, 49,152 environments normally used 88-95% GPU compute and
22.4-23.1 of 24.6 GB VRAM.

Three evaluator attempts were intentionally stopped and produced no acceptance
sample:

- `.build/docker/integration/eval-press.20260722T001111Z.193721` ran
  1,514.341 seconds before the per-reward CUDA synchronization bottleneck was
  diagnosed.
- `.build/docker/integration/eval-press.20260722T003728Z.208124` ran 309.002
  seconds before CPU TorchScript inference and the per-step CUDA boolean
  synchronization were removed.
- `.build/docker/integration/eval-press.20260722T004332Z.211388` used the final
  CUDA-native evaluator and was stopped at the user's request after 1,041.674
  seconds, before seed 101 completed.

Because the evaluator aggregates only complete episodes over all three seeds,
none of these interrupted attempts supplies a valid partial rate. The required
thresholds remain at least 80% confirmed latch success, 85% stable first
contact among contacted episodes, and 95% stable retreat among toggled
episodes. Resume exactly with:

```console
make -o generate-drone-usd eval-press
```

The evaluator restarts seeds 101, 202, and 303 with 100 deterministic episodes
each. If it fails, record the exact aggregate and reward-term means here before
changing parameters. Do not infer failure percentages from the training reward,
and do not start another training run without a new explicit decision. Before
validation, also rerun `make -o generate-drone-usd
test-integration-press-contact-chain`: the earlier passing evidence predates the
final 0.35 m retreat threshold.

### 14. Diagnose and repair the ten-hour evaluator hang

Plain-language view: The fourth evaluation launch did run the first group of
episodes, but it printed nothing while doing so. It then tried to build a new
simulator for the next random seed. Isaac Lab cannot safely stop and recreate
that simulator inside the same application process: its stop callback waited
for the timeline to start while the timeline was deliberately stopped. The
program therefore kept one CPU core busy forever without advancing physics.
This was a lifecycle hang, not slow policy inference and not evidence that the
GPU needed a larger evaluation batch.

Technical view: Run
`.build/docker/integration/eval-press.20260722T044137Z.6384` remained in seed
101's only logged scene for more than ten hours. The Python process accumulated
about ten CPU-hours at 100% of one core while its 32 Carb tasking workers slept;
the GPU held about 6.1 GB and was sampled near 16% utilization. A live GDB frame
walk resolved the active call chain through `SwitchPressEnv.__init__`,
`DirectRLEnv.__init__`, `SimulationContext.__init__`,
`SimulationContext.stop`, and `_app_control_on_stop_handle_fn`. The final
callback contains a `while not timeline.is_playing(): render()` loop. The
evaluator's former `_run_seed` created and closed a complete environment for
every seed, so the second construction emitted STOP during initialization and
entered that loop. The absence of a seed-completion message had hidden that
seed 101 had already returned far enough for the second construction to begin.

The evaluator now creates one 100-environment scene with the first configured
seed, validates and loads the CUDA TorchScript policy once, and calls
`env.reset(seed=...)` for seeds 101, 202, and 303. Isaac Lab's reset method
reseeds Torch, NumPy, Python, and Replicator before resetting all environment
instances, so this preserves the intended deterministic non-training seed
partitions without recreating `SimulationContext`. The scene is closed once in
an outer `finally` block. Each seed now logs start, ten evenly spaced progress
snapshots, elapsed time, cumulative contact/toggle/retreat counts, and a
per-seed summary. Progress snapshots consolidate their CUDA counters into one
host transfer, keeping the earlier synchronization fix intact.

The orphaned run was stopped only after confirming that its original
evaluate-then-shutdown shell was no longer present; no shutdown command was
executed. Host regression tests cover reseeding the existing environment,
single construction and close across all seeds, metric denominators, and close
on rollout failure. Acceptance percentages remain pending until the repaired
three-seed run completes; the partial in-memory seed-101 result was never
printed and is not used as acceptance evidence.

### 15. Use the repaired evaluator to target contact and retreat

Plain-language view: The repaired evaluation completed in 34 seconds and
showed that the policy reliably presses the switch, but presses too hard and
never backs away. All 300 switches toggled, only 24 contacts were gentle enough,
and none of the drones completed the stable retreat. This is useful failure
information: latching does not need a larger reward. The next policy instead
needs slower command authority, a clearer consequence for the first impact,
and a stronger reason to reverse after the lamp turns on.

Technical view: Seeds 101, 202, and 303 respectively measured 100%, 100%, and
100% confirmed toggles; 6%, 10%, and 8% first-contact stability; and 0%, 0%,
and 0% stable retreat. The aggregate is 100% toggle success, 8% stable first
contact among 300 contacted episodes, and 0% retreat among 300 toggled
episodes. Median time-to-toggle was 0.700 s and peak compression overshoot was
0.00200 m. The signed mean retreat contribution was negative (-0.000943 per
active sample), while success reward was exactly zero; the deterministic policy
continued pressing instead of discovering the post-toggle reversal.

The first evidence-based tuning cycle makes three changes. The press experiment
overrides `max_body_velocity_mps` from the shared navigation-scale 3.0 m/s to
0.5 m/s. The switch begins only 0.20-0.30 m away, so the larger scale made a
small normalized action capable of exceeding the 0.25 m/s first-contact limit;
0.5 m/s still permits the full 0.35 m retreat within the ten-second episode.
The symmetric one-shot `gentle_contact_weight` increases from 5 to 20, making
an unstable first impact cost the same magnitude that a stable one earns. The
signed `retreat_progress_weight` increases from 30 to 100, so any immediate
post-toggle motion away from the wall supplies a dense gradient before the
sparse 50-point success event. Toggle reward remains one-shot at 10: raising it
would reinforce the behavior that already passes at 100% and could further
favor aggressive impact. The changes are scoped to the press experiment and
press task, preserving the hover policy's action scale and artifact contract.

### 16. Correct the gentle-touch local optimum and preserve checkpoints

Plain-language view: The first tuned policy solved the safety problem
completely, but became too cautious to press the button. Every drone touched
the switch gently and then waited for the episode to end. The next run gives
the drone enough speed to compress the mechanism after contact, makes button
motion and the confirmed toggle more valuable than merely touching, and makes
backing away the largest remaining opportunity. It also saves several points
during training, so a good intermediate policy is not lost if later PPO
updates move away from it.

Technical view: The 300-update run
`.build/training/press/20260722T150005Z.19` completed in 2,806.864 seconds with
49,152 environments. GPU samples were normally 87-97% compute and 20.8-21.6
of 24.6 GB VRAM. Its initial export failed after training because Git rejected
the root-owned container's access to the UID-1000 bind-mounted repository as
unsafe. `model_299.pt` was already complete, so no PPO work was repeated. Git
provenance commands now set `safe.directory` only for their individual
invocation, and a tested `--export-checkpoint` path rebuilt the two-environment
runner, loaded policy weights without optimizer state, and exported the
original checkpoint in 9.363 seconds. Metadata records action scale 0.5,
source commit `c6ef12cadcedaf28c50bebdf3535c0f927b34ab9`, and
`source_dirty: true`.

Deterministic evaluation then measured 0% toggles, 100% stable first contact
among all 300 contacted episodes, and 0% retreat. Compression, button, toggle,
retreat, and success reward means were exactly zero. Gentle-contact reward
averaged +0.100503 per active sample while timeout averaged -0.062814, exposing
the local optimum: the +20 one-shot contact event plus alive/stability return
outweighed the -12.5 terminal timeout without requiring compression.

The second tuning cycle increases the press-only velocity scale from 0.5 to
1.0 m/s, still one third of the shared navigation scale. Gentle-contact weight
decreases from 20 to 10; its symmetric unsafe-contact penalty remains material,
but no longer dominates task completion. Compression and button progress each
increase from 50 to 100, confirmed toggle from 10 to 30, signed retreat
progress from 100 to 200, and terminal success from 50 to 100. Timeout rises
from 12.5 to 25, while unsafe failure rises further from 10 to 30 so crashing
cannot become the cheaper way to avoid timeout. These changes retain the
one-shot latch contract and signed progress anti-gaming behavior.

Checkpoint interval decreases from 250 to 50 updates. This does not change PPO
optimization; it preserves policies at updates 50, 100, 150, 200, 250, and the
final update. If the final deterministic mean misses a gate after an
intermediate policy learned it, each checkpoint can be exported and evaluated
in seconds instead of rerunning the entire 46-minute experience collection.

### 17. Pause before executing the second tuning cycle

Plain-language view: The second set of parameter changes has been checked but
has not been trained. A launch began while the stop request arrived and was
ended immediately, before it saved a model. The last usable model and its
evaluation are unchanged. This is an intentional pause point, not a failed
learning result, so the empty launch must not be compared with either trained
policy.

Technical view: The launch created integration directory
`.build/docker/integration/train-rl.20260722T155518Z.241125` and run directory
`.build/training/press/20260722T155523Z.19`. Its container was stopped after
about 19 seconds, before any checkpoint or artifact file was written. The
command exited with status 143, and no training or evaluation metrics were
produced. `.build/training/press/latest` still resolves to the first tuned run,
`20260722T150005Z.19`, whose deterministic result is 0% confirmed toggles, 100%
stable first contact, and 0% stable retreat.

At the pause, the unexecuted second-cycle configuration used action scale 1.0
m/s; gentle-contact weight 10; compression and button progress 100; toggle 30;
retreat progress 200; terminal success 100; failure 30; timeout 25; and
checkpoint interval 50. The contact-chain evidence in the next entry revises
only the staged action scale before any PPO update. Resume the current exact
experiment only after an explicit decision with:

```console
make train-rl HYDRA_CONFIG=configs/train_press_quad.yaml \
  TRAIN_ARGS='--num-envs 49152 --max-iterations 300' \
  -o generate-drone-usd
```

After training, run the full deterministic evaluator before changing another
parameter. If the final checkpoint fails, export and evaluate the saved
50-update checkpoints to determine whether PPO learned and later forgot a
passing policy. Do not treat the interrupted empty directory as a resumable
checkpoint.

### 18. Preserve enough action authority for the real latch

Plain-language view: Slowing the drone helped the learned policy touch safely,
but the next speed limit also has to let the physical vehicle push the real
button far enough. The deterministic hardware-chain test showed that limits of
1.0 and 1.5 m/s could touch and compress the button but could not reach its
10 mm switch point. A 2.0 m/s limit crossed the threshold once, turned on the
lamp, backed away, and stayed stable. This does not mean the drone should hit
the switch at 2.0 m/s: the script and policy can still make first contact at
0.20 m/s, then request more force only after contact.

Technical view: With identical seed, mechanics, 0.20 m/s initial approach, and
post-contact command ramp, the 1.0 m/s cap produced 4.99 mm maximum button
travel and 5.055 N peak contact force. The 1.5 m/s cap produced 7.69 mm and
7.693 N. Neither crossed the production `switches.press_threshold=0.01`, so
neither could advance the production five-sample latch or start scripted
retreat. At 2.0 m/s the gate measured 10.47 mm button travel, 12.19 mm pusher
compression, 10.432 N peak force, exactly one lamp transition to intensity
5000, stable retreat beyond 0.35 m, and 0.448 rad maximum tilt. The gate passed
in integration artifact
`.build/docker/integration/test-integration-press-contact-chain.20260722T160119Z.248449`.

The staged second-cycle action scale is therefore 2.0 m/s, below the original
3.0 m/s baseline but above the measured physical feasibility boundary. This
change was made before training and has no policy metric yet. The first-contact
reward and acceptance checks still use actual measured velocity, so the larger
action range does not redefine a gentle contact; it makes safe approach and
post-contact force control a learning problem instead of making the real latch
unreachable under action saturation.

### 19. Keep commanding generic and identify the policy interface in metadata

Plain-language view: The flight commander should not need a new command every
time a neural network learns a new skill. Hovering and pressing therefore both
remain `kind: policy`. The model's small metadata file now says which input
contract it expects (`hover-v1` or `press-v1`), and the runtime connects the
right observations and completion checks. This is similar to labeling a tool's
connector: the person scheduling the job still says "use this tool," while the
connector explains how this particular tool is wired.

Technical view: `policy_meta.json` now includes `policy_interface` in addition
to observation schema/version, action contract, policy rate, action scales,
`source_commit`, and `source_dirty`. The commanding layer receives a
`PolicyAdapter` and stores opaque frozen state. Each transition receives a
`PolicyInput(vehicle_state, sim_time_us, task_observation,
reference_position_enu)` and returns a
`PolicyResult(state, setpoint, complete, failure_reason)`, so commanding owns
cadence and OFFBOARD transport without importing `PressObservation` or
branching on task IDs. The runtime selects `hover-v1` or `press-v1` from
metadata. Exact 13-value hover schema-v1 artifacts without the new field retain
a narrow legacy inference; a 34-value press artifact without it fails closed
and must be re-exported. This avoids silently interpreting a future 34-value
policy as the switch task merely because its tensor has the same length.

No PPO parameters changed in this decision. The next exported training artifact
will carry `press-v1` automatically. The current first-tuned artifact predates
the field and remains unsuitable regardless because its accepted metrics are
0% toggles, 100% stable first contact, and 0% retreat.

### 20. Build the deploy-time observation and completion logic against Isaac Sim 5.1

Plain-language view: A trained policy cannot press a real simulated switch from
its model file alone. During the flight, something must measure where the
switch is, whether the pusher touched it, how far the pusher compressed, and
whether the lamp changed. A small runtime adapter now gathers exactly those 34
values. A separate, testable state machine prevents "success" from meaning only
that the drone touched the wall: it requires one lamp change and a controlled
back-away.

Technical view: The adapter was verified against the installed
`vv-isaac-sim-custom:5.1.0` APIs, specifically
`isaacsim.sensors.physics.ContactSensor`,
`isaacsim.core.prims.SingleArticulation`, `ArticulationAction`, and
`PhysxSchema.PhysxJointAPI.maxJointVelocity`. It applies the easy-task switch
and pusher settings before standalone physics starts, reads the real stage and
articulations, converts the source asset's local `-Y` outward normal to
canonical `+X` with `Rz(-pi/2)`, and packs the schema-v1 34-value
`PressObservation`. The previous normalized action is injected from frozen
runtime state rather than guessed from the physical state.

`PressPolicyExecutionState` records handoff, first-contact evidence, previous
lamp state, transition count, and consecutive stable-retreat samples. The transition
holds a zero action until the existing handoff gate passes; rejects missing or
non-finite observation groups, an initially-on lamp, malformed policy output,
and a second lamp transition; and completes only after one transition plus the
configured retreat. The action remains the existing four-value normalized
velocity action and maps through the same `press_action_to_offboard` contract.
This is deployment safety/completion logic, not a new reward and not a change
to PPO optimization.

### 21. Add the full-flight sanity gate, but stop before new training

Plain-language view: Passing the acceptance evaluator means the model can solve
the contact task inside its training-style environment. It does not prove that
the entire vehicle can take off, navigate to the switch, hand control to the
model, return, and land through PX4. A second test now covers that full journey.
It is intentionally waiting for a good model; running it with the current
gentle-touch policy would only reconfirm a known failure.

Technical view: `configs/policy_press_quad.yaml` and routine
`policy_press.yaml` define arm -> takeoff -> hover -> goto handoff -> hover ->
generic policy -> return goto -> land. `make test-integration-press-routine`
requires stable first contact, exactly one off-to-on toggle event, the configured
stable-retreat sample count, routine completion, landing, and disarm. The shared
`FleetState` now owns switch spawning/latching and ordered toggle events; the
light-switch extension only publishes those events to ROS, eliminating duplicate
physical ownership.

The acceptance evaluator remains the statistical model gate: 300 deterministic
episodes over three non-training seeds, with minimum 80% confirmed toggles, 85%
stable first contact among contacted episodes, and 95% retreat among toggled
episodes. The full-flight routine is a system-integration gate run only after
those percentages pass. This separation matters because a single successful
mission can miss a fragile policy, while 300 Lab episodes cannot expose PX4,
OFFBOARD, live-observation, or landing integration mistakes.

Per the pause request, no new training cycle, checkpoint export, acceptance
evaluation, or artifact-dependent mission run was started during this work.
Resume with the exact command in entry 17, run `make -o generate-drone-usd
eval-press`, and only then run `make test-integration-press-routine` if all
three evaluator thresholds pass.

The shared hover-policy regression gate initially exposed a narrow legacy
metadata bug: adapter selection inferred `hover-v1`, but vector validation
reread the unnormalized sidecar and rejected the same artifact. Both paths now
use one metadata reader, guarded by a host test. The rerun passed the complete
PX4 arm/takeoff/policy-hover/land routine in artifact
`.build/docker/integration/test-integration-press-policy.20260722T174538Z.335593`.
The real contact-chain gate also passed under Isaac Sim 5.1.0 in
`.build/docker/integration/test-integration-press-contact-chain.20260722T174404Z.332722`.
Neither command performed PPO updates.

### 22. Run the existing gentle-contact policy faithfully before tuning again

#### Diagnose the launch failure before changing policy behavior

Plain-language view: The failed launch did not prove that the model was broken.
The simulator stopped while deciding how to connect the model, before the model
received an observation or commanded the drone. The visible Fabric warning and
the warehouse's missing optional quadcopter payload were unrelated startup
messages. The actionable error was the missing policy-interface label in the
model's metadata.

Technical view: `spawn_fleet()` reached `load_policy_metadata()` for
`.build/training/press/20260722T150005Z.19` and raised because its schema-v1
sidecar predates `policy_interface`. The failure happened after stage mutation,
which also exposed a cleanup weakness. The loaded application and APIs were
verified against `vv-isaac-sim-custom:5.1.0`; no Isaac Sim 6.x behavior or
extension identifier was used.

#### Infer only the one unambiguous legacy press contract

Plain-language view: Rewriting the saved model's label by hand would make this
one copy start, but would hide the rule needed to interpret other older copies.
The loader now recognizes this old model only because all of its identifying
measurements agree with the press-policy connector. If any identifying value is
different, startup still stops and asks for a proper export.

Technical view: Missing-interface inference selects `press-v1` only for schema
version 1, observation dimension 34, policy rate 20 Hz, velocity action kind,
action dimension 4, and all four action-scale fields. It emits a runtime warning
and does not modify `policy_meta.json`. Explicit interfaces and the existing
13-value legacy-hover inference are unchanged; incomplete or ambiguous tuples
still fail closed.

#### Preserve the action scale that the network learned

Plain-language view: A neural network outputs a normalized request, not meters
per second directly. Multiplying its old output by 2.0 instead of the trained
0.5 would make every translation command four times stronger. That might look
more decisive, but it would no longer answer the question "what did the trained
policy learn?" The diagnostic flight therefore uses 0.5 m/s.

Technical view: The artifact sidecar records
`action_scales.max_body_velocity_mps=0.5`, while the staged next-training config
uses 2.0 m/s because the real contact-chain test found that authority necessary
to latch. `configs/policy_press_quad.yaml` now uses 0.5 so strict metadata
validation passes without rescaling the deployed policy. The future training
config remains at 2.0; promoting a future artifact requires updating the
deployment scale in the same change.

#### Make a failed attempt produce useful behavioral evidence

Plain-language view: "Timed out" alone cannot distinguish a motionless drone
from one that approached gently, touched the switch, and merely failed to push
far enough. The runtime now keeps a small scorecard of the attempt: whether the
handoff happened, how close the drone came, its largest action and pusher
compression, first-contact quality, lamp changes, and retreat samples.

Technical view: The frozen `PressPolicyExecutionState` adds initial and minimum
wall-normal range, peak pusher compression, and peak absolute normalized action.
These are deployment diagnostics only and do not alter the 34-value observation,
policy weights, rewards, or action mapping. Generic commanding still treats
state as opaque, but includes its final representation in a policy timeout
instead of the incorrect goto-specific "did not reach the waypoint" message.

#### Fail before launching containers and clean partial simulator state

Plain-language view: Configuration mistakes should be reported immediately,
not after waiting for Isaac and PX4 to start. If a later simulator-specific
operation fails, pressing Play again should not collide with half-created drones
or switches from the first attempt.

Technical view: `make preflight-policy-artifacts` loads the Hydra experiment and
validates every policy file, interface, schema, cadence, action contract, and
scale without importing Torch. `run` and `run-isaac` depend on that target. The
live runtime repeats preflight before stage mutation for non-Make launch paths;
`spawn_fleet()` tears down locally accumulated switches, drones, observers, and
bridge resources on exceptions. The Kit lifecycle boundary records one setup
failure, installs no PhysX callback, and supports a clean subsequent Play. The
extension manifest now directly declares the Isaac Sim 5.1
`isaacsim.core.prims` and `isaacsim.sensors.physics` dependencies it imports.

#### Keep the manual probe separate from the acceptance evaluator

Plain-language view: Watching one flight answers "does this model do anything
that resembles pressing?" It cannot answer "does it work reliably?" The
acceptance evaluator repeats the task across many initial conditions and seeds,
then counts success. It prevents a visually convincing lucky attempt from being
mistaken for a dependable policy.

Technical view: The diagnostic run classifies handoff, approach, contact,
compression, toggle, and retreat for one PX4/Isaac mission. The acceptance
evaluator remains the deterministic 300-episode statistical gate: at least 80%
confirmed toggles over all episodes, 85% stable first contact conditioned on
contacted episodes, and 95% stable retreat conditioned on toggled episodes.
Legacy compatibility does not waive those thresholds. The current artifact's
existing evaluator result remains 0% toggles, 100% stable first contact, and 0%
retreat, so a likely diagnostic outcome is gentle contact followed by timeout.
No new PPO training or acceptance-evaluation cycle is part of this decision.

#### Verification and the remaining manual Play step

Plain-language view: The corrected launch now accepts the saved model and opens
the Isaac Sim 5.1 warehouse without the old metadata failure. The automated
checks also proved that the physical switch can still latch and that the older
hover-policy route was not broken. The final live flight did not start because
the desktop required a human click to grant remote-control permission before an
automated Play click; leaving the simulators idle would provide no evidence, so
they were stopped cleanly. No claim about live PX4 press behavior is made from
that non-started attempt.

Technical view: The full pre-commit suite passed: 459 host tests passed and one
was skipped, with Ruff, format, mypy, vulture, shell, YAML, TOML, and binary
asset checks green. Taplo initially could not fetch the public pyproject schema
through the sandbox proxy and passed when rerun with network access. The real
contact-chain gate passed at
`.build/docker/integration/test-integration-press-contact-chain.20260722T182633Z.356810`
with 12.19 mm pusher compression, 10.47 mm button displacement, 10.432 N peak
force, one lamp toggle, and stable retreat. The hover-policy PX4 regression
passed at
`.build/docker/integration/test-integration-press-policy.20260722T182653Z.357371`.

The live preflight accepted the legacy artifact at 0.5 m/s and reported source
commit `c6ef12cadcedaf28c50bebdf3535c0f927b34ab9` with `source_dirty: true`.
Isaac Sim 5.1 reached "Press Play to fly the configured routine" in
`kit_20260722_183049.log` with no policy-interface, action-scale, or setup error.
The Wayland remote-control confirmation prevented an injected Play event, so no
drone was spawned and no `PressDeploymentState` metrics were produced. Isaac,
PX4, and ROS were then stopped with `make stop-all`.

To finish the one-flight diagnostic, run:

```console
make run-all-autorun HYDRA_CONFIG=configs/policy_press_quad.yaml
```

When Isaac displays the warehouse and logs that it is ready, click Play once.
The likely result, based on the 300-episode evaluator, is a completed handoff,
stable first contact, zero toggles, zero retreat, and a 20-second policy timeout.
The timeout now prints the exact approach, action, compression, contact, toggle,
and retreat progress needed to confirm or reject that expectation. Stop with
`make stop-all` after capturing it. Do not resume the staged 2.0 m/s training
cycle until this diagnostic record has been added here.

### 23. Replace the floating contact puck before any further training

#### Verify the observed missing-tool problem

Plain-language view: The drone did have a small simulated contact piece, and the
simulator extended it, but the observation that there was no visible pusher
stick was correct. The model contained only a tiny round puck floating ahead of
the frame. It had no bracket, boom, shaft, spring guide, or other visible
connection to the drone.

Technical view: The tracked X500 USDA contained only the moving rigid body
`/pusher/pad` with a 30 mm diameter by 8 mm cylinder at body X=0.11 m and a
15 mm prismatic joint. The cylinder retained OpenUSD's default Z axis while the
joint translated on X, so switch contact occurred on the cylinder's curved edge
instead of a flat forward face. Isaac Sim 5.1 evidence showed that the joint did
extend and that this simplified collision shape could toggle the switch; the
missing element was the intended mechanical geometry, not the PhysX joint.

#### Author the complete tool while preserving the software contract

Plain-language view: The drone now has a dark body bracket, an 80 mm carbon
boom, a visible metal-colored sliding plunger, and an orange flat pad facing the
switch. Existing code still looks for the same joint and contact-pad names, so
the mechanical correction does not create a second policy interface.

Technical view: The USD builder now authors fixed body geometry at
`/body/pusher/mount` and `/body/pusher/fixed_boom`, moving plunger geometry at
`/pusher/pad/plunger`, and the existing collidable sensor sink at
`/pusher/pad/geom`. The pad and plunger explicitly use cylinder axis X.
`/pusher_joint` remains a 15 mm X-axis spring-damper articulation and the
contact-sensor path is unchanged. The placeholder hexacopter receives the same
schema with an under-frame placement so its centerline front arm is not
overlapped.

#### Preserve mass and clearance rather than hiding a dynamics change

Plain-language view: Adding visible parts should not quietly make the vehicle
heavier. Weight that was previously bundled into generic frame numbers was
moved into the explicit bracket and boom. The complete X500 remains about
2.339 kg, and the pad stays safely below the propeller plane.

Technical view: The X500 fixed body is now 2.3062 kg and the moving
pad/plunger body is 0.033 kg, preserving the prior 2.3392 kg articulation mass.
The 5 g bracket and 12 g boom are included in body mass, while 10 g of the old
implicit pusher allowance was removed from the wiring residual. Static tests
require the mount to overlap the center-plate front edge, the 80 mm boom to
begin there, the plunger to remain captured at the full 15 mm extension, and at
least 10 mm vertical clearance between the pad and propeller plane.

#### Turn visible deployment into a prerequisite gate

Plain-language view: Future tests cannot pass merely because an invisible
collision shape happens to press the switch. The check now requires every
visible tool component, verifies that the pad faces forward, and confirms the
pusher starts fully extended before the drone approaches.

Technical view: `make test-integration-press-contact-chain` inspects the real
Isaac Sim 5.1 stage for a collidable mount and fixed boom, visible display
colors on mount/boom/plunger/pad, X-axis plunger and pad geometry, the preserved
pad collision API, and an initial pusher joint position equal to its upper
limit. It then runs the unchanged real contact, production latch, lamp, and
retreat assertions.

#### Verification result and training consequence

Plain-language view: The corrected tool started fully deployed, pressed the
real switch, turned on exactly one lamp, and retreated stably. No policy was
trained or evaluated. The old model learned with the floating puck, so its
results cannot validate the corrected tool; the next meaningful policy must be
trained from this geometry.

Technical view: The Isaac Sim 5.1 gate passed in
`.build/docker/integration/test-integration-press-contact-chain.20260723T035112Z.15799`.
It measured 15.00 mm initial deployment, 12.39 mm peak pusher compression,
10.61 mm button displacement, 10.471 N peak pad force, one off-to-on lamp
transition, stable retreat, and 0.454 rad peak tilt. The existing artifact at
`.build/training/press/20260722T150005Z.19` predates this contact geometry and
is now explicitly non-parity diagnostic evidence. Keep training paused until
the repository revision containing this geometry is the recorded clean source
revision for a new run.

### 24. Make the entire approach geometry collidable before retraining

#### Verify the reported collision gap

Plain-language view: The report was correct. The arm tubes had collision shapes
in the generated drone, but both press execution paths explicitly turned those
collisions off. The visible propellers had no collision shape at all. The Lab
training task had a real wall collider, while the warehouse deployment path
assumed the surrounding scene supplied a suitable wall and did not guarantee
one behind the chosen switch.

Technical view: `SwitchPressEnv._disable_boom_collisions()` and the live
`PressObservationIO` adapter set `physics:collisionEnabled=false` on every arm
boom. Propeller hubs and blades were cosmetic `UsdGeom` only. The Lab
`RigidObjectCfg` wall already used `CollisionPropertiesCfg`, but no equivalent
collision proxy was authored for `/Root/LightSwitch_<index>` during live policy
setup. This meant training and deployment could permit physically impossible
arm or rotor penetration, and their wall contracts differed.

#### Enable arm, swept-rotor, and wall collision as one contract

Plain-language view: The simulator now treats the drone's arms and the complete
area swept by each spinning propeller as solid. The training wall is checked,
and live warehouse policy execution creates an invisible solid wall immediately
behind the selected switch. A policy must therefore approach with the pusher
instead of passing the airframe or propellers through the surface.

Technical view: Both boom-disabling paths were removed. The USD builder now
authors one invisible cylinder collider at each
`/body/arms/*/prop_collision`, using the configured propeller radius and
thickness. The collider is a stationary sibling of the animated `/prop` marker;
parenting it under `/prop` made cosmetic spin rebuild the PhysX shape and
invalidate Isaac's tensor simulation view during the first manual run. Runtime
setup fails closed if any arm boom is disabled or the arm-to-prop-collider
count is not one-to-one. It also authors
`/Root/LightSwitch_<index>/PressWallCollider/geom`, positioned along the source
asset's local `+Y` behind its local `-Y` outward face. The proxy dimensions come
from `rl.press.reset.wall_half_extents_m`; no new hardcoded wall geometry was
introduced.

#### Lengthen the tool and move mass to the correct rigid body

Plain-language view: Once the propeller sweep is solid, the old short pusher
cannot reach the wall first. The X500 boom is now 265 mm long, placing the
compressed pad face about 7 mm ahead of the forward rotor envelope. The spring
housing and guide bushing stay on the fixed drone body; only the 10 g pad and
plunger move. This keeps hover behavior stable without pretending the hardware
is lighter.

Technical view: The forward rotor envelope is approximately 0.342 m from the
body origin. The compressed pad face is at 0.349 m and the deployed face at
0.364 m. The complete articulation mass is 2.3472 kg: 2.3372 kg fixed body plus
0.0100 kg moving pad/plunger. An intermediate model incorrectly placed 0.033 kg
of spring/bushing/plunger mass on the distant moving body; the hover gate then
missed its 0.10 m limit (`last1.0s_max=0.130 m`). Moving the fixed mechanism
mass back to the fixed boom restored the gate without changing total mass or
controller gains.

#### Move reset and handoff geometry outside the collision envelope

Plain-language view: A training episode must not begin with a propeller or
pusher already inside the wall. Starts are now 0.40-0.50 m from the switch, and
the navigation-to-policy handoff accepts up to 0.50 m. The warehouse routine's
handoff point is 0.40 m from its switch.

Technical view: `rl.press.reset.wall_normal_min_m/max_m` changed from
0.20-0.30 m to 0.40-0.50 m, `rl.press.handoff.switch_range_max_m` changed from
0.35 m to 0.50 m, and the configured warehouse goto changed correspondingly.
The hover/goto side-channel switch was also moved to 0.45 m so that this
non-contact gate does not hit the 0.364 m deployed tool. While rerunning that
gate, a stale source-asset orientation was exposed: its raw `-Y` normal now
uses `Rz(-pi/2)`, matching the provider's canonical conversion.

#### Verification result and training consequence

Plain-language view: Collision hardening passed both deterministic checks. The
drone hovers and navigates normally, then a separate scripted routine contacts
the real switch with the pusher, toggles one lamp, and retreats without the arms
or propellers crossing the wall. No training or policy evaluation was started.

Technical view: Isaac Sim 5.1 hover/goto evidence is under
`.build/docker/integration/test-integration-press-hover-goto.20260723T041342Z.34723`:
hover last-window error 0.084 m, goto last-window error 0.040 m, and
ground-truth range 2.397 m to 0.458 m. Contact-chain evidence is under
`.build/docker/integration/test-integration-press-contact-chain.20260723T041406Z.35316`:
four arm colliders, four swept-propeller colliders, one wall collider, 14.04 mm
peak pusher compression, 10.90 mm button travel, 11.962 N peak force, one lamp
toggle, stable retreat, and 0.514 rad peak tilt.

The previous artifacts were trained without this collision contract and are
not parity-valid. Keep training paused. Before continuing, commit this state,
confirm the exported metadata records that clean commit and
`source_dirty: false`, rerun both deterministic gates, then start the staged
training command already documented above. The separate acceptance evaluator
is still required afterward because one scripted contact-chain success proves
mechanical executability, not learned-policy reliability across the configured
300 episodes and three non-training seeds.

### 25. Keep physics collision outside the animated propeller marker

#### Diagnose the first manual policy-run failure

Plain-language view: The manual run did start, but it broke almost immediately
inside the simulator. Each propeller's invisible safety disc had been attached
to the visible propeller marker. The runtime rotates that marker every frame to
make the propeller look as if it is spinning. Moving a collision shape in this
way caused PhysX to replace the shape while the flight runtime was still using
it. Once that physics connection was invalid, later drone and switch errors
were consequences rather than independent failures.

Technical view: The first causal warning in the Isaac Sim 5.1 log
`/root/.nvidia-omniverse/logs/Kit/Isaac-Sim Full/5.1/kit_20260723_041940.log`
occurred at `2026-07-23T04:20:30Z`:
`prim '/Root/DroneSpawned/body/arms/arm_4/prop/collision' was deleted while
being used by a shape in a tensor view class`. The following
`physics.tensors simulationView was invalidated` message explains the repeated
transform and switch-joint read failures. `IsaacArticulationIO` intentionally
writes the `RotateZ` operation on each `/prop` marker for cosmetic rotor spin;
therefore a collider below that marker violates the invariant that tensor-view
physics shapes remain structurally stable during simulation.

#### Author the swept-disc collider as a stationary sibling

Plain-language view: The visible propeller can still spin, but its invisible
safety disc now stays fixed relative to the drone arm. A circular swept area is
rotationally symmetric, so visually rotating that collision disc never added
physical accuracy. Keeping it stationary prevents the simulator from replacing
the shape.

Technical view: The USD builder now authors
`/body/arms/arm_<index>/prop_collision` beside `/prop`, with the same configured
pose, radius, and thickness. Only the hub and blades remain descendants of the
animated marker. Runtime and deterministic geometry checks were updated to
require exactly one `prop_collision` sibling per arm. Both tracked drone USDA
files were regenerated from the existing YAML definitions; no binary asset was
changed. A host-test fake was also completed with the two commanding setpoint
types imported transitively by press deployment, eliminating an
order-dependent test isolation failure exposed by the focused verification
run.

#### Add a regression that reproduces the invalidation sequence

Plain-language view: The drone articulation test now does more than check that
the drone can be read and pushed. It spins the visible rotors while physics is
active, advances the simulator, and then reads and controls the drone again. A
collider accidentally placed under the animated marker will make this test fail
in the same way as the manual run.

Technical view: `test-integration-drone-articulation` now calls
`update_rotor_spin(...)` with live tensor views, advances Kit, reads the
articulation again, and applies rotor forces. It passed in Isaac Sim 5.1 with
artifacts at
`.build/docker/integration/test-integration-drone-articulation.20260723T042751Z.46232`.
The collision-hardened hover/goto gate passed at
`.build/docker/integration/test-integration-press-hover-goto.20260723T042807Z.47008`
with 0.084 m last-window hover error, 0.040 m last-window goto error, and a
ready handoff at 0.458 m. The real contact-chain gate passed at
`.build/docker/integration/test-integration-press-contact-chain.20260723T042833Z.47715`
with four arm colliders, four stationary propeller colliders, one wall
collider, 14.04 mm peak pusher compression, 10.90 mm button travel, one toggle,
11.962 N peak force, and stable retreat.

#### Training and continuation decision

Plain-language view: Do not train yet. The simulator fault is fixed, but these
checks demonstrate mechanics and runtime stability, not that the old learned
policy can press the switch. The next step is to rerun the manual
takeoff-to-policy-to-landing sanity routine and observe whether the existing
policy attempts a plausible approach. Retraining remains paused until that
diagnostic is reviewed.

Technical view: No training or acceptance evaluation was started for this
decision. The old artifact still lacks parity with the corrected pusher and
collision geometry. Resume with
`make run-all-autorun HYDRA_CONFIG=configs/policy_press_quad.yaml`; inspect the
structured Isaac Sim 5.1 log and routine telemetry for approach, contact,
toggle, retreat, and landing behavior. If its behavior is not useful, begin a
new clean-revision training run using the staged commands in this document,
then run the acceptance evaluator. The evaluator remains the statistical gate:
it measures learned-policy reliability over the configured non-training seeds,
whereas the deterministic contact-chain test only proves that one scripted
physical press is possible.

### 26. Reward an uninterrupted latch hold and a controlled return to the handoff range

#### Interpret the manual sanity run before tuning

Plain-language view: The drone's approach looked useful, but the latest run did
not actually reach the retreat stage. It tilted beyond its safety limit before
the lamp changed. What looked like a retreat or a second charge was therefore
still the old policy acting before a successful press. This distinction matters:
we should improve both the hold objective and the future retreat objective,
rather than assume the old policy demonstrated a bad learned retreat.

Technical view: The Isaac Sim 5.1 log
`kit_20260723_150118.log` records policy start at approximately 15:02:07 and an
envelope failure one second later: tilt 38.5 degrees exceeded the configured
35.0-degree limit. No lamp transition was confirmed. An earlier attempt also
timed out with `first_contact_seen=False`, `toggle_count=0`, an initial range of
0.49459 m, a closest range of 0.36435 m, and 4.74 mm peak compression. These
observations support the user's diagnosis that the approach is partly learned
but the physical latch hold is not.

#### Add signed confirmation progress

Plain-language view: The policy now receives a small part of the latch reward
for every consecutive sample that keeps the button past its threshold. If it
lets go before confirmation, those partial rewards are taken back. Holding
steadily is therefore useful, while tapping the button repeatedly cannot build
free reward.

Technical view: `PressTaskState.press_steps` remains the exact vectorized twin
of the production `apply_button_sample` latch. The new `latch_progress` reward
is `latch_progress_weight * (new_press_steps - old_press_steps) /
confirm_steps`, gated to the period before the first confirmed toggle. A normal
sample contributes positive progress, the confirming sample contributes the
last increment plus the existing one-shot toggle reward, and premature release
contributes the negative of all accumulated increments. The configured total
confirmation-progress budget is 10.0; the one-shot toggle remains 30.0. Lamp
state alone is not paid continuously because that would favor loitering after a
toggle instead of completing the task.

#### Define retreat as a reverse approach, not clearing an obsolete threshold

Plain-language view: The old success line was 0.35 m from the switch. With the
longer pusher, the drone can still touch the wall around 0.36 m, so crossing
that line is not a real retreat. Each episode must now return to at least where
it began, along a centered, slow, stable path. Flying back toward the wall loses
the signed distance reward, drifting across the wall or moving too fast is
penalized, and overshooting the starting point earns no extra distance reward.

Technical view: `PressTaskState` now stores `initial_range_m`. The retreat target
is `max(initial_range_m, termination.retreat_distance_m)`, retaining 0.35 m only
as a minimum. Signed wall-normal progress is capped at that target. Post-toggle
lateral/vertical error receives a quadratic penalty with weight 50.0, and
linear speed above 0.20 m/s receives a quadratic penalty with weight 50.0.
Success requires the lamp to remain on, contact to be absent, range to meet the
dynamic target, lateral/vertical error at most 0.05 m, linear speed at most
0.20 m/s, angular speed at most 0.50 rad/s, and five consecutive stable samples.
The deployment transition uses the same dynamic target and stability gates so
an evaluator success and a runtime routine completion mean the same thing.

#### Verification and continuation

Plain-language view: Unit tests cover early-release reward cancellation and
show that a drone starting 0.50 m away cannot succeed at 0.40 m, while a
centered, slow return to 0.50 m can. No new training was started. The existing
model cannot benefit from a changed reward; it must be retrained after the code
and deterministic gates are clean.

Technical view: Focused NumPy transition and deployment tests pass. Before a
new full run, complete the repository quality gate and the Isaac Sim 5.1
deterministic contact-chain/hover checks. Then train from a clean revision and
run the separate acceptance evaluator over the configured 300 non-training
episodes. Acceptance remains at least 80% confirmed-latch success, 85% stable
first contact among contacted episodes, and 95% stable retreat among toggled
episodes. The evaluator is still required because reward improvement and unit
tests establish objective semantics, not statistical policy reliability.

### 27. Remove passive hover credit and always publish the selected diagnostic

#### Interpret the first 300-update checkpoint sweep

Plain-language view: Run `20260723T165833Z.19` learned to hover near or touch
the switch but did not learn a reliable press. Checkpoints 50 and 100 contacted
the switch in all 300 evaluation episodes without toggling it. Checkpoint 150
contacted in 275 of 300 episodes and toggled once. Checkpoints 200, 250, and
299 regressed to no contact. This is a hover/contact local optimum followed by
collapse, not evidence that more updates under the same objective will solve
the task.

Technical view: The schema-1 diagnostic report for
`20260723T165833Z.19` records zero accepted checkpoints. Iteration 150 has the
best normalized-gate maximin tuple because its confirmed-toggle rate is
1/300 while every later checkpoint is zero on contact-conditioned progress.
PPO return is not used to reinterpret that evidence. Under the revised
selection contract, iteration 150 is the fallback winner.

#### Penalize tilt instead of paying for survival and upright hover

Plain-language view: The task no longer gives the drone points merely for
remaining alive or upright. It now charges an increasingly strong penalty as
the drone tilts, pays much more for approaching and moving the real pusher and
button, and puts more value on completing an uninterrupted latch hold. The
training failure angle now matches the 35-degree deployment gate that ended the
manual run.

Technical view: `alive_weight` changes from 0.025 to 0.0. Stability changes
from positive upright credit to
`-stability_weight * (1 - clamp(body_z_world_z, 0, 1))^2`, with
`stability_weight` 5.0. Approach progress changes from 10.0 to 50.0, gentle
first contact from 10.0 to 5.0, compression and button progress from 100.0 to
200.0 each, and latch progress from 10.0 to 30.0. Premature release still
cancels all accumulated latch-progress credit. `crash_tilt_cos` is 0.819152,
the cosine of 35 degrees. The 2.0 m/s action authority, toggle, retreat,
success, failure, timeout, action and retreat penalties, PPO settings,
300-update run length, and 50-update checkpoint interval are unchanged.

#### Publish the best evidence without weakening acceptance

Plain-language view: Every completed selection now leaves a usable artifact for
manual diagnosis. Passing candidates still win. If none pass, the same
balanced gate ranking chooses the best fallback, publishes it, and clearly
marks it rejected. Independent evaluation still fails the workflow, and only
an independently accepted artifact proceeds to the full PX4 press routine.

Technical view: Selection method `press-gated-maximin-v2` ranks accepted
candidates first and otherwise ranks every evaluated candidate by the existing
normalized-gate maximin tuple and tie-breakers. Schema-2
`checkpoint_selection.json` adds `selected_accepted`. `policy_meta.json` keeps
`selected_checkpoint_iteration` and `selection_method` and adds
`checkpoint_evaluation`, containing the selected acceptance result, seeds,
episodes per seed, thresholds, exact metrics, normalized margins, ranking
values, and reward-term means. Metadata is strict JSON; these diagnostic fields
do not alter the runtime policy contract. `latest` now means the newest
selected artifact and may point to an explicitly rejected fallback.

#### Next run and gate sequence

Start from scratch after committing this objective change and confirming a
clean revision; do not resume a checkpoint trained with the previous reward.
Preserve the user-owned `TODO.txt` change rather than reverting or silently
including it to obtain `source_dirty: false`. Run the hover/goto and real
contact-chain gates, then use the standard wrapper with 49,152 environments,
300 updates, and 50-update checkpoints. Selection evaluates every checkpoint
over seeds 101, 202, and 303 with 100 episodes per seed. Independent acceptance
still requires at least 80% toggles, 85% stable first contact, and 95%
post-toggle retreat. Only a passing artifact proceeds to
`test-integration-press-routine`; a fallback remains diagnostic evidence while
the workflow remains failed.

### 28. Remove the early-crash return shortcut and expose press progress

#### Diagnose the second objective run

Plain-language view: Run `20260723T213105Z.38183` did not merely fail to find
the switch. It learned that crashing quickly was cheaper than continuing an
unsuccessful attempt. Mean episode length briefly reached about 173 steps at
iteration 5, then collapsed to about 18 steps by iteration 10 and stayed there
through the stopped run. Mean reward settled near -28 while no press-specific
training signals were available. Raising the adaptive learning rate would make
that locally preferred behavior learn faster, not correct what the reward asks
for.

Technical view: The run began at a configured learning rate of `1e-4`; the
adaptive schedule ranged from `1e-5` to approximately `3.84e-4` and was about
`3.38e-5` near iteration 74. A terminal failure cost 30, while an unsuccessful
ten-second attempt could accumulate the 25 timeout cost plus stability and
control penalties. Approach progress before a crash could reduce the terminal
episode loss further. This reward ordering explains the stable short-episode
solution without invoking insufficient optimizer step size.

#### Make controlled failure preferable to crashing

Plain-language view: A crash now costs substantially more than reaching the
end of an unsuccessful but controlled attempt. The policy still receives no
passive survival or upright credit, and the existing approach, contact, button,
latch, retreat, action, and stability terms are unchanged. The correction is
therefore aimed only at removing the demonstrated shortcut.

Technical view: `reward.failure_weight` changes from 30.0 to 100.0;
`timeout_weight` remains 25.0. The crash angle remains 35 degrees and the PPO
learning rate, adaptive schedule, noise, rollout, checkpoint, and action
settings are unchanged. New training starts from scratch because checkpoints
from the stopped run optimize the old return.

#### Report task percentages while PPO is running

Plain-language view: Mean reward alone no longer has to stand in for whether
the drone is reaching or pressing the switch. TensorBoard now shows cumulative
percentages for completed episodes that contacted, toggled, fully succeeded,
or failed, plus active-episode progress. Each run is served by itself and its
URL is announced only after scalar data is actually loaded.

Technical view: `SwitchPressEnv` adds `Press/contact_percent`,
`Press/toggle_percent`, `Press/success_percent`, and
`Press/failure_percent`, along with completed-episode count and active contact
and toggle percentages. Key signed reward terms are logged under `Reward/`.
These counters are training diagnostics and do not change checkpoint ranking.
The training wrapper now waits for the exact run's non-empty TensorBoard event
file before launching the server, checks the scalar-tags endpoint, and then
prints the URL and run identity. This replaces starting TensorBoard against a
log directory that did not yet exist, which could leave it reporting no active
dashboards.

### 29. Establish a deterministic press baseline before retreat training

#### Stop after the press behavior is established

Plain-language view: Run `20260723T215635Z.61663` learned to press the switch
reliably. The active training population first showed confirmed toggles at
iteration 6, exceeded 25% near iteration 29, and spent later windows mostly
between about 75% and 92%. Mean episode length remained near the 200-step
horizon instead of collapsing to the previous 18-step crash behavior. Training
was stopped after checkpoint 150 because confirmed pressing was established
while full retreat remained at zero; the immediate goal was a measurable press
rate, not the full 80% end-to-end acceptance target.

Technical view: The run started clean from source commit `0f69dfc` with 49,152
environments and the corrected 100.0 terminal failure weight. Saved
checkpoints are iterations 0, 50, 100, and 150. The run-scoped TensorBoard
telemetry showed cumulative completed-episode toggles reaching 66.2% by
iteration 150, with active-window values peaking above 90%. Cumulative failure
fell from more than 20% early in training to 5.6%. The training process was
interrupted only after checkpoint 150 was safely written.

#### Publish checkpoint 50 as the diagnostic fallback

Plain-language view: Deterministic evaluation found that checkpoints 50 and
100 both pressed the switch in every episode. Checkpoint 50 made the gentler
first contact and therefore won the fallback ranking. It still does not back
away after pressing, so it is published for diagnosis and manual press testing,
not described as accepted.

Technical view: Selection evaluated each saved checkpoint for 100 episodes on
seeds 101, 202, and 303. Iteration 50 achieved a 1.0 confirmed-toggle rate,
0.3666666667 stable-first-contact rate, and 0.0 post-toggle-retreat rate.
Iteration 100 achieved 1.0, 0.2033333333, and 0.0 respectively. Iteration 150
achieved 0.7233333333, 0.23, and 0.0. The
`press-gated-maximin-v2` selector therefore published iteration 50 with
`accepted: false` and advanced `.build/training/press/latest` to run
`20260723T215635Z.61663`.

Independent evaluation of the selected artifact reproduced 300 confirmed
toggles in 300 episodes, a 0.3666666667 stable-first-contact rate, zero retreat
and full completion, a 1.075 s median toggle time, and 1.92 mm peak compression
overshoot. It correctly failed acceptance on first-contact stability and
post-toggle retreat. The next objective should preserve this confirmed-toggle
baseline while improving approach/contact gentleness and teaching the
post-toggle reverse trajectory; increasing the toggle target or learning rate
is not the next constraint.

### 30. Fine-tune stable contact and learned retreat from the press actor

#### Preserve the learned press while resetting PPO state

Plain-language view: The next run starts from checkpoint 50's deterministic
press behavior but learns with a fresh critic, optimizer, action noise, and
iteration count. It is a new run with explicit provenance, not a continuation
of the old optimization trajectory. Checkpoint 50 remains the protected
published fallback.

Technical view: `train_rl.py --initialize-checkpoint PATH` accepts a numeric
checkpoint below the selected policy namespace, verifies the stored
observation and action dimensions, and copies only compatible `actor.*`
parameters into a freshly built runner. The exact actor keys and tensor shapes
must match. Strict-JSON metadata records the `actor-only` method, source run,
checkpoint iteration/path, and source commit. Export-only checkpoint loading is
unchanged.

#### Correct the objective around contact and retreat

Plain-language view: The actor is now strongly encouraged to slow down near the
switch, make a stable first contact, release after the lamp turns on, and back
away without lateral drift. Lingering in contact, waiting after the toggle, or
toggling the lamp again is explicitly costly.

Technical view: Gentle-contact weight is 20.0. While precontact and within
0.42 m, squared linear overspeed above 0.20 m/s has weight 20.0 and squared
angular overspeed above 0.50 rad/s has weight 5.0. Retreat progress weight is
300.0. Retreat alignment is a squared hinge beyond the existing 0.05 m
tolerance with weight 50.0. Post-toggle contact costs 0.5 per step and delay
before stable retreat costs 0.05 per step. A later confirmed toggle costs
100.0 once on its transition. Success pays 150.0 only after a stable first
contact. Approach, compression, latch, toggle, failure, timeout, action
authority, tilt gate, and signed progress cancellation remain unchanged.

#### Conservative run and evidence plan

Plain-language view: Fine-tuning uses small, fixed updates and frequent
checkpoints so deterministic press behavior can be retained if later updates
forget it. TensorBoard now directly exposes the three acceptance populations
and repeated-toggle failures.

Technical view: The press config uses a fixed `1e-4` learning rate, 0.20 fresh
action noise, checkpoints every 25 iterations, and a 150-iteration maximum.
Network architecture, 32-step rollout, clipping, entropy, GAE, and minibatch
settings are unchanged. Cumulative telemetry reports stable first contact
among contacted episodes, retreat and repeat toggles among toggled episodes,
plus all new reward means. Run with 49,152 environments from
`20260723T215635Z.61663/logs/model_50.pt`. Stop for catastrophic forgetting if
active toggles remain below 50% for ten iterations after iteration 25, or as
unproductive if retreat is still exactly zero at iteration 50. Evaluate every
25-iteration checkpoint over seeds 101, 202, and 303. Only an independently
accepted selection proceeds to `test-integration-press-routine`.

If actor-only fine-tuning still has zero retreat at iteration 50 while toggles
remain high, stop it and run one 100-iteration from-scratch comparison under
the identical corrected objective. Selection may publish a better candidate,
but the existing checkpoint-50 fallback is not overwritten merely because a
comparison run completed.

### 31. Actor-only fine-tuning learned retreat but not stable contact

#### Complete the run without triggering either stop rule

Plain-language view: The actor-only run preserved reliable pressing and began
to learn retreat just before iteration 50, so it was allowed to finish. It did
not spend ten consecutive post-25 iterations below 50% active toggles. The
from-scratch contingency was therefore not triggered.

Technical view: Run `20260723T231822Z.117549` trained 49,152 environments for
150 iterations from source commit `f15597d`, with `source_dirty: false`.
Initialization metadata identifies method `actor-only`, source run
`20260723T215635Z.61663`, checkpoint iteration 50, and source commit
`0f69dfc`. At training iteration 50, active toggles were 91.65% and cumulative
retreat among toggled episodes was 0.026%, no longer exactly zero. The run
saved periodic checkpoints 0, 25, 50, 75, 100, and 125 plus rsl_rl's final
checkpoint 149. The run-scoped TensorBoard URL was
`http://127.0.0.1:6006/`; its environment endpoint was independently verified
to identify only this run's log directory. The wrapper stopped the server on
exit.

#### Select and independently reject the best fallback

Plain-language view: Checkpoint 75 learned essentially perfect retreat but made
unstable first contact. Later checkpoints traded retreat reliability for a
small contact-stability improvement. None met all three gates, so the selector
published the best balanced fallback rather than describing it as accepted.

Technical view: Deterministic 300-episode results for iterations
0/25/50/75/100/125/149 respectively were:

- Confirmed toggles: 0.9967 / 0.9967 / 1.0 / 0.9967 / 1.0 / 1.0 / 0.9833.
- Stable first contact among contacted episodes: 0.3700 / 0.1100 / 0.1667 /
  0.1767 / 0.2033 / 0.3000 / 0.2367.
- Retreat among toggled episodes: 0.0 / 0.0 / 0.0 / 1.0 / 0.4733 / 0.0 /
  0.3356.

The maximin selector chose iteration 149 because its weakest normalized
acceptance margin, contact stability at 0.2784 of target, exceeded iteration
75's 0.2078 and iteration 100's 0.2392. It published the artifact with
`accepted: false` and advanced `latest`; the original checkpoint-50 run remains
intact. Independent evaluation reproduced 0.9833 confirmed toggles, 0.2367
stable first contact, 0.3356 retreat, 0.3300 full completion, a 0.700 s median
toggle time, and 2.00 mm peak compression overshoot. It failed first-contact
stability and retreat acceptance. Consequently
`test-integration-press-routine` was not run.

### 32. Improve joint safe completion before changing actor capacity

#### Interpret the current completion metric correctly

Plain-language view: The current network has already demonstrated that it can
press and retreat, but it has not learned to combine a gentle first contact
with a reliable retreat. More hidden layers are therefore not the next
experiment. The immediate problem is objective alignment and optimization
stability across the precontact and post-toggle phases.

Technical view: `full_completion_rate` currently counts episodes that reached
stable retreat, regardless of whether first contact was stable. Checkpoint 75
therefore achieved 0.9967 raw full completion while achieving only 0.1767
stable first contact. Iterations 75/100/125/149 then achieved retreat rates of
1.0/0.4733/0.0/0.3356 while contact stability moved to
0.1767/0.2033/0.3000/0.2367. This rapid, partly opposing movement with the
unchanged `[256, 256, 128]` actor is evidence of phase interference and
destructive PPO updates, not an established representation-capacity limit.

Add a diagnostic `safe_full_completion_rate` that requires stable first
contact, confirmed toggle, and stable retreat in the same episode. Preserve
the three existing acceptance gates because their separate denominators explain
which phase failed. Do not publish checkpoint 75 merely for its raw completion:
it remains unsafe and rejected.

#### Diagnose the failing predicate before retuning

Extend deterministic evaluation diagnostics before the next training run.
Record wall-normal, lateral, total linear, and angular velocity at first
contact; approach speed by range; the delay from toggle to contact release and
first retreat command; and the final failed retreat predicate (contact release,
distance, alignment, linear speed, angular speed, crash, or timeout). Report
per-seed distributions and margins to the configured thresholds, not only
means or booleans. Re-evaluate existing checkpoints 0, 50, 75, 100, 125, and
149 with these diagnostics.

Also inspect PPO KL, clip fraction, action standard deviation, value loss, and
explained variance when available. A KL or clip-fraction spike indicates
oversized actor updates. Persistently poor value estimates, especially early in
actor-only fine-tuning with a fresh critic, indicate a critic warm-up or
separate actor/critic learning-rate experiment rather than a larger actor.

#### Align downstream credit with stable first contact

Plain-language view: An unstable hit can currently remain profitable because it
may still collect latch, toggle, retreat-progress, and physical retreat success
credit. Gating only the terminal bonus does not make gentle contact a
prerequisite for the learned task.

Technical view: In the next objective experiment, make stable first contact a
prerequisite for downstream task credit. Prefer terminating an unstable
first-contact episode with an explicit failure-scale cost because it produces a
clear advantage signal. If preserving post-contact diagnostic experience is
more important, continue the physical episode but gate latch, toggle, retreat,
and terminal-success reward on stable first contact. Continue to report the
physical toggle and retreat events independently.

Replace the short near-contact total-speed penalty with dense braking guidance
over the complete 0.40-0.50 m reset envelope. Use wall-normal closing velocity,
with a distance-dependent target that approaches the configured contact-speed
limit near the switch. Keep separate lateral and angular control terms. Do not
begin by increasing the one-shot gentle-contact weight again: the policy needs
a useful slowing gradient before contact, not only a larger reward after the
outcome is already determined.

#### Run a conservative checkpoint-75 experiment

Warm-start from run `20260723T231822Z.117549`, checkpoint 75, because that actor
already owns reliable post-toggle retreat. Keep the observation, action
authority, and `[256, 256, 128]` MLP unchanged for the first comparison. Use a
smaller actor update, initially comparing a fixed `5e-5` learning rate and 0.10
initial action noise, save every 5-10 iterations, and evaluate a 50-75
iteration run. If KL or clip fraction remains high, compare `2e-5`, fewer PPO
epochs, or a smaller clip range. Change one material variable per comparison
and rank deterministic checkpoints by the existing acceptance gates plus the
new joint diagnostic.

If retreat is still forgotten while contact improves, preserve checkpoint 75's
post-toggle behavior explicitly with an anchor/distillation loss on
post-toggle observations or balanced precontact/post-toggle reset sampling.
This is a more direct response to catastrophic forgetting than increasing
generic network depth.

#### Escalate architecture only with phase-interference evidence

Do not add hidden layers or recurrence initially. The existing observation
already supplies pose, velocity, angular velocity, contact, compression and
compression rate, lamp state, and previous action. There is no current evidence
that missing history requires an LSTM.

If corrected rewards, smaller updates, and retreat anchoring repeatedly show
that contact improvement destroys retreat, test a shared trunk with separate
pre-toggle and post-toggle actor heads selected by lamp state. This
phase-specific architecture directly isolates the behaviors that are
interfering. Test generic actor capacity only if training and deterministic
evaluation both plateau under an aligned objective while KL, clipping, and
critic learning remain stable. If value learning alone is inadequate, change
critic capacity before actor capacity.

Use these decision signals:

- High wall-normal impact speed: improve dense braking and stable-contact
  prerequisite.
- Near-threshold contact misses: adjust shaping margin and reduce update noise.
- Alignment-dominated retreat failures: retune retreat alignment behavior.
- Large checkpoint-to-checkpoint phase swings: reduce update size, checkpoint
  more frequently, and anchor the learned phase.
- High training performance but low deterministic evaluation: inspect
  exploration dependence and train/evaluation mismatch.
- Passing Lab evaluation but failing the live routine: inspect observation,
  controller, and simulator-to-deployment parity instead of model capacity.
- Stable optimization and a repeatable capacity plateau: only then compare a
  larger actor under identical seeds, objective, and sample budget.
