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

### 33. Replace the joint objective with an explicit approach curriculum stage

Date: 2026-07-26.

Decision: Retire the approach/contact/toggle/retreat transition and train one
small, named `approach` stage first. Keep the 34-value observation, four-value
action, pusher/switch assets, and `[256, 256, 128]` actor so a later stage can
initialize from the Stage 1 actor. Curriculum progression is manual through
separate Hydra options and policy namespaces; no automatic promotion or fixed
stage enum is introduced.

Reason: The earlier joint objective showed strong phase interference:
checkpoints alternated between contact quality and retreat reliability without
meeting the joint gates. Stage 1 isolates the prerequisite flight behavior:
reach 0.40-0.42 m, align, brake, and hold stably without touching the switch.
Its reward contains only range progress, alignment, braking, upright, action
delta, and terminal outcome terms. Contact is a failure, which removes
post-contact credit ambiguity.

Implementation contract:

- Hydra option:
  `configs/experiment/rl/press/approach.yaml`.
- Training config and namespace:
  `configs/train_press_approach_quad.yaml` and `press-approach`.
- Deployment config:
  `configs/policy_press_approach_quad.yaml`.
- Acceptance: at least 80% approach success and at most 5% contact failure on
  100 episodes for each seed 101, 202, and 303.
- Ranking: worst normalized gate margin, then fewer other flight failures,
  fewer timeouts, faster successful completion, and earlier iteration.
- Metadata: `policy_interface: press-v1` plus required
  `curriculum_stage: approach`; historical stage-less press artifacts are
  rejected.
- Physical contact validation remains in
  `test-integration-press-contact-chain`, using a test-local no-dones
  environment so approach termination cannot reset the physical scenario.

Initial training decision: preserve the existing PPO topology, conservative
fixed `1e-4` schedule, and 2.0 m/s action authority for the first Stage 1 run.
An attempted reduction to 0.5 m/s was rejected before training because the
physical contact-chain gate then produced only 2.31 mm of button travel against
the 5 mm latch threshold. Precision remains the dense braking objective's job;
the shared physical/action contract retains enough authority for later stages.
Do not initialize from the old joint press actor for the first run: its learned
contact/retreat behavior is not a clean approach baseline, while the new task
is small enough to train directly.

### 34. Reduce exploration scale after the first approach run

Date: 2026-07-26.

Decision: Reject run `approach-20260726a` and repeat Stage 1 from scratch with
0.05 initial action noise, a fixed `5e-5` learning rate, `0.0001` entropy
coefficient, and 300 iterations. Keep the 2.0 m/s action authority, reward,
reset distribution, network topology, rollout length, and acceptance gates
unchanged.

Evidence: The 150-iteration baseline completed in 545 seconds. Selection over
300 deterministic episodes per checkpoint found iteration 50 best, but it
achieved only 0.2033 approach success with zero contacts and 0.7967 timeouts.
Iteration 75 achieved 0.1933 success, iteration 100 achieved 0.1833, iteration
125 failed the flight envelope in every episode, and iteration 149 timed out
in every episode. The iteration-50 mean closest range was 0.40714 m, showing
that wall-normal braking was learned while lateral/vertical correction was
not. Its mean alignment contribution was -0.16143 per active step.

Diagnosis: A generated frame-aware proportional actor passed the unchanged
evaluator at 1.000 success, zero contacts, zero other flight failures, and zero
timeouts across all 300 episodes, with a 1.225 s median completion time. The
task geometry, contact frame, action mapping, goal predicate, and evaluator are
therefore capable of passing. At the configured 2.0 m/s action scale, the old
0.20 standard deviation injected roughly 0.40 m/s per-axis exploratory
commands around a goal requiring at most 0.10 m/s total speed and offsets no
larger than 0.08 m. Reduce fresh exploration to approximately the terminal
speed scale, halve the update size, and double the checkpoint horizon to avoid
the observed post-50 collapse while preserving the shared action contract.

Publication rule correction: The rejected selector result exposed that a
fallback checkpoint was still exported and advanced as `latest`. Selection may
rank rejected checkpoints for diagnostics, but it must now write the report,
fail the make gate, and leave publication unchanged unless the selected
checkpoint passes both acceptance thresholds.

### 35. Warm-start the approach actor with the task's affine flight prior

Date: 2026-07-26.

Decision: Reject run `approach-20260726b` and actor-only initialize the next
run from an architecture-compatible affine approach prior. Fine-tune for 25
iterations with `1e-6` fixed learning rate, 0.01 fresh action noise, zero
entropy bonus, and checkpoints every five iterations. Keep the final task,
reward, reset distribution, 2.0 m/s action authority, and actor/critic topology
unchanged.

Evidence: Run b completed 300 iterations in 1,128 seconds, but its strict
selector failed as intended. Iteration 125 ranked best at 0.1967 success, zero
contacts, zero other flight failures, and 0.8033 timeouts. Direct actor probes
showed that its lateral and vertical outputs changed by only about 0.01 across
the complete 0.16 m reset width, while retaining a much larger constant action
bias. More iterations and smaller random exploration therefore reproduced the
same local optimum instead of learning feedback.

Prior: Use the unchanged `[256, 256, 128]` ELU actor as a sparse affine
controller. Positive hidden biases keep three pass-through units in ELU's
linear region. The output commands a 0.42 m wall-normal equilibrium with
normalized gain 0.625 per meter and commands lateral/vertical normalized
velocities equal to their signed switch-relative offsets; yaw is zero. This is
an initialization, not a runtime fallback or an added reward term. The prior
passed the exact 300-episode evaluator at 1.000 success with zero contact,
flight failure, or timeout and 1.900 s median completion. The final artifact
must still come from PPO checkpoint selection and independently pass all
publication and deployment gates. Initialization provenance records policy,
stage, run, iteration, and checkpoint path through the standard actor-only
contract.

The first 1.25-gain fine-tune, run `approach-20260726c`, passed Lab selection
and independent evaluation at 1.000 success and zero failures. The live PX4
routine nevertheless contacted the switch at 0.3828 m while still moving
0.1122 m/s. Halving the prior's wall-normal gain limits its commanded speed to
0.10 m/s at 0.50 m and addresses the measured velocity-loop overshoot without
changing the configured 0.20 m/s reward target or any acceptance threshold.

### 36. Publish the slower Stage 1 actor after every gate passes

Date: 2026-07-26.

Decision: Publish run `approach-20260726d`, checkpoint 24, as the accepted
`press-approach` Stage 1 artifact. Its actor-only provenance is source policy
`press-approach`, stage `approach`, run `approach-affine-prior-slow`, checkpoint
0. One identical training attempt crashed during Isaac/Kit startup before
creating a run; the immediate retry completed all 25 PPO iterations in 77
seconds.

Selection evidence: Every saved checkpoint passed 300 deterministic episodes.
Iteration 24 ranked first with 1.000 approach success, zero contact failures,
zero other flight failures, zero timeouts, 0.41244 m mean closest range, and
1.500 s median completion. Its reward means were 0.103331 range progress,
-0.070378 alignment error, -0.088123 closing-speed error, effectively zero
upright/action-delta error, 4.980631 success, zero failure/timeout, and 4.925460
total.

Independent and deployment evidence: `make eval-press
RL_RUN_ID=approach-20260726d` reproduced 1.000 success, zero
contact/fault/timeout, 0.41244 m mean closest range, and 1.500 s median
completion. `make test-integration-press-approach-routine` then completed the
handoff, five-sample stable approach, contact-free terminal zero setpoint,
return waypoint, landing, and disarm without a lamp transition. The physical
contact-chain gate also remained independent and passed with 14.02 mm maximum
pusher travel, 10.90 mm button travel, one toggle, 11.821 N maximum contact
force, and stable retreat.

Final repository validation: the full host suite passed with 578 tests and one
skip; all pre-commit hooks passed; the aggregate container suite passed 8/8
targets; and the independent `test-integration-press-policy` gate passed.

### 37. Penalize collision through privileged Lab state

Date: 2026-07-27.

Decision: Keep the deployed `press-v1` contract unchanged and add collision
only to the Stage 1 Lab transition. The policy interface remains a schema-v1
34-value observation and four normalized body-FLU velocity/yaw-rate actions at
20 Hz, with 2.0 m/s velocity and 1.5 rad/s yaw-rate scaling. `contact_flag`
continues to mean pusher/pad contact only. No body, arm, propeller, frame, or
wall collision value enters `PressApproachState`, the actor observation, the
critic input, artifact contract metadata, or deploy-time sensing.

Training mechanics: Separate PhysX contact sensors monitor the drone body and
pusher pad. Their boolean collision facts latch across every one of the 12 or
13 physics substeps in a policy interval and enter only
`PressApproachInput`. The pure transition retains both sources for diagnostics,
applies one combined collision penalty, and terminates the episode. PPO stores
the reward and terminal flag, while reset prevents collision from entering a
successor policy state.

Reward decision:

- Range progress: `50 * (previous_range - current_range)`.
- Alignment: `-50 * ||relative_yz||^2`.
- Closing overspeed: `-100 * max(closing_speed - safe_target, 0)^2`.
- Upright error: `-5 * (1 - alignment)^2`.
- Action delta: `-0.004 * mean((action - previous_action)^2)`.
- Success: `+150`.
- Collision: `-300`.
- Ordinary flight failure: `-100`.
- Timeout: `-25`.

The safe closing-speed target is 0.08 m/s at 0.50 m and tapers linearly to zero
at 0.44 m. Slower motion has no closing-speed penalty. Collision overrides
success and the ordinary flight-failure reward; simultaneous body and pusher
contact still produces one collision penalty.

Evaluation decision: Selection and independent evaluation report combined
collision rate plus body and pusher source rates. Publication continues to
require at least 0.80 approach success and at most 0.05 simulated collision
rate over 100 episodes for each seed 101, 202, and 303. Artifact identity and
inference metadata do not gain a collision field; checkpoint evaluation
results retain the diagnostic rates.

Deployment decision: Continue using the deployable pusher compression/contact
and lamp providers. Do not introduce a real-world any-body collision detector
or new drone sensor. Velocity and angular-rate disturbances may affect the
policy indirectly, but they are not reliable collision sensing. If Lab passes
and the live PX4 approach still visibly contacts the wall, record a dynamics
mismatch and stop instead of changing the policy contract.

Training plan: Actor-only initialize run `approach-20260727a` from
`approach-20260726d/logs/model_24.pt`. Train for 50 iterations with fixed
`1e-6` learning rate, 0.01 initial action noise, zero entropy, and checkpoints
every five iterations. Then run checkpoint selection, independent approach
evaluation, and the live PX4 routine only after publication.

Training and selection evidence: The 50-iteration run completed in 75.982
seconds. Selection evaluated iterations 0, 5, 10, 15, 20, 25, 30, 35, 40, 45,
and 49 over all 300 deterministic episodes; every checkpoint passed both
publication gates. Iteration 49 ranked first and was published. It achieved
1.000 approach success; zero combined, body, and pusher collisions; zero other
flight failures and timeouts; 0.408096 m mean closest range; and 1.300 s median
completion. Its per-active-step reward means were 0.128475 range progress,
-0.081518 alignment error, -0.235154 closing overspeed, effectively zero
upright and action-delta error, 5.788526 success, zero collision,
flight-failure, and timeout terms, and 5.600329 total.

Independent evaluation evidence: `make eval-press
RL_RUN_ID=approach-20260727a` reproduced the selected checkpoint's complete
300-episode result: 1.000 success; zero combined, body, and pusher collisions;
zero other flight failures and timeouts; 0.40810 m mean closest range; and
1.300 s median completion.

Deployment result: Stop on a Lab-to-PX4 dynamics mismatch. The strict live
approach routine detected real pusher contact at 0.3807 m while the vehicle was
still moving 0.0782 m/s. Landing recovery began, then the PX4 bridge actuator
poll timed out after 1.000 s; PX4 reported invalid setpoints and blind-land
failsafe. This is a failed deployment gate even though Lab selection and
independent evaluation passed. Do not add body collision sensing to
`press-v1`, do not retrain around a deploy-only collision observation, and do
not claim the collision-aware artifact as live-deployment accepted. Preserve
the run and its diagnostics for controller/simulator dynamics investigation.

Repository and integration evidence: The host gate passed with 582 tests and
one skip, and every pre-commit hook passed. The deterministic contact-chain
gate passed with 14.02 mm maximum pusher travel, 10.90 mm button travel, one
toggle, 11.821 N maximum contact force, and stable retreat. The independent
press-policy gate passed. The first aggregate integration attempt passed seven
of eight targets; multi-drone navigation hit a one-off x500 actuator-poll
timeout after the hexacopter completed. The navigation target then passed in
isolation, and a complete serial aggregate rerun passed all eight targets.

### 38. Trial the official Isaac Lab thruster and a shared governor

Date: 2026-07-27.

Decision: Restore `approach-20260726d` as the deployment baseline. Use the
official `isaaclab_contrib.actuators.Thruster` from the already pinned Isaac Lab
v2.3.2 commit for batched motor realization, without replacing the
pusher-equipped articulation with `Multirotor`. Keep Pegasus and the live PX4
SITL path as the deployment oracle. Do not adopt `rlPx4Controller`: it is a
CPU/Eigen controller with hard-coded gains, incomplete yaw-rate behavior, and
no indexed reset contract, so maintaining it would not remove the project-owned
PX4-parity work.

Dynamics diagnosis: Paired bridge/ULog evidence from the failed live run showed
healthy 4 ms actuator lag but a braking mismatch. At 0.4459 m the configured
safe target was approximately 0.008 m/s, the actor still requested 0.084 m/s,
and PX4 measured 0.103 m/s. At 0.4032 m the actor requested 0.031 m/s while PX4
still measured 0.104 m/s. The vehicle reached 0.3712 m. This is controller
response mismatch, not missed privileged collision state.

Implementation: Convert controller rotor-speed references to desired per-rotor
thrust, pass them through the official actuator, then reconstruct the existing
per-rotor yaw reaction, drag, and body wrench. Rise/fall time constants and the
thrust-rate limit live in Hydra. They initially preserve the instantaneous
Pegasus motor response with zero time constants until a deterministic paired
trace supplies measured values.

Safety decision: Apply one deterministic near-wall governor in both Lab and
deployment. It follows the reward's safe closing-speed schedule, subtracts
measured overspeed with configured braking gain, and caps reverse speed. Physics
and PX4 receive the governed action. The actor observation, stored previous
action, and action-delta reward retain the raw actor action, preserving
`press-v1`, schema 1, dimension 34, and the existing reward contract. No body
collision signal or new real sensor is introduced.

Verification and publication decision: Re-run actor-only training from
`approach-20260726d/logs/model_24.pt` for 50 iterations at fixed `1e-6`, 0.01
noise, zero entropy, and five-iteration checkpoints. Serve only that run through
TensorBoard. Selection, independent evaluation, host/pre-commit/integration
gates, and three consecutive contact-free live approach routines are required
before publication. If Lab passes but a live run contacts the wall, stop and
record the remaining dynamics mismatch.

### 39. Pause retraining and isolate controller conformance

Date: 2026-07-27.

Training result: Run `20260727T195222Z.162535` completed all 50 requested
iterations, but it did not demonstrate learned collision avoidance. Combined,
body, and pusher collision rates were zero from the first iteration through the
last. Selection evaluated iterations 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, and
49 over 300 episodes each; all reported 1.000 success and zero collision,
failure, and timeout rates. Iteration 0 won
`press-approach-gated-maximin-v3`, with 0.417999 m mean closest range,
0.417608 m minimum closest range, and 4.050 s median completion. Independent
evaluation reproduced those results. Selecting the warm-start checkpoint and a
flat zero-collision curve means this run provides no evidence that PPO learned
to reconcile the Lab/PX4 approach behavior. Keep training paused.

Monitoring result: The run was exposed through a run-scoped TensorBoard on port
6007. Its event file did not contain governor-intervention scalars, so actor
intent could not be separated from the applied command after the run.

Live result: The first governed PX4 routine did not report approach contact, but
the later `land` step timed out after 30 simulated seconds and recovery disarmed
the vehicle. Do not use that end-to-end failure to tune approach-controller
gains; the mission contains unrelated navigation, landing, and lifecycle
behavior.

Isolation decision: Reconcile equivalent layers before another reward run.
Static rotor realization compares identical rotor commands through the
quadratic thrust/body-wrench path and the pinned Isaac Lab `Thruster`.
Controller conformance compares identical estimator state and velocity intent
at the physical rotor-speed/body-wrench boundary. Closed-loop conformance then
compares deterministic forward/stop/reverse velocity traces without a wall,
policy, reward, or mission landing step. Training may resume only after the
instantaneous and temporal comparisons have reviewed tolerances.

Initial controller replay: `flight_analysis/tools/controller_conformance.py`
replayed the governed live ULog window 28.3-30.7 s through the numpy twin of the
Isaac Lab controller. Across 24 matched samples, PX4 versus Isaac rotor-speed
RMSE was 3.212 rad/s, or 0.00612 of the configured 1100 rad/s full scale; maximum
absolute rotor error was 6.911 rad/s. Mean collective thrust was 23.012 N for
PX4 and 23.045 N for Isaac, with 0.0494 N collective RMSE. Body-torque RMSE was
0.0319, 0.0737, and 0.00088 N m. This provisional result says the instantaneous
controller/allocation boundary is already close. The remaining investigation
therefore targets cadence, delay, braking distance, overshoot, and settling in
the closed loop rather than blind gain retuning.

### 40. Remove the governor and require learned collision avoidance

Date: 2026-07-27.

Latest live evidence: The full autorun completed approach, return, landing, and
disarm. Its applied PX4 forward setpoint peaked at 0.0778 m/s, reversed at
0.4458 m range, and took approximately 0.8 s and 0.030 m to stop. It reached
0.4158 m without a logged pusher contact. In the earlier failed run, the actor
requested up to 0.1414 m/s and did not reverse until 0.3713 m; the vehicle had
already entered the collision region. The successful autorun therefore
demonstrates that the governor prevented the bump, not that PPO learned safer
braking.

Controller evidence: Replaying the latest approach window through the Isaac
controller produced 3.353 rad/s rotor-speed RMSE (0.00639 of full scale),
0.0209 N collective-thrust RMSE, and body-torque RMSE of 0.0281, 0.0792, and
0.00119 N m. The bridge remained healthy at 4 ms actuator lag. Instantaneous
controller and allocation math are already close; the remaining mismatch is
temporal closed-loop response.

Decision: Remove the governor from both training and deployment. The actor
action again drives physics and PX4 directly, and `previous_action` remains the
actual preceding command. Keep privileged body/pusher collision reward and
termination unchanged. A new actor-only run must show collision rate falling
through training and pass checkpoint selection plus independent evaluation
without a safety layer. Defer new safety and closed-loop conformance harnesses
until the raw policy has proved correct behavior.

### 41. Actor-only retraining still does not expose the live collision

Date: 2026-07-27.

Run `20260727T204645Z.223589` removed the governor from both training and
deployment and actor-only initialized from
`approach-20260726d/logs/model_24.pt`. The requested 50 iterations completed
with the fixed `1e-6` learning rate, 0.01 noise, zero entropy, and checkpoints
every five iterations. A run-scoped TensorBoard was available at port 6006
throughout training.

Result: Training collision, body-collision, and pusher-collision percentages
remained zero. All eleven checkpoints produced 1.000 success and zero
collisions across 300 selection episodes. Selection chose iteration 45 because
it completed faster than the earlier checkpoints. Independent evaluation
reproduced 1.000 success, zero failures, 0.408605 m mean closest range,
0.403935 m minimum closest range, and 1.300 s median completion.

Interpretation: Removing the safety layer was necessary, but it did not make
the Lab environment reproduce the PX4 failure. The unguided warm-start actor
already succeeds without contact in Lab, so PPO never receives the privileged
collision penalty and no falling collision curve can appear. Across
checkpoints, median completion improved from 1.500 s to 1.300 s while minimum
closest range moved from 0.411114 m to 0.403935 m. The selected actor is faster
and uses more of the range margin, which is not evidence of safer live braking.
Do not claim learned collision reconciliation from this run. Leave the raw
artifact available for the requested manual routine, and use that result before
changing training dynamics or selection policy.

### 42. Stage-filtered live comparison confirms a settling mismatch

Date: 2026-07-27.

Scope correction: Compare only the configured `policy[press-approach]` routine
step. The preceding arm, takeoff, hover, and waypoint steps establish the live
handoff state and are not learned-policy behavior; the return and landing steps
are also excluded. In the existing run, the PX4 trajectory-setpoint transition
from position to velocity identifies the policy interval at raw simulated time
28.120-46.720 s, or ULog-relative time 27.940-46.540 s.

Live result: The policy step began at 0.500645 m wall-normal switch range and
lasted 18.600 s. It first entered the 0.40-0.42 m goal band after 1.280 s at
0.416357 m, but speed was 0.107596 m/s, just above the 0.10 m/s stable-goal
limit. The applied forward setpoint did not reverse until 1.600 s, when the
ground-truth range was already approximately 0.380 m; the body was still
closing at 0.0400 m/s. Closest range was 0.380348 m at 1.680 s, and sampled
ground truth remained below 0.40 m for 16.8 s before the vehicle backed out and
completed at 0.402134 m. No pusher-contact failure was reported; this does not
exclude a body, arm, propeller, or wall collision.

Lab comparison: The independently evaluated selected checkpoint completes in a
1.300 s median with 0.408605 m mean closest range and 0.403935 m minimum closest
range. The live vehicle therefore reaches the goal band at approximately the Lab
completion time, but does not settle there and overshoots the Lab minimum by
23.6 mm. This is a policy-stage behavior mismatch, not an artifact of including
takeoff or waypoint travel in the comparison.

Controller comparison: Replaying the full policy-only ULog window through the
Isaac controller produced 3.225 rad/s rotor-speed RMSE, 0.00614 normalized
rotor-speed RMSE, 8.523 rad/s maximum rotor error, 0.0238 N collective-thrust
RMSE, and body-torque RMSE of 0.0241, 0.0765, and 0.00145 N m. Instantaneous
allocation remains close. Investigate the closed-loop transient, plant setup,
cadence, and handoff-state distribution before changing the reward or training
again.

Logging decision: Every routine step now emits structured `ROUTINE_STAGE`
records with start/complete/failed event, configured kind, label, index, and
exact simulation time. The fleet runtime mirrors them into the per-vehicle
bridge log so future analysis can select the policy interval directly rather
than infer it from setpoint type.

### 43. Matched policy entry isolates the remaining plant response

Date: 2026-07-27.

Entry reconciliation: Training resets and deployment now share one
`rl.press.handoff` actor-entry contract: 0.45-0.50 m wall-normal range, at most
0.08 m radial lateral/vertical error, at most 0.02 m/s wall-normal and lateral
speed, at most 0.05 rad/s angular speed, upright attitude, valid pose and
estimator state, no contact, zero previous action, and pre-contact mode. The
policy routine retains its 0.48 m waypoint, tightened to 0.02 m settling
tolerance, but transmits zero velocity until this complete gate passes.

Matched Lab result: The existing iteration-45 actor from run
`20260727T204645Z.223589` was independently evaluated over 300 episodes using
the reconciled reset envelope. It achieved 1.000 success, zero body or pusher
collisions, zero other flight failures and timeouts, 0.40864 m mean closest
range, 0.40726 m minimum closest range, and 1.300 s median completion. The
evidence is under
`.build/docker/integration/eval-press.20260727T220447Z.283068`. This passes the
80% success, 5% collision, and 0.40 m minimum-range acceptance thresholds.

Matched PX4 entry: The corrected manual routine entered the policy at HIL time
50.084 s with the following exact 34-value observation:

```text
[0.48916839599609224, -0.007876251824200153, 0.020658969879150394,
 0.0007149667336627738, -0.0009180963702619456, -0.0034416273301046685,
 0.9999934005396961, 1.0, 0.0, 1.0, -0.006910835427534126,
 0.005143429205436118, -0.001968258445456121, -0.003251545146200264,
 -0.006700190542363499, 1.736254838224173e-05, 0.0011547500034794211,
 -0.00014363434456754476, 0.7046685218811035, 0.7095357179641724,
 2.6640482246875763e-05, 0.0034139060880988836, 1.0, 0.0, 0.0, 1.0,
 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 7.0]
```

The raw normalized action was
`[0.0627968311, -0.0075850487, 0.0221121311, -0.0005383986]`, and the
transmitted velocity setpoint was 0.125594 m/s forward, 0.015170 m/s right,
-0.044224 m/s down, and 0.000808 rad/s yaw rate. The gate admitted the actor
with only 0.00691 m/s initial wall-normal speed. The policy accumulated four
stable samples at 0.415001, 0.411734, 0.407021, and 0.404009 m, but retained
0.08286-0.07400 m/s closing speed through those samples. It first commanded
reverse at 1.76 s, when range was already 0.387520 m and closing speed was
0.052732 m/s. Pusher contact followed at 1.86 s and 0.3834 m, so the policy
stage failed without completing five stable samples. The evidence is under
`.build/docker/integration/test-integration-press-approach-routine.20260727T220835Z.286190`.

Isolated replay: Reconstructing the captured contact-relative entry in Lab and
running the actor closed-loop completed in 1.45 s without collision, with a
0.409990 m closest range. Replaying the captured PX4 normalized-action sequence
through the same Lab plant also completed without collision in 1.40 s, with a
0.406557 m closest range. Lab policy intervals alternate 52/48 ms at 250 Hz;
the 50 Hz routine driver quantizes the same 20 Hz policy to alternating 60/40
ms intervals. Both average exactly 50 ms, while the materially different
closing-speed and braking traces remain.

Decision: Entry-state mismatch is reconciled, but it was not the complete
cause. Lab succeeds from the matched entry while PX4 overshoots, so do not
resume PPO and do not run another manual approach yet. The identical-action
replay and the already-close instantaneous controller/allocation comparison
localize the remaining work to temporal closed-loop plant response, including
velocity tracking, braking distance, and settling. Reconcile that response
before another wall run.

### 44. Fit temporal response, retrain, and stop before the wall

Date: 2026-07-27.

Trace capture: The wall-free production actor ran at nominal 0.460, 0.475, and
0.490 m entry ranges with switch and wall collision disabled. The measured
entry ranges were 0.456251, 0.481800, and 0.498089 m. PX4 closest ranges were
0.406414, 0.401356, and 0.326529 m; peak closing speeds were 0.062576,
0.089772, and 0.115816 m/s. The 0.490 m case required 0.057806 m of braking
distance and settled after 4.96 s. The policy cadence retained its exact 40/60
ms quantization around a 50 ms mean. Every routine step had a timeout of at
most 20 s; nearby staging points kept the complete 19-step, three-case mission
to 52.884 s of simulation time. Raw traces and the combined report are under
`.build/benchmarks/press-temporal/`.

Temporal fit: A first-order fit to the two outer PX4 cases estimated 0.260 s
command delay and a 0.660 s response time constant, with 0.017398 m/s training
velocity RMSE and 0.015558 m/s held-out-middle RMSE. Exact isolated-Lab replay
then evaluated the production controller in physics ticks and selected a
0.440 s Lab-only action delay from the outer cases. Outer-case velocity RMSE
was 0.017951 m/s; the held-out 0.475 m case was 0.023553 m/s. Its middle-case
braking-distance delta was 0.0 m and closest-range delta was +0.011787 m.
This is an equivalent closed-loop response parameter, not evidence of a
literal 440 ms wait in one PX4 component. PX4, actor cadence, and `press-v1`
remain unchanged.

Actor decision: The existing iteration-45 actor from
`20260727T204645Z.223589` failed reconciled Lab evaluation over 300 episodes:
0.517 success, 0.483 pusher collision, 0.39414 m mean closest range, and
0.38285 m minimum closest range. Run
`temporal-reconcile-440ms-20260728T0040Z` therefore actor-only initialized
from that iteration and completed the prescribed 50 iterations at fixed
`1e-6` learning rate, 0.01 initial noise, zero entropy, and five-iteration
checkpoints. Each iteration collected 8,192 transitions from 256 environments
and 32 policy steps. Run-scoped TensorBoard used
`.build/training/press-approach/temporal-reconcile-440ms-20260728T0040Z/logs`.

Selection outcome: No checkpoint passed all three gates. Iterations 45 and 49
were the two best. Iteration 49 ranked first with 0.940 success, 0.060
collisions, 0.404974 m mean closest range, and 0.383841 m minimum closest
range. It missed both the 0.050 collision ceiling and 0.400 m minimum-range
floor. Selection retained iteration 49 only as the rejected diagnostic and
did not export it or advance `latest`.

Publication override: At the user's explicit direction, iteration 49 was
manually exported and `.build/training/press-approach/latest` was advanced to
`temporal-reconcile-440ms-20260728T0040Z`. This makes the 94.0%-success model
available for later work but does not change its rejected selection result or
waive the virtual and physical acceptance gates.

Decision: Stop at the published diagnostic actor. The required three virtual
PX4 acceptance runs were not started, and the physical wall run was not
performed. Physical completion, five stable samples, closest range, pusher
contact, and visible body/arm/propeller contact therefore have no outcome for
this cycle. To continue actor-only work later, initialize a new run from
`.build/training/press-approach/temporal-reconcile-440ms-20260728T0040Z/logs/model_49.pt`;
the full rejected metrics remain in that run's `checkpoint_selection.json`.

### 45. Verify per-goto PX4 ground-speed control without the wall

Date: 2026-07-28.

Manual verification command:

```text
make test-integration-px4-navigation
```

Wall-free result: The dual-airframe navigation gate passed using a 0.5 m/s
PX4-guided reposition leg followed by the existing streamed OFFBOARD leg. The
x500 observed 0.568615 m/s peak and 0.453339 m/s average horizontal speed,
0.656407 m braking distance, 0.489171 m error when the waypoint predicate
completed, 0.069953 m final error after the following hover, zero overshoot,
and 0.035061 m maximum cross-track error. The s550 observed 0.538365 m/s peak
and 0.440375 m/s average horizontal speed, 2.129739 m braking distance,
0.489472 m completion error, 0.060672 m final error, 0.039235 m overshoot, and
0.056772 m maximum cross-track error. Both completion errors remained within
the existing 0.5 m position gate and both cross-track errors remained within
the existing 2.0 m gate. Run-scoped evidence is under
`.build/docker/integration/test-integration-px4-navigation.20260728T220018Z.1260490`.

Virtual approach result: The 0.5 m/s staging leg, 0.08 m/s final 0.11 m leg,
fresh OFFBOARD policy entry, policy completion, and 0.5 m/s return leg all
completed. The mission then failed only because the land step reached its
required 20.0 s timeout before PX4 disarmed; ULog showed that PX4 accepted the
Land command and the vehicle reached ground level well inside that interval.
Recovery subsequently disarmed the vehicle. The failed run is preserved under
`.build/docker/integration/test-integration-press-approach-routine.20260728T215628Z.1258179`;
the timeout was not raised above the required 20 s cap.

Decision: PX4-guided speed control is accepted for wall-free navigation, while
the physical approach remains unapproved because its complete virtual routine
did not pass. No physical-wall command was run. Do not infer a hard velocity
limit from the requested speed: the observed peak speeds demonstrate that it is
a PX4 trajectory target, and PX4 retains braking responsibility.

### 46. Record the successful manual mission without upgrading the actor

Date: 2026-07-28.

Reproducible manual command:

```text
make run-all-autorun HYDRA_CONFIG=configs/policy_press_approach_quad.yaml
```

Run result: The complete ten-stage routine finished, including the 0.5 m/s
staging leg, hover and final approach, policy, 0.5 m/s return, landing, and
disarm. The bridge evidence is
`px4/logs/2026-07-28/22_41_53_v0_bridge.log`; the paired ULog is
`px4/logs/0/2026-07-28/22_41_53.ulg`.

Policy-only result: The policy ran for 2.2 s over 45 actor samples. It moved
from 0.47949 to 0.40823 m switch range, with 0.03223 m/s average and
0.06621 m/s peak closing speed. Contact sensing remained valid and false,
the lamp stayed off, and maximum pusher compression was approximately
0.027 mm. Classify this policy interval as contact-free. The terminal actor
sample emitted zero velocity; the following 0.5 m/s PX4-guided `goto`, not the
policy, performed the controlled retreat.

Speed interpretation: The 0.08 m/s final `goto` completed after one 20 ms
driver tick because the preceding braking and hover had already brought the
vehicle inside the configured 0.02 m position tolerance. This run therefore
does not independently validate sustained PX4-guided motion at 0.08 m/s.

Acceptance decision: Preserve this successful manual mission separately from
the earlier formal approach-gate run in section 45, which failed on the landing
timeout before later recovery disarmed. The manual result does not convert
iteration 49 into an accepted actor, waive its rejected 6.0% collision and
0.383841 m minimum-range selection metrics, or replace the formal gates. No
additional warehouse-wall run is authorized by this evidence.

### 47. Retrain the held-out delay candidate and analyze the manual mission

Date: 2026-07-29.

Retraining command:

```text
make/scripts/run_training_evaluation_then_shutdown.sh \
  --hydra-config configs/train_press_approach_quad.yaml \
  --train-target train-rl \
  --select-target select-rl-checkpoint \
  --eval-target eval-press \
  --train-args '--initialize-checkpoint /isaac-sim/proj/.build/training/press-approach/latest/logs/model_49.pt' \
  --label press-approach-200ms
```

Run `20260729T024615Z.21141` actor-only initialized from iteration 49 of the
rejected 0.440 s diagnostic run and trained for 50 iterations with the
held-out-best 0.200 s Lab response proxy. All eleven checkpoints passed
selection. Iteration 49 was selected and published to
`.build/training/press-approach/latest`. Independent evaluation over 300
episodes achieved 1.000 success, zero body or pusher collisions, zero other
flight failures and timeouts, 0.406831 m mean closest range, 0.401545 m
minimum closest range, and 1.450 s median completion.

GPU observation: This run inherited the conservative 256-environment config
and sustained approximately 45-59% GPU utilization with about 6.5 GB of
24.6 GB VRAM. Earlier capacity measurements placed 32,768 environments near
the throughput peak at approximately 86-89% utilization and 17.5 GB VRAM,
while 49,152 used approximately 22.5 GB without improving throughput. New
training therefore uses 32,768 environments in the tracked Hydra config;
49,152 remains an explicit capacity probe.

Manual command:

```text
make run-all-autorun HYDRA_CONFIG=configs/policy_press_approach_quad.yaml
```

The complete ten-stage mission finished, including return, landing, and
disarm. Evidence is in
`px4/logs/2026-07-29/02_51_45_v0_bridge.log` and the paired ULog
`px4/logs/0/2026-07-29/02_51_46.ulg`. The policy ran for 2.2 s over 45 actor
samples, moved from 0.485541 to 0.404858 m, and had 0.036674 m/s average and
0.081813 m/s peak observed closing speed. Contact sensing stayed valid and
false, the lamp stayed off, and maximum pusher compression was approximately
0.028 mm, so classify the policy interval as contact-free.

The policy ended with a zero-velocity command. The following 0.5 m/s
PX4-guided `goto`, not the policy, performed the retreat. Residual momentum
continued for approximately 0.364 s and 14.6 mm after the return command,
placing the estimated closest post-policy range near 0.390 m before reversal.
There was no logged contact or lamp activation. This successful mission is
strong deployment evidence, but post-policy coasting below the Lab 0.40 m
floor keeps temporal plant alignment open.

Instantaneous ULog replay remained close: normalized rotor-speed RMSE was
0.00794, collective-thrust RMSE was 0.03701 N, and mean collective thrust was
23.0303 N in PX4 versus 23.0542 N in the Lab controller reference. These are
diagnostics, not accepted publication thresholds.

### 48. Revalidate the 0.200 s proxy on fresh wall-free traces

Date: 2026-07-29.

The calibrated wall-free temporal mission passed under
`.build/docker/integration/test-integration-press-temporal-conformance.20260729T031246Z.41668`.
The measured policy-entry ranges were 0.473103, 0.495020, and 0.499409 m;
the analyzer uses these measured observations even though the stable labels
remain nominal 0.460, 0.475, and 0.490 cases. PX4 closest ranges were
0.406176, 0.403244, and 0.401688 m, and peak closing speeds were 0.063719,
0.090268, and 0.096454 m/s.

Exact-action Isaac Lab replay evaluated zero, 0.200, and 0.440 s delay
candidates. Outer-trace RMSE was 0.031966, 0.024226, and 0.025795 m/s,
respectively. Held-out-middle RMSE was 0.030365, 0.020706, and 0.025401 m/s.
The fresh evidence therefore selects 0.200 s on both fitted and held-out
metrics. The first-order PX4-only fit estimated 0.164 s delay and a 0.612 s
response time constant; this supports interpreting the Lab delay as an
equivalent proxy rather than literal transport latency.

The 0.200 s Lab replay still ended 0.04117, 0.02319, and 0.01234 m closer than
PX4 across the three traces. Decision: retain the accepted 0.200 s actor and
configured proxy, but do not call sim-to-sim alignment complete. The remaining
work is to identify response bandwidth, per-axis dynamics, and braking
separately from delay, then validate on new held-out commands. Do not
automatically run another warehouse-wall mission.

### 49. Sweep horizontal controller response and retain the hover gate

Date: 2026-07-29.

Identification input: Reuse the wall-free PX4 traces from section 48 while
holding the actor, captured normalized actions, 40/60 ms cadence, initial
observations, and 0.200 s command-delay proxy fixed. The first diagnostic
swept 30 combinations of velocity proportional- and damping-gain scales. A
second 15-combination sweep changed only the two horizontal axes, leaving the
unexcited vertical gains unchanged. Reports are
`.build/benchmarks/press-temporal/20260729T031246Z/controller-grid-conformance.json`
and
`.build/benchmarks/press-temporal/20260729T031246Z/horizontal-controller-grid-conformance.json`.

Selected horizontal candidate: Scale the horizontal proportional gains by
0.75 and horizontal damping gains by 1.50. Against the current controller,
outer-trace velocity RMSE fell from 0.024226 to 0.012599 m/s and
held-out-middle RMSE fell from 0.020706 to 0.007104 m/s. Closest-range deltas
changed from -0.04117, -0.02319, and -0.01234 m to -0.01641, +0.00333, and
+0.01200 m. The all-axis exploratory sweep produced materially identical
forward-response metrics, confirming that the observed improvement came from
the horizontal gains.

Independent flight gate: Applying the selected absolute gains
`controller_kp_vel=[1.5, 1.5, 3.0]` and
`controller_kd_vel=[1.8, 1.8, 1.5]` failed
`make test-integration-press-hover-goto`: the final hover error was 0.093 m,
but the last-one-second maximum was 0.135 m against the 0.100 m limit. A
conservative damping-only candidate kept horizontal proportional gain at 2.0
and raised horizontal damping to 2.4. It also failed, with 0.087 m final error
and 0.129 m last-window maximum. Evidence is under
`.build/docker/integration/test-integration-press-hover-goto.20260729T040349Z.12744`
and
`.build/docker/integration/test-integration-press-hover-goto.20260729T040426Z.13390`.

A 1.50x damping candidate, absolute horizontal damping 1.8, narrowly failed:
final hover error was 0.065 m, but its last-window maximum was 0.101 m. The
1.25x candidate, absolute horizontal damping 1.5, passed. Its last-window
maximum hover error was 0.086 m, settled goto error was 0.016 m, and the press
side-channel handoff was ready at 0.464 m. Evidence is under
`.build/docker/integration/test-integration-press-hover-goto.20260729T040516Z.14088`
and
`.build/docker/integration/test-integration-press-hover-goto.20260729T040540Z.14769`.
The retained candidate reduces outer/held-out velocity RMSE from
0.024226/0.020706 to 0.020522/0.016336 m/s.

Decision: Retain `controller_kp_vel=[2.0, 2.0, 3.0]` and
`controller_kd_vel=[1.5, 1.5, 1.5]`. This is the strongest measured response
change that preserves the independent hover/goto precondition. Retraining may
proceed from the latest accepted actor; it must repeat selection and
independent evaluation before PX4 acceptance.

Bounded delay refinement: With the retained damping fixed, delays 0.120,
0.160, 0.200, 0.240, and 0.280 s produced held-out RMSE of 0.019700,
0.017700, 0.016336, 0.015804, and 0.016014 m/s. The nominal 0.240 s winner
improves on 0.200 s by only 0.000532 m/s, or approximately 3.3%. Because the
same middle trace already informed the delay and damping decisions, this
marginal result is insufficient independent evidence for another config
change. Keep `controller_command_delay_s=0.200`; validate any future delay
change on a new command shape.

### 50. Retrain the hover-safe damping plant at production parallelism

Date: 2026-07-29.

Run command:

```text
make/scripts/run_training_evaluation_then_shutdown.sh \
  --hydra-config configs/train_press_approach_quad.yaml \
  --train-target train-rl \
  --select-target select-rl-checkpoint \
  --eval-target eval-press \
  --train-args '--initialize-checkpoint /isaac-sim/proj/.build/training/press-approach/latest/logs/model_49.pt' \
  --label press-approach-damped-200ms
```

Run `20260729T041316Z.27604` used source commit `468e408`, 32,768 parallel
environments, the retained 0.200 s delay, and horizontal damping 1.5. It
actor-only initialized from iteration 49 of accepted run
`20260729T024615Z.21141`. Fifty iterations collected 52,428,800 transitions
in 328.8 s. Mean reported throughput across iterations 1-49 was approximately
174,104 steps/s, ranging from 159,369 to 186,917 steps/s. Live GPU sampling
showed 84% utilization, 16.8-18.0 GB of 24.6 GB VRAM, and 229-278 W. This
confirms that the tracked 32,768-environment default materially uses the GPU
while retaining VRAM headroom.

All eleven saved checkpoints passed the 300-episode selection gates with
1.000 success and zero collisions, other flight failures, or timeouts.
Maximin selection chose iteration 20 and published it to
`.build/training/press-approach/latest`. Independent evaluation of that exact
artifact reproduced 1.000 success, zero body and pusher collisions, zero other
flight failures and timeouts, 0.405855 m mean closest range, 0.400629 m
minimum closest range, and 1.400 s median completion.

Decision: Accept iteration 20 as the Lab artifact for the retained
delay-and-damping plant. This completes retraining, checkpoint selection, and
independent Lab evaluation, but it does not by itself complete sim-to-sim
alignment or authorize an automatic warehouse-wall run. The next acceptance
evidence is a fresh wall-free PX4 temporal capture using this actor, followed
by exact-action Lab replay on a command shape not used to select the current
delay and damping.

### 51. Reject the retrained actor at fresh PX4 temporal validation

Date: 2026-07-29.

Command:

```text
make test-integration-press-temporal-conformance
```

The fresh wall-free mission used selected iteration 20 from run
`20260729T041316Z.27604`. Its first two policy stages completed, but their
PX4 trajectories overshot the Lab safety floor before recovering:

| Trace | Measured entry | Closest range | Terminal range | Peak closing speed | Duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| `temporal-0.460` | 0.470444 m | 0.355108 m | 0.403326 m | 0.083668 m/s | 4.66 s |
| `temporal-0.475` | 0.491539 m | 0.348231 m | 0.402060 m | 0.106641 m/s | 4.80 s |

Both stages eventually accumulated five stable-goal samples, but closest
range was approximately 45-52 mm below the 0.40 m Lab collision floor. This
is a material PX4/Lab braking-response mismatch and rejects the actor for a
warehouse-wall run despite its perfect Lab evaluation.

The third stage did not exercise the actor. Its `goto`/hover sequence entered
at 0.523123 m, outside the shared 0.45-0.50 m handoff envelope. The policy
correctly transmitted zero velocity with `handoff_reason=too_far` until the
20 s stage timeout. Landing recovery was accepted but exhausted its 30 s
disarm wait. Treat this separately as a temporal-harness staging failure, not
an actor or contact failure. Evidence is under
`.build/docker/integration/test-integration-press-temporal-conformance.20260729T042714Z.46121`.

Decision: The 0.200 s delay plus horizontal damping 1.5 improves trace RMSE
for the previous actor but is insufficient to transfer the retrained actor's
braking behavior. Keep iteration 20 as a Lab-accepted diagnostic artifact,
not a PX4-accepted actor. Do not run a warehouse-wall mission. The next
alignment cycle should use policy-independent step/ramp/reversal excitation
and fit response bandwidth and braking dynamics separately from pure delay;
identified ranges should then be randomized during training. The temporal
staging harness also needs a wall-normal entry predicate that does not rely on
one three-dimensional waypoint tolerance before it can provide three valid
acceptance traces.

### 52. Reject the scratch actor's physical transfer and start alignment v2

Date: 2026-07-29.

The newest scratch Stage 1 actor passed its Isaac Lab validation with a
0.4114 m minimum range. In the physical warehouse transfer it nevertheless
contacted the switch at 0.3832 m. Its trace still requested approximately
0.078 m/s forward at 0.400 m range. This is a downstream physical-transfer
failure: the actor is not accepted and the complete approach mission must not
run from this evidence.

The continued forward command means controller lag cannot be assumed to be the
only cause. Preserve this trace as a diagnostic design input, never as an
alignment fit or validation trace. The next diagnosis first replays identical
initial state, cadence, and actions through PX4 and Lab. If that exact-action
response passes while closed-loop observations or actions diverge, classify the
remaining issue as observation/state distribution or policy behavior instead
of continuing to tune controller gains.

The new comparison generation is `x500-velocity-px4-v2`, artifact schema 2,
with policy-scale linear/yaw and combined probes. Its staged beam fits velocity
delay/filter/PID terms before attitude/rate terms and ranks held-out braking,
residual motion, velocity, yaw, and endpoint error. Schema-1 artifacts are not
upgraded. Instantaneous actuator response remains fixed because the fresh
diagnostics from section 47 measured normalized rotor-speed RMSE 0.00794 and
collective-thrust RMSE 0.03701 N, both below the v2 third-stage triggers.

No v2 selected settings, advisory result, retraining provenance, or final gate
distribution exist at this entry. Record them only from a completed fresh
alignment run. Retraining is authorized only after v2 advisory quality and the
exact-action diagnosis pass; it must then be a new 300-iteration scratch run
with unchanged reward and PPO hyperparameters. Checkpoint selection,
independent Lab evaluation, the three-range physical transfer gate, and the
complete mission remain ordered stop-on-failure gates.

### 53. Reject alignment v2 at the advisory exact-action gate

Date: 2026-07-30.

Fresh run `20260730T012000Z` captured all 11 PX4 probe traces and evaluated 799
unique Isaac Lab candidates. The deterministic staged beam selected candidate
587 with:

- zero command delay;
- velocity P/I/D `[1.35, 1.35, 3.0]`, `[0, 0, 0]`, and
  `[0.3, 0.3, 0]`;
- velocity and derivative cutoffs 0.95493 and 2.5 Hz;
- attitude P `[6, 6, 4.2]` with yaw weight 0.3;
- the pinned rate P/I `[0.15, 0.15, 0.2]` and
  `[0.2, 0.2, 0.1]`, with zero rate D/feed-forward;
- the pinned 20 Hz rate-derivative cutoff and a 67.5 degree tilt limit.

The selected exact-action response failed every advisory v2 category:

| Held-out metric | Result | Limit |
| --- | ---: | ---: |
| Braking-distance error | 5.346 mm | 5 mm |
| Residual-speed error | 0.01125 m/s | 0.01 m/s |
| Stop-time error | 0.200 s | 0.10 s |
| Worst-axis velocity NRMSE | 58.30% | 10% |
| Yaw-rate NRMSE | 48.36% | 10% |
| Heading error | 56.25 deg | 3 deg |
| Combined endpoint error | 206.25 mm | 10 mm |

Candidate-level GPU batching evaluated the largest expansion as 48 settings
times 11 probes, or 528 independent environments. The sweep completed in
297.4 s; the interrupted sequential implementation had reached only 620
candidates after approximately 48 minutes. Representative GPU sampling on
this hardware averaged approximately 35% utilization, peaked at 47%, and used
about 5.24/24.56 GB VRAM. A measured 4x scene replication experiment did not
increase GPU utilization and slowed equal search progress by approximately
33%, so it was removed as semantically meaningless load. The remaining ceiling
is the single Isaac/PhysX process; future improvement requires useful
multi-process candidate sharding with deterministic beam coordination.

Decision: Reject v2 run `20260730T012000Z` at the first advisory gate. Preserve
its reference, candidates, artifact, and integration logs under `.build/`.
Because identical-action controller response itself fails, do not start the
closed-loop actor comparison, do not retrain Stage 1, and do not run checkpoint,
physical-transfer, or mission gates. Continue controller/plant diagnosis from
these traces before expanding the search or changing the actor.

Proceed in this order:

1. Use the preserved per-probe PX4 and Lab traces to localize the velocity and
   yaw failures. Verify frames, initial state, command cadence, controller-state
   reset, filter and anti-windup transitions, saturation, attitude/rate
   response, thrust allocation, and the shared mass, rotor geometry, thrust
   curve, and drag contract. Fix a confirmed plant or controller
   implementation mismatch directly; do not absorb it into fitted gains.
2. Before another broad GPU search, add useful multi-process candidate sharding
   around the deterministic beam coordinator and measure aggregate throughput
   and GPU utilization. Preserve candidate ordering and score equivalence
   across worker counts. Do not restore replicated environments that perform no
   additional candidate work.
3. If the implementation and plant contracts are correct, revise the
   declarative `ParameterSpec` ranges or staging using the failed-trace
   diagnosis, then perform a fresh reference capture, instantaneous actuator
   replay, target sweep, and held-out fit. Start a new comparison fingerprint
   if the probe suite, metrics, environment, image, schema, or controller
   interface changes.
4. Require every v2 advisory threshold to pass on identical actions before
   comparing the closed-loop actor. Report observation, action, and state
   divergence separately; if exact-action response passes but actions diverge,
   classify the remaining failure as observation/state distribution or policy
   behavior instead of tuning the controller again.
5. Once alignment and the closed-loop diagnosis pass, retrain Stage 1 from
   scratch for 300 iterations with the selected v2 artifact embedded as
   provenance. Existing actors are incompatible with the newly selected
   controller dynamics. Keep rewards, PPO hyperparameters, domain
   randomization, and the policy interface unchanged.
6. Run automatic checkpoint selection, independent Lab evaluation, the
   three-range physical warehouse transfer gate, and then the complete approach
   mission. Stop at the first failed gate and append its artifact references and
   metric distribution here.

### 54. Reject the corrected alignment v2 run at the exact-action gate

Date: 2026-07-30.

The diagnosis from section 53 confirmed two reference-capture defects. PX4
expects NED yaw-rate setpoints, but the FLU probe yaw rate had been sent without
the required sign inversion. In addition, each probe captured its initial state
while the vehicle could still be settling from the preceding priming motion.
The capture now sends a neutral setpoint before every profile and requires both
a minimum neutral duration and a continuous stable hold before recording. The
stability predicate covers linear and angular speed plus tilt, has an explicit
timeout, and the yaw probe rejects a response with the wrong sign.

The first corrected capture stopped at profile 5 because its original
0.01 m/s linear and 0.02 rad/s angular stability bounds were below the measured
PX4 hover noise. The preserved ULog showed linear speed up to 0.0140 m/s,
angular speed up to 0.0286 rad/s, and tilt below 0.117 degrees during the
failed precondition. The bounds were therefore changed to 0.02 m/s and
0.03 rad/s while retaining a 3.0 s minimum neutral period and a 0.5 s
continuous stable hold. These are capture-validity tolerances, not controller
fit parameters. The next capture completed all 11 profiles:

- run: `.build/alignment/x500-velocity-px4-v2/20260730T035141Z`;
- integration artifact:
  `.build/docker/integration/alignment-capture-reference.20260730T035143Z.21779`;
- PX4 ULog: `px4/logs/0/2026-07-30/03_51_54.ulg`.

The corrected instantaneous search evaluated 818 candidates and selected
candidate 558. Its exact-action result improved yaw substantially but still
failed residual speed, stop time, worst-axis velocity, heading, and combined
endpoint quality. A fresh controller-state replay measured normalized
rotor-speed RMSE 0.00865 and collective-thrust RMSE 0.08518 N. The latter
exceeded the 0.05 N trigger, so the declared third search stage was activated
for actuator rise, fall, and thrust-slew settings.

The complete three-stage run
`.build/alignment/x500-velocity-px4-v2/20260730T041659Z` reused the byte-identical
fresh reference, evaluated 921 candidates, and selected candidate 849 with:

- command delay `0.036 s`;
- velocity P/I/D `[2.7, 2.7, 6.0]`, `[0.3, 0.3, 1.5]`, and
  `[0.2, 0.2, 0.0]`;
- velocity and velocity-derivative cutoffs `0.95493 Hz` and `2.5 Hz`;
- attitude P `[4.0, 4.0, 2.8]` with yaw weight `0.5`;
- rate P/I/D `[0.15, 0.15, 0.2]`, `[0.2, 0.2, 0.1]`, and
  `[0.0015, 0.0015, 0.0]`, with zero feed-forward;
- rate-derivative cutoff `30 Hz` and tilt limit `45 degrees`;
- actuator rise/fall time constants `0.01/0.0 s` and effectively unlimited
  thrust slew (`100000 N/s`).

Its held-out result was:

| Held-out metric | Result | Limit |
| --- | ---: | ---: |
| Braking-distance error | 2.725 mm | 5 mm |
| Residual-speed error | 0.01841 m/s | 0.01 m/s |
| Stop-time error | 0.840 s | 0.10 s |
| Worst-axis velocity NRMSE | 80.32% | 10% |
| Yaw-rate NRMSE | 4.23% | 10% |
| Heading error | 3.761 deg | 3 deg |
| Combined endpoint error | 101.55 mm | 10 mm |

The x/y/z velocity NRMSE values were 80.32%, 35.13%, and 22.06%;
aggregate validation velocity RMSE was 0.04137 m/s. Braking distance and yaw
rate pass, but the other five criteria do not. This corrected run therefore
replaces section 53 as the current exact-action evidence and is rejected at the
same first advisory gate. The comparison fingerprint is
`0dab2650c46eb4c6e15552704970759770154c1442be191f12cfb3d8c41cbb34`.

Candidate sharding is now configurable and preserves deterministic candidate
partitioning and merge order. A three-worker run averaged 95.49% GPU
utilization but processed only 1.10 candidates/s over 743 s. The one-worker
three-stage run averaged 37.07% GPU utilization but processed
3.60 candidates/s over 256 s. The three-worker configuration contends in
Isaac/PhysX and is 3.27 times slower in useful throughput, so one worker is the
current default. Retain the sharding architecture for hardware where measured
candidate throughput improves; GPU utilization alone is not an acceptance
metric.

Decision: Do not run the alignment integration target, closed-loop actor
comparison, retraining, checkpoint selection, physical transfer, or complete
mission. Exact-action controller response still fails before those gates.
Existing actors also remain incompatible with any eventually accepted new
controller settings, so a fresh 300-iteration Stage 1 training run is required
only after alignment passes.

### 55. Authorize alignment v3 attempt 1: physical rate-output scaling

Date: 2026-07-30.

Candidate-wide analysis of the section 54 artifact found that no v2 candidate
passed stop time, worst-axis velocity NRMSE, heading, or combined endpoint
quality. The best observed values were 0.300 s, 55.09%, 3.038 degrees, and
61.87 mm respectively. None passed three or more advisory gates. Nevertheless,
the lexicographic v2 metric selected the globally smallest braking-distance
error even though 375 candidates already passed that gate. The next search must
therefore preserve candidates that lead each independent gate and rank overall
feasibility before improving an already-passing first metric.

The controller-state replay also confirmed a structural unit mismatch. PX4's
rate PID produces normalized torque demand for its control allocator, while
the Lab cascade applied the copied numeric gains directly as physical N-m.
Regressing `vehicle_torque_setpoint` against the physical body wrench
reconstructed from logged motor output over the usable ULog window estimated
per-axis scales `[6.3051, 5.8242, 1.2801]` N-m per normalized unit, with fit
RMSE `[0.02774, 0.01230, 0.00006]` N-m. Attempt 1 therefore starts a new
`x500-velocity-px4-v3` comparison and schema 3, centers the scale candidates at
`[6.5, 6.5, 1.3]`, and retains `[1, 1, 1]` as the exact v2 behavior.

The v3 metric minimizes failed-gate count, largest normalized gate excess, and
total normalized excess before applying the v2 tie-breakers.
`staged-beam-v2` retains the per-gate champions in a deterministic 16-setting
beam. Existing v2 artifacts remain loadable with an implicit unit scale and
are not reinterpreted as v3.

This campaign permits at most five completed alignment attempts. An attempt is
material progress when it lowers the failed-gate count, or lowers the largest
normalized gate excess by at least 20% without regressing an already-passed
gate. A no-progress result triggers an evidence-backed expansion of up to four
controller parameters or ranges, rather than repeated fitting of the same
space. Two consecutive no-progress attempts after expansion stop the campaign
early. Every attempt receives a separate commit before execution and a result
commit afterward. Advisory acceptance requires every exact-action gate to pass;
only then may the alignment integration and closed-loop divergence gates run.

Proceed in this order:

1. Diagnose the per-probe x-axis residual and stop-time traces before expanding
   the search. The selected horizontal and vertical velocity P gains reached
   the current candidate maxima and the derivative cutoff reached its minimum;
   treat those boundary selections as evidence to inspect, not automatic
   authority to widen ranges.
2. Verify that the compared state and filter inputs represent the same PX4
   estimator quantities and that actuator-realized collective thrust is
   compared at the correct boundary. Confirm the shared mass, rotor geometry,
   thrust curve, allocation, drag, saturation, and anti-windup contracts.
3. Fix any confirmed controller or plant mismatch directly. If the contracts
   are correct, revise only declarative `ParameterSpec` ranges or staging from
   the failed-trace evidence and rerun capture, replay, sweep, and held-out fit
   under the resulting comparison fingerprint.
4. Require all v2 advisory thresholds to pass before the closed-loop actor
   comparison. Then report observation, action, and state divergence
   separately.
5. Only after exact-action and closed-loop alignment pass, retrain Stage 1 from
   scratch for 300 iterations with the accepted artifact embedded as
   provenance, unchanged reward and PPO parameters, and no domain
   randomization.
6. Continue with automatic checkpoint selection, independent Lab evaluation,
   the three-range physical transfer gate, and the complete approach mission,
   stopping and preserving diagnostics at the first failed gate.

### 56. Reject v3 attempt 1 and authorize structural attempt 2

Date: 2026-07-30.

Attempt 1 ran from source commit `5fb9481` and completed the fresh
`x500-velocity-px4-v3/20260730T050034Z` capture, sweep, and fit. It evaluated
2161 unique candidates in 36 useful GPU batches and selected candidate 1674.
The comparison fingerprint is
`ffa062a9c94c751be09b4f633f2bef8fb696254d605fda8b3df05b9f4c6c364d`.

| Held-out metric | Attempt 1 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 26.432 mm | 5 mm |
| Residual-speed error | 0.01606 m/s | 0.01 m/s |
| Stop-time error | 0.540 s | 0.10 s |
| Worst-axis velocity NRMSE | 9.27% | 10% |
| Yaw-rate NRMSE | 4.04% | 10% |
| Heading error | 2.856 deg | 3 deg |
| Combined endpoint error | 89.891 mm | 10 mm |

This is material progress relative to section 54: the failed-gate count fell
from five to four, and worst-axis velocity NRMSE fell from 80.32% to 9.27%.
Velocity, yaw rate, and heading now pass. Across all usable candidates, 293
failed four gates and none failed fewer. The candidate-wide best stop-time
error was 0.340 s and the best combined endpoint error was 11.453 mm. The
selected candidate's remaining braking, residual, and stop failures all
localized to the held-out lateral reversal; its target settled 0.54 s earlier
than PX4 on the worst event. The combined trace under-traveled the PX4 lateral
path by approximately 90 mm.

The useful single-worker sweep averaged 32.29% GPU utilization, 5.26 candidate
evaluations/s, and 4450 MiB peak device memory. Three-worker sharding remains
disabled because section 54 measured 3.27 times lower useful throughput from
Isaac/PhysX contention. Attempt 2 instead increases the deterministic beam from
16 to 32, creating independent useful candidate work and enough memory
headroom to measure whether larger batches improve coverage and throughput.

Pinned PX4 v1.17 source inspection found three more structural discrepancies
before another gain-range expansion:

- `MPC_VEL_LP=0` disables the position-controller velocity low-pass. The Lab
  base had incorrectly substituted the unrelated `EKF2_TAU_VEL=0.25 s`
  equivalent and did not support zero as pass-through.
- PX4 uses a thrust-prioritized reduced-quaternion attitude law, weighted yaw
  recombination, and body-frame projection of world-Z yaw-rate feed-forward.
  Lab used a geometric skew-matrix proxy and added feed-forward only on body Z.
- PX4 computes the current rate output with the previous integral, applies its
  large-error integral factor, and advances the integral for the next cycle.
  Lab applied the newly advanced integral in the same cycle.

Attempt 2 corrects those equations in both NumPy and Torch, uses PX4's exact
`dt / (tau + dt)` AlphaFilter coefficient, sets the pinned velocity cutoff to
disabled, and keeps the existing gain ranges. Sequential NumPy/Torch parity is
tested with non-level attitudes. This attempt isolates confirmed controller
structure before adding asymmetric lateral parameters.

### 57. Reject v3 attempt 2 and authorize axis-separated attempt 3

Date: 2026-07-30.

Attempt 2 ran from source commit `e2b0c8d` and completed
`x500-velocity-px4-v3/20260730T052836Z`. The sweep evaluated 3905 unique
candidates and selected candidate 3468 under comparison fingerprint
`1834b08a114bebba455663763459ffc98f311f86cc44a1b1be52881f96ea5276`.

| Held-out metric | Attempt 2 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 12.100 mm | 5 mm |
| Residual-speed error | 0.00820 m/s | 0.01 m/s |
| Stop-time error | 0.240 s | 0.10 s |
| Worst-axis velocity NRMSE | 9.75% | 10% |
| Yaw-rate NRMSE | 5.27% | 10% |
| Heading error | 2.818 deg | 3 deg |
| Combined endpoint error | 30.486 mm | 10 mm |

The attempt is material progress: failed gates fell from four to three,
residual speed newly passed, and the largest normalized excess fell from 7.99
to 2.05. Braking, stop time, and endpoint remain the only failures. Among 3809
usable candidates, 100 failed exactly those three gates and none failed fewer.
The independent gate champions reached 6.413 mm braking distance, 0.140 s stop
time, and 15.799 mm endpoint error, so no declared candidate passed any of the
three remaining gates individually.

Per-event diagnostics localized the selected candidate's worst braking error
to the lateral reversal: Lab traveled 40.44 mm after braking versus PX4's
52.54 mm. The worst stop-time mismatch moved to the vertical reversal, where
Lab stopped 0.24 s later. On the combined turn, Lab under-traveled the final
PX4 position by approximately 6.7 mm in X, 20.6 mm in Y, and 21.5 mm in Z.
The existing vector candidates always tie horizontal axes and change the
vertical gain with both, so these conflicting axis requirements cannot be
combined even though their individual champions are close.

Attempt 3 adds three evidence-backed candidate degrees of freedom:
axis-separated velocity P, I, and D vectors. The new candidates preserve the
passing forward settings while varying lateral persistence and vertical
bandwidth. No plant field, probe, gate, reward, actor, or policy interface
changes. The deterministic 32-setting beam remains.

Attempt 2 also validated the useful-parallelism change. One worker averaged
30.22% GPU utilization and 4580 MiB peak memory, but processed 6.75 candidate
evaluations/s, 28.1% more than attempt 1's 5.26 candidates/s. This is retained
because it increased independent useful throughput; additional worker
processes remain disabled because their measured contention reduced it.

### 58. Reject v3 attempt 3 and authorize hover-thrust attempt 4

Date: 2026-07-30.

Attempt 3 ran from source commit `1f91de7` and completed
`x500-velocity-px4-v3/20260730T055107Z`. The sweep evaluated 4767 unique
candidates and selected candidate 4743 under comparison fingerprint
`54ced12bbeb14d57048b5ae7850f2b72daf09e5a28bd15f01aaa9600bac244de`.

| Held-out metric | Attempt 3 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 4.281 mm | 5 mm |
| Residual-speed error | 0.00395 m/s | 0.01 m/s |
| Stop-time error | 0.060 s | 0.10 s |
| Worst-axis velocity NRMSE | 25.87% | 10% |
| Yaw-rate NRMSE | 4.77% | 10% |
| Heading error | 2.491 deg | 3 deg |
| Combined endpoint error | 38.534 mm | 10 mm |

This is material progress: failed gates fell from three to two, and braking
distance and stop time newly passed without regressing residual speed, yaw
rate, or heading. Of 4662 usable candidates, 605 failed exactly two gates and
every one of those candidates failed the same pair: velocity NRMSE and
combined endpoint. Individual candidates passed each remaining gate
(candidate-wide minima were 8.87% NRMSE and 9.166 mm endpoint), but no
candidate combined them. The selected NRMSE is vertical-dominated at 25.87%;
X and Y were 12.24% and 4.34% respectively.

The selected held-out combined turn ended 13.8 mm beyond PX4 in X, 16.9 mm
beyond it in Y, and 31.8 mm above it in Z. PX4 lost 44.6 mm of altitude over
that nominally horizontal trace while the reset Lab controller lost only
12.8 mm. This and the vertical NRMSE frontier identify the collective
acceleration mapping as the next structural seam, not another reason to widen
the already independent velocity gains.

Pinned PX4 v1.17 computes normalized collective thrust from the active hover
thrust estimate and vertical acceleration, then divides by the desired body-Z
vertical projection. The Lab cascade instead uses ideal physical
`mass * (acceleration + gravity)` feed-forward and has no aligned hover-thrust
setting. The diagnostic ULog from the last manual run records
`MPC_THR_HOVER=0.5`, `MPC_USE_HTE=1`, and an in-flight hover estimate spanning
0.5488 to 0.5812 with median 0.5721. This ULog remains diagnostic evidence
only; attempt 4 will fit the new setting exclusively from the alignment
probes.

Attempt 4 therefore adds the missing PX4 collective-thrust equation and a
declarative hover-thrust-fraction parameter centered on 0.57. It starts a new
controller generation and artifact schema so v3 artifacts retain their exact
physical-feed-forward behavior. NumPy and Torch use the same equation and
receive sequential parity and legacy-artifact tests. This is the first
new-field expansion in the campaign and is evidence-backed rather than a
repeat of the attempt-3 search space.

Attempt 3 used one worker, 37 GPU batches, and 938.31 s for the useful sweep,
or 5.08 candidate evaluations/s. GPU utilization averaged 26.01%, reached
37% at p95 and 49% at peak, and used 4776 MiB peak memory. The retained large
batches already contain independent candidate environments; Isaac/PhysX
sequential stepping is the measured bottleneck. Multi-process sharding remains
disabled because it previously reduced useful throughput by 3.27 times.

### 59. Reject v4 attempt 4 and correct the hover-thrust plant boundary

Date: 2026-07-30.

Attempt 4 ran from source commit `1354344` and completed
`x500-velocity-px4-v4/20260730T062943Z`. The sweep evaluated 4900 unique
candidates and selected candidate 4700 under comparison fingerprint
`e355bbe5ca7d2c8f116091e0db312797faa356ade28d2eb29a5306a9843485ed`.

| Held-out metric | Attempt 4 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 398.587 mm | 5 mm |
| Residual-speed error | 0.27203 m/s | 0.01 m/s |
| Stop-time error | 1.240 s | 0.10 s |
| Worst-axis velocity NRMSE | 487.38% | 10% |
| Yaw-rate NRMSE | 3.80% | 10% |
| Heading error | 2.451 deg | 3 deg |
| Combined endpoint error | 864.570 mm | 10 mm |

This is the first no-progress attempt after a new-field expansion. All 4741
usable candidates failed at least five gates. The selected hover-thrust
fraction reached the search minimum of 0.45, and even the independent minima
were 395.6 mm braking error, 1.240 s stop-time error, 178.66% velocity NRMSE,
and 320.3 mm endpoint error. The range, rather than another gain tradeoff, was
therefore invalid.

The invalid range came from comparing two different thrust boundaries.
PX4's hover-thrust estimate is a normalized rotor command. The Lab field is
consumed after allocation as a physical collective-thrust fraction. With
`THR_MDL_FAC=0`, bridge input scaling of 1000 rad/s, rotor maximum 1100 rad/s,
and the shared quadratic thrust curve, normalized estimate `h` maps to the
physical fraction `(h * 1000 / 1100)^2`. The diagnostic estimate range
0.5488 to 0.5812 therefore maps to 0.2489 to 0.2791, with median 0.2705.
Attempt 4 incorrectly searched 0.45 to 0.65, guaranteeing excessive thrust.

Attempt 5 is the final campaign attempt. It preserves the schema-4 equation
and corrects only this confirmed plant-boundary conversion, searching physical
fractions 0.23 through 0.32 around 0.27. The ULog values choose the range but
remain diagnostic only; selection still uses fresh held-out alignment traces.
If attempt 5 does not pass every exact-action gate, record the terminal
frontier and stop the campaign without training.

Attempt 4 used one worker, 39 GPU batches, and 983.21 s for the useful sweep,
or 4.98 candidate evaluations/s. GPU utilization averaged 26.73%, reached
38% at p95 and 49% at peak, and used 4840 MiB peak memory. This repeats the
measured Isaac/PhysX sequential-stepping bottleneck; adding worker processes
would reintroduce the previously measured throughput regression.

### 60. Stop after v4 attempt 5: exact-action alignment remains rejected

Date: 2026-07-30.

Attempt 5 ran from source commit `9bcc79e` and completed
`x500-velocity-px4-v4/20260730T065643Z`. The sweep evaluated 4831 unique
candidates and selected candidate 4495 under comparison fingerprint
`fc79958ed6fcd7a1f492f7879aead990fa658f78c0e7e1ab9b622ff18a95d0d9`.

| Held-out metric | Attempt 5 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 758.484 mm | 5 mm |
| Residual-speed error | 0.28091 m/s | 0.01 m/s |
| Stop-time error | 2.200 s | 0.10 s |
| Worst-axis velocity NRMSE | 185.95% | 10% |
| Yaw-rate NRMSE | 3.36% | 10% |
| Heading error | 2.741 deg | 3 deg |
| Combined endpoint error | 1394.719 mm | 10 mm |

This is the second consecutive no-progress attempt after the hover-thrust
field expansion, and it reaches the campaign limit of five completed attempts.
All 4713 usable candidates failed at least five gates. Candidate-wide minima
were 42.57 mm braking error, 0.01495 m/s residual speed, 1.040 s stop-time
error, 185.71% velocity NRMSE, and 1394.717 mm endpoint error. Only yaw rate
and heading could pass. The selected hover fraction reached the corrected
range maximum of 0.32, while attempt 4 selected its original range minimum of
0.45. A scalar physical hover fraction between those ranges is not enough to
represent the PX4 normalized collective, control-allocation output, actuator
command, and quadratic rotor-thrust chain.

The terminal diagnosis is that the new schema-4 collective mapping was placed
at the wrong abstraction boundary. A future campaign should not widen this
scalar again. It should capture PX4 normalized thrust setpoint, allocated motor
commands, and realized collective on the same timestamps; model or fit the
normalized-collective-to-per-rotor command curve before physical allocation;
and include any required thrust-model factor or allocator normalization as
declarative settings. The v3 attempt-3 artifact remains the strongest
diagnostic frontier at two failed gates, but it is advisory-failed and must not
authorize training.

The exact-action gate failed, so the alignment integration target,
closed-loop observation/action/state divergence comparison, retraining,
checkpoint selection, physical transfer, and complete mission were not run.
No policy or reward conclusion can be drawn from this campaign.

Attempt 5 used one worker, 39 GPU batches, and 1011.74 s for the useful sweep,
or 4.77 candidate evaluations/s. GPU utilization averaged 28.07%, reached
41% at p95 and 52% at peak, and used 4776 MiB peak memory. Independent
candidate environments were already batched; sequential Isaac/PhysX stepping
remained the limiting factor.

Decision: stop the alignment campaign under both terminal conditions: five
completed attempts and two consecutive no-progress attempts after parameter
expansion. Preserve every trace, candidate set, fit artifact, and runtime log
under `.build/`, commit this terminal diagnosis, keep the accepted training
pointer unchanged, and shut down the machine. A future run must begin from
the actuator-boundary diagnosis above rather than silently continuing this
search.

### 61. Correct the attempt-5 plant diagnosis and authorize a fresh campaign

Date: 2026-07-30.

Section 60 used the generic bridge defaults, `omega = 100 + 1000*u` with a
1100 rad/s maximum, instead of the composed Holybro X500 configuration. The
actual vehicle mapping is `omega = 50 + 475*u` with a 525 rad/s maximum. Its
composed 2.3472 kg mass and 61.389629 N maximum collective thrust require a
physical hover fraction of 0.37495 at standard gravity. Independently, the
latest PX4 hover estimate maps through the composed affine motor boundary and
quadratic rotor curve to approximately 0.37505. Attempts 4 and 5 therefore
searched above and below the evidenced boundary without testing it.

This correction is appended rather than rewriting the historical attempt-5
record. Reopen alignment as a fresh campaign of at most five completed
attempts. Attempt 1 is a schema-4 calibration control: keep
`x500-velocity-px4-v4`, every probe, advisory gate, controller equation, policy
interface, reward, and training setting unchanged; set the base physical hover
fraction to 0.375 and search
`[0.35, 0.36, 0.365, 0.375, 0.385, 0.39, 0.40]`. Compare material progress with
v3 attempt 3, which failed two gates with a largest normalized excess of 2.853.
This control does not start the consecutive schema-5 stall counter.

Every fresh capture now records absolute per-profile PX4 simulation start
times and keeps `reference.json`, `reference.ulg`, `bridge.log`,
`actuator_audit.json`, `target_candidates.json`, `gpu_utilization.json`, and
`alignment.json` under one run directory described by `manifest.json`. The
actuator-boundary audit must cover thrust and torque setpoints, motor controls,
bridge rotor speed and reconstructed wrench, and hover estimate for every
profile. It fails on incomplete coverage, timestamp error above 4 ms,
normalized motor RMSE above 0.01, or collective RMSE above 0.05 N.

If this corrected control passes all seven exact-action limits, proceed to the
alignment integration gate and the diagnostic closed-loop actor comparison
without schema 5. If it remains rejected, implement the normalized PX4
allocator boundary as schema 5 before beginning the no-progress streak. Do not
retrain during this campaign.

### 62. Reject the corrected schema-4 control and authorize schema 5

Date: 2026-07-30.

The calibration control ran from clean source commit `66d4468` and completed
`x500-velocity-px4-v4/20260730T164305Z`. The sweep evaluated 5099 unique
candidates and selected candidate 4110 under comparison fingerprint
`16448ce91e56aaacba2a36eb60991b8be30167fa6888b7581fa25103525e1c67`.
The selected physical hover fraction is 0.375.

| Held-out metric | Attempt 1 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 15.066 mm | 5 mm |
| Residual-speed error | 0.00925 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 9.95% | 10% |
| Yaw-rate NRMSE | 4.51% | 10% |
| Heading error | 2.730 deg | 3 deg |
| Combined endpoint error | 30.211 mm | 10 mm |

The control remains rejected with two failed gates. Its largest normalized
excess fell from the v3 attempt-3 frontier's 2.853 to 2.021, a 29.2% reduction,
but braking distance regressed from passed to failed. It therefore does not
meet the campaign definition of material progress. This calibration control
does not start the schema-5 consecutive-stall counter.

All 5000 usable candidates failed at least two gates; 518 failed exactly
braking distance and combined endpoint, and none failed fewer. Candidate-wide
minima were 3.927 mm braking distance, 0.00076 m/s residual speed, 0.060 s stop
time, 6.34% velocity NRMSE, 2.66% yaw-rate NRMSE, 2.303 degrees heading error,
and 14.137 mm endpoint error. Thus no declared schema-4 candidate passed the
endpoint gate, and no candidate combined the passing braking champion with
the endpoint frontier.

The actuator audit passed every fit, validation, and stability profile.
Timestamp joins were exact, normalized motor RMSE ranged from 0.00027 to
0.00030, and reconstructed collective RMSE ranged from 0.0088 N to 0.0109 N.
The captured hover estimate ranged from 0.57077 to 0.57316 with median 0.57162,
while normalized vertical thrust ranged from -0.64765 to -0.44687. Motor
controls, the composed `50 + 475*u` mapping, rotor speed, and reconstructed
wrench agree, so the first unresolved target boundary remains the normalized
thrust-setpoint and PX4 allocator behavior that schema 4 does not represent.
The next attempt must implement schema 5 at that boundary; the confirmed plant
mapping must remain unchanged.

Attempt 1 used one worker, 39 GPU batches, and 1036.29 s for the useful sweep,
or 4.92 candidate evaluations/s. GPU utilization averaged 28.49%, reached 40%
at p95 and 57% at peak, and used 5632 MiB peak memory. The recorded bottleneck
remains sequential Isaac/PhysX stepping, so worker count stays at one.

### 63. Start schema-5 normalized allocator attempt

Date: 2026-07-30.

Attempt 2 starts `x500-velocity-px4-v5` and artifact schema 5 at the first
unresolved boundary from the corrected control. The velocity path now produces
normalized thrust from the acceleration request and
`hover_thrust_normalized`, sends normalized rate-PID torque directly to
allocation, reproduces the pinned PX4 v1.17 `CA_METHOD=2`, `MC_AIRMODE=0`
pseudo-inverse normalization and sequential desaturation, clips motor controls
to `[0, 1]`, then applies the composed bridge mapping
`omega = 50 + 475*u` and the existing quadratic rotor and actuator models.
Schemas 2 through 4 load with a zero normalized-hover sentinel and keep their
physical-force behavior.

The normalized-hover candidates are
`[0.54, 0.55, 0.56, 0.57, 0.572, 0.58, 0.59, 0.60]`, bracketing the captured
0.57077-0.57316 estimator interval. The existing velocity, attitude, rate,
delay, filter, tilt, and actuator stages remain unchanged. Schema 5 no longer
fits either the legacy physical hover fraction or physical rate-output scale.
The probes, seven advisory gates, policy interface, reward, training
configuration, and plant definition are unchanged.

As a pre-execution boundary check, replaying the captured PX4 normalized thrust
and torque setpoints through the generated X500 effectiveness matrix and the
new allocator against 1221 joined `actuator_motors` samples produced normalized
motor RMSE 0.00512. This is below the 0.01 audit limit and supports the
allocator signs, normalization, and desaturation rather than using gain fitting
to hide a plant mismatch. The prior actuator audit independently passed the
bridge mapping, rotor speed, and collective boundary.

This is the first schema-5 attempt and the first entry in the consecutive
no-progress streak. It will run capture, actuator audit, the staged sweep, and
held-out fit from the clean implementation commit with one worker. Any rejected
result will be routed from the first failed intermediate boundary before a
further structural or declarative search change is authorized.

### 64. Reject schema-5 attempt 2 at normalized thrust

Date: 2026-07-30.

Attempt 2 ran from clean source commit `9942dbc` and completed
`x500-velocity-px4-v5/20260730T173924Z`. The sweep evaluated 4933 unique
candidates, 4837 of them usable, and selected candidate 4457 under comparison
fingerprint
`46ac0d7f46a7c9d681857af061ef3e4bd98ae0ca9158bd1647a65cb73521211e`.
The selected normalized hover thrust is 0.572.

| Held-out metric | Attempt 2 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 20.789 mm | 5 mm |
| Residual-speed error | 0.00691 m/s | 0.01 m/s |
| Stop-time error | 0.300 s | 0.10 s |
| Worst-axis velocity NRMSE | 16.41% | 10% |
| Yaw-rate NRMSE | 4.42% | 10% |
| Heading error | 2.854 deg | 3 deg |
| Combined endpoint error | 7.666 mm | 10 mm |

The attempt is rejected with three failed gates. Its largest normalized excess
is 3.158, worse than both the corrected schema-4 control's 2.021 and the v3
attempt-3 frontier's 2.853, while stop time and worst-axis NRMSE regress from
passed to failed. This is the first consecutive schema-5 no-progress result.

No usable candidate passed fewer than three gates: 628 failed three, 1101
failed four, 2764 failed five, 343 failed six, and one failed all seven.
Candidate-wide minima were 8.424 mm braking distance, 0.00019 m/s residual
speed, 0.260 s stop time, 9.54% worst-axis velocity NRMSE, 2.83% yaw-rate
NRMSE, 2.354 degrees heading error, and 2.966 mm endpoint error. Thus no
declared candidate passed either braking distance or stop time.

The fresh actuator audit again passed every profile. Timestamp joins stayed
within one 4 ms physics tick, normalized motor RMSE ranged from 0.00030 to
0.00123, and reconstructed collective RMSE ranged from 0.0078 N to 0.0120 N.
Matched-state replay of the selected schema-5 controller remained below the
0.01 normalized-motor limit on the forward, lateral, and vertical validation
profiles at 0.00793, 0.00870, and 0.00922 respectively. Collective RMSE,
however, rose to 0.255 N on the vertical profile and just exceeded 0.05 N on
the yaw and combined profiles. The selected velocity low-pass reached its
5 Hz upper bound and the velocity-derivative filter reached its 2.5 Hz lower
bound, while the selected normalized hover value remained inside the measured
0.57077-0.57316 estimator interval.

The force-application seam was inspected because controller outputs were close
while motion still failed. The pinned Isaac Lab wrench composer and Isaac Sim
wrapper apply a position-less local force at the body-frame origin, matching
the target's origin-reduced rotor torque and the PX4 force application. No
center-of-mass or drag application mismatch was confirmed, so the plant remains
unchanged. The first evidenced unresolved boundary is normalized thrust.
Attempt 3 may extend the evidenced velocity-filter ranges and add declarative
velocity-integral limits while retaining the current normalized-hover range.
Hover-estimate dynamics are not authorized because its captured variation is
small and does not correlate with the error.

Attempt 2 used one worker, 36 GPU batches, and 1020.76 s for the useful sweep,
or 4.83 candidate evaluations/s. GPU utilization averaged 30.50%, reached 42%
at p95 and 50% at peak, and used 5633 MiB peak memory. Sequential Isaac/PhysX
stepping remains the bottleneck, so worker count stays at one.

### 65. Expand the evidenced normalized-thrust search for attempt 3

Date: 2026-07-30.

Attempt 3 retains schema 5, the normalized allocator implementation, the
complete 0.54-0.60 normalized-hover range, every probe and advisory gate, and
the composed plant unchanged. It makes no structural controller or plant
change. At the first failed normalized-thrust boundary it adds the required
declarative velocity-integral-limit stage with lower, pinned, and upper
candidates `[1, 1, 2]`, `[2, 2, 4]`, and `[3, 3, 6]` m/s2. Four new
evidence-routed range candidates are introduced in total: the lower and upper
integral-limit candidates, a 10 Hz velocity low-pass candidate beyond the
selected 5 Hz boundary, and a 1 Hz velocity-derivative candidate below the
selected 2.5 Hz boundary.

The hypothesis is that the candidate-wide braking and stop-time floor comes
from insufficient velocity-loop state and filter range rather than allocator
or plant scaling. Material progress still requires fewer failed gates or at
least a 20% reduction from the 3.158 largest normalized excess without
regressing a passed gate. This is the second consecutive schema-5 attempt; if
it does not make material progress, the campaign terminates at the required
two-attempt no-progress limit. It will use one worker and run from a new clean
source commit after the complete host quality gates.

### 66. Continue after schema-5 attempt 3 makes material progress

Date: 2026-07-30.

Attempt 3 ran from clean source commit `2dedeca` and completed
`x500-velocity-px4-v5/20260730T181602Z`. The sweep evaluated 4991 unique
candidates, 4892 of them usable, and selected candidate 4780 under comparison
fingerprint
`0d0da659af0ec8fcfd2fb252eb8eeb92f89f50f5816a59492b16fabd7ea367cb`.
The selected velocity-derivative filter is the new 1 Hz candidate.

| Held-out metric | Attempt 3 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 4.935 mm | 5 mm |
| Residual-speed error | 0.00796 m/s | 0.01 m/s |
| Stop-time error | 0.060 s | 0.10 s |
| Worst-axis velocity NRMSE | 12.97% | 10% |
| Yaw-rate NRMSE | 4.74% | 10% |
| Heading error | 2.724 deg | 3 deg |
| Combined endpoint error | 76.228 mm | 10 mm |

The attempt remains rejected, but it makes material progress by reducing the
failed-gate count from three to two without regressing any gate that passed in
attempt 2. This resets the no-progress streak. There are 431 candidates with
exactly the same two failures, 428 with three, 751 with four, 2333 with five,
and 949 with six. Candidate-wide minima pass all seven limits independently:
3.073 mm braking error, 0.00022 m/s residual speed, 0.040 s stop-time error,
6.96% velocity NRMSE, 2.82% yaw-rate NRMSE, 2.245 degrees heading error, and
8.553 mm endpoint error. No candidate in the declared table combines the
passing velocity-NRMSE and endpoint regions.

All 431 two-failure candidates share the same stage-1 controller settings:
zero delay, `kp_vel=[1.35, 1.35, 3]`, `ki_vel=[0.2, 0.2, 1]`,
`kd_vel=[0.2, 0.2, 0]`, a 5 Hz velocity filter, a 1 Hz derivative filter,
`velocity_integral_limit_mps2=[2, 2, 4]`, and normalized hover thrust 0.572.
The held-out forward profile has 0.01095 m/s x-velocity RMSE and 0.00407 m/s
bias. The combined profile has 0.01403 m/s y-velocity RMSE and 0.01245 m/s
lateral bias; that persistent bias accounts for most of its 76 mm endpoint
error. The existing asymmetric proportional-gain candidates all force vertical
gain to 6 and do not test the evidenced horizontal frontier while preserving
the selected vertical gain of 3.

The fresh actuator audit passed every fit, validation, and stability profile.
Timestamp joins stayed within one 4 ms tick, normalized motor RMSE ranged from
0.00028 to 0.00259, and reconstructed collective RMSE ranged from 0.0083 N to
0.0112 N. Matched-state replay of the selected controller stayed within both
limits on the forward and lateral validation profiles. Its dynamic mismatch
first crossed the collective limit on the vertical profile and the motor limit
on yaw/combined profiles; stability crossed both. Because the direct ULog to
bridge audit passes those same profiles, this does not evidence a plant-owned
offset, scale, rotor curve, mass, or geometry error.

Attempt 4 therefore makes no structural change and adds four complete
axis-separated `kp_vel` candidates around the selected frontier:
`[1.575, 1.8, 3]`, `[1.575, 2.25, 3]`, `[1.8, 1.8, 3]`, and
`[1.8, 2.25, 3]`. The hypothesis is that modestly higher forward bandwidth
will pass worst-axis NRMSE while independent lateral bandwidth removes the
combined-profile bias without discarding the passing vertical, braking, and
stop-time response.

The user has explicitly authorized additional attempts for as long as an
attempt makes progress or trace evidence supports another meaningful test.
That instruction supersedes the earlier five-attempt and two-no-progress
terminal limits for this active campaign. Every attempt still requires a
committed hypothesis, clean-source capture, audit, sweep, fit, result record,
and the unchanged exact-action acceptance gates.

Attempt 3 used one worker, 39 GPU batches, and 1112.21 s for the useful sweep,
or 4.49 candidate evaluations/s. GPU utilization averaged 29.86%, reached 42%
at p95 and 50% at peak, and used 5506 MiB peak memory. Sequential
Isaac/PhysX stepping remains the measured bottleneck, so worker count stays at
one.

### 67. Reject attempt 4 and locate the per-profile controller-state boundary

Date: 2026-07-30.

Attempt 4 ran from clean source commit `5a35258` and completed
`x500-velocity-px4-v5/20260730T185057Z`. The sweep evaluated 5463 unique
candidates, 5364 of them usable, and selected candidate 5279 under comparison
fingerprint
`6d5ed3ee77c0fce0bbf5fe52b6892815b2dd06a1f6d8fb1b981e4745f6a76fb8`.

| Held-out metric | Attempt 4 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 12.472 mm | 5 mm |
| Residual-speed error | 0.00362 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 8.82% | 10% |
| Yaw-rate NRMSE | 4.72% | 10% |
| Heading error | 2.710 deg | 3 deg |
| Combined endpoint error | 54.270 mm | 10 mm |

The attempt remains rejected with two failed gates. Worst-axis NRMSE now
passes and the largest normalized excess falls 33.2%, from 6.623 to 4.427,
but braking distance regresses from passed to failed. This is therefore not
material progress under the strict campaign definition.

Only two candidates failed exactly two gates. Candidate-wide minima were
2.551 mm braking error, 0.00019 m/s residual speed, zero stop-time error,
8.30% velocity NRMSE, 2.81% yaw-rate NRMSE, 2.250 degrees heading error, and
26.303 mm endpoint error. Unlike attempt 3, no candidate passed the endpoint
gate. The four new proportional-gain vectors did not create an endpoint
frontier: their endpoint minima ranged from 50.5 mm to 60.9 mm. The selected
candidate returned to `kp_vel=[1.35, 1.35, 3]`.

This fresh-capture change is evidence of a missing initial-state contract, not
another gain range. The combined profile began with world velocity
`[-0.0110, 0.0028, -0.0067]` m/s, whereas attempts 2 and 3 began at
`[0.0031, -0.0022, -0.0023]` and
`[0.0108, -0.0124, -0.0007]` m/s. PX4 preconditions each profile without
resetting its controller. Lab restores the recorded rigid-body state but
clears velocity filters, acceleration, velocity integral, rate derivative,
and rate integral to zero. Thus the two backends do not start at the same
controller boundary even though they start at the same kinematic state.

The ULog exposes the missing boundary. `vehicle_local_position_setpoint`
records PX4's velocity-loop acceleration output,
`vehicle_local_position` records estimated acceleration,
`vehicle_angular_velocity` records angular acceleration, and
`rate_ctrl_status` records the rate integrals. At every held-out profile start,
the PX4 pitch-rate integral was approximately 0.0122 rather than zero.
Derived velocity-loop bias terms ranged to approximately 0.036 m/s2. The
selected-controller replay consequently missed collective thrust by
0.106-0.220 N even on the forward, lateral, and vertical profiles, despite
the direct ULog-to-bridge audit passing all profiles at 0.0080-0.0112 N.

Attempt 5 makes one structural change and no declarative parameter additions.
Reference trace schema 4 will record the per-profile PX4 acceleration
setpoint, measured linear and angular acceleration, and rate-integral state
at the absolute profile start. The target will restore these values after
kinematic reset. For each candidate, its velocity integral will be initialized
so its first neutral-command desired acceleration equals the logged PX4
acceleration setpoint; rate integral and derivative state will be restored in
body-FLU coordinates. Schemas 2 and 3 retain zero-state replay. The target
controller ID and definition fingerprint will advance because replay behavior
changes, while artifact schema 5, probes, gates, plant, allocator, settings,
policy interface, reward, and training configuration remain unchanged.

Attempt 4 used one worker, 39 GPU batches, and 1332.64 s for the useful sweep,
or 4.10 candidate evaluations/s. GPU utilization averaged 29.03%, reached 42%
at p95 and 50% at peak, and used 5441 MiB peak memory. Sequential
Isaac/PhysX stepping remains the bottleneck, so worker count stays at one.

### 68. Reject attempt 5 at the force-application boundary

Date: 2026-07-30.

Attempt 5 ran from clean source commit `16d234c` and completed
`x500-velocity-px4-v5/20260730T194712Z`. Its schema-4 capture contains valid
controller-state snapshots for all 11 profiles. The actuator audit passed every
profile: timestamp joins stayed within one 4 ms physics tick, normalized motor
RMSE ranged from 0.00027 to 0.00084, and reconstructed collective RMSE ranged
from 0.0081 N to 0.0122 N. The sweep evaluated 4971 unique candidates, 4866 of
them usable, and selected candidate 4520 under comparison fingerprint
`60472abacac866598f97caf5f1d24c00281f6d1ea74000943030029b0bd36c5d`.

| Held-out metric | Attempt 5 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 30.402 mm | 5 mm |
| Residual-speed error | 0.00340 m/s | 0.01 m/s |
| Stop-time error | 0.640 s | 0.10 s |
| Worst-axis velocity NRMSE | 22.78% | 10% |
| Yaw-rate NRMSE | 5.41% | 10% |
| Heading error | 2.650 deg | 3 deg |
| Combined endpoint error | 51.681 mm | 10 mm |

The attempt is rejected with four failed gates and a largest normalized excess
of 5.400, regressing from attempt 4. None of the usable candidates failed fewer
than four gates: 816 failed four, 3019 failed five, and 1031 failed six.
Candidate-wide minima were 13.718 mm braking error, 0.00065 m/s residual speed,
0.300 s stop-time error, 11.12% velocity NRMSE, 3.13% yaw-rate NRMSE,
2.362 degrees heading error, and 11.246 mm endpoint error. Thus restoring the
complete logged state cannot pass braking, stop time, velocity NRMSE, or
endpoint error anywhere in the declared parameter table.

The regression appears before the first maneuver rather than at a tunable
controller range. During each one-second neutral lead-in, the restored PX4
pitch-rate integral drives the Lab vehicle approximately 0.01-0.03 m/s in
negative X while the PX4 reference remains near trim. The generated X500 body
has a 3.067 mm positive-X center-of-mass offset. PX4 applies rotor forces at
their local positions, so collective thrust at the body origin contributes the
corresponding trim torque. Lab reduces the rotor forces to an origin-referenced
wrench but calls `set_external_force_and_torque` without a position; the pinned
API forwards a null position to PhysX, which applies the force at the center of
mass. This drops approximately 0.071 N m of hover pitch moment, consistent with
the approximately 0.012 normalized PX4 pitch-rate integral and its measured
torque scale.

The first failed boundary is therefore force application, after normalized
controller output, allocation, bridge mapping, rotor speed, and collective
reconstruction have passed. The next attempt will make one plant-seam change:
apply the reduced Lab wrench explicitly at the body-frame origin. Controller
state, candidates, probes, gates, allocator, mass, geometry, drag, policy
interface, reward, and training configuration remain unchanged.

Attempt 5 used one worker, 39 GPU batches, and 1303.68 s for the useful sweep,
or 3.81 candidate evaluations/s. GPU utilization averaged 29.58%, reached 43%
at p95 and 49% at peak, and used 5761 MiB peak memory. Sequential Isaac/PhysX
stepping remains the measured bottleneck, so worker count stays at one.

### 69. Preserve attempt 6 semantics after a positioned-force infrastructure stall

Date: 2026-07-30.

Attempt 6's first execution from commit `ba11e77` completed the fresh capture
and actuator audit, then stalled in the third target batch. The first two
batches retained the historical approximately 30-second duration, but the
first expanded batch remained inside PhysX for more than ten minutes without a
simulation completion marker. The worker and GPU remained allocated, so this
is preserved as an incomplete infrastructure run rather than an alignment
result.

The implementation used an explicit zero body-frame force position to retain
the X500 body's COM-offset moment. The pinned Isaac Lab wrench composer accepts
that position but incurs a pathological positioned-force path at the
4224-environment alignment capacity. Attempt 6 therefore retains the same
physical hypothesis while using the equivalent COM-referenced wrench:
`torque_at_com = torque_at_origin - com_position x force`. The position-less
force can then use the previously stable high-throughput PhysX path while
producing the same net force and moment as the bridge's rotor forces. No
controller, candidate, probe, gate, plant property, or worker setting changes.
The existing fresh capture and passed actuator audit remain valid; the target
sweep will restart from the new clean implementation commit.

### 70. Continue after attempt 6 confirms the body-origin correction

Date: 2026-07-30.

Attempt 6 completed from clean source commit `79be38b` as
`x500-velocity-px4-v5/20260730T204909Z`. The final run used a fresh
exact-commit capture rather than reusing the valid pre-correction capture. Its
actuator audit passed all profiles: timestamp joins stayed within one 4 ms
tick, normalized motor RMSE peaked at 0.00074, and reconstructed collective
RMSE peaked at 0.0118 N. The sweep evaluated 5146 unique candidates, 5035 of
them usable, and selected candidate 5092 under comparison fingerprint
`0156fc635a3648195652e0d3754e91919f0cdade3d32f0d3889cb7c749aea80f`.

| Held-out metric | Attempt 6 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 13.769 mm | 5 mm |
| Residual-speed error | 0.00498 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 8.91% | 10% |
| Yaw-rate NRMSE | 3.17% | 10% |
| Heading error | 2.982 deg | 3 deg |
| Combined endpoint error | 25.813 mm | 10 mm |

The attempt remains rejected with three failed gates; stop time exceeds its
limit only by floating-point representation at 0.10000000000000053 s. It
nevertheless makes material progress: the failed-gate count falls from four to
three, worst-axis velocity NRMSE passes, and the largest normalized excess
falls 67.5%, from 5.400 to 1.754. This resets the no-progress streak. There
were 867 usable candidates with three failed gates, 2421 with four, 1325 with
five, 412 with six, and 10 with all seven.

Candidate-wide minima were 11.435 mm braking error, 0.00091 m/s residual
speed, 0.100 s stop-time error, 8.32% velocity NRMSE, 3.11% yaw-rate NRMSE,
2.309 degrees heading error, and 5.899 mm endpoint error. Braking and stop time
therefore remain candidate-wide floors, while endpoint passes independently.
The selected delay is the 0.072 s upper boundary, normalized hover is 0.572,
and the selected horizontal velocity integral is the lowest nonzero
axis-separated candidate `[0, 0.2, 1]`.

Matched-state controller replay stays within both motor and collective limits
on the Stage 1 brake and lateral validation profiles. Their remaining motion
error is therefore downstream of the logged controller boundary, although the
body-origin correction reduced it substantially. Replay first exceeds the
motor limit on yaw and combined profiles; validation combined reaches 0.0176
normalized-motor RMSE and 0.102 N collective RMSE. The direct ULog-to-bridge
actuator audit continues to pass those profiles, so this is an evidenced
normalized torque/controller mismatch rather than an affine motor or rotor
curve error.

Attempt 7 makes no structural change and adds four evidence-routed declarative
entries: 0.084 s and 0.096 s command delays beyond the selected boundary, a
0.5715 normalized-hover candidate matching the captured held-out estimator
level, and a rate-integral-limit search around the pinned 0.30 normalized
limit. No hover-estimate dynamics are added because held-out variation is only
approximately 0.0002 and does not correlate with the failing profiles. The
existing probes, gates, controller state, allocator, plant, worker count,
policy interface, reward, and training configuration remain unchanged.

Attempt 6 used one worker, 39 GPU batches, and 1351.32 s for the useful sweep,
or 3.81 candidate evaluations/s. GPU utilization averaged 30.42%, reached 45%
at p95 and 61% at peak, and used 9120 MiB peak memory. Sequential Isaac/PhysX
stepping remains the measured bottleneck, so worker count stays at one.

### 71. Continue after attempt 7 isolates command-frame latching

Date: 2026-07-30.

Attempt 7 completed from clean source commit `93b579a` as
`x500-velocity-px4-v5/20260730T213019Z`. Its fresh capture restored valid
controller state for all 11 profiles, and the actuator audit passed every
profile within one 4 ms physics tick. The sweep evaluated 5715 unique
candidates, 5616 of them usable, and selected candidate 5344 under comparison
fingerprint
`246dab4718de078c0a6da086c38f90e548be34841a06134f2595d5f6301fab2c`.

| Held-out metric | Attempt 7 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 3.957 mm | 5 mm |
| Residual-speed error | 0.00469 m/s | 0.01 m/s |
| Stop-time error | 0.040 s | 0.10 s |
| Worst-axis velocity NRMSE | 3.02% | 10% |
| Yaw-rate NRMSE | 4.57% | 10% |
| Heading error | 3.728 deg | 3 deg |
| Combined endpoint error | 35.172 mm | 10 mm |

The attempt remains rejected with two failed gates, but it makes material
progress: the failed-gate count falls from three to two while braking and stop
time pass without regressing any attempt-6 passed gate. This resets the
no-progress streak. Among usable candidates, 1709 failed two gates, 710 failed
three, 1745 failed four, 933 failed five, and 519 failed six. Candidate-wide
minima were 2.383 mm braking error, 0.00021 m/s residual speed, 0.040 s
stop-time error, 2.59% velocity NRMSE, 3.01% yaw-rate NRMSE, 2.311 degrees
heading error, and 11.142 mm endpoint error. Every gate except endpoint
therefore passes independently in this search, but no evaluated candidate
combines fewer than two failures.

The selected settings return to zero command delay and 0.570 normalized hover.
The new rate-integral search selects `[0.4, 0.4, 0.4]`, but the 0.45 candidate
does not improve the failure count. Matched-state replay of the selected
controller stays below 0.01 normalized motor RMSE on the held-out yaw and
combined profiles, while collective RMSE reaches 0.111 N and 0.120 N. The
direct ULog-to-bridge actuator audit remains below 0.012 N on the same
profiles, locating the remaining difference before the plant-owned actuator
mapping.

The combined trace exposes a command-interface seam. The target ends 34.10 mm
positive in world Y relative to PX4 even though held-out Y-velocity NRMSE is
only 3.02%. PX4 v1.17 handles each `MAV_FRAME_BODY_NED` message by reading
attitude, rotating its body velocity into the local frame, and publishing that
local velocity once. The Lab controller instead rotates the retained body
command by the current yaw on every 4 ms physics tick. The discrepancy is
active only while body velocity and yaw rate are commanded together, matching
the remaining endpoint and heading failures.

Attempt 8 will make one structural command-boundary correction: latch the
world-frame velocity setpoint from body action and attitude once per policy
command update, then retain that local setpoint through the intervening
physics/controller ticks exactly as PX4 does. The normalized yaw-rate command,
controller candidates, controller state, allocator, plant, probes, gates,
worker count, policy interface, reward, and training configuration remain
unchanged.

Attempt 7 used one worker, 41 GPU batches, and 1134.52 s for the useful sweep,
or 5.04 candidate evaluations/s. GPU utilization averaged 32.04%, reached 50%
at p95 and 58% at peak, and used 9301 MiB peak memory. Sequential Isaac/PhysX
stepping remains the measured bottleneck, so worker count stays at one.

### 72. Refine attempt 8 latching to the measured transport cadence

Date: 2026-07-30.

Attempt 8 completed from clean source commit `a63eef7` as
`x500-velocity-px4-v5/20260730T221628Z`. Its fresh capture restored valid
controller state for all 11 profiles. The actuator audit passed every profile:
timestamp joins stayed within one 4 ms physics tick, normalized motor RMSE
peaked at 0.00083, and reconstructed collective RMSE peaked at 0.0123 N. The
sweep evaluated 5593 unique candidates, 5470 of them usable, and selected
candidate 5549 under comparison fingerprint
`20bafd5a1429ae9b71d900d310cba9866fd693c6d4e911e2c0cee405a18e2873`.

| Held-out metric | Attempt 8 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 4.459 mm | 5 mm |
| Residual-speed error | 0.00990 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 8.83% | 10% |
| Yaw-rate NRMSE | 7.50% | 10% |
| Heading error | 5.750 deg | 3 deg |
| Combined endpoint error | 38.437 mm | 10 mm |

The attempt remains rejected with the same two failed gates and a largest
normalized excess of 2.844, regressing from attempt 7's 2.517. It is the first
consecutive no-progress attempt. Among usable candidates, 1400 failed two
gates, 910 failed three, 1462 failed four, 1258 failed five, 438 failed six,
and 2 failed all seven. Candidate-wide minima were 1.781 mm braking error,
0.00026 m/s residual speed, 0.040 s stop-time error, 4.81% velocity NRMSE,
3.03% yaw-rate NRMSE, 2.358 degrees heading error, and 20.707 mm endpoint
error. The endpoint floor therefore worsened rather than improving.

The attempt-8 implementation latched the body-frame action when the 20 Hz
policy action entered the Lab environment. The capture log provides the
missing cadence evidence: the routine driver retransmits the retained
`MAV_FRAME_BODY_NED` setpoint every 20 ms, at its configured 50 Hz control
rate. PX4 v1.17 rotates the body velocity at every received message, not only
when the policy computes a new action. The attempt-8 target consequently held
each local-frame velocity setpoint for 50 ms, while PX4 refreshes it every
20 ms. The pre-attempt continuous 4 ms reinterpretation was closer to that
transport behavior, which explains the observed regression.

Attempt 9 will keep the local-frame latching correction but refresh it from the
retained body action and current attitude on the existing 50 Hz OFFBOARD
setpoint-stream cadence.
The 20 Hz policy action cadence remains unchanged. The cadence will use the
shared exact-average scheduler so it remains derived from configured physics
and driver rates. Controller candidates, controller state, allocator, plant,
probes, gates, worker count, policy interface, reward, and training
configuration remain unchanged.

Attempt 8 used one worker, 41 GPU batches, and 1480.14 s for the useful sweep,
or 3.78 candidate evaluations/s. GPU utilization averaged 31.78%, reached 45%
at p95 and 59% at peak, and used 9154 MiB peak memory. Sequential Isaac/PhysX
stepping remains the measured bottleneck, so worker count stays at one.

### 73. Return to the strongest frontier after attempt 9 rejects stream latching

Date: 2026-07-30.

Attempt 9 completed from clean source commit `a59fd94` as
`x500-velocity-px4-v5/20260730T225539Z`. Its fresh capture restored valid
controller state for all 11 profiles. The actuator audit passed every profile:
timestamp joins stayed within one 4 ms physics tick, normalized motor RMSE
peaked at 0.00076, and reconstructed collective RMSE peaked at 0.0123 N. The
sweep evaluated 5907 unique candidates, 5796 of them usable, and selected
candidate 4998 under comparison fingerprint
`df0f4a31f93781cceeefb34e9a66a3c8efe29f6900f83b792a53e3aebbdc1c40`.

| Held-out metric | Attempt 9 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 8.262 mm | 5 mm |
| Residual-speed error | 0.00301 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 7.31% | 10% |
| Yaw-rate NRMSE | 4.34% | 10% |
| Heading error | 2.805 deg | 3 deg |
| Combined endpoint error | 52.359 mm | 10 mm |

The attempt remains rejected with two failed gates. Although heading now
passes, braking regresses and the largest normalized excess rises to 4.236.
This is a second consecutive no-progress result for the stream-latching
branch. Among usable candidates, 1488 failed two gates, 727 failed three, 1847
failed four, 1214 failed five, and 520 failed six. Candidate-wide minima were
1.652 mm braking error, 0.00063 m/s residual speed, 0.060 s stop-time error,
4.38% velocity NRMSE, 3.18% yaw-rate NRMSE, 2.493 degrees heading error, and
24.154 mm endpoint error.

Refreshing the local setpoint at the measured 50 Hz OFFBOARD stream cadence
improves heading relative to attempt 8 but does not recover attempt 7's
braking or endpoint frontier. The structural branch is therefore rejected.
Attempt 10 will remove stream latching and return to attempt 7's controller-rate
body-action interpretation, whose selected candidate passed braking and whose
candidate-wide endpoint floor was 11.142 mm.

Attempt 7's retained frontier also exposes two confounded parameter tables.
Yaw feed-forward `[0.025, 0.025, 0.025]` produced passing heading but regressed
braking because roll and pitch feed-forward changed simultaneously. The
combined trace under-traveled world Y by 34.10 mm, while the available
Y-gain candidates also changed X or Z. Attempt 10 will therefore add exactly
four declarative candidates: yaw-only rate feed-forward `[0, 0, 0.025]`,
Y-only velocity proportional gain `[1.8, 2.25, 4.0]`, and Y-only velocity
integral gains `[0.4, 0.5, 2.0]` and `[0.4, 0.6, 2.0]`. No other controller,
allocator, plant, probe, gate, worker, policy-interface, reward, or training
change is made.

Attempt 9 used one worker, 41 GPU batches, and 1426.14 s for the useful sweep,
or 4.14 candidate evaluations/s. GPU utilization averaged 28.88%, reached 43%
at p95 and 53% at peak, and used 8742 MiB peak memory. Sequential Isaac/PhysX
stepping remains the measured bottleneck, so worker count stays at one.

### 74. Pause after attempt 10 rejects the isolated-axis expansion

Date: 2026-07-30.

Attempt 10 completed from clean source commit `4177a0a` as
`x500-velocity-px4-v5/20260730T233356Z`. Its fresh capture restored valid
controller state for all 11 profiles. The actuator audit passed every profile:
timestamp joins stayed within one 4 ms physics tick, normalized motor RMSE
peaked at 0.00143, and reconstructed collective RMSE peaked at 0.0131 N. The
sweep evaluated 6187 unique candidates, 6044 of them usable, and selected
candidate 6089 under comparison fingerprint
`be61546aa96c19ae78e34b296fd697dc191ee84034ef8d5b33da689a9957af7b`.

| Held-out metric | Attempt 10 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 11.525 mm | 5 mm |
| Residual-speed error | 0.00991 m/s | 0.01 m/s |
| Stop-time error | 0.440 s | 0.10 s |
| Worst-axis velocity NRMSE | 6.95% | 10% |
| Yaw-rate NRMSE | 4.17% | 10% |
| Heading error | 3.861 deg | 3 deg |
| Combined endpoint error | 44.701 mm | 10 mm |

The attempt is rejected with four failed gates and a largest normalized excess
of 3.470. It does not improve on attempt 7's two failures and 2.517 largest
excess. Among usable candidates, 2173 failed four gates, 2435 failed five,
1431 failed six, and 5 failed all seven; no candidate failed fewer than four
gates. Candidate-wide minima were 7.324 mm braking error, 0.00127 m/s residual
speed, 0.260 s stop-time error, 6.40% velocity NRMSE, 3.12% yaw-rate NRMSE,
2.186 degrees heading error, and 3.233 mm endpoint error. The isolated Y
integral candidate can therefore clear the endpoint gate, but it does not
combine with a braking or stop-time frontier. The isolated Y proportional
gain and yaw-only feed-forward likewise do not reduce the minimum failed-gate
count.

The selected settings use 0.024 s command delay, velocity P/I/D
`[1.35, 1.35, 3.0]`, `[0.3, 0.3, 1.5]`, and `[0.3, 0.3, 0]`, 2 Hz velocity
filtering, 1 Hz velocity-derivative filtering, normalized hover thrust 0.57,
attitude P `[5, 5, 3.5]`, yaw weight 0.6, rate P/I/D
`[0.15, 0.15, 0.2]`, zero, and `[0.0045, 0.0045, 0]`, zero rate
feed-forward, 30 Hz rate-derivative filtering, 0.4 rate-integral limits, and
0.01 s symmetric actuator time constants.

Direct ULog state replay places the first unresolved boundary at controller
collective output. The selected settings pass normalized motor RMSE for the
braking and combined validation windows at 0.00248 and 0.00874, but their
collective RMSE is 0.111 N and 0.0946 N, above the 0.05 N limit. The yaw
window additionally reaches 0.01157 normalized motor RMSE and 0.105 N
collective RMSE. The fresh reference-side bridge audit passes, so the composed
plant offset, scale, and rotor curve remain internally consistent. A resumed
campaign should inspect the normalized thrust-setpoint and allocator inputs
for the braking window before adding candidates. The candidate-wide stop-time
floor of 0.260 s is stronger evidence than a selected parameter touching a
range boundary. Preserve the passing 3.233 mm endpoint champion when routing
that investigation.

Attempt 10 used one worker, 41 GPU batches, and 1482.81 s for the useful
sweep, or 4.17 candidate evaluations/s. GPU utilization averaged 28.83%,
reached 43% at p95 and 51% at peak, and used 8741 MiB peak memory. Sequential
Isaac/PhysX stepping remains the measured bottleneck, so the one-worker
decision remains supported.

The campaign is paused at the user's request after this completed fitting
attempt. The source implementation, hypothesis, capture, ULog, bridge log,
reference trace, target candidates, actuator audit, GPU telemetry, manifest,
and rejected fit artifact are preserved. No attempt 11, integration acceptance
test, actor diagnostic, training run, or machine shutdown was started.

### 75. Correct the schema-5 attitude-to-collective boundary

Date: 2026-07-31.

Attempt 11 corrects one source-confirmed schema-5 controller boundary. PX4
constructs the desired normalized thrust vector in position control, converts
its direction into the attitude setpoint, and passes the vector magnitude as
the body-Z thrust setpoint. The Lab cascade instead projected the desired
thrust vector onto the current body Z. That made normalized collective depend
on transient attitude error even though PX4 preserves it while the attitude
loop rotates toward the desired direction.

The schema-5 NumPy and Torch paths will now send the limited desired thrust
vector's magnitude to the normalized allocator. Schemas 2 through 4 retain
their original physical-force projection behavior. The target controller
identifier advances to `isaac-lab-x500-velocity-v12`; artifact schema 5,
controller settings, candidates, probes, gates, worker count, policy
interface, reward, and training configuration remain unchanged.

Controller replay now consumes the recorded absolute start and duration of
every fit, validation, and stability profile. For unsaturated samples it
inverts the generated normalized allocator mix and reports collective and
torque error before reporting motor, rotor-speed, and physical-wrench error.
Saturated samples remain in output-boundary metrics but cannot be attributed
upstream through the inverse.

An offline replay of attempt 10 with the corrected controller shows that the
source correction is necessary but not sufficient by itself. The braking
profile's physical collective RMSE changes from 0.11080 N to 0.11057 N, while
its normalized collective RMSE is 0.00163 and normalized motor RMSE is
0.00274. No braking sample is saturated. The reference hover estimate for
that profile is 0.57153-0.57157, while attempt 10 selected normalized hover
0.570 and the unchanged search already contains 0.5715. Attempt 11 therefore
keeps the candidate table unchanged and tests whether the source-correct
boundary lets the measured-hover candidate survive the complete staged
ranking.

### 76. Attempt 11 materially improves the retained schema-5 frontier

Date: 2026-07-31.

Attempt 11 completed from clean source commit `6a0335d` as
`x500-velocity-px4-v5/20260731T045809Z`. Its fresh capture completed all 11
profiles with valid initial controller state. The reference actuator audit
passed every profile: timestamp joins stayed within one 4 ms physics tick,
normalized motor RMSE peaked at 0.00081, and reconstructed collective RMSE
peaked at 0.0117 N. The relative manifest preserves the ULog, bridge log,
reference trace, target candidates, actuator audit, controller-boundary
replay, and GPU telemetry in the run directory.

The sweep evaluated 5718 unique candidates, 5619 of them usable, and selected
candidate 5417 under comparison fingerprint
`7b93f29eca2dfb9b34728f3ec92e3e9dbd5a03bf4dd6b59442bb5eacab43241b`.

| Held-out metric | Attempt 11 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 4.479 mm | 5 mm |
| Residual-speed error | 0.00351 m/s | 0.01 m/s |
| Stop-time error | 0.10000000000000053 s | 0.10 s |
| Worst-axis velocity NRMSE | 8.92% | 10% |
| Yaw-rate NRMSE | 4.51% | 10% |
| Heading error | 3.130 deg | 3 deg |
| Combined endpoint error | 9.956 mm | 10 mm |

The attempt remains rejected with two failed gates. The largest normalized
excess is 1.043, down 58.6% from attempt 7's retained 2.517 frontier without
regressing a passed gate, so this is material progress and resets the
no-progress streak. The stop-time result lies one floating-point
representation step above the unchanged limit; the heading result exceeds its
limit by 0.130 degrees. Among usable candidates, 1432 failed two gates, 1057
failed three, 1071 failed four, 1493 failed five, and 566 failed six.
Candidate-wide minima were 2.208 mm braking error, 0.00263 m/s residual speed,
0.040 s stop-time error, 5.49% velocity NRMSE, 3.02% yaw-rate NRMSE, 2.265
degrees heading error, and 0.334 mm endpoint error.

The selected settings use zero command delay, velocity P/I/D
`[1.35, 1.35, 3.0]`, `[0.6, 0.6, 3.0]`, and `[0.3, 0.2, 0]`, no velocity
filter, a 1 Hz velocity-derivative filter, normalized hover thrust 0.57,
attitude P `[4, 4, 2.8]`, yaw weight 0.5, rate P/I/D
`[0.15, 0.15, 0.2]`, `[0.3, 0.3, 0.15]`, and
`[0.003, 0.003, 0]`, zero rate feed-forward, a 20 Hz rate-derivative filter,
0.3 rate-integral limits, 33.75 degrees maximum tilt, 0.02 s actuator rise,
and zero actuator fall time.

Automatic controller-state replay keeps low-speed normalized motor RMSE below
0.01, but the yaw and combined profiles first exceed that limit at the
normalized torque boundary. Validation yaw and combined normalized motor RMSE
are 0.0144 and 0.0133; their normalized yaw-torque RMSE is 0.0138 and 0.0122.
The stability envelope has broader normalized collective, torque, and motor
error. Physical collective RMSE also remains above 0.05 N despite small
low-speed normalized collective errors, peaking at 0.400 N in the vertical fit
profile and 1.933 N in stability. The reference-side bridge audit passes, so
the first controller-side boundary for the remaining heading gate is
normalized torque rather than plant mapping.

The source-correct collective change also exposes a narrow complementary
frontier. A yaw-only rate feed-forward of 0.025 lowers heading to 2.468 degrees
but moves the endpoint to 10.513 mm, while the selected zero-feed-forward
candidate passes endpoint at 9.956 mm and misses heading. The next attempt
should preserve the selected velocity frontier, fit an evidenced intermediate
yaw-only torque response, and add a matching isolated lateral candidate only
if needed to recover the 0.513 mm endpoint margin. Gates, probes, controller
interface, allocator, plant, worker count, policy interface, reward, and
training remain unchanged.

Attempt 11 used one worker, 41 GPU batches, and 1411.39 s for the useful sweep,
or 4.05 candidate evaluations/s. GPU utilization averaged 26.81%, reached 41%
at p95 and 48% at peak, and used 4441 MiB peak memory. Sequential
Isaac/PhysX stepping remains the measured bottleneck, so worker count stays at
one.

### 77. Retain and combine the complementary Attempt 11 champions

Date: 2026-07-31.

Attempt 12 expands the staged beam from 32 to 64 because Attempt 11 produced
complementary two-gate champions that did not coexist in the final retained
settings. Of 5619 usable candidates, 1286 failed only stop time and heading,
while 144 failed only stop time and endpoint. The selected zero-yaw-feed-
forward candidate passes endpoint at 9.956 mm and reaches 3.130 degrees
heading; the yaw-only 0.025 feed-forward champion reaches 2.468 degrees
heading and 10.513 mm endpoint error. The latter under-travels the final
negative-Y segment by 8.24 mm. The selected candidate also places stop-time
error one floating-point representation step above 0.10 s, while the
candidate-wide floor is 0.04 s.

The target controller identifier advances to
`isaac-lab-x500-velocity-v13`. Attempt 12 adds exactly four declarative
candidates: isolated Y proportional gain `[1.35, 1.575, 3.0]`, isolated X
integral gain `[0.625, 0.6, 3.0]`, and yaw-only rate feed-forward
`[0, 0, 0.01]` and `[0, 0, 0.015]`. The isolated Y candidate follows the
measured endpoint residual, the X-integral candidate tests a real stop-event
crossing without changing the gate, and the two feed-forward candidates fit
the normalized yaw-torque boundary between the measured heading and endpoint
champions.

No controller equation, allocator, plant, actuator, probe, gate, worker,
policy interface, reward, or training setting changes. The wider beam is the
one structural search change for this attempt; the same single worker remains
appropriate because Attempt 11 measured sequential Isaac/PhysX stepping as
the bottleneck.

### 78. Attempt 12 crosses stop and heading but loses the endpoint basin

Date: 2026-07-31.

Attempt 12 completed from clean source commit `813187c` as
`x500-velocity-px4-v5/20260731T054522Z`. Its fresh capture completed all 11
profiles with valid initial controller state. The reference actuator audit
passed: timestamp joins stayed within one 4 ms physics tick, normalized motor
RMSE peaked at 0.00101, and collective RMSE peaked at 0.0122 N.

The 64-wide sweep evaluated 12597 unique candidates, 12351 of them usable, and
selected candidate 12027 under comparison fingerprint
`65d63f76f21e02e840b87b08d55c8185c1d693b223fd6db16ea5f9529cac4c7a`.

| Held-out metric | Attempt 12 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 6.350 mm | 5 mm |
| Residual-speed error | 0.00581 m/s | 0.01 m/s |
| Stop-time error | 0.060 s | 0.10 s |
| Worst-axis velocity NRMSE | 8.10% | 10% |
| Yaw-rate NRMSE | 4.65% | 10% |
| Heading error | 3.000 deg | 3 deg |
| Combined endpoint error | 47.260 mm | 10 mm |

The attempt is rejected with two failed gates and a largest normalized excess
of 3.726. It crosses both Attempt 11 failures, stop time and heading, but
regresses the previously passed braking and endpoint gates. It is therefore a
no-progress result against Attempt 11's two failures and 1.043 largest excess.
Among usable candidates, 924 failed two gates, 3520 failed three, 2658 failed
four, 1961 failed five, 3283 failed six, and 5 failed all seven. Candidate-wide
minima were 1.855 mm braking error, 0.00035 m/s residual speed, 0.060 s
stop-time error, 5.73% velocity NRMSE, 3.02% yaw-rate NRMSE, 2.323 degrees
heading error, and 10.849 mm endpoint error. Unlike Attempt 11, this fresh
reference produced no evaluated endpoint-passing candidate.

The selected settings use 0.060 s command delay, velocity P/I/D
`[1.8, 2.25, 6]`, `[0, 0.4, 2]`, and `[0.3, 0.3, 0]`, a 1 Hz
velocity-derivative filter, normalized hover thrust 0.572, attitude P
`[4, 4, 2.8]`, yaw weight 0.3, rate P/I/D
`[0.1125, 0.1125, 0.15]`, `[0.3, 0.3, 0.15]`, and
`[0.0015, 0.0015, 0]`, yaw-only feed-forward 0.025, a 15 Hz
rate-derivative filter, 0.45 rate-integral limits, and 0.04 s actuator fall
time.

Controller replay again places yaw and combined error first at normalized
torque: validation normalized motor RMSE is 0.0137 and 0.0131, with normalized
yaw-torque RMSE of 0.0134 and 0.0126. The selected braking and lateral
profiles pass the normalized motor and physical collective replay limits.
Vertical collective RMSE remains 0.254 N and the stability envelope has
broader collective and torque error. The passing fresh reference actuator
audit continues to exclude the bridge mapping as the first seam.

The global beam expansion is not retained as evidence of a better frontier.
Attempt 13 should restore the 32-wide search and preserve Attempt 11's
zero-delay, high-integral braking/endpoint basin explicitly while combining
the evidenced intermediate yaw-torque and isolated lateral candidates. It
should not broaden the controller, allocator, plant, actuator, probes, gates,
worker count, policy interface, reward, or training configuration.

Attempt 12 used one worker, 41 GPU batches, and 2025.57 s for the useful
sweep, or 6.22 candidate evaluations/s. GPU utilization averaged 25.21%,
reached 48% at p95 and 60% at peak, and used 5409 MiB peak memory.
Sequential Isaac/PhysX stepping remains the measured bottleneck.

### 79. Seed the staged search from the retained Attempt 11 basin

Date: 2026-07-31.

Attempt 13 restores the 32-wide beam and makes Attempt 11's complete selected
settings the target definition's base settings. Attempt 12 showed that the
new yaw-torque and X-integral candidates can pass heading and stop time, but
its global expansion never evaluated the full Attempt 11 setting combination
against the fresh reference. The generic staged search builds combinations
one declared field at a time; a wider beam does not guarantee that a later
complementary combination survives every earlier ranking stage.

The base now uses zero command delay, velocity P/I/D
`[1.35, 1.35, 3]`, `[0.6, 0.6, 3]`, and `[0.3, 0.2, 0]`, a 1 Hz
velocity-derivative filter, velocity-integral limits `[3, 3, 6]`, normalized
hover thrust 0.57, yaw weight 0.5, rate integral `[0.3, 0.3, 0.15]`,
33.75 degrees maximum tilt, and a 0.02 s actuator rise time. These are all
existing declared candidates and reproduce candidate 5417 from Attempt 11;
the change ensures their complete combination is evaluated as the initial
candidate rather than reconstructed only if every intermediate survives.

The target controller identifier advances to
`isaac-lab-x500-velocity-v14`. The isolated Y proportional, isolated X
integral, and two intermediate yaw-only feed-forward candidates introduced in
Attempt 12 remain available. No new candidate, controller equation,
allocator, plant, actuator, probe, gate, worker, policy interface, reward, or
training change is made.

### 80. Attempt 13 exposes a profile-reset actuator-state leak

Date: 2026-07-31.

Attempt 13 completed from clean source commit `849d3d9` as
`x500-velocity-px4-v5/20260731T063918Z`. Its fresh capture completed all 11
profiles with valid initial controller state. The reference actuator audit
passed: timestamp joins stayed within one 4 ms physics tick, normalized motor
RMSE peaked at 0.00313, and collective RMSE peaked at 0.0118 N.

The sweep evaluated 5943 unique candidates, 5841 of them usable, and selected
candidate 5459 under comparison fingerprint
`03e41d2d3a728062ef33bc2ced2b58363d65acb6293491c58ca51d99eb6e189f`.

| Held-out metric | Attempt 13 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 12.171 mm | 5 mm |
| Residual-speed error | 0.00126 m/s | 0.01 m/s |
| Stop-time error | 0.260 s | 0.10 s |
| Worst-axis velocity NRMSE | 20.31% | 10% |
| Yaw-rate NRMSE | 6.01% | 10% |
| Heading error | 2.980 deg | 3 deg |
| Combined endpoint error | 15.484 mm | 10 mm |

The attempt is rejected with four failed gates and a largest normalized excess
of 1.600. No usable candidate failed fewer than four gates: 3543 failed four,
1932 failed five, and 366 failed six. Candidate-wide minima were 11.671 mm
braking error, 0.00027 m/s residual speed, 0.200 s stop-time error, 19.19%
velocity NRMSE, 3.04% yaw-rate NRMSE, 2.204 degrees heading error, and 9.713
mm endpoint error. This is a second no-progress result, but it identifies a
source-confirmed target reset defect rather than exhausting the controller
frontier.

The exact Attempt 11 settings were evaluated as candidate 0 as intended.
Their validation vertical RMSE regressed from 0.00745 m/s in Attempt 11 to
0.04191 m/s in Attempt 13. The stored target trace locates the discontinuity
at the first sample after profile reset: vertical velocity changes from
+0.0075 to -0.2261 m/s in 20 ms. Under the identical settings in Attempt 11,
the corresponding first change is -0.0060 to -0.0072 m/s.

The pinned `isaaclab_contrib.actuators.Thruster` explains the run-dependent
behavior. Every profile calls `reset_idx`, which restores rotor speed from
`init_thruster_rps` and re-samples rise and fall constants from the ranges
fixed when the environment was constructed. The replay constructs the
thruster with zero initial rotor speed. Attempt 11's target base had zero rise
time, so profile reset silently erased candidate rise times back to zero.
Attempt 13's base rise time is 0.02 s, so the same reset both preserves that
construction-time value and starts the ongoing-hover profile from stopped
rotors, producing the measured impulse.

Attempt 14 must fix this plant/controller replay seam before adding candidates:
preserve each fitted rise/fall value across profile reset and initialize
thruster state from the seeded neutral-command controller boundary rather than
zero rotor speed. The NumPy/Torch controller, allocator, physical rotor curve,
probes, gates, worker count, policy interface, reward, and training
configuration remain unchanged.

Attempt 13 used one worker, 41 GPU batches, and 1494.83 s for the useful
sweep, or 3.98 candidate evaluations/s. GPU utilization averaged 26.13%,
reached 40% at p95 and 47% at peak, and used 4569 MiB peak memory.

### 81. Restore the ongoing-flight thruster boundary on profile reset

Date: 2026-07-31.

Attempt 14 makes the one structural correction demonstrated by Attempt 13.
Profile reset continues to clear dynamic Thruster state, but it now preserves
the fitted per-environment rise/fall constants across the pinned
`reset_idx()` call. After the PX4 velocity and rate controller states are
seeded, the replay reconstructs the neutral-command rotor thrust and uses it
as the Thruster's current state. The controller state itself is not advanced
by this priming calculation.

This matches the probe contract: every profile starts while the captured
vehicle is already armed and hovering, so stopped rotors are not a valid
initial actuator boundary. It also makes the declared actuator candidates
independent of the target YAML's construction-time base values. Host tests
guard preservation of distinct per-environment rise/fall constants and
nonzero seeded rotor thrust without controller-state advancement.

The target controller identifier advances to
`isaac-lab-x500-velocity-v15`. Attempt 13's base settings, 32-wide search,
declarative candidates, controller equations, allocator, physical rotor
curve, probes, gates, worker count, policy interface, reward, and training
configuration remain unchanged.

### 82. Attempt 14 reaches the first one-gate frontier

Date: 2026-07-31.

Attempt 14 completed from clean source commit `1a17650` as
`x500-velocity-px4-v5/20260731T072630Z`. Its fresh capture completed all 11
profiles with valid initial controller state. The reference actuator audit
passed: timestamp joins stayed within one 4 ms physics tick, normalized motor
RMSE peaked at 0.00184, and collective RMSE peaked at 0.0121 N.

The reset correction is directly visible in the stored candidate-0 vertical
validation trace. Its first velocity samples are +0.00490, +0.00267, +0.00074,
and -0.00015 m/s, replacing Attempt 13's first-step drop to -0.2261 m/s.

The sweep evaluated 6095 unique candidates, 6093 of them usable, and selected
candidate 5662 under comparison fingerprint
`de23489cd9ac85ca582698f8a87fd117e43937963e42c288108e8ce20a453c68`.

| Held-out metric | Attempt 14 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 4.417 mm | 5 mm |
| Residual-speed error | 0.00394 m/s | 0.01 m/s |
| Stop-time error | 0.200 s | 0.10 s |
| Worst-axis velocity NRMSE | 5.66% | 10% |
| Yaw-rate NRMSE | 5.29% | 10% |
| Heading error | 2.949 deg | 3 deg |
| Combined endpoint error | 9.845 mm | 10 mm |

This is the campaign's first one-gate frontier and therefore material
progress. Among usable candidates, 698 failed one gate, 929 failed two, 931
failed three, 1649 failed four, 1299 failed five, 586 failed six, and 1 failed
all seven. Candidate-wide minima were 3.974 mm braking error, 0.00019 m/s
residual speed, 0.10000000000000009 s stop-time error, 5.32% velocity NRMSE,
2.58% yaw-rate NRMSE, 1.792 degrees heading error, and 1.854 mm endpoint
error.

The selected settings use zero command delay, velocity P/I/D
`[1.35, 1.35, 3]`, `[0.4, 0.6, 2]`, and `[0.25, 0.25, 0]`, 10 Hz velocity
filtering, a 1 Hz velocity-derivative filter, normalized hover thrust 0.57,
attitude P `[5, 5, 3.5]`, yaw weight 0.6, rate P/I/D
`[0.1875, 0.1875, 0.25]`, `[0.2, 0.2, 0.1]`, and
`[0.00375, 0.00375, 0]`, yaw-only feed-forward 0.025, 20 Hz rate-derivative
filtering, 0.4 rate-integral limits, 56.25 degrees maximum tilt, and a 0.02 s
actuator rise time.

The candidate-wide stop-time floor is mathematically the unchanged 0.10 s
limit, but subtracting two late floating-point timestamps reports
0.10000000000000009 and rejects it. Attempt 15 should correct only this
measurement artifact by quantizing the elapsed-time difference to nanosecond
precision before applying the unchanged gate. This does not loosen the limit
or change the event definition.

Controller replay continues to place yaw and combined residuals at normalized
torque, but all seven motion gates except stop time pass and no controller
candidate expansion is warranted before the numerical boundary is evaluated
correctly. The physical collective diagnostic remains above 0.05 N despite
small normalized collective errors; the fresh reference bridge audit still
passes.

Attempt 14 used one worker, 41 GPU batches, and 1494.83 s for the useful
sweep, or 4.08 candidate evaluations/s. GPU utilization averaged 26.41%,
reached 40% at p95 and 47% at peak, and used 4377 MiB peak memory.

### 83. Quantize stop-event timestamp differences at the exact gate

Date: 2026-07-31.

Attempt 15 corrects the sole remaining Attempt 14 rejection without changing
its event definition or 0.10 s limit. Stop indices are still the first samples
at or below the registered speed threshold. Their trace timestamps are
converted to integer nanoseconds before subtraction, and the exact elapsed
difference is converted back to seconds for the existing metric and gate.

This removes cancellation from subtracting two late binary floating-point
timestamps. In particular, the candidate-wide Attempt 14 floor of
0.10000000000000009 s represents exactly 0.10 s on the trace grid and should
not be rejected. A host regression test covers that boundary.

The target controller identifier advances to
`isaac-lab-x500-velocity-v16` so the corrected definition has a new comparison
fingerprint. No controller setting, candidate, equation, allocator, plant,
actuator, probe, gate, worker, policy interface, reward, or training
configuration changes.

### 84. Attempt 15 validates the boundary correction but loses the retained basin

Date: 2026-07-31.

Attempt 15 completed from clean source commit `a03ea00` as
`x500-velocity-px4-v5/20260731T080850Z`. Its fresh capture completed all 11
profiles with valid initial controller state. The reference actuator audit
passed: timestamp joins stayed within one 4 ms physics tick, normalized motor
RMSE peaked at 0.00451, and collective RMSE peaked at 0.0176 N.

The sweep evaluated 6373 unique candidates, 6369 of them usable, and selected
candidate 5990 under comparison fingerprint
`b0c8d121dbba45b3a7525cac967946ebfa521548ac6d4c9bde12274e618c1bbd`.

| Held-out metric | Attempt 15 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 16.005 mm | 5 mm |
| Residual-speed error | 0.00687 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 8.08% | 10% |
| Yaw-rate NRMSE | 4.87% | 10% |
| Heading error | 2.883 deg | 3 deg |
| Combined endpoint error | 46.364 mm | 10 mm |

The stop-time correction behaves as intended: the exact 0.10 s trace-grid
result passes. The fresh fit nevertheless rejects with braking and endpoint
failures and a largest normalized excess of 3.636, so it is a no-progress
result against Attempt 14's one failure and 1.000 excess. All 1198 candidates
that failed only two gates failed the same braking-plus-endpoint pair; no
candidate failed fewer. The remaining usable candidates comprise 1109 with
three failures, 615 with four, 2275 with five, and 1172 with six.
Candidate-wide minima were 3.923 mm braking error, 0.00076 m/s residual speed,
0.100 s stop-time error, 5.90% velocity NRMSE, 2.54% yaw-rate NRMSE, 1.864
degrees heading error, and 5.969 mm endpoint error.

The selected settings use 0.024 s command delay, velocity P/I/D
`[1.35, 1.575, 3]`, `[0.6, 0.6, 3]`, and zero, 5 Hz velocity filtering, a
3.75 Hz velocity-derivative filter, normalized hover thrust 0.57, attitude P
`[4, 4, 2.8]`, yaw weight 0.3, rate P/I/D
`[0.225, 0.225, 0.3]`, `[0.3, 0.3, 0.15]`, and
`[0.00225, 0.00225, 0]`, symmetric rate feed-forward 0.025, a 15 Hz
rate-derivative filter, 0.15 rate-integral limits, 45 degrees maximum tilt,
and a 0.08 s actuator rise time.

Controller replay places the low-speed braking profile within the normalized
motor limit at 0.00390 RMSE, but its physical collective RMSE is 0.112 N.
The combined profile reaches 0.02182 normalized motor RMSE, with its largest
normalized torque residual on yaw, and 0.127 N collective RMSE. The
reference-side bridge audit passes, so the plant mapping remains excluded as
the first seam.

The complete Attempt 14 settings are all still represented by declarative
candidates, but their combination is absent from Attempt 15's evaluated
candidate set. The generic staged beam pruned an intermediate before that
cross-stage combination could be reconstructed. Attempt 16 should therefore
make the exact Attempt 14 one-gate settings the base candidate. This is the
smallest evidenced way to guarantee evaluation of the retained basin after
the reset and stop-time corrections; no new candidate or controller equation
is justified yet.

Attempt 15 used one worker, 41 GPU batches, and 1498.40 s for the useful
sweep, or 4.25 candidate evaluations/s. GPU utilization averaged 26.28%,
reached 41% at p95 and 54% at peak, and used 4379 MiB peak memory.
Sequential Isaac/PhysX stepping remains the measured bottleneck.

### 85. Seed the corrected search from the Attempt 14 one-gate frontier

Date: 2026-07-31.

Attempt 16 makes the complete Attempt 14 selected settings the target
definition's base candidate. Attempt 15 proved the stop-time correction, but
its fresh staged search did not evaluate that complete setting combination
even though every individual value remains declared. The beam pruned an
intermediate cross-stage combination before reconstruction.

The base now uses zero command delay, velocity P/I/D
`[1.35, 1.35, 3]`, `[0.4, 0.6, 2]`, and `[0.25, 0.25, 0]`, 10 Hz velocity
filtering, a 1 Hz velocity-derivative filter, normalized hover thrust 0.57,
attitude P `[5, 5, 3.5]`, yaw weight 0.6, rate P/I/D
`[0.1875, 0.1875, 0.25]`, `[0.2, 0.2, 0.1]`, and
`[0.00375, 0.00375, 0]`, yaw-only feed-forward 0.025, a 20 Hz
rate-derivative filter, 0.4 rate-integral limits, 56.25 degrees maximum tilt,
and a 0.02 s actuator rise time. These are exactly candidate 5662 from
Attempt 14, which passed six gates before stop-time quantization.

The target controller identifier advances to
`isaac-lab-x500-velocity-v17`. No candidate, controller equation, allocator,
plant, actuator, probe, gate, beam width, worker, policy interface, reward, or
training configuration changes. Attempt 16 is an evidence-routed retained-
basin control; its base candidate is guaranteed to be evaluated against the
fresh reference before any staged replacement is pruned.

### 86. Attempt 16 exposes stale controller-state joins at profile reset

Date: 2026-07-31.

Attempt 16 completed from clean source commit `73e920a` as
`x500-velocity-px4-v5/20260731T085246Z`. Its fresh capture completed all 11
profiles with valid initial controller state. The reference actuator audit
passed: timestamp joins stayed within one 4 ms physics tick, normalized motor
RMSE peaked at 0.00155, and collective RMSE peaked at 0.0141 N.

The sweep evaluated 6680 unique candidates, 6679 of them usable, and selected
candidate 6390 under comparison fingerprint
`3c6eb47f4eeb7e35a2c995b3240d52b3192045e23da5b2b00a04c13b108d7930`.

| Held-out metric | Attempt 16 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 21.138 mm | 5 mm |
| Residual-speed error | 0.00598 m/s | 0.01 m/s |
| Stop-time error | 0.600 s | 0.10 s |
| Worst-axis velocity NRMSE | 8.39% | 10% |
| Yaw-rate NRMSE | 4.40% | 10% |
| Heading error | 2.962 deg | 3 deg |
| Combined endpoint error | 61.254 mm | 10 mm |

The attempt is rejected with three failures and a largest normalized excess
of 5.125, so it does not improve the Attempt 14 frontier. Among usable
candidates, 3123 failed three gates, 2668 failed four, 761 failed five, and
127 failed six; no candidate failed fewer than three. Candidate-wide minima
were 6.505 mm braking error, 0.00107 m/s residual speed, 0.060 s stop-time
error, 6.32% velocity NRMSE, 2.78% yaw-rate NRMSE, 2.236 degrees heading
error, and 6.781 mm endpoint error.

The retained Attempt 14 settings were evaluated as candidate 0 as intended.
On this fresh reference they fail braking at 11.447 mm, stop time at 0.700 s,
and endpoint at 86.110 mm while passing the remaining four gates. This
excludes staged pruning as the cause of the run-to-run regression.

The selected settings use 0.024 s command delay, velocity P/I/D
`[1.35, 1.575, 3]`, `[0.625, 0.6, 3]`, and `[0.25, 0.25, 0]`, 2 Hz
velocity filtering, a 1 Hz velocity-derivative filter, normalized hover
thrust 0.57, attitude P `[4, 4, 2.8]`, yaw weight 0.3, rate P/I/D
`[0.1125, 0.1125, 0.15]`, `[0.2, 0.2, 0.1]`, and
`[0.00375, 0.00375, 0]`, yaw-only feed-forward 0.015, 30 Hz
rate-derivative filtering, 0.2 rate-integral limits, 56.25 degrees maximum
tilt, zero actuator time constants, and a 500 N/s actuator rate limit.

Controller replay keeps the braking and combined normalized motor RMSE within
the 0.01 limit at 0.00319 and 0.00990, but their physical collective RMSE is
0.0997 N and 0.1128 N. The fresh bridge audit passes, so the plant mapping is
still internally consistent.

The capture's initial controller-state report identifies the first
reproducibility seam. It currently holds the latest
`vehicle_local_position_setpoint` and `rate_ctrl_status` samples at each
absolute profile start. Their recorded ages are 44-144 ms in this run and
were 44-184 ms across the recent campaign, far beyond one 4 ms physics tick.
The ULog topics are recorded at approximately 10 Hz and 5 Hz, respectively.
Meanwhile, linear and angular acceleration are already interpolated to the
exact start timestamp. The held setpoint and integral snapshots therefore
inject a run-dependent controller-state phase error into every otherwise
identical isolated Lab replay.

Attempt 17 should interpolate the acceleration-setpoint and rate-integral
vectors to the absolute profile start just like the existing kinematic
derivatives. It should retain the raw source-age evidence in the report,
advance the controller-state extraction method and target controller
identifier, and add a midpoint regression test. No gain or actuator candidate
is justified while the candidate-wide braking floor remains above its gate.

Attempt 16 used one worker, 41 GPU batches, and 1198.21 s for the useful
sweep, or 5.57 candidate evaluations/s. GPU utilization averaged 28.80%,
reached 46% at p95 and 51% at peak, and used 4377 MiB peak memory.

### 87. Interpolate controller-state vectors to the absolute profile start

Date: 2026-07-31.

Attempt 17 corrects the controller-state join demonstrated by Attempt 16.
Initial acceleration setpoint and rate-integral vectors are now linearly
interpolated to each profile's recorded absolute simulation start, matching
the existing linear-acceleration and angular-acceleration treatment. The
latest prior sample ages remain in the controller-state report as coverage
evidence; they no longer select a stale held value for Lab initialization.

This is appropriate for the recorded low-rate controller topics: the
acceleration setpoint is continuous during the neutral command that opens
every profile, and the rate integral is a slowly evolving controller state.
A host regression places both source topics around a profile-start midpoint
and verifies the interpolated ENU/FLU vectors while preserving the raw 4 ms
prior-sample ages.

The controller-state extraction method advances to
`px4-ulog-controller-boundary-v2`, and the target controller identifier
advances to `isaac-lab-x500-velocity-v18` for a new comparison fingerprint.
The retained Attempt 14 base, candidates, controller equations, allocator,
plant, actuator, probes, gates, beam width, worker count, policy interface,
reward, and training configuration remain unchanged.

### 88. Attempt 17 restores a one-gate frontier after exact state joining

Date: 2026-07-31.

Attempt 17 completed from clean source commit `809fb44` as
`x500-velocity-px4-v5/20260731T092906Z`. Its fresh capture completed all 11
profiles. The reference actuator audit passed: timestamp joins stayed within
one 4 ms physics tick, normalized motor RMSE peaked at 0.00407, and collective
RMSE peaked at 0.01391 N.

The staged sweep evaluated 6306 unique usable candidates and selected candidate
6091 under comparison fingerprint
`584958d7626a39d466291b402ebc5441ddb77f78f9712c8999a91111a515159f`.

| Held-out metric | Attempt 17 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 4.172 mm | 5 mm |
| Residual-speed error | 0.00227 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 9.47% | 10% |
| Yaw-rate NRMSE | 4.07% | 10% |
| Heading error | 2.410 deg | 3 deg |
| Combined endpoint error | 13.250 mm | 10 mm |

The attempt is rejected with endpoint error as its only failure and a largest
normalized excess of 0.325. This is material progress over Attempt 14's
one-gate frontier and largest normalized excess of 1.0. Among all candidates,
568 failed one gate, 1671 failed two, 1424 failed three, 2060 failed four, 496
failed five, and 87 failed six. Every one-gate candidate failed endpoint error
only. Candidate-wide minima were 2.738 mm braking error, 0.000181 m/s residual
speed, 0.040 s stop-time error, 5.24% velocity NRMSE, 2.78% yaw-rate NRMSE,
1.844 degrees heading error, and 1.968 mm endpoint error.

The selected settings use zero command delay, velocity P/I/D
`[1.35, 1.575, 3]`, `[0.3, 0.3, 1.5]`, and `[0.25, 0.25, 0]`, 2 Hz
velocity filtering, a 1 Hz velocity-derivative filter, normalized hover thrust
0.572, attitude P `[4, 4, 2.8]`, yaw weight 0.5, rate P/I/D
`[0.1125, 0.1125, 0.15]`, `[0.3, 0.3, 0.15]`, and zero, symmetric
rate feed-forward 0.025, 20 Hz rate-derivative filtering, 0.15 rate-integral
limits, 56.25 degrees maximum tilt, zero rise time, 0.08 s fall time, and an
effectively unlimited 100000 N/s actuator rate limit.

Controller replay now passes the isolated braking boundary with normalized
motor RMSE 0.00452 and collective RMSE 0.0214 N. The validation combined-turns
boundary is the first remaining miss: normalized motor RMSE is 0.01066 and
collective RMSE is 0.0551 N. Its normalized torque RMSE is
`[0.00425, 0.00698, 0.00882]`; no sample saturates. The fresh reference audit
passes, so the residual is in target motor response rather than the composed
bridge mapping.

The one-gate population supplies matching expansion evidence. All 568 use the
same selected rate P, I, and feed-forward gains. In contrast, their minimum
endpoint error improves monotonically with actuator fall time: 36.109 mm at
zero, 33.072 mm at 0.01 s, 30.242 mm at 0.02 s, 24.657 mm at 0.04 s, and
13.250 mm at the current 0.08 s ceiling. The selected fall time is therefore
not merely a range hit; its motor-boundary replay and candidate-wide endpoint
trend both support extending it. Attempt 18 should retain the complete Attempt
17 selected settings as its base and add 0.10, 0.12, and 0.16 s fall-time
candidates. It should leave controller gains and all other contracts unchanged.

Attempt 17 used one worker, 41 GPU batches, and 1496.39 s for the useful
sweep, or 4.21 candidate evaluations/s. GPU utilization averaged 26.73%,
reached 41% at p95 and 47% at peak, and used 4377 MiB peak memory. A second
worker is still not justified because the existing worker already batches all
candidate work on the GPU and the limiting serial stages do not create
independent useful batches.

### 89. Retain the corrected frontier and extend actuator fall time

Date: 2026-07-31.

Attempt 18 makes the complete Attempt 17 selected settings the target
definition's base candidate, guaranteeing that the six-gate frontier is
evaluated on the next fresh reference before staged pruning. The target
controller identifier advances to `isaac-lab-x500-velocity-v19`.

The only candidate expansion is the evidenced actuator fall-time range. It adds
0.10, 0.12, and 0.16 s above the prior 0.08 s ceiling. This follows both the
monotonic endpoint improvement in the one-gate population and the combined
motor/collective replay miss. The intermediate values test whether the endpoint
gate crosses before a slower actuator regresses braking, stop time, or
worst-axis velocity response.

No controller equation, allocator, plant mapping, probe, gate, beam width,
worker count, policy interface, reward, or training configuration changes.
Attempt 18 is an evidence-routed declarative expansion within schema 5.

### 90. Attempt 18 crosses endpoint but exposes the braking-side tradeoff

Date: 2026-07-31.

Attempt 18 completed from clean source commit `04951ac` as
`x500-velocity-px4-v5/20260731T104305Z`. Its fresh capture completed all 11
profiles. The reference actuator audit passed: timestamp joins stayed within
one 4 ms physics tick, normalized motor RMSE peaked at 0.000693, and collective
RMSE peaked at 0.01204 N.

The sweep evaluated 6144 unique candidates, 6142 of them usable, and selected
candidate 6143 under comparison fingerprint
`22aac97a46608682a3e2a0e60998464af655b5a5c4fdd323d586eff65fee99c2`.
The two unusable candidates failed the existing 35 degree tilt and 2 m lateral
flight-envelope guards rather than numerical execution.

| Held-out metric | Attempt 18 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 12.799 mm | 5 mm |
| Residual-speed error | 0.00306 m/s | 0.01 m/s |
| Stop-time error | 0.100 s | 0.10 s |
| Worst-axis velocity NRMSE | 9.03% | 10% |
| Yaw-rate NRMSE | 2.86% | 10% |
| Heading error | 2.327 deg | 3 deg |
| Combined endpoint error | 9.890 mm | 10 mm |

The attempt is rejected with braking distance as its only failure and a largest
normalized excess of 1.560. The actuator expansion crossed the endpoint gate,
but it regressed the braking gate that Attempt 17 passed, so it is not material
progress under the campaign definition. Among usable candidates, 1186 failed
one gate, 1161 failed two, 1256 failed three, 1900 failed four, 615 failed
five, and 24 failed six. Every one-gate candidate failed braking distance only.
No candidate passed braking; its candidate-wide floor was 6.791 mm.
Candidate-wide minima for the remaining gates were 0.000169 m/s residual
speed, 0.100 s stop-time error, 6.39% velocity NRMSE, 2.48% yaw-rate NRMSE,
1.963 degrees heading error, and 1.081 mm endpoint error.

The selected settings use 0.060 s command delay, velocity P/I/D
`[1.35, 1.575, 3]`, `[0.4, 0.4, 2]`, and `[0.3, 0.3, 0]`, 5 Hz
velocity filtering, a 1 Hz velocity-derivative filter, normalized hover thrust
0.57, attitude P `[5, 5, 3.5]`, yaw weight 0.3, rate P/I/D
`[0.1875, 0.1875, 0.25]`, `[0.3, 0.3, 0.15]`, and zero, symmetric
rate feed-forward 0.025, 15 Hz rate-derivative filtering, 0.4 rate-integral
limits, 67.5 degrees maximum tilt, 0.01 s rise time, 0.10 s fall time, and an
effectively unlimited actuator rate limit.

The selected replay passes braking normalized motor RMSE at 0.00836 but misses
physical collective RMSE at 0.1001 N; its normalized X torque RMSE is 0.01139.
The captured PX4 hover estimate averages 0.57156 over that profile while the
selected target uses 0.57. This places the first unresolved seam at the
normalized thrust and horizontal braking boundary, not the bridge's composed
plant mapping.

Candidate 6137 is a stronger retained basin for the next attempt. It has the
same 12.799 mm braking failure but 4.434 mm endpoint error, leaving 5.566 mm of
endpoint margin. It uses 1/1/2 velocity-integral limits, yaw weight 0.6,
25 Hz rate filtering, 0.45 rate-integral limits, a 500 N/s actuator rate limit,
and otherwise preserves the selected velocity gains and 0.10 s fall time.

Attempt 18 used one worker, 41 GPU batches, and 1493.55 s for the useful
sweep, or 4.11 candidate evaluations/s. GPU utilization averaged 26.59%,
reached 40% at p95 and 48% at peak, and used 4377 MiB peak memory. The same
serial orchestration and trace-materialization bottleneck remains; adding a
worker would not create additional useful GPU batches.

### 91. Retain endpoint margin and soften the derivative braking boundary

Date: 2026-07-31.

Attempt 19 makes Attempt 18 candidate 6137 the target definition's base
candidate. This guarantees evaluation of its one-gate response with 4.434 mm
endpoint error before staged pruning. The target controller identifier advances
to `isaac-lab-x500-velocity-v20`.

The braking diagnostic resolves the sign hidden by the absolute advisory
metric. In the held-out X braking event, PX4 travels 57.148 mm after command
reversal while candidate 6137 travels 44.349 mm. The target therefore stops
12.799 mm too early. All 1186 one-gate candidates use the current 1 Hz
velocity-derivative filter floor, and their common 0.3 horizontal derivative
gain is needed by the other passed gates.

Attempt 19 extends only that evidenced filter range with 0.5, 0.625, 0.75, and
0.875 Hz candidates below 1 Hz. Lowering the cutoff smooths the acceleration
feedback and reduces the excess target braking action while the retained
candidate's endpoint margin permits some additional travel. The captured PX4
hover estimate is higher than the target setting, but increasing hover would
increase horizontal braking authority and move the signed distance in the
wrong direction, so no hover expansion is made.

No controller equation, allocator, plant mapping, actuator range, probe, gate,
beam width, worker count, policy interface, reward, or training configuration
changes. Attempt 19 remains a schema-5 declarative filter expansion.

### 92. Attempt 19 exposes stochastic estimator input as the reproducibility seam

Date: 2026-07-31.

Attempt 19 completed from clean source commit `fb67199` as
`x500-velocity-px4-v5/20260731T112716Z`. Its fresh capture completed all 11
profiles. The reference actuator audit passed: timestamp joins stayed within
one 4 ms physics tick, normalized motor RMSE peaked at 0.00119, and collective
RMSE peaked at 0.03187 N.

The sweep evaluated 6919 candidates, 6918 of them usable, and selected candidate
6789 under comparison fingerprint
`9386022ee613433ec70762fda22191266cc7bf634753a0d51a629b476358d6f1`.

| Held-out metric | Attempt 19 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 9.861 mm | 5 mm |
| Residual-speed error | 0.00374 m/s | 0.01 m/s |
| Stop-time error | 0.700 s | 0.10 s |
| Worst-axis velocity NRMSE | 9.88% | 10% |
| Yaw-rate NRMSE | 3.67% | 10% |
| Heading error | 2.791 deg | 3 deg |
| Combined endpoint error | 78.662 mm | 10 mm |

The attempt is rejected with braking, stop-time, and endpoint failures and a
largest normalized excess of 6.866. This is not material progress. Among usable
candidates, 2392 failed three gates, 3679 failed four, 723 failed five, and 124
failed six. No candidate failed fewer than three gates. Candidate-wide minima
were 5.237 mm braking error, 0.000287 m/s residual speed, 0.140 s stop-time
error, 7.23% velocity NRMSE, 2.42% yaw-rate NRMSE, 1.845 degrees heading error,
and 49.756 mm endpoint error.

Candidate 0 retained Attempt 18 candidate 6137 exactly, but on this fresh
reference it failed four gates: 8.118 mm braking error, 0.400 s stop-time
error, 11.29% velocity NRMSE, and 94.000 mm endpoint error. This excludes the
new derivative-filter values and staged pruning as causes of the regression.

The selected profile replay usually keeps normalized motor RMSE below 0.01,
including 0.00432 on the braking profile, but collective RMSE is 0.10090 N
there. The stability replay has 0.03499 normalized motor RMSE and 1.32018 N
collective RMSE. These are consequences of fitting a substantially changed
reference response rather than evidence for another controller range
expansion.

Three consecutive captures identify why the closed-loop frontier does not
reproduce. The PX4 `vehicle_local_position` estimator velocity differs from
the captured Isaac ground-truth velocity by approximately 0.005-0.010 m/s RMS
per profile, with maxima up to 0.026 m/s. The signed mean bias changes between
captures and profiles. This is the same scale as the 0.005 m braking and
0.010 m endpoint gates once integrated over a five-to-six-second probe.
Correspondingly, the PX4 combined validation displacement changed by about
80 mm between Attempts 18 and 19 even though the controller, plant, probes,
and capture code were unchanged.

The alignment experiment currently inherits
`px4.sensor_noise_enabled: true`. That stochastic estimator input is absent
from Isaac Lab, whose aligned controller consumes deterministic simulated
kinematics. Attempt 20 should disable PX4 sensor noise only in the alignment
experiment, advancing the experiment and comparison fingerprints. This keeps
the controller, allocator, composed plant, probe suite, gates, and candidate
ranges unchanged while making the input boundary being compared reproducible.

Attempt 19 used one worker, 41 GPU batches, and 1514.77 s for the useful sweep,
or 4.57 candidate evaluations/s. GPU utilization averaged 26.24%, reached 41%
at p95 and 48% at peak, and used 4377 MiB peak memory. The measured
trace-materialization and serial orchestration bottleneck still does not
justify a second worker.

### 93. Make the alignment reference input deterministic

Date: 2026-07-31.

Attempt 20 disables `vehicles.x500_01.bridge.sensor_noise_enabled` only in
`configs/alignment_px4_x500_velocity.yaml`. The default remains enabled for
normal integration and runtime experiments. Controller alignment is intended
to compare deterministic controller, allocator, actuator, and plant response;
stochastic estimator divergence remains a downstream PX4-versus-Lab
diagnostic.

This is the single structural change evidenced by Attempt 19. The target
controller identifier advances to `isaac-lab-x500-velocity-v21`, and the
experiment change produces a new comparison fingerprint. The Attempt 18
candidate 6137 base, all declarative candidates, controller equations,
allocator, composed plant, actuator model, probe suite, gates, beam width,
worker count, policy interface, reward, and training configuration remain
unchanged.

A host configuration regression verifies that the dedicated alignment
experiment disables stochastic PX4 sensor models. Attempt 20 will run a fresh
capture, actuator audit, unchanged target sweep, and held-out fit from the
committed source.

### 94. Retain valid HIL noise with a deterministic alignment seed

Date: 2026-07-31.

The noise-free Attempt 20 source commit `4a6804a` did not reach capture. Two
launches under run IDs `20260731T121830Z` and `20260731T121927Z` both stopped
before arming at 5.18 s simulation time. The bridge sent fresh IMU,
magnetometer, barometer, and GPS data every 4 ms through the final tick, but
PX4 switched barometer instances, declared the constant noise-free pressure
stream stale, and stopped returning actuator messages. The bridge then raised
its existing one-second lockstep timeout. These are preserved infrastructure
failures, not completed fitting attempts.

Attempt 20 therefore retains the default stochastic Pegasus sensor models and
seeds their shared NumPy sequence from a new
`alignment.capture.sensor_seed` immediately before the scenario constructs
bridge hosts. The configured seed is 42. This preserves changing HIL sensor
values while making the reference estimator perturbations reproducible across
captures. A host test verifies seeding occurs before scenario startup, and
configuration validation restricts the seed to NumPy's unsigned 32-bit range.

The target controller identifier advances to
`isaac-lab-x500-velocity-v22`. The controller, allocator, composed plant,
actuator model, probe suite, gates, candidate ranges, beam width, worker count,
policy interface, reward, and training configuration remain unchanged.

### 95. Attempt 20 proves a shared seed is insufficient

Date: 2026-07-31.

Attempt 20 completed from clean source commit `f9b2976` as
`x500-velocity-px4-v5/20260731T122442Z`. Its fresh capture completed all 11
profiles. The reference actuator audit passed: timestamp joins stayed within
one 4 ms physics tick, normalized motor RMSE peaked at 0.00112, and collective
RMSE peaked at 0.01010 N.

The staged sweep evaluated 6505 unique candidates, 6504 of them usable, and
selected candidate 6454 under comparison fingerprint
`2306adcff2e907bcf4cfaca968421e1d5feaf21a3b3793e5c8e3c9184b2647ad`.

| Held-out metric | Attempt 20 | Limit |
| --- | ---: | ---: |
| Braking-distance error | 26.063 mm | 5 mm |
| Residual-speed error | 0.00844 m/s | 0.01 m/s |
| Stop-time error | 0.540 s | 0.10 s |
| Worst-axis velocity NRMSE | 33.43% | 10% |
| Yaw-rate NRMSE | 2.45% | 10% |
| Heading error | 2.959 deg | 3 deg |
| Combined endpoint error | 9.359 mm | 10 mm |

The attempt is rejected with braking, stop-time, and worst-axis velocity
failures and a largest normalized excess of 4.4. Although that excess is lower
than Attempt 19's 6.866, Attempt 20 regresses the previously passed velocity
gate and therefore is not material progress under the campaign definition.
Among usable candidates, 1914 failed three gates, 3553 failed four, 985 failed
five, and 52 failed six. No candidate failed fewer than three gates.
Candidate-wide minima were 11.682 mm braking error, 0.000310 m/s residual
speed, 0.240 s stop-time error, 11.06% velocity NRMSE, 2.38% yaw-rate NRMSE,
1.816 degrees heading error, and 0.670 mm endpoint error.

Candidate 0 retained the Attempt 18 candidate 6137 settings exactly but failed
four gates on this reference: 18.942 mm braking error, 1.100 s stop-time
error, 17.68% velocity NRMSE, and 50.966 mm endpoint error. This again excludes
the added derivative-filter candidates and staged pruning as the cause.

The selected replay fails before the plant boundary. Its braking profile keeps
normalized motor RMSE within the 0.01 limit at 0.00452, but collective RMSE is
0.0930 N. Across the complete replay, normalized motor RMSE reaches 0.05660
and collective RMSE reaches 1.9004 N on the stability profile. The captured
bridge audit passes, so these are controller-output differences rather than an
incorrect bridge affine or rotor-curve reconstruction.

A second capture from the same commit and configured seed, preserved as
`20260731T122749Z`, received the same initial sensor sequence and used identical
sample-time arrays. It nevertheless differed from the fitted capture by up to
11.367 mm of relative position and 5.504 mm/s of velocity on the held-out
vertical profile, and by 11.367 mm and 4.528 mm/s on the held-out braking
profile. Later profiles began 0.2 s earlier after the neutral-stability
predicate crossed on a different tick. The shared NumPy seed therefore
controls injected random samples but does not make the complete PX4/PhysX
closed loop reproducible at the exact-action gate scale.

The two runs first select different quantized PX4 motor values while the
vehicle is still on the ground, then integrate that difference through takeoff
and the sequential probes. PX4 requires a changing barometer stream to remain
healthy, but the failed noise-free launches do not establish that every other
sensor must remain stochastic. Attempt 21 should split the barometer noise
amplitude from the aggregate sensor-noise switch, disable stochastic IMU, GPS,
and magnetometer terms only for alignment, and retain the seeded nonzero
barometer stream. It should rerun two captures before committing another full
sweep.

Attempt 20 used one worker, 41 GPU batches, and 1525.92 s for the useful
sweep, or 4.26 candidate evaluations/s. GPU utilization averaged 27.27%,
reached 42% at p95 and 48% at peak, and used 5244 MiB peak memory. The measured
serial orchestration and trace-materialization bottleneck still does not
justify a second worker.

### 96. Isolate required barometer excitation from estimator noise

Date: 2026-07-31.

Attempt 21 separates the barometer pressure-noise amplitude from the aggregate
Pegasus sensor-noise switch. Normal runtime retains stochastic IMU, GPS,
magnetometer, and 1 Pa barometer models. The alignment experiment disables the
IMU, GPS, and magnetometer stochastic terms, retains the seeded 1 Pa pressure
stream PX4 requires for barometer health, and continues to seed the sequence
before bridge-host construction.

This is the single structural change evidenced by Attempt 20's replicate. It
removes estimator perturbations absent from Isaac Lab without repeating the
constant-pressure failure of the noise-free launches. A configuration
regression verifies the dedicated alignment override, a bridge sensor test
verifies exact IMU/GPS/magnetometer values alongside changing pressure, and
configuration validation rejects negative or non-finite pressure noise.

The target controller identifier advances to
`isaac-lab-x500-velocity-v23`. The controller, allocator, composed plant,
actuator model, probe suite, gates, candidate ranges, beam width, worker count,
policy interface, reward, and training configuration remain unchanged.
Attempt 21 will run two captures first. Another full sweep is justified only if
the capture succeeds and materially reduces held-out trace variation.

### 97. Attempt 21 requires magnetometer excitation for yaw alignment

Date: 2026-07-31.

The Attempt 21 source commit `79b2266` launched capture
`x500-velocity-px4-v5/20260731T131601Z` but did not reach the probe routine.
PX4 received the arm command at 12.132 s and denied it with unresolved system
health failures until the existing ten-second arm timeout expired. The ULog is
preserved; this is an infrastructure failure rather than a completed fitting
attempt.

The ULog localizes the failure. Barometer height fusion remained active and
GPS checks passed, but every magnetometer observation was rejected. The exact
Pegasus field's Z innovation test ratio was 2.730, above the PX4 gate, so yaw
never aligned and GNSS position/velocity fusion never started. The prior
stochastic magnetometer stream intermittently entered that gate and allowed
normal estimator initialization.

Attempt 22 should therefore retain seeded magnetometer noise as well as the
seeded 1 Pa pressure stream, while keeping IMU and GPS stochastic terms
disabled. This remains the same sensor-boundary isolation: it preserves the two
HIL excitations demonstrated to be required by PX4 and removes the IMU/GPS
perturbations most directly coupled to velocity and position estimation.

### 98. Retain only the two HIL excitations PX4 requires

Date: 2026-07-31.

Attempt 22 adds an independent magnetometer-noise scale alongside the
independent barometer amplitude introduced by Attempt 21. Normal runtime keeps
the Pegasus unit scale. The alignment experiment keeps that seeded unit scale
and 1 Pa pressure noise while disabling stochastic IMU and GPS terms.

The target controller identifier advances to
`isaac-lab-x500-velocity-v24`. Configuration validation rejects negative or
non-finite magnetic noise scales, and bridge tests verify that the alignment
combination produces exact IMU/GPS values alongside changing barometer and
magnetometer samples. The controller, allocator, composed plant, actuator
model, probe suite, gates, candidate ranges, beam width, worker count, policy
interface, reward, and training configuration remain unchanged. Two successful
captures with materially reduced trace spread remain the prerequisite for
another full sweep.

### 99. Attempt 22 restores capture but not reproducibility

Date: 2026-07-31.

Attempt 22 completed two capture controls from clean source commit `1dd1c3d` as
`x500-velocity-px4-v5/20260731T132413Z` and
`x500-velocity-px4-v5/20260731T132715Z`. Both armed, took off, and completed all
11 profiles. The first capture's actuator audit passed: timestamp joins stayed
within one 4 ms physics tick, normalized motor RMSE peaked at 0.000327, and
collective RMSE peaked at 0.01278 N.

The replicate's profile starts were consistently 20 ms earlier, and every
trace retained identical sample counts and relative-time arrays. Removing IMU
and GPS noise nevertheless failed to reduce closed-loop spread at the gate
scale. Maximum absolute relative-position differences were 12.071 mm on the
held-out braking profile, 4.297 mm lateral, 57.004 mm vertical, 28.619 mm yaw,
29.265 mm combined, and 21.898 mm on the stability profile. Corresponding
velocity differences reached 4.544, 2.445, 13.913, 10.765, 10.575, and
13.385 mm/s.

This is worse overall than the Attempt 20 replicate, so a full target sweep is
not justified. The remaining full-scale stochastic input is the seeded
magnetometer model required to avoid PX4's exact-field stuck-data path. Attempt
23 should reduce only `magnetometer_noise_scale` from 1.0 to 0.1 and rerun the
two-capture control. The nonzero scale should retain sensor health while
reducing magnetic estimator perturbation by one order of magnitude.

### 100. Reduce magnetic excitation to the health-preserving minimum

Date: 2026-07-31.

Attempt 23 changes only the alignment experiment's
`magnetometer_noise_scale` from 1.0 to 0.1. Normal runtime remains at unit
scale. The target controller identifier advances to
`isaac-lab-x500-velocity-v25`, producing a new comparison fingerprint.

The seeded barometer amplitude, exact alignment IMU/GPS configuration,
controller, allocator, composed plant, actuator model, probe suite, gates,
candidate ranges, beam width, worker count, policy interface, reward, and
training configuration remain unchanged. Attempt 23 will again require two
successful captures with materially reduced trace spread before a full sweep.

### 101. Attempt 23 improves stability but not held-out reproducibility

Date: 2026-07-31.

Attempt 23 completed two capture controls from clean source commit `5010c4f` as
`x500-velocity-px4-v5/20260731T133426Z` and
`x500-velocity-px4-v5/20260731T133724Z`. Both armed, took off, and completed all
11 profiles. The first capture's actuator audit passed: timestamp joins stayed
within one 4 ms physics tick, normalized motor RMSE peaked at 0.000338, and
collective RMSE peaked at 0.01401 N.

The replicate stabilized 0.36-0.40 s later and one braking trace contained one
additional 20 Hz sample. After interpolation on their common profile-relative
time domain, maximum absolute relative-position differences were 13.428 mm on
the held-out braking profile, 30.768 mm lateral, 40.709 mm vertical, 32.719 mm
yaw, 19.083 mm combined, and 6.047 mm on the stability profile. The stability
spread materially improved from Attempt 22's 21.898 mm, but the held-out
profiles did not improve overall, so a full sweep is not justified.

The remaining full-amplitude stochastic input is the seeded 1 Pa barometer
stream. One pascal corresponds to centimeter-scale pressure-altitude
perturbation, and the exact pressure stream is already proven unhealthy.
Attempt 24 should retain the nonzero health signal but reduce only
`barometer_noise_std_pa` from 1.0 to 0.1 before repeating the capture control.

### 102. Reduce pressure excitation to the health-preserving range

Date: 2026-07-31.

Attempt 24 changes only the alignment experiment's
`barometer_noise_std_pa` from 1.0 to 0.1. Normal runtime remains at 1 Pa. The
target controller identifier advances to `isaac-lab-x500-velocity-v26`,
producing a new comparison fingerprint.

The 0.1 magnetic scale, exact alignment IMU/GPS configuration, controller,
allocator, composed plant, actuator model, probe suite, gates, candidate
ranges, beam width, worker count, policy interface, reward, and training
configuration remain unchanged. Attempt 24 will require two successful
captures with materially reduced trace spread before a full sweep.

### 103. Attempt 24 improves braking but not overall reproducibility

Date: 2026-07-31.

Attempt 24 completed two capture controls from clean source commit `a136bfc` as
`x500-velocity-px4-v5/20260731T134410Z` and
`x500-velocity-px4-v5/20260731T134724Z`. Both armed, took off, and completed all
11 profiles. The first capture's actuator audit passed: timestamp joins stayed
within one 4 ms physics tick, normalized motor RMSE peaked at 0.000544, and
collective RMSE peaked at 0.01215 N.

The replicate's profile starts were 200-240 ms earlier. After interpolation on
the common profile-relative time domain, maximum absolute relative-position
differences were 9.297 mm on the held-out braking profile, 17.182 mm lateral,
69.894 mm vertical, 31.216 mm yaw, 20.643 mm combined, and 19.431 mm on the
stability profile. Corresponding velocity differences reached 5.963, 8.070,
14.977, 11.947, 8.906, and 6.155 mm/s.

Reducing pressure noise improved braking spread relative to Attempt 23, but
vertical and stability spread regressed and remain larger than the exact-action
gates. A full target sweep is therefore not justified. Attempts 21-24 show that
exact barometer or magnetometer samples fail PX4 health checks, while
stochastic health-preserving samples make repeated captures diverge. Attempt 25
should replace only those alignment-specific stochastic samples with small
deterministic, simulation-time-based excitation. This keeps the required
sensor variation while removing random estimator input from the reference
boundary.

### 104. Use simulation time for repeatable PX4 sensor health excitation

Date: 2026-07-31.

Attempt 25 adds one bridge sensor-excitation mode. Normal runtime remains
`stochastic`. The alignment experiment selects `deterministic`, which evaluates
small fixed-frequency barometer and three-axis magnetometer signals from
absolute simulation time. Their amplitudes remain the existing 0.1 Pa and 0.1
magnetometer scale, pressure and pressure altitude remain physically
consistent, and the Pegasus IMU/GPS paths remain exact.

The target controller identifier advances to
`isaac-lab-x500-velocity-v27`, producing a new comparison fingerprint.
Configuration and direct-port validation reject unknown modes, and the bridge
sensor regression verifies identical barometer and magnetometer output under
different NumPy random seeds. The controller, allocator, composed plant,
actuator model, probe suite, gates, candidates, worker count, policy interface,
reward, and training configuration remain unchanged. Attempt 25 will run two
captures before deciding whether a full target sweep is justified.

### 105. Attempt 25 removes random input but not PhysX spread

Date: 2026-07-31.

Attempt 25 completed two capture controls from clean source commit `41a37b3` as
`x500-velocity-px4-v5/20260731T135947Z` and
`x500-velocity-px4-v5/20260731T140246Z`. Both deterministic sensor streams
passed PX4 health checks, both flights completed all 11 profiles, and the first
capture's actuator audit passed. Timestamp joins stayed within one 4 ms physics
tick, normalized motor RMSE peaked at 0.000326, and collective RMSE peaked at
0.01296 N.

Every replicate profile began exactly 40 ms earlier and retained the same
sample count. After interpolation on the common profile-relative time domain,
maximum absolute relative-position differences were 7.279 mm on held-out
braking, 27.832 mm lateral, 95.660 mm vertical, 33.346 mm yaw, 5.735 mm
combined, and 13.434 mm on stability. Corresponding velocity differences
reached 7.887, 10.952, 21.094, 11.950, 3.169, and 4.104 mm/s.

Deterministic excitation therefore preserves estimator health and improves
braking and combined reproducibility, but it does not eliminate closed-loop
variation. With sensor output deterministic, the first unresolved seam is now
the GPU PhysX execution used by the PX4 reference capture. Attempt 26 should
enable PhysX enhanced determinism for the alignment experiment only, preserving
the same controller, plant, probes, and sensor excitation, then repeat the
two-capture control before running a full sweep.

### 106. Enable enhanced determinism at the reference plant boundary

Date: 2026-07-31.

Attempt 26 adds the PhysX enhanced-determinism scene flag to the canonical
simulation configuration. Normal experiments retain the default disabled
state. The alignment experiment enables it before spawning the X500, so
PhysX uses its stronger deterministic simulation ordering for the entire
reference flight.

The target controller identifier advances to
`isaac-lab-x500-velocity-v28`, producing a new comparison fingerprint. Host
tests verify that the authored flag reaches the PhysX scene API, and the
standalone simulation-settings integration probe verifies the applied scene
attribute. The deterministic estimator excitation, controller, allocator,
composed plant, actuator model, probes, gates, candidates, worker count, policy
interface, reward, and training configuration remain unchanged. Attempt 26
will again require two captures before a target sweep is justified.

### 107. Attempt 26 improves vertical repeatability but shifts sensor phase

Date: 2026-07-31.

Attempt 26 completed two capture controls from clean source commit `d386104` as
`x500-velocity-px4-v5/20260731T141045Z` and
`x500-velocity-px4-v5/20260731T141339Z`. Both completed all 11 profiles with
PhysX enhanced determinism active. The first capture's actuator audit passed:
timestamp joins stayed within one 4 ms physics tick, normalized motor RMSE
peaked at 0.000514, and collective RMSE peaked at 0.01226 N.

Every replicate profile began 380 ms earlier and retained the same sample
count. Maximum absolute relative-position differences were 7.240 mm on
held-out braking, 24.851 mm lateral, 33.164 mm vertical, 36.179 mm yaw,
15.528 mm combined, and 40.116 mm on stability. Corresponding velocity
differences reached 3.572, 7.268, 8.501, 7.984, 7.492, and 16.079 mm/s; the
stability angular-rate difference reached 0.154 rad/s.

Enhanced determinism materially improved Attempt 25's 95.660 mm vertical
spread but did not make the reference fit-ready. The timing shift identifies a
remaining deterministic input mismatch: profiles start only on the 20 ms
policy cadence, while the 7/17/23/29 Hz health sinusoids present different
phases after such shifts. Attempt 27 should retain enhanced determinism and
change only the deterministic excitation to a 50 Hz sequence. It will still
vary across the five 4 ms physics samples in each policy interval, while every
profile shifted by whole policy ticks receives the same profile-local sequence.

### 108. Phase-lock health excitation to the policy cadence

Date: 2026-07-31.

Attempt 27 changes only the deterministic barometer and magnetometer excitation
period to 20 ms, or 50 Hz. The pattern is computed from integer simulation
microseconds, repeats exactly at each 20 ms policy tick, and still presents five
distinct values across the 250 Hz physics samples. A sensor regression verifies
both different adjacent physics samples and exact equality one policy interval
later.

The target controller identifier advances to
`isaac-lab-x500-velocity-v29`, producing a new comparison fingerprint.
Enhanced PhysX determinism, excitation amplitudes, controller, allocator,
composed plant, actuator model, probes, gates, candidates, worker count, policy
interface, reward, and training configuration remain unchanged. Attempt 27
will run two capture controls before deciding whether to resume fitting.

### 109. Attempt 27 passes exact action but fails the Lab hover gate

Date: 2026-07-31.

Attempt 27 completed two captures from clean source commit `92ba81d` as
`x500-velocity-px4-v5/20260731T142154Z` and
`x500-velocity-px4-v5/20260731T142456Z`. Both completed all profiles. The first
capture's actuator audit passed with 0.000520 maximum normalized motor RMSE,
0.01183 N maximum collective RMSE, and timestamp joins within one 4 ms physics
tick.

Phase locking aligned every held-out and stability profile to the same absolute
PX4 start time, while fit profiles differed by 40 ms. It improved stability
relative-position spread to 8.485 mm, but held-out vertical and combined spread
remained 74.212 mm and 45.635 mm. This rules out profile-relative sensor phase
as the cause of those remaining differences.

The full one-worker sweep evaluated 6,431 unique candidates in 41 GPU batches
over 1,469 seconds and selected candidate 6,172. GPU utilization averaged 31%,
with sequential staged coordination remaining the measured bottleneck. The
held-out fit passed all seven exact-action limits: 4.325 mm braking error,
0.00328 m/s residual-speed error, 0.00 s stop-time error, 7.19% worst-axis
velocity NRMSE, 2.88% yaw-rate NRMSE, 2.458 degree heading error, and 9.141 mm
combined endpoint error. Stability traces were finite and complete.

Final acceptance nevertheless failed. Two unchanged runs of
`test-integration-alignment-target` produced the identical hover result:
0.068 m final error and 0.129 m last-window maximum against the 0.100 m limit.
Isaac Lab simultaneously warned that external forces were not applied every
TGS solver iteration and explicitly associated that setting with noisy
velocity updates. Exact-action boundaries pass while sustained Lab motion
fails, so this warning identifies the first unresolved plant seam. Attempt 28
should enable external forces every solver iteration in both Lab flight
environments, rerun the unchanged controller through the fast hover gate, and,
if confirmed, repeat capture, sweep, fit, and acceptance under the corrected
plant fingerprint.

### 110. Correct the Lab TGS external-force application seam

Date: 2026-07-31.

Attempt 28 authors `sim.external_forces_every_iteration` and enables it by
default. The standalone Isaac scene writes the corresponding PhysX scene
attribute, while both Isaac Lab flight environment factories pass it through
`PhysxCfg`. The existing alignment-only enhanced-determinism flag is also
passed through the Lab configuration, closing a previously incomplete
simulation-setting seam.

This is the single structural change required by Attempt 27's first failed
acceptance boundary and Isaac Lab's runtime warning. It changes force
integration at the plant boundary rather than absorbing the error into
controller gains. The target controller identifier advances to
`isaac-lab-x500-velocity-v30`, producing a new comparison fingerprint. Host
tests cover configuration and standalone scene application; the existing
Isaac Lab config test covers both authored PhysX flags when that dependency is
available. Controller candidates, probes, gates, worker count, policy
interface, reward, and training hyperparameters remain unchanged.

### 111. Attempt 28 falsifies the force-iteration seam and isolates horizontal authority

Date: 2026-07-31.

Attempt 28's unchanged-candidate integration control ran from clean source
commit `2306168`. Applying external rotor forces during every TGS iteration did
not materially change the failed boundary: candidate 6,172 ended at 0.069 m
position error with a 0.130 m last-second maximum, compared with Attempt 27's
0.068 m and 0.129 m. The warning-driven plant correction remains part of the
shared plant contract, but it does not explain the acceptance failure.

A time-resolved 16-environment replay localized the error to horizontal
settling. The worst environment began 0.578 m from its target, fell to 0.066 m
at 2.0 seconds, overshot to 0.131 m at 3.0 seconds, and returned to 0.069 m at
4.0 seconds. Altitude error stayed small, commanded tilt stayed below
0.067 rad, and rotor speeds remained finite without saturation. The accepted
exact-action candidate therefore has sufficient plant authority but too little
large-error horizontal damping for the four-second position-return envelope.

The stored sweep contained no exact-action-passing candidate above horizontal
velocity proportional gain 1.35. Candidate 1,382 used gain 1.8 but changed
multiple later-stage controller and actuator settings and terminated
out-of-bounds, so it did not isolate the first failed boundary. Holding all
candidate 6,172 settings fixed and changing only horizontal proportional gain
to 1.575 passed the hover envelope at 0.098 m last-second maximum, but its
integral overshoot ended the goto at 0.430 m wall range, below the 0.450 m
handoff minimum. Lowering only the horizontal integral gain from 0.4 to 0.2
then passed the complete integration target: hover error was 0.046 m final and
0.090 m last-second maximum, goto error was 0.024 m final and 0.041 m
last-second maximum, and the final wall range was 0.456 m.

The probe artifacts and logs are preserved under
`x500-velocity-px4-v5/20260731T142154Z` and indexed by its relative-path
manifest. Attempt 29 should add the evidenced `[1.575, 1.575, 3.0]`
proportional and `[0.2, 0.2, 2.0]` integral vectors to the declarative search,
seed the complete successful integration combination as the base setting, and
run a fresh capture, actuator audit, sweep, fit, and acceptance sequence.

### 112. Seed the integration-passing horizontal P/I combination

Date: 2026-07-31.

Attempt 29 makes no structural controller or plant change. It advances the
target controller identifier to `isaac-lab-x500-velocity-v31`, adds the two
complete velocity-gain vectors evidenced by Attempt 28 to the declarative
stage-1 candidates, and seeds their integration-passing combination as the
base setting. All remaining base fields come from Attempt 27 candidate 6,172,
which passed every exact-action gate.

The base horizontal proportional gain rises from 1.35 to 1.575 while its
integral gain falls from 0.4 to 0.2. The former cleared the four-second hover
settling limit; the latter removed the goto overshoot and restored the handoff
range. The fresh staged fit must independently prove that the combination
still passes all seven exact-action gates. The schema, allocator, plant,
probes, gates, beam width, worker count, policy interface, reward, and training
configuration remain unchanged.

### 113. Attempt 29 exposes a reference-force regression before fitting

Date: 2026-07-31.

Attempt 29 started the fresh reference capture
`x500-velocity-px4-v5/20260731T154418Z` from clean source commit `766d861`.
PX4 armed and entered takeoff, but the vehicle remained on the ground until the
30-second takeoff timeout. The preserved ULog shows local vertical position
remaining between -0.0003 m and 0.0064 m while PX4 drove normalized collective
to 1.0 and motor controls to 0.999-1.0. Landing recovery completed and
disarmed the vehicle cleanly.

The target-only gain additions cannot affect the standalone PX4 reference.
The first reference capture since Attempt 28 instead proves that enabling
PhysX `external_forces_every_iteration` broke the bridge's force-application
contract. Its target control had already shown no measurable benefit. The
failed capture, ULog, Isaac log, and PX4 log are preserved under the run's
relative-path manifest; no target sweep or fit was justified.

Attempt 30 should remove the external-force iteration setting from the shared
plant contract and restore PhysX's default one-application-per-step behavior.
The useful Lab enhanced-determinism pass-through remains independent and
should be retained. The evidenced velocity-gain candidates also remain
unchanged, after which the complete capture and acceptance sequence restarts
from a new clean source commit.

### 114. Restore one external-wrench application per physics step

Date: 2026-07-31.

Attempt 30 removes `sim.external_forces_every_iteration` and its standalone
and Lab plumbing. The bridge and Lab physics sinks each already submit one
complete rotor wrench per physics tick; PhysX's default one-application-per-
step behavior is therefore the shared plant contract. The alignment
experiment's enhanced-determinism flag remains passed to both standalone
PhysX and the Lab `PhysxCfg`.

The target controller identifier advances to
`isaac-lab-x500-velocity-v32`, producing a new comparison fingerprint. The
integration-evidenced P/I candidates and base setting remain unchanged. The
controller, allocator, rotor model, probes, gates, search, policy interface,
reward, and training configuration also remain unchanged. Attempt 30 must
first prove that a fresh PX4 capture can take off and pass its actuator audit
before another target sweep is justified.

### 115. Attempt 30 restores flight and isolates the normalized-hover boundary

Date: 2026-07-31.

Attempt 30 completed the fresh capture
`x500-velocity-px4-v5/20260731T155322Z` from clean source commit `50d0283`.
The vehicle took off and completed all 11 profiles. The actuator audit passed:
timestamp joins stayed within one 4 ms physics tick, normalized motor RMSE
peaked at 0.000337, and collective-thrust RMSE peaked at 0.01212 N. This
confirms that restoring one external-wrench application per step fixed Attempt
29's reference regression.

The one-worker sweep evaluated 6,331 usable candidates in 41 GPU batches over
1,665.7 seconds. Useful throughput was 3.80 candidates/s. GPU utilization
averaged 28.7%, reached 45% at the 95th percentile, and peaked at 65%, with
6,176 MiB peak memory use. Candidate 5,749 was selected. It missed only the
braking-distance gate at 6.808 mm: residual speed was 0.00135 m/s, stop-time
error was 0.10 s, worst-axis velocity NRMSE was 6.88%, yaw-rate NRMSE was
3.27%, heading error was 2.971 degrees, and combined endpoint error was
8.829 mm. The failed-gate distribution was 1,192 candidates with one failure,
1,521 with two, 2,069 with three, 982 with four, 535 with five, and 32 with six.
Every individual gate had passing candidates, but no single candidate passed
all seven.

The integration-evidenced base candidate 0 missed only exact-action braking at
12.629 mm and passed the complete Lab acceptance target: its hover error was
0.045 m final and 0.089 m over the last second, goto error was 0.019 m final
and 0.031 m over the last second, and the final handoff range was 0.461 m.
Candidate 5,749 also passed that target when only its horizontal proportional
gain was raised from 1.35 to 1.575, with 0.096 m hover and 0.044 m goto
last-window maxima. This preserves an evidenced, minimal large-error gain
schedule for a later attempt; it is not yet the first failed boundary.

Controller-state replay first failed normalized collective. The selected
controller's validation braking profile had normalized motor RMSE 0.00256 but
collective RMSE 0.10495 N. Across the capture, PX4's logged hover estimate
declined from 0.57307 to 0.57075 while the fixed target used 0.57; the roughly
0.0015 neutral offset predicts the observed 0.10 N collective bias. Yaw and
combined profiles also expose later normalized-torque discrepancies, but they
are downstream of this thrust-setpoint failure. Attempt 31 should restore each
profile's logged initial hover estimate as controller state, retain the
authored setting for ordinary environment resets and legacy traces, and rerun
the unchanged candidates before adding large-error gain scheduling.

The candidate trace was compressed losslessly from 5.9 GiB to a verified
1.2 GiB Zstandard archive. The capture, audit, replay, integration probes, GPU
telemetry, fit, and compressed candidates are preserved under the run's
relative-path manifest.

### 116. Restore the profile-start PX4 hover estimate in Attempt 31

Date: 2026-07-31.

Attempt 31 addresses Attempt 30's first failed controller boundary without
changing the allocator, plant, probe suite, gates, search ranges, or ordinary
runtime reset. Alignment trace schema 5 records the normalized PX4
`hover_thrust_estimate` interpolated at each profile's absolute simulation
start. The NumPy and Torch controller states restore that value for profile
replay and use it in the schema-5 acceleration-to-normalized-thrust mapping.
Ordinary environment resets and trace schemas 2 through 4 retain the
artifact-authored normalized hover thrust through a concrete zero sentinel.

The target controller identifier advances to
`isaac-lab-x500-velocity-v33`, producing a new comparison fingerprint. This is
one structural controller-boundary correction supported by the ULog drift and
the 0.10495 N validation-braking collective residual. The complete candidate
set remains unchanged so the next fresh capture measures only the restored
boundary before any large-error gain scheduling is considered.

### 117. Prime airborne actuator state in Attempt 32

Date: 2026-07-31.

Attempt 31's fresh capture and actuator audit passed, and candidate 6,072 became
the first candidate to pass all seven exact-action gates. It achieved 4.289 mm
braking error, 0.00222 m/s residual-speed error, 0.10 s stop-time error, 6.96%
worst-axis velocity NRMSE, 4.83% yaw-rate NRMSE, 2.892 degrees heading error,
and 7.301 mm combined endpoint error. The complete integration target rejected
it immediately because its 80 ms rise time and 250 N/s slew began from zero
realized thrust. Fast-actuator candidates 6,241 and 6,446 removed the crash but
settled at 0.104 m versus the 0.100 m hover limit.

The reset diagnostic confirmed a structural boundary mismatch: the Lab task
spawns every vehicle airborne while clearing the actuator's realized thrust to
zero. Attempt 32 primes ordinary airborne resets to equal, weight-balanced
per-rotor thrust while continuing to clear controller and command-delay state.
Alignment profile replay still overwrites this neutral value with its logged
ongoing-flight rotor boundary. The controller, allocator, actuator dynamics,
plant, probes, gates, search ranges, policy interface, reward, and training
configuration remain unchanged. The target controller identifier advances to
`isaac-lab-x500-velocity-v34` for the new reset contract.

### 118. Attempt 32 removes the reset crash and isolates large-error settling

Date: 2026-07-31.

Attempt 31's fresh run `x500-velocity-px4-v5/20260731T165205Z` evaluated
6,448 unique candidates in 41 one-worker batches over 1,706.5 seconds. Useful
throughput was 3.78 candidates/s; GPU utilization averaged 30.1%, reached 46%
at the 95th percentile, and peaked at 64%, with 6,209 MiB peak memory use.
Candidate 6,072 was selected and 238 candidates passed all seven exact-action
gates. Its actuator audit joined every boundary at zero timestamp error;
normalized motor RMSE ranged from 0.000285 to 0.000329 and collective RMSE
ranged from 0.00866 N to 0.01278 N.

Attempt 32 ran from clean source commit `9d33c60`. Priming the airborne
actuator to weight-balanced thrust removed candidate 6,072's reset crash. The
standard 16-environment integration gate then missed only hover settling at
0.101 m versus the unchanged 0.100 m limit. A time-resolved replay localized
the maximum to environment 13, which began 0.580 m from its target with
horizontal error `(-0.353, +0.449)` m. Its vertical error stayed at or below
0.101 m and had fallen to 0.029 m when total error peaked at 0.102 m; the
remaining miss is horizontal overshoot rather than thrust or altitude error.

Holding every candidate 6,072 setting fixed and raising only horizontal
velocity proportional gain from 1.575 to 1.8 passed the same complete
16-environment target. Hover last-window error fell to 0.080 m, goto
last-window error was 0.024 m, and final handoff range was 0.463 m. Existing
constant-1.8 candidates regress exact low-speed braking and endpoint gates, so
Attempt 33 should preserve gain 1.575 in the exact-action envelope and blend
to 1.8 only for larger horizontal velocity error. This is the previously
deferred large-error gain schedule, now supported by the first failed
acceptance boundary.

### 119. Blend horizontal gain only above the exact-action envelope

Date: 2026-07-31.

Attempt 33 introduces comparison generation `x500-velocity-px4-v6` and
artifact schema 6. It adds a smooth horizontal velocity-error gain schedule
to the NumPy and Torch controller twins. The selected low-speed gain 1.575 is
unchanged through 0.15 m/s horizontal error, blends with a cubic smoothstep,
and reaches the integration-evidenced gain 1.8 at 0.30 m/s. Vertical gain is
unchanged. The effective gain is shared by acceleration control, profile-state
seeding, and horizontal tracking anti-windup.

The three new fields are declarative stage-1 parameters. Gain scale candidates
span 1.0 through 1.285714286, transition starts span 0.10 through 0.20 m/s,
and full-gain boundaries span 0.25 through 0.40 m/s. Schemas 2 through 5 load
with scale 1.0 and preserve their original behavior. Attempt 33 seeds all
remaining base values from exact-action-passing candidate 6,072. Probes,
advisory limits, the standard integration gate, allocator, plant, policy
interface, reward, training configuration, beam width, and one-worker runtime
remain unchanged.

### 120. Attempt 33 isolates fresh-reference stop timing

Date: 2026-07-31.

Attempt 33 completed fresh run
`x500-velocity-px4-v6/20260731T180706Z` from clean source commit `dffc203`.
The capture completed all 11 profiles and the plant-owned actuator audit
passed. Timestamp joins stayed within one 4 ms physics tick, normalized motor
RMSE peaked at 0.000634, and collective-thrust RMSE peaked at 0.01230 N.

The one-worker sweep evaluated 7,625 unique candidates, of which 7,602 were
usable, in 47 GPU batches over 1,495.1 seconds. Useful GPU-stage throughput was
5.10 candidates/s. GPU utilization averaged 30.1%, reached 49% at the 95th
percentile, and peaked at 83%, with 5,833 MiB peak memory use. Candidate 7,412
was selected. It passed braking at 3.531 mm, residual speed at 0.00422 m/s,
worst-axis velocity NRMSE at 4.91%, yaw-rate NRMSE at 4.35%, heading at 2.618
degrees, and combined endpoint at 9.575 mm. It missed only stop time at 0.20 s
versus the unchanged 0.10 s limit. The failed-gate distribution was 1,218
candidates with one failure, 2,394 with two, 2,557 with three, 1,068 with
four, 343 with five, and 22 with six. Every one-failure candidate missed only
stop time; no candidate passed all seven gates.

The selected validation braking controller boundary is already within the
replay limits at 0.00191 normalized motor RMSE and 0.00183 N collective RMSE.
Later vertical, yaw, combined, and stability controller replay boundaries
remain diagnostic failures, but they are downstream of the immediate motion
acceptance boundary. Of the 725 stop-time-passing candidates, the closest
still failed both braking and heading. The best missed braking by 1.401 mm and
heading by 0.600 degrees.

Trace replay also disproved schedule leakage as the cause of the fresh
regression. The seeded candidate's largest horizontal velocity error was
0.111 m/s across every fit and validation profile, below the 0.15 m/s schedule
onset. The selected controller instead places horizontal derivative gain at
the searched upper boundary of 0.30 while still stopping one 0.10 s reporting
interval late. Attempt 34 should seed candidate 7,412 and expand only the
evidenced horizontal braking neighborhood before considering another
structural change.

The candidate trace was compressed losslessly from 7,593,801,257 bytes to a
verified 1.38 GiB Zstandard archive. The capture, audit, controller replay,
GPU telemetry, fit, and compressed candidates are preserved under the run's
relative-path manifest.

### 121. Expand the selected horizontal braking neighborhood

Date: 2026-07-31.

Attempt 34 retains schema 6, comparison generation `x500-velocity-px4-v6`,
the gain schedule, controller and plant contracts, probes, gates, search
strategy, and one-worker execution. Its base settings become Attempt 33
candidate 7,412 so every staged replacement is evaluated locally around the
one-failure frontier instead of around the now weaker historical seed.

The matching validation trace supports a declarative braking expansion. The
selected candidate passes the validation braking actuator boundary but stops
0.10 s late, and its horizontal derivative gain is the existing range maximum
of 0.30. The `kd_vel` candidates therefore add x-axis values 0.325, 0.35,
0.375, and 0.40 while preserving y at 0.30. Nearby x-axis proportional values
1.9, 2.0, and 2.1 and integral values 0.45, 0.50, and 0.60 are also added so
the beam can preserve braking distance and endpoint position if derivative
gain alone shifts the discrete stop crossing. All existing candidates remain
available. No allocator, actuator, policy, reward, training, integration, or
advisory behavior changes in this attempt.

### 122. Accept schema 6 candidate 8,083

Date: 2026-07-31.

Attempt 34 completed fresh run
`x500-velocity-px4-v6/20260731T185547Z` from clean source commit `4e3e005`.
The one-worker sweep evaluated 8,300 unique candidates in 47 GPU batches over
1,742.3 seconds and selected candidate 8,083. Useful throughput was 4.76
candidates/s. GPU utilization averaged 30.3%, reached 50% at the 95th
percentile, and peaked at 73%, with 6,316 MiB peak memory use. The measured
deterministic simulation throughput remains the bottleneck; additional workers
are not justified by the prior representative benchmark.

Candidate 8,083 passes all seven exact-action limits: braking-distance error
is 3.192 mm, residual-speed error is 0.00199 m/s, stop-time error is 0.06 s,
worst-axis velocity NRMSE is 4.38%, yaw-rate NRMSE is 3.89%, heading error is
2.816 degrees, and combined endpoint error is 7.489 mm. Its stability trace is
finite, complete, and usable. The plant-owned actuator audit passes all 11
profiles: timestamp joins are at most one 4 ms physics tick, normalized motor
RMSE peaks at 0.000494, and collective-thrust RMSE peaks at 0.01275 N.

The unchanged standard 16-environment integration target also passes. Hover
position error is 0.036 m final and 0.075 m over the last second. Goto error is
0.006 m final and 0.006 m over the last second. The press handoff finishes
ready at 0.483 m wall-normal range. This satisfies alignment acceptance, and
both `latest` and `best` resolve to the accepted run.

Controller-state replay remains a diagnostic rather than an acceptance gate.
Its validation braking profile is within the replay limits, while vertical
collective and yaw, combined, and stability motor boundaries still exceed the
diagnostic limits. Those later discrepancies do not reopen controller tuning
because identical-action motion and the downstream integration response pass.

The existing actor cannot produce a post-acceptance divergence trace. Every
available actor embeds legacy alignment schema 1. Lab evaluation rejects it as
unsupported provenance, and the PX4 transfer path independently rejects it at
spawn preflight before the first actor evaluation. Observation divergence,
action divergence, and actor-driven state divergence are therefore unavailable,
not zero. The accepted identical-action result remains valid.

The candidate trace is preserved as a verified Zstandard archive with an
uncompressed size of 8,282,853,039 bytes under the run's relative-path
manifest. Training now resolves `x500-velocity-px4-v6/best`. The next session
must run a new 300-iteration Stage 1 scratch training with the accepted schema-6
record embedded, followed by checkpoint selection, independent Lab evaluation,
PX4 transfer, and the complete approach mission. No retraining belongs to this
alignment session.

### 123. Retain only the accepted alignment evidence

Date: 2026-07-31.

Alignment storage now retains only
`x500-velocity-px4-v6/20260731T185547Z`, with both `best` and `latest`
resolving to that accepted run. The final capture, sweep, integration, and
actor-preflight runtime diagnostics were consolidated under its
`acceptance/` directory and added to the relative-path manifest. A
`SHA256SUMS` index covers all 21 retained payloads, and both checksum
verification and Zstandard archive verification pass. The public artifact
loader confirms schema 6, candidate 8,083, advisory acceptance, clean source
commit `4e3e005`, and run ID `20260731T185547Z`.

Rejected alignment generations v1 through v5, rejected v6 run
`20260731T180706Z`, 59 capture workspaces, 43 sweep workspaces, 24 integration
acceptance workspaces, and the two copied final actor-preflight workspaces were
deleted. Unrelated training, benchmark, cache, and integration artifacts were
not touched. Alignment storage decreased by 114,588,796,441 bytes and duplicate
Docker integration storage decreased by 3,946,965,565 bytes, reclaiming
118,535,762,006 bytes in total.

The retained 1.6 GiB bundle includes the lossless candidate archive with an
exact uncompressed size of 8,282,853,039 bytes. It can be rehydrated to
`target_candidates.json` for offline rescoring when tighter gates or new
metrics use already-captured signals. A future gate needing new topics, probes,
controller or plant behavior, images, or environments requires a fresh capture
and comparison fingerprint seeded from candidate 8,083; the accepted evidence
bundle remains immutable.

### 124. Reproduce the accepted alignment from fresh evidence

Date: 2026-07-31.

Run `x500-velocity-px4-v6/20260731T201319Z` tests whether the retained accepted
artifact is transient or whether the committed alignment workflow reproduces
acceptance from scratch. It retains schema 6, the complete controller and plant
contracts, all probe definitions, the 8,300-candidate deterministic search,
one-worker execution, and every advisory and integration limit. It reuses no
reference trace, ULog, candidate trace, fit result, or selected settings from
the accepted run; only committed source and configuration are shared.

The baseline remains run `20260731T185547Z`, candidate 8,083, which passes all
seven exact-action gates and the standard 16-environment integration target.
The reproduction succeeds only if a fresh capture passes its actuator audit,
the fresh fit passes all seven exact-action gates with finite completed
stability traces, and the new artifact itself passes
`make test-integration-alignment-target`. Candidate identity and individual
metric improvements are diagnostic comparisons, not added acceptance
requirements. Publication behavior is not used as proof: the integration gate
will receive the new run path directly even if `best` continues to resolve to
the retained baseline.

### 125. Fresh reproduction does not recover acceptance

Date: 2026-07-31.

Reproduction run `x500-velocity-px4-v6/20260731T201319Z` started from clean
source commit `9970110` and reused no runtime artifact from the accepted run.
Its fresh PX4 capture completed all 11 profiles. The actuator audit passed with
at most one 4 ms timestamp tick of join error, peak normalized motor RMSE
0.000335, and peak collective-thrust RMSE 0.01334 N.

The one-worker fresh sweep evaluated 8,126 unique candidates, of which 8,106
were usable, in 47 GPU batches over 1,670.2 seconds. Useful throughput was 4.87
candidates/s. GPU utilization averaged 29.3%, reached 48% at the 95th
percentile, and peaked at 83%, with 6,812 MiB peak memory use. Candidate 6,425
was selected. It passed residual speed at 0.00104 m/s, stop time at 0.10 s,
worst-axis velocity NRMSE at 5.46%, yaw-rate NRMSE at 4.70%, heading at 2.867
degrees, and combined endpoint at 9.360 mm. It failed braking distance at
5.571 mm versus the unchanged 5 mm limit. No candidate passed all seven gates:
1,672 usable candidates had one failure, 2,877 had two, 2,362 had three, 878
had four, 293 had five, and 24 had six; 20 candidates were unusable.

The fresh selected settings differ materially from accepted candidate 8,083,
and the accepted settings fingerprint is absent from the fresh candidate file.
The staged beam therefore cannot regenerate the accepted combination on demand
even though every individual value remains declarative. Candidate 6,425's
controller replay has the same known later-profile diagnostic failures as the
accepted run; its immediate validation-braking boundary remains within replay
limits, so that diagnostic does not explain away the motion-gate miss.

Direct integration confirms the rejection. Candidate 6,425 settles hover at
0.059 m final and 0.098 m last-second maximum error, then goto at 0.033 m final
and 0.043 m last-second maximum error. Its handoff ends at 0.447 m wall-normal
range and fails as too close. The retained candidate 8,083 was rerun
contemporaneously in the same current environment and reproduced its accepted
result exactly: hover 0.036 m final and 0.075 m last-second maximum, goto 0.006
m final and last-second maximum, and a ready handoff at 0.483 m.

Controller, reference, target, probe-suite, and image fingerprints are equal
between the runs. The comparison fingerprint nevertheless changed because the
resolved experiment fingerprint includes the downstream training artifact
path, which acceptance commit `eba4ffe` changed from v2 `best` to v6 `best`.
This behavior-only-irrelevant change initialized the rejected reproduction as
`best`; publication was restored manually so `best` resolves to accepted run
`20260731T185547Z` and `latest` resolves to the reproduction.

The result disproves treating the accepted artifact as transient under the
current workflow. Keep the accepted run. Before reconsidering deletion, make
its complete settings an unconditional evaluated seed, remove downstream-only
training selection from the alignment comparison fingerprint, and require
multiple independent clean campaigns to pass both exact-action and integration
acceptance. The reproduction's capture, audit, replay, fit, GPU telemetry,
integration comparison, and losslessly compressed candidate traces are retained
under its checksummed relative-path manifest.
