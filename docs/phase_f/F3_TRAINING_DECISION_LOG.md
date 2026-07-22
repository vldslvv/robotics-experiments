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
