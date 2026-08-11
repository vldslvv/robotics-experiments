# Press policy training decision log

Status: SIXTH SCRATCH SELECTION REJECTED; LATE HARD-CASE CURRICULUM PREDECLARED
(2026-08-11).

This is the chronological record for press-policy scratch training, checkpoint
selection, acceptance, and deployment decisions. Controller fitting and its
evidence remain in `docs/phase_f/F3_TRAINING_DECISION_LOG.md`. Each entry states
the practical intuition first, followed by the implementation-level reasoning
and measurements. Every training attempt records enough provenance to explain
what changed and why; post-run fine-tuning and checkpoint-initialized retries
are unsupported.

## Run ledger

| Run ID | Controller run | Scratch schedule | Held-out result | Decision |
| --- | --- | --- | --- | --- |
| `20260811T030638Z-scratch-aligned` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 11 at `3e-5` | Iteration 126 selected; score -1.604688, 63.0% success, advisory rejected | Diagnose the timeout/failure boundary before the next scratch run |
| `20260811T034512Z-scratch-hard30-coarse` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 51 at `3e-5` | Iteration 150 selected; score -0.566693, 94.7% success, advisory rejected | Fine-sweep updates 145-153 from scratch |
| `20260811T041833Z-scratch-hard30-fine` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 33 at `3e-5` with every-update saves near the peak | Iteration 149 selected; score -0.566482, 94.7% success, advisory rejected | Expand the evidenced hard-case reset mixture in the next scratch lineage |
| `20260811T045313Z-scratch-hard40-coarse` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 31 at `3e-5` with 40% difficult resets | Iteration 140 selected; score -0.519048, 96.0% success, advisory rejected | Fine-sweep updates 136-144 from scratch |
| `20260811T052107Z-scratch-hard40-fine` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 24 at `3e-5` with every-update saves near the 40% peak | Iteration 142 selected; score -0.516978, 96.0% success, advisory rejected | Cache identical held-out evaluations, then expand to the evidenced upper reset bracket |
| `20260811T060552Z-scratch-hard50-coarse` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 31 at `3e-5` with 50% difficult resets | Iteration 150 selected; score -0.527187, 95.7% success, advisory rejected | Preserve the stronger 40% learning path, then introduce 50% difficult resets only after its peak |

### 1. Start one scratch lineage from the jointly certified controller

Date: 2026-08-11.

Plain-language view: The first policy run after alignment starts with a new
network and optimizer. It does not reuse or fine-tune an older policy. The
accepted controller is treated as a fixed dependency so policy results cannot
silently change when alignment labels move.

Technical view: Run `20260811T030638Z-scratch-aligned` embeds controller run
`20260810T-accepted-reproduction`, controller digest
`5324c2e822886c49cc9765ec58cc413d13689873ca84e48ffdc3bc255c018bc1`,
and certified `velocity` plus `ctbr` control modes. The policy uses the strict
schema-1 `press-approach` interface with a 34-value observation, four velocity
actions, a 20 Hz policy cadence, and scratch-neutral actor output. Its first
phase ran 121 PPO updates at learning rate `1e-4`, entropy coefficient `1e-4`,
and the full handoff reset distribution. The same optimizer lineage then ran
11 updates at `3e-5`, zero entropy, and a 30% difficult-reset mixture. Training
completed all 132 updates in 847 seconds and wrote checkpoints 0, 30, 60, 90,
120, and 121 through 130.

The run is preserved under
`.build/training/press-approach/20260811T030638Z-scratch-aligned`. TensorBoard
uses its `logs` directory and is available at `http://127.0.0.1:6006` while the
run is active.

### 2. Repair evaluation bookkeeping without changing the trained policy

Date: 2026-08-11.

Plain-language view: The first evaluation launch stopped because some episode
counters had not been initialized. This was an evaluator bookkeeping defect,
not evidence that the trained weights were bad, so the completed scratch run
should be evaluated rather than discarded or retrained.

Technical view: `SwitchPressEnv` initialized collision, pusher, flight-failure,
and timeout counters inside the training hard-case setter instead of its
constructor. Checkpoint selection therefore failed before scoring iteration 0.
The counters now initialize with the other episode counters in `__init__`, and
the hard-case setter only updates the reset-mixture fraction. An AST regression
test verifies that all seven counters are constructor-owned and none are reset
by a curriculum phase change. The focused tests and complete host check passed.
No checkpoint, policy metadata, controller record, or evaluation threshold was
changed.

### 3. Evaluate every checkpoint on the held-out handoff distribution

Date: 2026-08-11.

Plain-language view: Later checkpoints are substantially better than the early
ones, but the policy did not reach the 95% advisory success threshold. The
result is diagnostic evidence for the next scratch run, not an accepted policy.

Technical view: Selection evaluates 100 episodes for each held-out seed 101,
202, and 303 with a 200-step horizon. Iterations 0, 30, 60, 90, and 120 improve
from 0% to 31.3% success. The short second phase then reaches 40.3% at iteration
123, 57.3% at 124, 62.7% at 125, and a best of 63.0% at 126 with score
-1.604688. Collision, body-collision, and pusher-collision rates are zero. The
remaining terminal outcomes at iteration 126 are 9.7% other flight failure and
27.3% timeout. Iterations 127 through 130 regress below that frontier. The
selector publishes iteration 126 as `latest` and as the first `best` for
comparison fingerprint
`bf2f671cf5b76483975f16e43f98117e64ebe5f0eecc4c169a8a9824280db8fe`,
but its metadata records `advisory_accepted: false`.

Training telemetry supports more learning rather than an evaluator or
collision fix. The first phase improves continuously but reaches only 5.6%
cumulative training success by update 120. The difficult-reset phase raises
that cumulative phase-local measure from 0% to 7.6% over 11 updates while
held-out success jumps from 31.3% to the 63.0% frontier. Policy noise remains
stable near 0.024, the surrogate loss stays small, and no collision boundary
appears. Before the next scratch attempt, compare the phase-local failure modes
and schedule length against the earlier accepted scratch recipe; extend or
reshape the in-run schedule only from that evidence.

Independent evaluation reproduces the selected metrics exactly and retains
the full worst-range trajectories under
`.build/docker/integration/eval-press.20260811T033955Z.45066`. The difficult
seed-101, seed-202, and seed-303 trajectories cross 0.400 m without contact
while correcting initial tangential error. At the crossings, normalized
wall-normal action is near zero or slightly negative, but residual forward
speed remains about 0.018-0.035 m/s. This confirms incomplete learned braking
and alignment coordination rather than collision or an unaligned controller.

### 4. Extend the improving hard-case phase as a coarse sweep

Date: 2026-08-11.

Plain-language view: The first difficult-reset phase helped substantially but
ended after only 11 updates. Before changing the task or optimizer, give the
same scratch lineage more updates and sample its progress at a moderate
interval. If a promising region appears, a later scratch run can inspect only
that interval at every update.

Technical view: Keep the accepted controller, scratch-neutral initialization,
121-update full-reset phase, PPO parameters, reward, task, fixed `3e-5` second
phase rate, zero entropy, and 30% difficult-reset mixture unchanged. Extend the
second phase from 11 to 51 updates and change its save interval from one to
five. This evaluates iterations 125 through 170 at five-update spacing while
retaining the broad phase checkpoints. It changes only duration and checkpoint
resolution, tests whether the 63% frontier is an undertrained consolidation
phase, and bounds held-out evaluation cost before any targeted fine sweep.

### 5. The coarse sweep brackets a near-acceptance interval

Date: 2026-08-11.

Plain-language view: More difficult-reset training was the right direction.
The policy reached almost 95% success, but later updates first became too
aggressive and then oscillated. The useful region is narrow enough that the
next run should inspect every update there instead of extending training again.

Technical view: Run `20260811T034512Z-scratch-hard30-coarse` started from a
clean `63d5491` source commit and reproduced checkpoints 0 through 125 exactly.
It completed 172 scratch updates in 1,073.5 seconds at approximately
169k-177k frames/s. Selection evaluated 15 checkpoints in 640.1 seconds.
Checkpoint 150 was selected with score -0.566693, 94.67% success, 5.33% other
flight failure, zero timeout and collision, 0.41418 m mean closest range, and
0.39858 m minimum closest range. It narrowly misses the 95% success boundary
and still fails the zero-flight-failure and 0.400 m minimum-range gates.

The new checkpoint curve is diagnostic rather than monotonic. Success rises
from 69.7% at 135 to 82.7% at 140, 87.3% at 145, and 94.7% at 150, then falls
to 83.0% at 155. Later checkpoints recover only to 91.0% and 92.0% before
ending at 88.7%. Collisions stay zero throughout, while timeout becomes zero
from 145 onward. Relative to the first run's 63.0% frontier, checkpoint 150
reduces success-gate shortfall by about 98.4% and is material progress.

The coarse five-update spacing brackets the best interval but cannot show
whether updates 145-149 or 151-153 pass. Reproduce the same scratch path, keep
coarse checkpoints through 144, and save every update from 145 through 153.
Do not change controller, task, reward, PPO values, reset mixture, acceptance
gates, or initialization. This is the targeted precision stage justified by
the broad sweep.

### 6. Resolve the coarse peak with an every-update scratch sweep

Date: 2026-08-11.

Plain-language view: Rerun from new weights under the identical learning path,
but record every policy update around the best coarse checkpoint. This tests
the narrow promising region without fine-tuning or evaluating dozens of
unpromising checkpoints.

Technical view: Keep phase 1 at 121 updates and split the unchanged difficult
curriculum into 25 coarse updates followed immediately by 10 fine updates on
the same runner, optimizer, environment, rate, entropy, and 30% reset mixture.
The coarse phase saves every five updates through model 140, followed by the
shared phase-boundary model 144. The fine phase saves models 145 through 153
individually. Total training falls from 172 to
156 updates, while the selector still evaluates 15 candidates and adds exact
resolution only in the evidenced interval. Because the split does not reload
state or change any optimization value, checkpoints through 144 must reproduce
the coarse run before the new candidates can be trusted.

### 7. The exact checkpoint sweep confirms a reset-boundary failure

Date: 2026-08-11.

Plain-language view: The every-update sweep found no hidden passing policy.
The best policy is repeatable, but it still carries too much forward momentum
in a small set of difficult handoffs. Training longer makes that behavior
worse, so the next run should expose the policy to more of those hard starts
instead of adding epochs.

Technical view: Run `20260811T041833Z-scratch-hard30-fine` started from clean
source commit `b4a5b53`, a new network, and a new optimizer. It completed its
configured 156-iteration schedule in 952.2 seconds at approximately 175k-178k
frames/s and produced terminal model 153. The
phase-boundary checkpoint numbering is one lower than the initial estimate:
model 144 exactly reproduced coarse model 145, and model 149 reproduced the
coarse peak. Selection evaluated 19 total checkpoints in 814.2 seconds.

Checkpoint 149 was selected with score -0.566482, 94.67% success, 5.33% other
flight failure, zero collision and timeout, 0.41414 m mean closest range, and
0.39874 m minimum closest range. Models 145-148 progressed from 85.67% to
94.0% success, but model 150 fell to 93.67%; models 151-153 then degraded to
88.67%, 80.67%, and 80.33%. Every candidate from model 144 onward had zero
timeouts, so the post-peak loss is entirely additional range-undershoot flight
failure. The exact sweep therefore rejects checkpoint spacing and insufficient
duration as explanations.

Relative to the first scratch run's 63.0% frontier, this remains material and
repeatable progress, but it misses all three coupled acceptance boundaries:
95% success, zero other flight failure, and 0.400 m minimum closest range. The
first failed behavioral boundary remains learned braking on the difficult
0.45-0.47 m handoff range with 0.06-0.08 m tangential error. Historical
accepted-policy evidence brackets hard-case exposure between 30% and 50% for
eliminating undershoots. Change only that training reset mixture in the next
scratch lineage; retain the controller, reward, optimizer, cadence, full-reset
held-out distribution, and acceptance gates.

### 8. Coarse-sweep the reset-mixture bracket from scratch

Date: 2026-08-11.

Plain-language view: Increase the share of difficult starts halfway from 30%
to the historical 50% upper bracket. Save at broad intervals first so a long
held-out sweep is not spent measuring nearly identical policies. A later
scratch run will add every-update precision only if this run exposes a narrow
promising interval without passing directly.

Technical view: Keep phase 1, the accepted controller, reward, policy network,
PPO parameters, optimizer lineage, full-distribution held-out evaluation, and
all acceptance thresholds unchanged. Replace the 30% consolidation mixture
with the bracket midpoint of 40%. Run 31 consolidation iterations at the same
fixed `3e-5` learning rate and zero entropy, saving every five updates. This
produces a six-candidate coarse response after model 120 while taking four
fewer total updates than the rejected fine run and avoiding nine low-value
every-update evaluations. The only behavioral training change is the reset
mixture supported by the persistent undershoot traces.

### 9. The midpoint mixture passes success but not exact range safety

Date: 2026-08-11.

Plain-language view: More difficult starts improved braking enough to pass the
success threshold, but four percent of held-out flights still crossed the hard
range boundary. The broad checkpoints show a sharp improvement followed by a
clear regression, so inspect the missing individual updates around that peak
before changing how the policy learns again.

Technical view: Run `20260811T045313Z-scratch-hard40-coarse` started from clean
source commit `b7d0aaa`, a new network, and a new optimizer. Training completed
its 152-iteration schedule in 927.3 seconds at approximately 176k-178k
frames/s. Selection evaluated 11 checkpoints in 493.4 seconds, compared with
19 checkpoints and 814.2 seconds for the preceding fine run.

Model 140 was selected with score -0.519048, 96.0% success, 4.0% other flight
failure, zero collision and timeout, 0.41489 m mean closest range, and 0.39902 m
minimum closest range. It passes the success, collision, and timeout gates and
improves score by 8.4% relative to the 30% model-149 frontier. Model 135 was
still split between 63.67% success, 20.67% flight failure, and 15.67% timeout;
model 145 regressed to 93.67% success and 6.33% flight failure, while model 150
fell to 89.33% success and 10.67% flight failure. No collision boundary
appeared.

The response therefore brackets a narrow transition between models 136 and
144. The exact 95% success gate has passed, but acceptance still requires zero
other flight failure and minimum range at least 0.400 m. Run one every-update
scratch sweep only across that bracket with all training and evaluation values
unchanged. If none passes, checkpoint resolution is exhausted and the next
attempt must address the remaining braking boundary structurally or with the
already evidenced upper reset-mixture bracket.

### 10. Resolve the 40% coarse peak at every update

Date: 2026-08-11.

Plain-language view: Repeat the improving scratch recipe from new weights, but
save every policy update only between the last weak model and the first
regressing model. This is the least expensive way to determine whether the
coarse sweep skipped a policy that obeys the exact range boundary.

Technical view: Keep phase 1 and every optimization, controller, task, reward,
reset, held-out, and acceptance value identical to run
`20260811T045313Z-scratch-hard40-coarse`. Split the unchanged 40% difficult
phase after 16 iterations. Its first segment retains five-update checkpoints
through model 135; its second segment immediately continues the same runner and
optimizer for ten configured iterations, saving models 136-144 individually.
This lowers training from 152 to 147 configured iterations and evaluates 17
checkpoints while adding exact resolution only inside the evidenced bracket.

### 11. Exact resolution exhausts the 40% checkpoint boundary

Date: 2026-08-11.

Plain-language view: No skipped checkpoint passes. Five consecutive models
reach or exceed the success threshold, but all retain essentially the same
unsafe range crossings. The next training attempt must strengthen braking
exposure rather than save more checkpoints at 40%.

Technical view: Run `20260811T052107Z-scratch-hard40-fine` started from clean
source commit `9a049f9`, a new network, and a new optimizer. Training completed
its 147-iteration schedule in 905.1 seconds at approximately 175k-177k
frames/s. Selection evaluated 17 checkpoints in 755.1 seconds. Model 142 was
selected with score -0.516978, 96.0% success, 4.0% other flight failure, zero
collision and timeout, 0.41477 m mean closest range, and 0.39861 m minimum
closest range.

Models 138-142 all pass the 95% success threshold. Their flight-failure rates
remain confined to 4.0%-4.33%, and their minimum closest ranges remain below
0.400 m. Model 143 falls back below the success gate and model 144 regresses
further. The fine sweep therefore improves score by only 0.4% over the coarse
frontier and does not improve the first failed boundary at all. Stop repeating
40% checkpoint searches.

The scratch prefix through model 135 is byte-identical across the coarse and
fine runs, but selection spent about six minutes reevaluating eight identical
checkpoints. Before the next scratch attempt, add a content-addressed held-out
evaluation cache keyed by the checkpoint digest and the complete evaluator,
controller, policy, environment, seed, and threshold contract. Cache reuse is
only a selection optimization; final independent evaluation remains fresh and
uncached. After verification, test the existing 50% upper reset-mixture bracket
because 40% improved success but left the undershoot rate unchanged.

### 12. Deduplicate byte-identical held-out checkpoints

Date: 2026-08-11.

Plain-language view: Repeated scratch runs intentionally reproduce a long
common training prefix. Do not fly the same policy through the same statistical
selection workload again when its checkpoint bytes and every evaluation input
are identical. Reuse that selection measurement and keep the final independent
acceptance run fresh.

Technical view: Add a transient content-addressed cache under
`.build/training/<policy>/.evaluation-cache`. Its key includes checkpoint
SHA-256, policy comparison fingerprint, controller digest, and an explicit
press-evaluator revision. The comparison fingerprint already owns the policy
interface, observation, task, scene, vehicle, physics, goal, reset, handoff,
termination, held-out seeds, episode count, weights, and thresholds. Cache
payloads retain metrics, margins, reward means, acceptance, source run, and
source iteration; current reports identify each candidate's evaluation source
as `fresh` or `cache`.

At selection startup, same-fingerprint and same-controller reports seed missing
entries, so the next run benefits without a sacrificial reevaluation. Invalid
cache entries emit a warning and fall back to a fresh evaluation. The cache is
never consulted by `eval_press.py`, PX4 transfer, or deployment. Host tests
cover content and controller isolation, cached selection without environment
execution, report seeding, and iteration-local ranking reconstruction.

### 13. Cache benchmark removes repeated selection cost

Date: 2026-08-11.

Plain-language view: The optimized selector reproduced the exact same winner
and scores in seconds instead of minutes. It can now spend simulator time only
on checkpoints that contain genuinely new policy state.

Technical view: Reselect run `20260811T052107Z-scratch-hard40-fine` from clean
implementation commit `66c32c5`. Startup primed 50 unique entries from the
five same-fingerprint, same-controller scratch reports. All 17 candidates were
content-addressed cache hits. Model 142 remained selected with score -0.516978
and unchanged metrics. The selector test stage fell from 755.1 seconds to 9.55
seconds, a 98.7% reduction; total container time was 10.0 seconds. The report
and selected policy metadata mark the cached evaluation source, while the
independent acceptance command remains fresh by contract.

Proceed to the already evidenced 50% upper reset-mixture coarse sweep. Retain
the byte-identical phase-1 prefix, accepted controller, reward, optimizer,
held-out distribution, and gates. Cache hits should eliminate its five repeated
baseline evaluations; only new post-transition models require simulator work.

### 14. Coarse-sweep the upper reset-mixture bracket

Date: 2026-08-11.

Plain-language view: Half of the consolidation starts now come from the exact
difficult corner that still crosses the range boundary. Search broadly first:
the stronger exposure may fix undershoot earlier, but too much can also make
the actor brake so conservatively that it times out.

Technical view: Keep phase 1 at 121 updates and retain the accepted controller,
reward, policy architecture, PPO parameters, full-distribution evaluation, and
all gates. Change only the difficult-reset fraction from 40% to the historical
50% upper bracket. Run 31 consolidation iterations at fixed `3e-5`, zero
entropy, and five-update checkpoint spacing. The six new post-transition
checkpoints cover models 125-150. Content-addressed cache hits cover the five
byte-identical phase-1 checkpoints, reducing the expected held-out workload
from eleven fresh candidates to six. Fine resolution is justified only if the
coarse response exposes a new promising interval without passing directly.

### 15. The upper bracket does not remove the range undershoot

Date: 2026-08-11.

Plain-language view: Starting the whole consolidation phase with half difficult
starts is slightly worse than the 40% recipe. It still learns a high-success
approach, but it does not eliminate the unsafe final centimeters. Keep the
stronger 40% path and apply the harder mixture only after that path reaches its
best region.

Technical view: Run `20260811T060552Z-scratch-hard50-coarse` started from clean
source commit `698069a`, a new network, and a new optimizer. Training completed
its 152-update schedule in 927.4 seconds at approximately 175k-178k frames/s.
Model 150 was selected with score -0.527187, 95.67% success, 4.33% other flight
failure, zero collision and timeout, 0.41510 m mean closest range, and 0.39872 m
minimum closest range. Model 145 reached 95.33% success with 4.67% other flight
failure. Both pass the success threshold but remain below the 0.400 m hard
range boundary. Relative to the 40% model-142 frontier, the selected score
regresses by 2.0% and the first failed boundary does not improve.

Selection took 319.5 seconds. Models 0, 30, 60, and 90 were content-addressed
cache hits; model 120 was fresh because the checkpoint serialization differed,
even though its policy metrics remained comparable. The cache correctly reused
only identical bytes and reduced the fresh simulator workload from eleven to
seven candidates.

Do not fine-sweep a static 50% mixture: its coarse curve exposes no better
boundary. The two completed brackets instead support an ordered curriculum in
one scratch lineage. Reproduce the 40% consolidation through its evidenced
model-142 peak, then increase difficult starts to 50% for a short final phase
without resetting the runner or optimizer. This tests whether stronger late
braking exposure can remove the remaining undershoots without paying the early
learning penalty observed when 50% is active for the whole consolidation.

### 16. Increase hard-case exposure only after the 40% peak

Date: 2026-08-11.

Plain-language view: Learn the broadly successful approach with 40% difficult
starts first, then spend the final updates concentrating on the unsafe corner.
This remains one training run from new weights; no checkpoint is loaded and the
optimizer is never replaced.

Technical view: Reproduce the byte-identical phase-1 prefix and the prior 40%
fine-sweep schedule through model 142: 121 full-distribution updates at `1e-4`,
16 coarse 40% updates at `3e-5`, then eight every-update 40% updates at the same
rate. Continue the same runner and optimizer for 12 every-update iterations
with only `hard_case_fraction` increased from 0.4 to 0.5. Keep the accepted
controller, reward, network, optimizer values, held-out distribution, and
acceptance gates unchanged.

The first three phases deliberately reproduce the selected 40% frontier before
the new intervention. Cached identical checkpoints make that reproduction
cheap to select. Every-update saves in the 50% phase will show whether late
hard exposure monotonically improves braking, exposes a narrow passing policy,
or immediately regresses. If it does not improve the zero-flight-failure and
minimum-range gates, reset-mixture scheduling is exhausted and the next change
must directly strengthen the already identified zero wall-normal velocity
boundary in the task objective.
