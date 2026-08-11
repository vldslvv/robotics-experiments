# Press policy training decision log

Status: FIFTEENTH SCRATCH SELECTION REJECTED; DIRECT COMMAND TARGET NEXT
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
| `20260811T063515Z-scratch-late-hard50` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 24 at `3e-5` with 40% difficult resets and 12 at 50% | Iteration 150 selected; score -0.507785, 96.0% success, advisory rejected | Preserve independent worst traces, then strengthen the evidenced braking objective |
| `20260811T070651Z-scratch-barrier40-coarse` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 31 at `3e-5` with 40% difficult resets; 0.040 m range barrier | Iteration 145 selected; score -1.579391, 65.0% success, advisory rejected | Restore the 0.020 m barrier and fit a one-sided unsafe closing-speed term |
| `20260811T073726Z-scratch-closing300-coarse` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 31 at `3e-5` with 40% difficult resets; closing-speed weight 300 | Iteration 140 selected; score -0.580549, 94.0% success, advisory rejected | Increase the evidenced but underpowered one-sided term by one coarse decade |
| `20260811T080541Z-scratch-closing3000-coarse` | `20260810T-accepted-reproduction` | Same scratch schedule; closing-speed weight 3000 | Iteration 125 selected; score -0.639254, 93.0% success, advisory rejected | Stop fitting the state-only term and add predictive braking evidence |
| `20260811T083710Z-scratch-predictive400ms-w10` | `20260810T-accepted-reproduction` | Same scratch schedule; predictive horizon 0.4 s at weight 10 | Iteration 150 selected; score -0.527009, 95.33% success, advisory rejected | Extend the still-improving low-rate phase in a new scratch lineage |
| `20260811T090631Z-scratch-predictive-extended61` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 61 at `3e-5`; predictive horizon 0.4 s at weight 10 | Iteration 160 selected; score -0.472188, 96.67% success, advisory rejected | Stop extending and move prediction into the dense velocity reference |
| `20260811T093944Z-scratch-predictive-reference400ms` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 31 at `3e-5`; 0.4 s predictive velocity reference | Iteration 135 selected; score -0.471879, 97.0% success, advisory rejected | Strengthen the evidenced but underweighted dense tracking target |
| `20260811T100846Z-scratch-predictive-reference-vtrack900` | `20260810T-accepted-reproduction` | Same scratch schedule and predictive reference; velocity-tracking weight 900 | Iteration 145 selected; score -1.447104, 67.7% success, advisory rejected | Refine the overbraking bracket at weight 600 |
| `20260811T103705Z-scratch-predictive-reference-vtrack600` | `20260810T-accepted-reproduction` | Same scratch schedule and predictive reference; velocity-tracking weight 600 | Iteration 135 selected; score -0.518753, 95.7% success, advisory rejected | Stop the scalar sweep and teach the physical velocity command directly |

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

### 17. Late hard exposure improves score but not the exact boundary

Date: 2026-08-11.

Plain-language view: Switching to more difficult starts late does not solve the
last four percent of unsafe approaches. Some checkpoints get better overall,
but the same flights still carry forward speed across the minimum range. Stop
changing only the reset mixture and strengthen the braking lesson itself.

Technical view: Run `20260811T063515Z-scratch-late-hard50` started from clean
source commit `38ef479`, a new network, and a new optimizer. Its four phases
completed 157 configured updates in 958.1 seconds. The selector reused models
0-141 from the content-addressed cache and evaluated models 142-153 fresh in
504.7 seconds.

Model 150 was selected with score -0.507785, 96.0% success, 4.0% other flight
failure, zero collision and timeout, 0.41494 m mean closest range, and 0.39852 m
minimum closest range. Models 147 and 149-151 have the same 96.0% success and
4.0% failure split. The intervention initially regresses to 89.0% success and
11.0% failure at model 146, recovers without crossing the exact boundary, then
regresses again after model 151. Compared with the prior 40% model-142
frontier, selected score improves by 1.8%, but flight-failure rate and minimum
range do not improve. This is not material progress under the exact safety
gates.

Static 30%, 40%, and 50% mixtures, exact checkpoint resolution, and a late
40%-to-50% curriculum have now exhausted the reset-exposure hypothesis. The
first failed behavior remains the target-velocity boundary already visible in
the retained worst traces: when tangential error exceeds the alignment gate,
the wall-normal reference is zero, yet the policy retains small positive
wall-normal velocity long enough to cross 0.400 m. Run a fresh independent
evaluation of model 150 to preserve the current worst traces, then change one
task-owned braking signal supported by those traces while keeping the
controller and acceptance contract fixed.

### 18. Independent traces locate the braking signal too late

Date: 2026-08-11.

Plain-language view: The selected policy is already trying to stop, but the
strong range warning begins after the vehicle has too much forward speed for
the remaining distance. Start that warning earlier; do not change the aligned
controller or relax the safety boundary.

Technical view: Fresh independent evaluation of model 150 reproduced 96.0%
success, 4.0% other flight failure, zero collision and timeout, 0.41491 m mean
closest range, and 0.39852 m minimum closest range. The uncached evidence is
preserved under
`.build/docker/integration/eval-press.20260811T070218Z.262417`.

The seed-101 worst trace enters 0.420 m at 0.045 m/s wall-normal speed and
crosses 0.400 m at 0.022 m/s. Seed 202 enters at 0.050 m/s and crosses at
0.039 m/s; seed 303 enters at 0.052 m/s and crosses at 0.039 m/s. Actor
wall-normal actions are already near zero or slightly negative before the
crossing, so the failure is residual closing motion rather than a positive
approach command or controller tracking reversal. The existing normalized
range barrier begins only at 0.420 m because its margin is 0.020 m. Its dense
braking signal therefore starts after the retained traces require substantial
deceleration.

### 19. Coarse-sweep an earlier range barrier from scratch

Date: 2026-08-11.

Plain-language view: Double the warning distance from two to four centimeters.
This gives the policy the same shaped safety lesson before the dangerous final
segment, while success and forward progress still reward reaching the existing
0.400-0.420 m goal.

Technical view: Change only `range_barrier_margin_m` from 0.020 m to 0.040 m.
The barrier now begins at 0.440 m and retains its weight, normalized quadratic
shape, 0.400 m hard boundary, and 0.420 m goal maximum. Revert the exhausted
late-mixture experiment to the strongest simple schedule: 121 full-distribution
updates at `1e-4`, then 31 updates with 40% difficult starts at `3e-5`, saving
every five updates. Start a new actor, critic, runner, and optimizer.

This is a broad task-parameter step supported by the measured stopping
distance, not checkpoint fine-tuning. Keep controller settings, policy
architecture, PPO parameters, all other rewards, reset distributions,
held-out seeds, and acceptance gates unchanged. If the broader barrier reduces
range failures but causes timeouts, target the margin between 0.020 and 0.040 m;
if the first failed boundary does not improve, add a distinct one-sided
closing-speed signal instead of increasing the barrier again.

### 20. A four-centimeter barrier suppresses useful approach learning

Date: 2026-08-11.

Plain-language view: Starting the full range penalty earlier makes the policy
too reluctant without making its completed approaches safe. Restore the
original barrier and teach only the specific mistake: carrying more forward
speed than the remaining stopping distance permits.

Technical view: Run `20260811T070651Z-scratch-barrier40-coarse` started from
clean source commit `c9f42ab`, a new network, and a new optimizer. Training
completed 152 updates in 927.1 seconds. The changed reward produced a new
scratch trajectory, so the selector evaluated all 11 checkpoints fresh in
501.4 seconds.

Model 145 was selected with score -1.579391, 65.0% success, 13.67% other
flight failure, 21.33% timeout, zero collision, 0.41415 m mean closest range,
and 0.39940 m minimum closest range. Model 150 regressed to 59.67% success,
15.67% flight failure, and 24.67% timeout. The original 0.020 m barrier's
frontier was 96.0% success with 4.0% flight failure, so doubling the margin is
a large regression in every behavioral boundary and does not improve exact
range safety. Do not target an intermediate barrier margin: the broad signal
penalizes position in the valid 0.400-0.420 m goal region regardless of whether
the vehicle is moving safely.

### 21. Penalize only closing speed above a stopping envelope

Date: 2026-08-11.

Plain-language view: A vehicle moving slowly near the switch is safe; a vehicle
moving quickly at the same range is not. Add a penalty for only the excess
forward speed, while leaving retreat, slow approach, and the original goal
reward untouched.

Technical view: Restore `range_barrier_margin_m` to 0.020 m. Add one signed
`range_closing_speed` reward contribution. Its allowed forward speed is
`wall_normal_gain_per_s * max(range - range_min, 0)`; only positive actual
wall-normal speed above that envelope is penalized quadratically. Negative
speed and forward speed below the distance-scaled envelope receive zero. This
uses the existing goal gain and adds one declarative
`range_closing_speed_weight`, set to 300 for the first coarse scratch run.

At 0.450 m the envelope permits 0.0625 m/s, so the handoff distribution is not
suppressed. At 0.420 m it permits 0.025 m/s, directly penalizing the retained
0.045-0.052 m/s traces; at 0.400 m it requires zero forward speed. Keep the
121-plus-31 update 40% schedule, controller, all other rewards, PPO values,
held-out distribution, and gates fixed. Host tests cover one-sided behavior
and NumPy/Torch parity. Increment the selection evaluator revision because its
reported reward-term schema gains a field; independent acceptance remains
uncached.

### 22. The first closing-speed weight is too weak

Date: 2026-08-11.

Plain-language view: The selective braking idea does not suppress approach
learning like the wide barrier, but its first weight is too small to change the
unsafe final motion. Increase the same signal substantially before rejecting
the structure.

Technical view: Run `20260811T073726Z-scratch-closing300-coarse` started from
clean source commit `5f5a893`, a new network, and a new optimizer. Training
completed 152 updates in 931.4 seconds. Evaluator revision 2 intentionally
invalidated old diagnostic records, so all 11 checkpoints were evaluated fresh
in 433.6 seconds.

Model 140 was selected with score -0.580549, 94.0% success, 6.0% other flight
failure, zero collision and timeout, 0.41363 m mean closest range, and 0.39857 m
minimum closest range. Model 125 reached 92.67% success and 7.33% failure;
models 135 and 140 reached the best 94.0%/6.0% split before later regression.
This avoids the wide barrier's timeout failure, but it regresses the original
96.0%/4.0% frontier and does not improve minimum range.

Fresh independent evaluation reproduced model 140 exactly and preserved its
traces under `.build/docker/integration/eval-press.20260811T080140Z.334113`.
All three worst profiles still cross 0.400 m with approximately 0.027-0.032 m/s
forward speed; no collision or different flight-envelope boundary appears.
The mean `range_closing_speed` reward is only -0.00473, compared with -0.24769
for the range barrier and -1.25904 for velocity tracking. The signal is
correctly localized but underpowered.

### 23. Coarse-sweep one decade of closing-speed authority

Date: 2026-08-11.

Plain-language view: Make the selective stopping lesson ten times stronger.
It will still be zero for safe approach and retreat, so this tests braking
authority without reintroducing the wide barrier's position-only penalty.

Technical view: Increase only `range_closing_speed_weight` from 300 to 3000.
Retain the same stopping envelope, 0.020 m range barrier, reward terms,
controller, network, 121-plus-31 update 40% schedule, optimizer values,
held-out distribution, and acceptance gates. This order-of-magnitude step is
supported by the retained reward telemetry; it should raise the localized term
toward a meaningful fraction of the existing barrier while remaining far below
the global velocity-tracking contribution on average.

If range failure falls materially, target the passing interval between 300 and
3000 with a narrower weight sweep. If the failure is unchanged or worsens,
stop fitting this reward term and inspect whether the policy action-to-velocity
controller delay requires an explicit predictive target rather than a
state-only penalty.

### 24. A decade more closing-speed weight worsens the frontier

Date: 2026-08-11.

Plain-language view: Making the same warning ten times louder does not teach a
safer stop. It first makes the actor reluctant to finish, then produces more
range failures once it learns to approach. Stop tuning this weight and move the
warning earlier by accounting for the distance the vehicle will travel while
the controller is still slowing it down.

Technical view: Run
`20260811T080541Z-scratch-closing3000-coarse` started from clean source commit
`434cf8c`, a new network, and a new optimizer. Training completed 152 updates
in 932.9 seconds and evaluated all 11 checkpoints on the unchanged held-out
set.

Model 125 was selected with score -0.639254, 93.0% success, 6.67% other flight
failure, 0.33% timeout, zero collision, 0.41434 m mean closest range, and
0.39904 m minimum closest range. Earlier checkpoints show the stronger signal
suppressing completion (model 60 has 11.0% success and 88.33% timeout), while
later checkpoints trade timeout for unsafe approaches: model 120 has 73.0%
success and 26.67% flight failure. No checkpoint passes the exact range or
flight-failure gates.

Weights 300 and 3000 both regress the original 96.0% success, 4.0% failure
frontier, so there is no supported interval to refine. The failed boundary is
still residual wall-normal motion after the actor starts braking. Replace this
state-only excess-speed term with a predictive range barrier based on current
range, positive wall-normal speed, and an evidenced stopping horizon. Keep the
controller, current-range barrier, held-out distribution, and acceptance gates
unchanged.

### 25. Predict range through the observed braking horizon

Date: 2026-08-11.

Plain-language view: Reward the drone for leaving enough room to stop, not just
for being slow at its current location. At every training step, project its
forward motion 0.4 seconds ahead and warn only about the extra range risk that
motion creates.

Technical view: Replace the exhausted `range_closing_speed` term with
`range_prediction`. Project only positive actual wall-normal speed over a
0.4-second horizon, then compute how much this projected position increases the
existing normalized range-barrier cost relative to the current position. This
difference is exactly zero for retreat, zero wall-normal speed, and projected
positions outside the barrier. It therefore does not duplicate the current
position penalty or suppress a safe approach.

The 0.4-second coarse horizon comes directly from retained failures that travel
from about 0.420 m to below 0.400 m while decelerating. Use weight 10, equal to
the current-range barrier, so a 0.050 m/s approach at 0.420 m receives the same
penalty as already being at the 0.400 m hard boundary. Remove the failed
closing-speed field rather than preserving an unused training interface.
Increment the evaluator revision because reward diagnostics change. Keep the
controller, network, PPO schedule, 40% difficult-reset phase, held-out set, and
acceptance gates fixed for a new scratch run.

### 26. Predictive braking improves safety but has not converged

Date: 2026-08-11.

Plain-language view: The predictive warning helps: the final policy succeeds
slightly more than 95% of the time and reduces unsafe approaches. It begins
learning the useful tradeoff only at the end of the run, however, and stops
before eliminating the remaining range crossings. Give the same scratch
training procedure more low-rate consolidation updates; do not weaken the gate
or fine-tune its checkpoint.

Technical view: Run
`20260811T083710Z-scratch-predictive400ms-w10` started from clean source commit
`cc67ba8`, a new network, and a new optimizer. Training completed 152 updates
in 925.2 seconds. Evaluator revision 3 evaluated all 11 checkpoints fresh in
502.3 seconds.

Model 150 was selected with score -0.527009, 95.33% success, 3.67% other flight
failure, 1.0% timeout, zero collision, 0.41560 m mean closest range, and
0.39859 m minimum closest range. Independent evaluation reproduced every
aggregate metric and preserved traces under
`.build/docker/integration/eval-press.20260811T090143Z.409191`. The predictive
term is material at a mean -0.18620 versus -0.07405 for the current-range
barrier and -1.10805 for velocity tracking.

This is a small exact-frontier improvement over the 96.0% success and 4.0%
flight-failure baseline without predictive braking: flight failures decrease
by 8.3% relative while the success gate remains passed. More importantly, the
learning frontier is still rising at the terminal checkpoint. Model 140 has
21.33% success and 67.67% timeout, model 145 jumps to 88.67% success and 8.33%
timeout, and model 150 reaches 95.33% success and 1.0% timeout. The run ended
during rapid consolidation rather than after a plateau.

Worst traces confirm the same residual-action boundary: at the final crossing,
actual forward speed remains approximately 0.030-0.039 m/s while the actor
requests only a small negative normalized wall-normal action. The controller
responds in the correct direction; the policy has not yet learned enough
braking command. Preserve the predictive structure and extend the second phase
inside a new scratch invocation.

### 27. Extend the still-rising predictive consolidation frontier

Date: 2026-08-11.

Plain-language view: Keep training the same lesson longer inside one scratch
run. The useful behavior appeared only in the last ten updates, so the next run
must show whether it continues to improve or has merely shifted the narrow peak.

Technical view: Change only the second phase from 31 to 61 updates at the same
fixed `3e-5` learning rate, zero entropy, 40% difficult-reset fraction, and
five-update checkpoint interval. Keep phase 1 at 121 updates and preserve the
predictive horizon, reward weights, controller, actor/critic architecture, PPO
parameters, held-out seeds, and exact acceptance gates. The resulting 182
updates remain one new runner and optimizer initialized from scratch; this is
not checkpoint resume or post-run fine-tuning.

The unchanged prefix should reproduce content-identical checkpoints through
model 150. Let the content-addressed evaluation cache reuse only those exact
artifacts, then evaluate models 155-180 fresh. If success stabilizes while
range failures fall to zero, accept and run independent evaluation. If the
frontier regresses after model 150, reduce the predictive weight rather than
adding more updates, because the 0.4-second structure has already demonstrated
the intended safety effect.

### 28. Extended consolidation reaches and leaves its best frontier

Date: 2026-08-11.

Plain-language view: More updates improve the policy for ten steps, then the
result oscillates and finally collapses. The best policy is safer than every
earlier scratch run, but it still crosses the boundary in ten of 300 episodes.
Stop extending the schedule. The actor needs a direct negative braking target,
not more exposure to a state-only warning.

Technical view: Run
`20260811T090631Z-scratch-predictive-extended61` started from clean source
commit `b9a851f`, a new network, and a new optimizer. Training completed 182
updates in 1106.3 seconds. Checkpoints through model 150 were byte-identical to
the preceding scratch run; the content-addressed selector reused all 11 and
evaluated only six new checkpoints in 251.1 seconds.

Model 160 was selected with score -0.472188, 96.67% success, 3.33% other
flight failure, zero collision and timeout, 0.41557 m mean closest range, and
0.39860 m minimum closest range. Fresh independent evaluation reproduced the
rates with a 0.41547 m mean and 0.39822 m minimum; traces are preserved under
`.build/docker/integration/eval-press.20260811T093121Z.447015`. This reduces
the original four-failure-percent frontier by 16.7% relative and is the best
scratch result so far, but exact acceptance still rejects range undershoot and
flight failure.

The extension is exhausted: models 165, 170, and 175 alternate between 3.67%
and 4.0% failure, then model 180 collapses to 22.33% failure. In the independent
worst traces, positive wall-normal speed remains 0.014-0.039 m/s at the final
crossing while the actor requests only about -0.0014 to -0.0024 normalized
wall-normal action. The controller follows that small braking request; it is
not reversing or creating the command.

The predictive reward is already substantial at mean -0.22541, roughly 2.5
times the current-range barrier. Lowering its weight would mostly restore the
pre-predictive frontier without correcting the indirect credit assignment.
Instead, route the same 0.4-second projection through the existing dense
velocity target so a projected undershoot requests negative wall-normal
velocity directly. Remove the separate predictive reward term to keep the
task contract minimal, return to the shorter 121-plus-31 schedule, and retain
the controller, reset mixture, held-out distribution, and exact gates.

### 29. Make predictive braking the dense velocity reference

Date: 2026-08-11.

Plain-language view: Tell the policy how fast it should be moving after
accounting for stopping distance. If its current forward motion projects past
the neutral hover range, the target now asks for retreat instead of merely
adding another penalty to the same unsafe state.

Technical view: Add a required 0.4-second prediction horizon to the strict-v1
wall-normal velocity-reference contract. Project only positive contact-frame
wall-normal speed, use that projected range in the existing proportional and
clipped reference, and retain the alignment gate: unsafe projected motion can
request retreat even while tangentially misaligned, while positive approach
still waits for alignment. Evaluation uses the same contact-frame velocity and
reference, so its trajectory cost remains definitionally identical to
training.

Remove `range_prediction` and its two reward settings rather than retaining a
redundant or disabled compatibility path. The task returns ten named reward
terms again, and evaluator revision 4 prevents stale diagnostic-cache reuse.
Return to 121 full-reset updates plus 31 low-rate updates with 40% difficult
starts. Keep the aligned controller, network, remaining rewards, PPO values,
held-out seeds, and exact gates unchanged. Start a new actor, critic, runner,
and optimizer.

### 30. Direct prediction improves the frontier but remains underweighted

Date: 2026-08-11.

Plain-language view: Asking directly for retreat works better than adding a
separate warning, but the policy still ignores part of that request in the
hardest starts. Make the existing tracking lesson stronger before changing the
prediction horizon.

Technical view: Run
`20260811T093944Z-scratch-predictive-reference400ms` started from clean source
commit `d8b7833`, a new network, and a new optimizer. Training completed 152
updates in 929.8 seconds and evaluator revision 4 evaluated all 11 checkpoints
fresh in 463.3 seconds.

Model 135 was selected with score -0.471879, 97.0% success, 3.0% other flight
failure, zero collision and timeout, 0.41448 m mean closest range, and
0.39876 m minimum closest range. Independent evaluation reproduced every
aggregate metric and preserved its traces under
`.build/docker/integration/eval-press.20260811T100339Z.488702`. This removes
one of attempt 12's ten deterministic failures and is the strongest scratch
frontier so far. Models 130 and 135 share nine failures; models 140-150
regress, so more updates at the same objective are unsupported.

The trace provides a sharper boundary than the horizon alone. At one crossing,
the 0.4-second projection already requests approximately -0.033 m/s
wall-normal velocity, yet the actor emits a slightly positive normalized
wall-normal action and actual forward speed is 0.030 m/s. Another trace crosses
at 0.016 m/s while the actor remains positive. The predictive reference is
therefore active early enough, but the policy underweights its dense tracking
cost relative to progress and completion.

Keep the 0.4-second reference and coarse-sweep only
`velocity_tracking_weight` from 300 to 900. This mirrors the previously proven
threefold dense-target intervention while testing the new scratch contract.
Keep the 121-plus-31 schedule, reset mixture, controller, all other rewards,
held-out distribution, and exact gates fixed. If 900 overbrakes into timeout,
refine the bracket at 600; if it reduces failures without timeout, retain it
and refine only if exact acceptance still fails.

### 31. Threefold tracking weight overbrakes without closing the boundary

Date: 2026-08-11.

Plain-language view: Tripling the tracking lesson makes the early policy too
cautious, but the later policy still crosses the range boundary after it learns
to approach. The useful setting lies between the original and tripled weights.

Technical view: Run
`20260811T100846Z-scratch-predictive-reference-vtrack900` started from clean
source commit `2e845aa`, a new network, and a new optimizer. It kept the fixed
controller, 0.4-second predictive velocity reference, 121-plus-31 schedule,
40% difficult-reset fraction, held-out seeds, and exact gates. Training
completed 152 updates in 926.7 seconds; evaluator revision 4 evaluated all 11
checkpoints fresh in 497.7 seconds. A representative GPU sample measured 77%
utilization and 15.7/24.6 GB allocated, with environment stepping and PPO
synchronization still limiting useful throughput.

Model 145 was selected with score -1.447104, 67.67% success, 8.67% other
flight failure, 23.67% timeout, zero collision, 0.41220 m mean closest range,
and 0.39873 m minimum closest range. Model 90 demonstrates the overbraking side
of the tradeoff at 10.67% success, 3.67% failure, and 85.67% timeout. Later
updates exchange timeout for unsafe completion: model 150 collapses to 46.0%
success and 39.67% failure. The attempt is substantially worse than the
weight-300 frontier of 97.0% success and 3.0% failure, so weight 900 is
rejected rather than extended.

Preserve the predictive reference and all other settings, then test the
predeclared midpoint `velocity_tracking_weight: 600`. This is a targeted
bracket refinement between the high-success weight-300 result and the
overbraking weight-900 result. If 600 restores high success but leaves the
same three deterministic crossings, stop fitting this scalar and inspect the
action/reference boundary for a structural training target. If it materially
reduces failures, refine only the supported side of the bracket.

### 32. Midpoint confirms an indirect credit-assignment boundary

Date: 2026-08-11.

Plain-language view: The midpoint learns to finish, but it does not learn the
sign of the final braking command reliably. Stop making the indirect motion
penalty larger and teach the policy the controller command it should issue.

Technical view: Run
`20260811T103705Z-scratch-predictive-reference-vtrack600` started from clean
source commit `d67d695`, a new network, and a new optimizer. It kept the
accepted controller, 0.4-second predictive reference, 121-plus-31 schedule,
40% difficult-reset fraction, and exact held-out contract. Training completed
152 updates in 934.1 seconds and all 11 checkpoints were evaluated fresh in
437.4 seconds.

Model 135 was selected with score -0.518753, 95.67% success, 4.33% other
flight failure, zero collision and timeout, 0.41432 m mean closest range, and
0.39862 m minimum closest range. The midpoint is worse than the weight-300
frontier at 97.0% success and 3.0% failure. Models 125 and 130 both reach
95.0% success with 5.0% failure, and models 140-150 regress. The scalar bracket
is therefore exhausted rather than refined further.

Independent evaluation reproduced the selected result exactly and preserved
traces under `.build/docker/integration/eval-press.20260811T110113Z.562750`.
At a final crossing, range falls from 0.40050 m to 0.39954 m while forward
speed remains 0.0210 m/s and normalized wall-normal action is positive
0.00316. With the 2.0 m/s action scale, this requests positive 0.0063 m/s even
though the 0.4-second predictive reference requests approximately -0.029 m/s.
The same sign error appears across seeds. The state-velocity loss describes the
desired motion, but its gradient reaches the action only through the simulated
controller and plant.

Make one structural training change: retain the actual velocity-tracking term
at its strongest weight-300 frontier and add a physical command-tracking term
that compares the actor's three-axis velocity command directly with the same
predictive target. Carry the physical command in `PressApproachInput` so the
pure task remains independent of policy action scaling. Start the new term at
weight 100; at the evidenced crossing this contributes roughly 0.12 per step
and, unlike the state term, has a direct action gradient. Keep every other
reward, controller, schedule, reset, held-out metric, and gate unchanged.
