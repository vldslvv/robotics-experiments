# Press policy training decision log

Status: PRESS-HOVER OBJECTIVE SIMPLIFIED; METRIC CONTRACT REPAIRED; CLEAN
CAMPAIGN PENDING (2026-08-23).

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
| `20260811T111012Z-scratch-command-target-w100` | `20260810T-accepted-reproduction` | Same scratch schedule; velocity tracking 300 plus direct physical command tracking 100 | Iteration 150 selected; score -1.884269, 53.0% success, advisory rejected | Reduce the direct command weight by one coarse decade |
| `20260811T113819Z-scratch-command-target-w10` | `20260810T-accepted-reproduction` | Same scratch schedule; velocity tracking 300 plus direct physical command tracking 10 | Iteration 150 selected; score -0.636139, 93.3% success, advisory rejected | Restrict the direct target to the evidenced wall-normal boundary |
| `20260811T120750Z-scratch-wall-normal-command-w10` | `20260810T-accepted-reproduction` | Same scratch schedule; velocity tracking 300 plus wall-normal command tracking 10 | Iteration 140 selected; score -0.509376, 96.0% success, advisory rejected | Coarse-test wall-normal weight 30 before changing the target or curriculum |
| `20260811T123932Z-scratch-wall-normal-command-w30` | `20260810T-accepted-reproduction` | Same scratch schedule; velocity tracking 300 plus wall-normal command tracking 30 | Iteration 140 selected; score -0.796698, 89.3% success, advisory rejected | Stop the symmetric-weight sweep and supervise only evidenced retreat commands |
| `20260811T131001Z-scratch-one-sided-braking-w100` | `20260810T-accepted-reproduction` | Same scratch schedule; one-sided retreat-command supervision at weight 100 | Iteration 140 selected; score -0.491165, 96.3% success, advisory rejected | Reduce the oscillating one-sided gradient to weight 30 |
| `20260811T133733Z-scratch-one-sided-braking-w30` | `20260810T-accepted-reproduction` | Same scratch schedule; one-sided retreat-command supervision at weight 30 | Iteration 130 selected; score -0.480194, 96.7% success, advisory rejected | Stop the reward-weight sweep and introduce difficult starts during phase 1 |
| `20260811T140551Z-scratch-early-hard10-braking-w30` | `20260810T-accepted-reproduction` | 121 updates at `1e-4` with 10% difficult resets, then 31 at `3e-5` with 40%; one-sided braking weight 30 | Iteration 130 selected; score -0.632643, 93.3% success, advisory rejected | Restore the proven easy phase and introduce difficult resets through a short graduated transition |
| `20260811T143832Z-scratch-graduated-hard20-40-braking-w30` | `20260810T-accepted-reproduction` | 121 easy updates at `1e-4`, 11 at `3e-5` with 20% difficult resets, then 31 at `1e-5` with 40% | Iteration 160 selected; score -0.528887, 95.3% success, advisory rejected | Stop fitting the unstable schedule and narrow the task-owned velocity action scale |
| `20260811T150929Z-scratch-action-scale050-noise012` | `20260810T-accepted-reproduction` | Same graduated schedule; 0.5 m/s translation scale and 0.12 normalized exploration noise | Iteration 160 selected; score -1.715155, 59.3% success, advisory rejected | Preserve the late improvement and double the high-rate learning phase in one new scratch run |
| `20260811T154439Z-scratch-action050-main241` | `20260810T-accepted-reproduction` | 241 easy updates at `1e-4`, then the unchanged 20% and 40% refinement phases | Iteration 150 selected; cost 0.870737, 87.7% success, advisory rejected | Stop extending the fixed high-rate phase; merge full evaluation into training before the next attempt |
| `20260811T-merged-loop-smoke2` | `20260810T-accepted-reproduction` | One-update procedural smoke at the representative 32,768 environments | Iteration 1 selected from two fresh exact evaluations; cost 4.006851, intentionally rejected | Merged loop passes; verify stable cache identity across equivalent TorchScript exports |
| `20260811T-merged-loop-content-cache-replay` | `20260810T-accepted-reproduction` | Deterministic replay of the one-update smoke | Both evaluations reused by executable-content digest; identical cost and selection in 34.3 s | Procedure and cache optimization validated; next execution is a full scratch attempt |
| `20260812T191858Z-push-ready-scratch` | `20260810T-accepted-reproduction` | Strict 100-sample dwell; broad phase then 20% difficult-reset refinement | Stopped at phase-2 iteration 85; peak success 0.3%, and best cost 3.767904 came from 92.3% timeout behavior | Replace conflicting and sparse shaping before another scratch campaign |
| `20260812T203643Z-push-ready-readiness-smoke` | `20260810T-accepted-reproduction` | Three-seed, three-phase procedural smoke for dense readiness and settling resets | Passed in 954.2 s; distinct worker PIDs 57, 1231, and 2402; no reserved test workload | Start one monitored scratch attempt with dwell-based early stopping |
| `20260812T205331Z-push-ready-readiness` | `20260810T-accepted-reproduction` | Dense readiness with 30% tightly valid resets and normalized action noise 0.12 | Stopped at iteration 60; success and mean maximum dwell stayed exactly zero while cost plateaued near 5.94 | Lower exploration below the terminal speed scale and make phase 1 a predominantly centered settling curriculum |
| `20260812T211331Z-push-ready-lownoise-smoke` | `20260810T-accepted-reproduction` | Three-seed procedural smoke for 0.03 noise and the 80%/40%/15% settling schedule | Passed in 754.1 s with distinct processes and no reserved test workload | Start one full attempt and require early training completions |
| `20260812T212630Z-push-ready-lownoise` | `20260810T-accepted-reproduction` | Normalized action noise 0.03 with 80% tightly centered settling resets | Stopped after iteration-20 validation; zero training and held-out completion, zero held-out dwell | Add an explicit short-to-full training hold curriculum while keeping validation strict |

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

### 33. Direct command target is structurally valid but overweighted

Date: 2026-08-11.

Plain-language view: The new lesson keeps the actor cautious for too long. It
does not eliminate the final unsafe cases, and the policy learns to finish only
at the end of the run. Keep the direct lesson but make it much lighter.

Technical view: Run `20260811T111012Z-scratch-command-target-w100` started from
clean source commit `315df06`, a new network, and a new optimizer. The task
mapped normalized actions through the production Lab velocity-command seam,
rotated the resulting physical command into the contact frame, and compared it
with the predictive velocity reference at weight 100. Actual state-velocity
tracking returned to the strongest weight-300 frontier. The controller,
121-plus-31 schedule, reset mixture, held-out seeds, and gates stayed fixed.
Training completed 152 updates in 929.5 seconds and evaluator revision 5 scored
all 11 checkpoints fresh in 503.5 seconds.

Model 150 was selected with score -1.884269, 53.0% success, 3.33% other flight
failure, 43.67% timeout, zero collision, 0.42012 m mean closest range, and
0.39894 m minimum closest range. Success rises only from 20.0% at model 135 to
53.0% at model 150, while the failure floor remains comparable to the original
weight-300 state-only frontier. The selected mean command-tracking contribution
is -0.28232 per active step, so the new term is not merely diagnostic; at
weight 100 it materially dominates the completion tradeoff.

Do not extend this schedule: the first phase is already suppressed and the
final checkpoint remains far below the success gate. Preserve the structural
term and perform a coarse decade step to `command_tracking_weight: 10`. This
tests whether a direct sign-correction signal can coexist with the proven
high-success learning path. If weight 10 reproduces the state-only frontier
without reducing failures, refine upward at 30; if it improves the boundary,
refine the supported interval. Keep every other value unchanged.

### 34. Lighter three-axis command imitation still regresses the frontier

Date: 2026-08-11.

Plain-language view: A ten-times lighter command lesson lets the policy learn
to finish again, but it makes both completion and safety worse than the policy
without that lesson. The evidence only calls for correcting forward/backward
braking, not imitating all three command axes.

Technical view: Run `20260811T113819Z-scratch-command-target-w10` started from
clean source commit `ab517be`, a new network, and a new optimizer. It changed
only the physical command-tracking weight from 100 to 10. Training completed
152 updates in 930.3 seconds and evaluator revision 5 scored all 11 checkpoints
fresh in 499.5 seconds.

Model 150 was selected with score -0.636139, 93.33% success, 6.67% other
flight failure, zero collision and timeout, 0.41456 m mean closest range, and
0.39853 m minimum closest range. Model 120 briefly reaches 92.33% success and
7.33% failure before the hard-start phase oscillates. The selected mean
command-tracking contribution is only -0.04183 per active step, yet the run is
worse than the state-only 97.0% success and 3.0% failure frontier. Refining the
same three-axis scalar upward is contradicted; reducing it toward zero merely
returns to the already rejected state-only behavior.

The trace-supported first failed boundary is exclusively wall-normal range and
command sign. Tangential imitation was not evidenced, and it competes with the
policy's alignment strategy across most of each rollout. Narrow the structural
term to the contact-frame wall-normal command and rename its input, reward, and
Hydra setting so the strict-v1 contract states that scope explicitly. Retain
weight 10 for the first scoped test: at the crossing it contributes about 0.012
per step, while no longer constraining lateral or vertical command strategy.
Keep actual three-axis state tracking at 300 and preserve every other setting
and gate.

### 35. Wall-normal-only imitation preserves completion but remains too weak

Date: 2026-08-11.

Plain-language view: Teaching only the forward/backward command avoids the
large regression caused by imitating all three axes, but the lesson is still
too faint to make the actor brake before the exact range limit. Test one
stronger wall-normal weight before changing the target or curriculum.

Technical view: Run
`20260811T120750Z-scratch-wall-normal-command-w10` started from clean source
commit `2fd0706`, a new network, and a new optimizer. It retained the accepted
controller, predictive 0.4-second velocity reference, actual-state velocity
tracking weight 300, 121-plus-31 schedule, 40% difficult-reset fraction,
held-out seeds, and exact gates. The only learned-command target was the
physical contact-frame wall-normal velocity at weight 10. Training completed
152 updates in 933.5 seconds and evaluator revision 6 scored all 11 checkpoints
fresh in 455.1 seconds.

Model 140 was selected with score -0.509376, 96.0% success, 4.0% other flight
failure, zero collision and timeout, 0.41451 m mean closest range, and
0.39935 m minimum closest range. Models 125 through 145 form a stable plateau
from 95.0% to 96.0% success and 4.0% to 5.0% failure; model 150 regresses to
93.7% success and 6.3% failure. The selected mean wall-normal command reward is
only -0.002258 per active step. This is substantially better than three-axis
imitation at the same weight, but it does not beat the state-only frontier of
97.0% success and 3.0% failure, so it is not material progress.

Independent evaluation reproduced the selected metrics and retained the worst
traces under
`.build/docker/integration/eval-press.20260811T123222Z.678351`. At the seed-202
crossing, range moves from 0.40057 m to 0.39935 m with 0.0264 m/s residual
forward speed while normalized wall-normal action is positive 0.00165. The
0.4-second reference at that state requests approximately -0.032 m/s. The
seed-101 and seed-303 traces show the same sign error. The first failed boundary
therefore remains actor wall-normal command, not controller response,
collision, tangential alignment, or vertical control.

Preserve the scoped target and coarse-test
`wall_normal_command_tracking_weight: 30`. This is the geometric midpoint
between the underpowered weight-10 signal and the earlier weight-100 command
scale, without reintroducing tangential or vertical imitation. If 30 lowers
the exact failure count without losing the success gate, refine the supported
side. If it overbrakes or leaves the plateau unchanged, stop fitting this
weight and reshape the single-run hard-case curriculum or braking target from
the retained traces.

### 36. Stronger symmetric command tracking increases unsafe momentum

Date: 2026-08-11.

Plain-language view: A stronger forward/backward lesson makes the drone follow
the fast approach request more closely before it reaches the braking zone. It
then carries more momentum through the range limit. The policy only needs the
retreat half of this lesson, so remove the reward for imitating approach
commands.

Technical view: Run
`20260811T123932Z-scratch-wall-normal-command-w30` started from clean source
commit `0af5bb6`, a new network, and a new optimizer. It changed only the
wall-normal command-tracking weight from 10 to 30. Training completed 152
updates in 928.3 seconds and evaluator revision 6 scored all 11 checkpoints
fresh in 497.7 seconds.

Model 140 was selected with score -0.796698, 89.33% success, 10.67% other
flight failure, zero collision and timeout, 0.41408 m mean closest range, and
0.39888 m minimum closest range. Its median completion time increased to
6.125 seconds and its mean wall-normal command reward was -0.007995 per active
step. Model 120 begins at 86.0% success and 13.67% failure; models 125 through
145 remain between 84.67% and 89.33% success, and model 150 collapses to 70.67%
success with 27.0% failure. The stronger symmetric target is decisively worse
than both weight 10 and the state-only frontier, so this scalar sweep is
exhausted.

The first failed boundary remains the actor's wall-normal command, but the
symmetric squared target also rewards matching the positive approach reference
through most of a rollout. That changes momentum before the evidenced braking
boundary and explains why a larger weight can increase both completion time
and undershoot. Replace symmetric command tracking with a one-sided braking
term: it is active only when the predictive wall-normal reference is negative
and penalizes commanded velocity above that retreat reference. Keep the
physical command input and every controller, observation, state reward,
schedule, held-out distribution, and gate fixed. Start the scoped braking term
at weight 100 because it has no gradient during ordinary approach; if it
preserves the high-success path but remains weak, fit its weight. If it causes
timeouts, reduce it before changing the curriculum.

### 37. One-sided braking recovers the frontier but weight 100 oscillates

Date: 2026-08-11.

Plain-language view: Restricting the direct lesson to braking restores most of
the strong policy, but the lesson is still too strong: consecutive checkpoints
alternate between nearly useful and severe boundary crossing. Reduce its
weight before changing when difficult examples are introduced.

Technical view: Run
`20260811T131001Z-scratch-one-sided-braking-w100` started from clean source
commit `f276f54`, a new network, and a new optimizer. The reward term was active
only when the predictive reference requested retreat and penalized physical
wall-normal commands above that target. It retained the accepted controller,
actual-state velocity tracking weight 300, 121-plus-31 schedule, 40% difficult
resets, held-out distribution, and exact gates. Training completed 152 updates
in 932.8 seconds and evaluator revision 7 scored all 11 checkpoints fresh in
467.0 seconds.

Model 140 was selected with score -0.491165, 96.33% success, 3.67% other
flight failure, zero collision and timeout, 0.41465 m mean closest range, and
0.39871 m minimum closest range. Its mean one-sided braking contribution was
-0.003219 per active step. Model 130 has the same success/failure rates;
model 135 collapses to 54.0% success and 46.0% failure, model 145 recovers to
96.0% and 4.0%, and model 150 collapses again to 77.67% and 22.33%. More
updates at this weight are therefore unsupported. The result is directionally
better than symmetric weight 30 but does not beat the 97.0%/3.0% state-only
frontier and fails exact range safety.

Preserve the one-sided structure and reduce
`wall_normal_braking_command_weight` to 30. This keeps a direct gradient at the
evidenced sign error while reducing the phase-2 oscillation by more than three
times. If weight 30 preserves high success and lowers the failure count, refine
the supported bracket. If it merely returns to the existing plateau, stop
fitting the reward scalar and change the in-run hard-case learning schedule.

### 38. Lower braking weight nearly restores the frontier but not acceptance

Date: 2026-08-11.

Plain-language view: The lighter braking lesson avoids the worst oscillations
and gets close to the strongest policy again, but it does not remove the final
range crossings. The policy sees difficult boundary starts too abruptly and
too late; introduce a small fraction during its main learning phase.

Technical view: Run
`20260811T133733Z-scratch-one-sided-braking-w30` started from clean source
commit `278db1d`, a new network, and a new optimizer. It changed only the
one-sided braking-command weight from 100 to 30. Training completed 152 updates
in 934.7 seconds and evaluator revision 7 scored all 11 checkpoints fresh in
459.6 seconds.

Model 130 was selected with score -0.480194, 96.67% success, 3.33% other
flight failure, zero collision and timeout, 0.41476 m mean closest range, and
0.39814 m minimum closest range. Its mean braking-command contribution was
only -0.000788 per active step. Model 120 starts at 89.0% success and 11.0%
failure; model 125 falls to 77.33% and 22.67%; model 130 recovers sharply;
models 135 through 150 then range from 86.33% to 95.0% success. The lower
weight reduces but does not remove hard-phase instability. It neither beats
the 97.0%/3.0% frontier nor changes the first failed boundary, so further
reward-weight refinement is unsupported.

Keep one-sided weight 30 and introduce a 10% difficult-reset fraction during
the 121-update main phase, followed by the existing 40% hard phase. This keeps
90% of the proven easy curriculum while providing boundary-action gradients
before the abrupt update-121 transition. Preserve both learning rates,
iteration counts, controller, reward semantics, held-out distribution, and
exact gates. If early exposure improves the exact failure count without losing
success, refine the curriculum fraction; otherwise replace the abrupt two-step
schedule with a graduated intermediate phase.

### 39. Early difficult starts disrupt the main learning phase

Date: 2026-08-11.

Plain-language view: Mixing boundary starts into the entire main phase makes
the policy substantially less reliable. Keep the strong contact-free learning
phase intact and teach the boundary through a short transition before the full
hard phase.

Technical view: Run
`20260811T140551Z-scratch-early-hard10-braking-w30` started from clean source
commit `201ce62`, a new network, and a new optimizer. It changed only the main
phase difficult-reset fraction from zero to 10%, retaining the accepted
controller, predictive reference, one-sided braking weight 30, learning rates,
phase lengths, held-out seeds, and exact gates. Training completed 152 updates
in 967.9 seconds and evaluator revision 7 scored all 11 checkpoints fresh in
514.2 seconds.

Model 130 was selected with score -0.632643, 93.33% success, 6.67% other
flight failure, zero collision and timeout, 0.41475 m mean closest range, and
0.39855 m minimum closest range. The end of the main phase already has only
87.67% success and 12.33% failure at model 120. Models 125 through 150 then
oscillate between 80.0% and 93.33% success with 6.67% to 20.0% failure. This is
worse than both the 97.0%/3.0% strongest frontier and the immediately preceding
96.67%/3.33% run. Difficult starts throughout the high-rate phase are therefore
ruled out.

Restore zero difficult resets for the 121-update main phase. Add a short
intermediate phase with 20% difficult resets at `3e-5`, then enter the existing
40% mixture at a lower `1e-5` learning rate. This preserves the proven
contact-free representation, exposes boundary states without a single 0-to-40%
distribution jump, and reduces destructive updates after full hard exposure.
Keep the controller, rewards, observations, held-out distribution, gates, and
single scratch optimizer lineage unchanged. If this still oscillates, inspect
the intermediate and final checkpoint trajectory before changing either phase
length or fraction.

### 40. A graduated hard curriculum recovers but does not stabilize safety

Date: 2026-08-11.

Plain-language view: Introducing difficult starts gradually helps the policy
recover after each bad update, but it still alternates between safer and unsafe
commands. The policy's velocity action range is much wider than this task ever
needs, so the next run should give the learned output useful resolution around
the small braking commands.

Technical view: Run
`20260811T143832Z-scratch-graduated-hard20-40-braking-w30` started from clean
source commit `01e7182`, a new network, and a new optimizer. It restored the
proven 121-update easy phase, added 11 updates with 20% difficult resets at
`3e-5`, and used 31 updates with 40% difficult resets at `1e-5`. The accepted
controller, reward, observation, held-out distribution, and exact gates stayed
fixed. Training completed 163 updates in 1045.8 seconds at approximately
166,000 environment frames per second. A representative GPU sample was 75%
utilization, 15.9/24.6 GB memory, and 204 W; the simulator pipeline, not memory,
remained the measured throughput limit.

Content-addressed selection reused exact results for models 0 through 90 and
scored the nine curriculum-affected checkpoints fresh in 345.8 seconds. Model
160 was selected with score -0.528887, 95.33% success, 4.67% other flight
failure, zero collision and timeout, 0.41426 m mean closest range, and 0.39841 m
minimum closest range. Models 125, 135, 140, 145, 150, and 160 stay between
93.67% and 95.33% success, while models 130 and 155 collapse to 86.0% and
85.33%. The lower final learning rate therefore limits neither the destructive
oscillation nor the exact range failures. Extending or refining this schedule
is unsupported.

Independent evaluation reproduced the selected result and preserved its log
under `.build/docker/integration/eval-press.20260811T150243Z.885315`. At the
worst crossings the drone is still closing at about 0.022 to 0.040 m/s while
the deterministic actor commands only weak retreat, zero, or a small approach
velocity. Lateral, vertical, attitude, collision, and controller response
remain within their envelopes. The first failed boundary is still the actor's
wall-normal command.

The task reference never requests more than 0.10 m/s on any translation axis,
but the artifact currently maps a unit action to 2.0 m/s. Coarse-test a
task-owned maximum of 0.5 m/s and raise normalized exploration noise from 0.03
to 0.12 so its physical 0.06 m/s standard deviation is unchanged. This gives
the deterministic mean four times more normalized resolution around the
evidenced braking commands without reducing exploration or changing the
controller, reward, reset distribution, learning schedule, held-out suite, or
deployment contract. If it improves safety, refine the action-scale bracket;
if not, restore the scale and add an explicit policy-output safety projection
owned by the task interface rather than controller alignment.

### 41. Narrower actions need more high-rate scratch learning

Date: 2026-08-11.

Plain-language view: The smaller velocity range gives the actor finer control,
but the old training length is too short for it to learn decisive motion. Since
training is fast and the last checkpoints are still improving, give the same
scratch procedure substantially more high-rate learning before rejecting the
representation.

Technical view: Run `20260811T150929Z-scratch-action-scale050-noise012`
started from clean source commit `5ec3a27`, a new network, and a new optimizer.
It changed the translation action scale from 2.0 to 0.5 m/s and normalized
exploration noise from 0.03 to 0.12, preserving the initial 0.06 m/s physical
standard deviation. The accepted controller, reward, three-phase curriculum,
held-out distribution, and exact gates stayed fixed. Training completed 163
updates in 1058.2 seconds and all 13 checkpoints were evaluated fresh in 606.1
seconds because the action contract changed.

Model 160 was selected with score -1.715155, 59.33% success, 0.67% body
collision, 3.33% other flight failure, 36.67% timeout, 0.42033 m mean closest
range, and 0.39911 m minimum closest range. Model 120 first reaches 54.33%
success; the transition then falls to 45.0%, and models 130 through 150 stay
between 49.33% and 53.33%. Models 155 and 160 recover to 58.0% and 59.33%, so
the run is far below acceptance but still improving at its terminal boundary.
The new body-collision and timeout modes arise from insufficient learned motion
authority, not from controller response or a newly discovered simulator seam.

Preserve the 0.5 m/s scale, 0.12 exploration noise, and all other semantics.
Extend the zero-hard-case `1e-4` phase from 121 to 241 updates, then retain the
11-update 20% transition and 31-update 40% final phase unchanged. This remains
one new scratch network and optimizer; it is not checkpoint continuation or
fine-tuning. Content-addressed selection should reuse checkpoints 0 through 120
and evaluate the extended learning frontier fresh. If the longer phase restores
high success and improves safety, refine only the supported training-length
side. If it plateaus far below the old action contract, restore 2.0 m/s and
implement the evidenced task-owned braking projection.

### 42. Doubling the high-rate phase finds a better checkpoint, then collapses

Date: 2026-08-11.

Plain-language view: More high-rate learning recovered much of the performance
lost when the action range was narrowed, but the best behavior appeared early
and then disappeared. A fixed training endpoint therefore wastes computation
and cannot reliably represent the best policy produced during the run.

Technical view: Run `20260811T154439Z-scratch-action050-main241` started from a
new network and optimizer with the 0.5 m/s translation action scale, 0.12
normalized exploration noise, 241-update easy phase, and unchanged 11-update
20% plus 31-update 40% refinement phases. The held-out evaluator selected model
150 with raw task cost 0.870737 (the previous report's negated score was
-0.870737), 87.67% success, 1.0% body collision, 11.33% other flight failure,
zero timeout, and 0.398361 m minimum range. It remains rejected by success,
collision, flight-envelope, and range gates.

The later trajectory is materially worse: model 180 falls to 84.67% success,
model 210 times out in every episode, and the refinement checkpoints finish
near 26-27% success with roughly 19-20% body collision and 50-51% timeout. The
run disproves the assumption that a longer fixed phase or the terminal
checkpoint is sufficient. Preserve the learning configuration while replacing
the disconnected train-then-select workflow with complete held-out evaluation
inside the scratch loop. Use the resulting curve to advance phases formally and
retain the run-wide best exact actor bytes.

### 43. Merge training and exact held-out evaluation behind explicit contracts

Date: 2026-08-11.

Plain-language view: Training now checks the real held-out task at regular
intervals, records every result immediately, and moves on only after a declared
maximum or a formal plateau. Acceptance remains a final yes/no decision and
cannot distort the learning-progress rule.

Technical view: Replace the checkpoint-selector process with one scratch
orchestrator owning a single runner and optimizer across all configured phases.
Each phase declares `max_iterations`, `evaluation_interval`, optimizer values,
and reset mixture. A persistent spawned worker owns the exact three-seed by
100-episode evaluator. At each boundary the trainer exports `policy.pt`, hashes
those exact bytes, evaluates or reuses the content-addressed result, writes raw
metrics and phase state to `training_report.json`, and emits TensorBoard
scalars. The worker returns only `PolicyEvaluation`; `advance_plateau` owns
phase progress, `select_best_evaluation` owns run-wide selection, and
`evaluate_press_acceptance` applies the unchanged gates once to the selected
stored metrics. No second automatic evaluation stage remains.

The configured progress scalar is the raw weighted `held_out_task_cost`, lower
being better. It is not an absolute value, a negated quality score, or the PPO
reward, so changing the reward does not require changing the plateau threshold.
A material step is a cumulative 0.003 cost reduction from the phase reference.
The same rule and patience of 30 consecutive stale evaluations apply to every
phase; the broad phase evaluates every 10 updates and both refinement phases
every five. A retrospective replay over 70 preserved checkpoint curves showed
that patience 6 would have stopped before a known later best in six runs,
patience 8 in three, and patience 10 in none. The deliberately more relaxed
value 30 therefore avoids making an aggressive early-stopping claim while the
new procedure gathers higher-resolution curves.

Do not launch the next policy attempt until host checks and a GPU procedural
smoke prove checkpoint numbering, persistent-worker lifetime, exact-byte cache
identity, incremental reporting, final selection, and publication. The next
full attempt must start from scratch and is the first evidence for whether this
merged procedure can reach acceptance without post-run fine-tuning.

### 44. Representative smoke validates the merged procedure and stable deduplication

Date: 2026-08-11.

Plain-language view: The new procedure works at the real training scale. It
keeps the exact best actor it evaluated, and a deterministic replay now skips
all repeated held-out flights. The next run can focus on learning rather than
training/evaluation plumbing.

Technical view: The first smoke exposed that RSL-RL's public checkpoint save
writes model zero and then accesses an external-logger field that does not exist
until learning starts. The adapter now writes the same model, optimizer,
iteration, and info state directly, without invoking the logger. Run
`20260811T-merged-loop-smoke2` then completed at 32,768 training environments:
it evaluated model 0 fresh at cost 4.290002, trained and numbered exactly one
new update as model 1, evaluated model 1 fresh at cost 4.006851, selected model
1, applied acceptance once, and published the exact already-evaluated bytes.
The final policy SHA-256 matched the evaluated model-1 SHA-256. Both
`training_report.json` records and the phase terminal reason were present.
Peak sampled GPU memory with trainer and evaluator alive was approximately
18.5/24.6 GB, with 52% utilization at that sample; the short smoke is not a
throughput benchmark, but it proves the representative memory footprint fits.
The complete fresh container run took 134.1 seconds.

A direct replay initially saved evaluation time but still paid for the second
simulator. Making the evaluator lazy removed that startup on a full cache hit.
The next replay revealed that equivalent TorchScript files can differ only in
PyTorch's generated `.data/serialization_id`; raw whole-file SHA-256 is
therefore too strict for useful deduplication even when every executable archive
member and tensor is identical. Cache schema 2 now keys the sorted executable
archive content while excluding only that generated identifier. The raw
whole-file SHA-256 remains attached to each current `PolicyEvaluation` and
published artifact, so exact-byte provenance is unchanged.

Run `20260811T-merged-loop-content-cache-replay` then reused both model 0 and
model 1 evaluations, reproduced costs 4.290002 and 4.006851 and the model-1
selection exactly, never launched the evaluator simulator, and completed in
34.3 seconds. This is a 74% wall-time reduction from the corresponding fresh
run. The pre-smoke `latest` and `best` publication labels were restored after
each diagnostic. Host gates must be rerun after the final cache change; after
they pass, the merged-loop implementation is ready for the next full scratch
policy attempt.

### 45. Replace single-lineage selection with a validation-test campaign

Date: 2026-08-11.

Plain-language view: Model 150 was the best point in one run, but it failed four
safety/reliability gates and later training collapsed. It is canceled as a
deployable candidate. Training now keeps the best behavior within each attempt,
tries independent seeds when validation fails, and reserves a separate final
exam that can only approve or reject the frozen selected actor.

Technical view: The fixed-endpoint evidence in entry 42 is decisive: model 150
reached only 87.67% success with collision, flight-envelope, and range failures;
model 210 timed out in every episode; and the refinement endpoint fell to about
26-27% success. A lower learning rate is therefore a validation-selected branch
from the attempt-wide best complete learner checkpoint, not a continuation from
the latest weights. Actor, critic, optimizer, and iteration are restored;
training environments are intentionally reset before the new phase settings are
used.

One campaign now tries fresh PPO seeds 42, 43, and 44, stopping after the first
completed attempt with an eligible validation checkpoint. All checkpoints use
the same 300 validation episodes from seeds 101, 202, and 303. Gate-passing
validation candidates rank ahead of failures, then raw held-out task cost is
minimized with the earlier exact tie retained. This multi-attempt evidence is
reported in full rather than hiding failed seeds, following the motivation in
[empirical design guidance for deep RL](https://papers.nips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

The selected actor bytes are then frozen and evaluated freshly, without cache,
over 100 episodes each for seeds 404, 505, and 606. The resulting raw
`PolicyEvaluation` alone enters the five binding acceptance gates. Test failure
ends the command without another attempt or fine-tuning. Acceptance publishes a
byte-for-byte copy of the tested actor and advances labels only after digest
equality is checked. Report schema 2 preserves attempts, phase parents,
validation/test workloads, raw metrics, thresholds, individual gates, reasons,
and selected/tested/published digests. `make eval-press` is now a post-training
audit of that published digest and exact test workload, never a selection step.

The optimizer restoration follows the normal complete checkpoint pattern in
[PyTorch checkpoint guidance](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training).
No new policy result is claimed by this contract change; GPU smoke and the first
full campaign remain qualification evidence to be recorded separately.

### 46. Three-phase GPU smoke proves restoration and reserved-test isolation

Date: 2026-08-11.

Plain-language view: The new smoke caught a real reset bug at the first
refinement boundary, then proved the corrected campaign crosses both boundaries
without touching the final-exam seeds or changing deployment labels.

Technical view: Diagnostic runs `20260811T-campaign-contract-smoke` and
`20260811T-campaign-contract-smoke2` each trained model 1 and restored its full
learner checkpoint successfully, but failed while explicitly resetting the
32,768 training environments. Isaac Lab's live training buffers were inference
tensors after RSL-RL collection, so the direct reset attempted an in-place
update outside inference mode. The phase branch now performs the intentional
environment reset under `torch.inference_mode()` after checkpoint restore,
phase optimizer configuration, and reset-mixture configuration. A host test
locks down that order and context.

Run `20260811T-campaign-contract-smoke3` then completed one update in each of
the three configured phases in 230.9 seconds. Phase 2 restored model 1 at
iteration 1, applied `3e-5` and 20% difficult resets, and reset every
environment. Phase 3 restored model 2 at iteration 2, applied `1e-5` and 40%
difficult resets, and reset again. Report schema 2 records all three parent
checkpoints, exact parent digests, restored iterations, applied phase configs,
and reset confirmations under status `procedural_smoke_complete`.

The smoke evaluated only validation seeds 101, 202, and 303. It did not start a
test worker, its `test_evaluation` and `acceptance` fields remain null, and logs
contain no execution of seeds 404, 505, or 606. Existing `latest` and `best`
links were unchanged. This qualifies phase restoration and campaign reporting;
it is not policy acceptance and does not spend the reserved final workload.

### 47. Isolate attempts after the first campaign exposes simulator retention

Date: 2026-08-11.

Plain-language view: The first full campaign completed seed 42 but could not
create seed 43 reliably in the same simulator process. Explicit Python and CUDA
cleanup did not solve that lifecycle boundary. Each attempt now runs in its own
Isaac process, so process exit performs the isolation cleanup before the next
seed starts.

Technical view: Campaign `20260811T195121Z.1232485` completed seed 42 and
selected checkpoint 191 with held-out task cost 0.690247, 92% success, 1%
collision, and 7% other flight failure. It correctly failed validation and did
not run the reserved test workload. The transition to seed 43 then remained in
environment construction for more than 16 minutes. Diagnostic smoke
`20260811T-attempt-boundary-smoke2` reproduced the transition after attempt
teardown itself completed in 0.9 seconds; dropping runner closures, collecting
garbage, closing the environment, and emptying the CUDA cache therefore did not
make repeated DirectRLEnv construction reliable inside one Isaac process. Both
runs were stopped without publication, retry through test evidence, or label
changes.

The campaign coordinator now owns no simulator. It launches exactly one seed in
each worker, validates the worker's report identity and unique PID, aggregates
all completed attempts, and starts another process only when validation fails.
A validation-passing actor still enters the existing separate fresh-test worker;
test failure cannot start another attempt. Report schema 2 records
`attempt_execution: isolated_process` and each attempt's worker PID.

GPU smoke `20260811T-isolated-attempt-smoke` exercised all three configured
seeds and every phase boundary in 712.3 seconds. Seeds 42, 43, and 44 ran in
worker PIDs 53, 639, and 1222; their environment setups took 17.0, 16.4, and
16.2 seconds. All phase restores, optimizer/configuration changes, and explicit
environment resets completed. The aggregate report contains three attempts,
three phase records per attempt, null test/acceptance evidence, and no execution
of seeds 404, 505, or 606. `latest` and `best` remained unchanged.

TensorBoard retains one million scalar samples per tag and reloads every five
seconds so sparse validation points, including `Validation/action_delta_cost`,
remain available for hover inspection. TensorBoard 2.21 owns smoothing in one
browser-global local setting rather than a server flag; setting Time Series
Smoothing to zero applies to every namespace. The wrapper states that contract,
supports `RL_TENSORBOARD_PORT` for a run-scoped collision-free address, and
points to `training_report.json` for exact validation X/Y values.

### 48. Replace approach completion with a push-ready five-second dwell

Date: 2026-08-12.

Plain-language view: The manual production flight exposed a task-definition
problem rather than a reason to tune the old actor. The approach stage could
finish after only 0.25 seconds inside loose position, speed, and upright bounds,
then allow return navigation even though the pose was not ready for a later
switch push. The replacement objective asks the new actor to hold the actual
handoff pose for five continuous seconds before the routine can advance.

Technical view: Preserve the schema-1 34-value observation and four-value
velocity action, but use the relative quaternion already present in that
observation. Success now requires 100 consecutive 20 Hz samples at 0.40-0.42 m
range, at most 0.01 m combined lateral/vertical error, at most 1 degree full
quaternion-geodesic attitude error, at most 0.03 m/s linear speed, at most 0.05
rad/s angular speed, and no contact. Training and deployment call the same pure
goal predicate, and completion retains the zero terminal setpoint.

The reward keeps the range, predictive velocity, braking, upright,
action-change, terminal success, collision, flight-failure, and timeout terms.
Alignment weight increases from 50 to 200; squared full-attitude and
angular-speed penalties enter at weight 10 each; every strict-goal sample earns
one dwell reward. Episodes increase from 10 to 15 seconds. Held-out checkpoint
cost adds mean full-attitude and angular-velocity costs at weight 0.10 each.
Training schedule, PPO settings, action scaling, and independent seeds remain
unchanged so the next scratch campaign isolates this task and reward change.

Navigation evidence moves to schema 2. Ten new production waypoint-and-settle
runs must retain both relative position and a normalized relative quaternion.
The accepted rotation envelope is the maximum full attitude error plus 0.25
degree; the existing 0.005 m position margin and position envelope remain unless
the new samples require expansion. Every training and held-out reset samples
the complete measured position envelope plus one volume-uniform axis-angle
rotation ball, including difficult-position resets. Training-report schema 4,
evaluator revision 4, metric-registry revision 3, and comparison revision 3
prevent reuse of evaluations from the looser objective. No new campaign result
is claimed until measurement, host gates, procedural GPU smoke, validation, and
the reserved-seed final test complete.

The schema-2 measurement campaign
`20260812T181320Z.140147` completed ten fresh production routes successfully.
It measured range 0.464594-0.498991 m, maximum combined lateral/vertical error
0.033746 m, and maximum full relative-attitude error 0.0163389954 rad
(0.93616 degree). The existing 0.44-0.52 m range and 0.08 m alignment envelope
already covers the new positions with the 0.005 m margin. Adding exactly 0.25
degree to the measured attitude maximum sets the shared live-handoff and Lab
reset rotation envelope to 0.0207023186 rad (1.18616 degree). The accepted
schema-2 summary is
`benchmarks/press_navigation_handoff/waypoint_settle_20260812.json`.

### 49. Replace the zero-dwell timeout attractor with dense readiness

Date: 2026-08-12.

Plain-language view: The first five-second campaign learned mostly to wait
until timeout. Stop it early rather than spend two more seeds repeating a
reward conflict. Give the actor useful credit for becoming push-ready and show
it successful settling states early, without changing what counts as success.

Technical view: Campaign `20260812T191858Z-push-ready-scratch` completed the
180-update broad phase and 85 updates of the 20% difficult-reset refinement for
seed 42 before operator-directed early stop. Across 300-episode validations,
strict success was normally zero and peaked at one episode (0.3%). The best
reported cost, 3.767904 at iteration 215, had zero success, 7.7% other flight
failure, and 92.3% timeout. Later one-success checkpoints alternated between
37.7% and 86.7% flight failure. Zero collision was expected because crossing
the 0.400 m range boundary terminates before sustained contact. No reserved
test seed ran and no deployment label advanced.

The reward diagnosis is structural. At the configured 0.410 m goal center,
the former current-position barrier contributed -2.5 per sample while the
complete-goal dwell reward contributed +1. Over 100 samples that canceled the
+150 terminal success reward before motion, attitude, or action penalties.
Meanwhile the new raw-unit attitude, angular-speed, and alignment terms were
small at their strict thresholds, and the all-condition dwell reward supplied
no gradient until every predicate was already true.

Keep the 100-sample goal and every acceptance gate unchanged. Make the range
barrier zero for a stationary pose anywhere in 0.40-0.42 m and apply it only
when positive closing motion projected through the existing 0.4-second horizon
would cross below 0.400 m. Add one bounded readiness reward using range,
alignment, full attitude, linear speed, and angular speed normalized by their
strict tolerances. Add a training-only push-ready reset tranche strictly inside
all five limits at 30%, 20%, and 10% across the existing phases; the remaining
samples keep the full handoff and difficult-position mixtures, and held-out
resets remain full-envelope only.

Evaluation now reports mean maximum continuous dwell progress as a fraction of
100 and weights its missing fraction by 2.0 in held-out cost. Evaluator revision
5, metric-registry revision 4, comparison revision 4, and validation-cache v5
prevent reuse of the stopped campaign's evaluations. TensorBoard's primary
filter includes training return, signed total reward, dwell progress, success,
and held-out cost so learning direction is visible without knowing internal tag
ordering. Qualify this change with host gates and the three-seed procedural
smoke before starting another scratch campaign; stop that campaign early if
dwell progress and success remain flat after the settling curriculum has had a
representative learning window.

### 50. Make the settling curriculum learnable under stochastic actions

Date: 2026-08-12.

Plain-language view: Dense readiness improved ordinary reward and taught the
drone to avoid many flight failures, but it still learned to wait instead of
holding the switch-ready pose. Stop after the promised evidence window. Start
the replacement from easier, quieter hover states so complete five-second
dwell trajectories occur early, then taper those states as approach training
takes over.

Technical view: The three-seed procedural smoke
`20260812T203643Z-push-ready-readiness-smoke` passed every phase restore in
distinct worker processes and did not run reserved seeds. Full campaign
`20260812T205331Z-push-ready-readiness` was then stopped during seed 42 after
the iteration-60 validation. Iterations 0, 10, 20, 30, 40, 50, and 60 all had
zero strict success and exactly zero mean maximum dwell progress over 300 held-
out episodes. Cost reached 5.945210 at iteration 20, stayed near that value
through iteration 40, regressed to 6.212895 at iteration 50, and only narrowly
improved to 5.935788 at iteration 60. The actor was selecting a low-failure
timeout regime: iteration 60 ended 12.3% in other flight failure and 87.7% in
timeout.

Training reward and exploration isolate the remaining problem. Readiness rose
from about 0.10 to 0.33 per sample and the signed total reward improved, but
strict-goal samples remained near 0.01% and no training episode completed. The
learned action standard deviation was still 0.0888 after starting at 0.12. At
the configured 0.5 m/s translation scale, that initial normalized deviation is
about 0.06 m/s, twice the strict 0.03 m/s terminal speed limit. The 30% settling
tranche also began as far as half of each strict pose and speed tolerance, so
noise normally broke the predicate before a long dwell could supply terminal
credit.

Keep the goal, reward weights, PPO optimizer, held-out distribution, and all
acceptance gates unchanged. Reduce scratch action noise to 0.03. Tighten
settling resets around the 0.415 m reference to 0.412-0.418 m, 0.002 m
alignment, 0.2 degree attitude error, 0.005 m/s linear speed, and 0.01 rad/s
angular speed. Use settling fractions 80%, 40%, and 15% across the existing
three phases; difficult fractions remain 0%, 20%, and 40%, leaving the balance
for complete handoff-envelope resets. Re-run host gates and the procedural
smoke before a new scratch campaign. The next early decision must first see
training completions from the phase-1 settling tranche, followed by nonzero
held-out dwell progress; otherwise stop and redesign rather than merely extend
the schedule.

### 51. Stage the training hold without weakening validation

Date: 2026-08-12.

Plain-language view: Easier starting poses and quieter actions were still not
enough: the actor never saw a completed five-second training episode. Teach the
temporal behavior directly by requiring a short continuous hold first, then a
longer hold, and finally the complete five seconds. Every held-out checkpoint
still has to satisfy the full five seconds, so this changes learning difficulty
without weakening acceptance.

Technical view: The low-noise procedural smoke
`20260812T211331Z-push-ready-lownoise-smoke` passed all three seeds and phase
boundaries in 754.1 seconds. Campaign
`20260812T212630Z-push-ready-lownoise` was then stopped after the iteration-20
validation. Despite 80% tightly centered settling resets and normalized action
noise 0.03, cumulative training success stayed exactly zero over more than
423,000 completed episodes. Strict-sample reward occupied only 0.06-0.29% of
samples, decreased during the first updates, and held-out mean maximum dwell
remained zero at iterations 0, 10, and 20. Cost improved from 6.375080 to
6.026219 because flight failures fell, which is the same non-dwell attractor in
a less severe form.

Add a generic optional `success_hold_steps` phase field. The press training
environment alone applies it; evaluation and test environments retain the
immutable 100-step goal. Configure 10, 40, and 100 samples over phases of 60,
80, and 240 updates, preserving the 380-update campaign total and the existing
learning rates. Lower normalized action noise once more to 0.01, while keeping
the 80%/40%/15% settling mixture and 0%/20%/40% difficult mixture. TensorBoard
reports the active hold as `train/04_success_hold_steps` and labels episode
completion as `train/10_cumulative_training_success_rate`; the strict held-out
signals remain `val/01_approach_success_rate` and
`val/06_mean_max_stable_goal_fraction`. Configuration requires positive,
nondecreasing hold lengths and a final value equal to the strict goal.

The first temporal procedural smoke was stopped after seed 42 phase 1 because
even the 10-sample target had zero training completion. A one-update diagnostic
run, `20260812T215000Z-push-ready-predicate-diagnostic`, then measured the
individual training predicates: 67.9% range, 18.8% alignment, 96.3% attitude,
5.0% linear speed, 89.1% angular speed, and 0.007% complete-goal samples. The
maximum-streak signal remained far below one complete 10-sample hold. This
rules out attitude, angular speed, and exploration as the primary early
blockers and identifies controller settling velocity plus translational
alignment.

Extend the training-only phase contract with `success_tolerance_scale`. Keep
the unsafe 0.400 m lower range boundary unchanged, but scale the safe upper
range margin and alignment, attitude, linear-speed, and angular-speed limits by
10x, 3x, and 1x across the 10-, 40-, and 100-sample phases. Scales must be
finite, at least one, nonincreasing, and exactly one in the final phase.
TensorBoard reports each predicate's sample rate and mean/maximum goal-streak
fraction so another diagnostic can be stopped immediately if the targeted
linear-speed and alignment bottlenecks do not respond. Strict validation and
acceptance remain unchanged at every checkpoint.

One update of `20260812T215744Z-push-ready-tolerance-diagnostic` passed that
go/no-go check. The 10x/10-sample phase produced 69.5% cumulative training
success, 76.1% complete-goal occupancy, 81.9% range occupancy, 90.1% alignment
occupancy, and 100% attitude, linear-speed, and angular-speed occupancy. Mean
goal-streak fraction was 40.0%, compared with effectively zero before the
tolerance curriculum. Strict held-out success appropriately remained zero
after one update. This is sufficient evidence to qualify all phase transitions
with a complete procedural smoke before committing to another full campaign.

### 52. Contract the loose phase as soon as it saturates

Date: 2026-08-12.

Plain-language view: The loose first lesson is now easy, but practicing it
longer does not improve the actual five-second hover. Move to the tighter lesson
after 20 updates and spend the saved updates on the final strict task.

Technical view: The complete three-seed procedural smoke
`20260812T220133Z-push-ready-tolerance-smoke` passed every phase and checkpoint
restore without using reserved test seeds. Full campaign
`20260812T221555Z-push-ready-curriculum` was stopped during seed 42 after the
iteration-20 validation. By iteration 15 the 10x/10-sample phase had 90.7%
cumulative training completion, and it remained near 90% through iteration 23.
Strict validation nevertheless had zero success and exactly zero maximum dwell
progress at iterations 0, 10, and 20. Held-out cost only moved from 6.375080 to
6.286316; trajectory alignment cost stayed near 0.999 and velocity-tracking
cost stayed near 1.000. The improvement was dominated by other flight failure
falling from 59.0% to 49.3%, not by acquiring the strict goal.

End the 10x/10-sample phase at 20 rather than 60 updates. Keep the 3x/40-sample
phase at 80 updates and increase the final 1x/100-sample phase from 240 to 280,
preserving the total 380-update budget, learning rates, evaluation cadence, PPO
settings, reset mixtures, and all strict validation and acceptance gates. The
replacement campaign must show nonzero strict dwell during the intermediate
phase; otherwise stop and revise the curriculum again.

### 53. Remove the early-termination and timeout incentives

Date: 2026-08-12.

Plain-language view: The tighter lesson revealed that the drone earns a better
return by ending early, while model selection prefers waiting safely until
timeout. Reward becoming ready strongly enough to justify staying airborne and
make waiting to timeout no cheaper than another flight failure.

Technical view: Replacement campaign
`20260812T222721Z-push-ready-shortphase` was stopped after the iteration-40
validation in seed 42. Its first phase reproduced the earlier checkpoints and
restored iteration 20 into the 3x/40-sample phase. Fresh strict validations at
iterations 25, 30, 35, and 40 all had zero success and exactly zero maximum
dwell. Held-out cost fell from 6.229248 to 6.100258 only as other flight failure
fell from 43.0% to 30.0% and timeout rose from 57.0% to 70.0%. Strict alignment
and velocity costs remained approximately 0.999 and 1.000.

Training diagnostics show why. At iteration 40, the mean per-sample reward was
-13.717, including -7.999 alignment, -4.433 velocity tracking, and only +0.621
readiness. The cumulative mean episode lasted 48.18 samples and returned
-764.4, so an early terminal penalty of 300 could be cheaper than remaining
alive through the 300-sample horizon. The phase-2 alignment occupancy peaked at
5.6% before falling near 1-2%; complete-goal occupancy stayed near 1%, and no
environment held it longer than three of the required 40 samples. Validation
therefore selected the 25-point timeout instead of the 300-point flight
failure.

Keep every goal predicate, hold length, reset distribution, PPO setting, and
acceptance gate unchanged. Raise the training experiment's bounded readiness
weight from 5 to 100 so progress toward all five normalized predicates can
outweigh ordinary approach costs and supplies strong hold credit without an
unbounded error term. Raise its timeout weight from 25 to 300, equal to the
configured flight-failure weight, so model selection and PPO no longer benefit
from replacing attempts with timeouts. Equalize held-out timeout and flight-
failure costs at 3.0 as well. Evaluator revision 6, metric-registry revision 5,
comparison revision 5, and validation-cache v6 prevent the changed ranking
metric from reusing old evaluations. Qualify the reward and ranking change with
host gates and the three-seed procedural smoke before another scratch campaign.

### 54. Remove the unselectable 10x warm-up

Date: 2026-08-13.

Plain-language view: The easy lesson was learned, but the strict evaluator
correctly refused to carry it into the next phase. Start close enough to the
real task that short-hold learning can improve the strict checkpoint, then
lengthen the hold without tightening the pose at the same boundary.

Technical view: The revised incentive procedural smoke
`20260812T224803Z-push-ready-incentive-smoke` passed all three isolated seeds
and phase restores in 987.6 seconds without using reserved test seeds. Campaign
`20260813T002633Z-push-ready-incentive` was stopped after the seed-42 phase-2
iteration-20 validation. Phase 1 learned the 10x/10-sample task with about 91%
cumulative training completion and positive mean per-sample reward, but strict
held-out cost regressed from 6.785080 to 6.790523. Strict selection therefore
restored model zero at the phase boundary.

The 3x/40-sample phase consequently trained from scratch. At its updates 5, 10,
15, and 20, strict success and maximum dwell stayed exactly zero. The longest
training hold remained three samples, alignment occupancy stayed approximately
1-5%, and held-out cost worsened to 6.796284 while flight failure converted to
timeout. Stronger readiness fixed the negative loose-task return but could not
make an actor state cross a boundary where that actor was not selected.

Set phase 1 to 3x tolerance with its existing 10-sample hold; phase 2 remains
3x with a 40-sample hold. Keep the reset mixtures, learning rates, phase lengths,
final strict 1x/100-sample phase, PPO settings, validation workload, and every
acceptance gate unchanged. First run a one-seed, one-update diagnostic. Continue
to the required procedural smoke only if the 3x short-hold task produces useful
training occupancy without corrupting the strict held-out comparison.

### 55. Make readiness a cost that success can end

Date: 2026-08-13.

Plain-language view: A positive readiness payment accidentally made lingering
near the target profitable, even when the episode eventually failed. Make
readiness a cost that reaches zero at the ideal pose; successful completion now
ends future costs instead of ending future income.

Technical view: The one-update 3x diagnostic
`20260813T004211Z-push-ready-3x-diagnostic` improved strict held-out cost from
6.785080 to 6.783064 and produced 13.1% complete-goal occupancy, but no training
completion. A ten-update confirmation,
`20260813T004530Z-push-ready-3x-checkpoint-diagnostic`, rejected the curriculum:
training success stayed zero, 98.0% of completed episodes ended in ordinary
flight failure, complete-goal occupancy fell to 2.6%, and strict cost regressed
to 6.795912. Despite those failures, reported mean episode return remained
positive through update 10 because each near-ready sample earned positive
readiness before the terminal penalty.

Keep readiness normalized over all five predicates and bounded at the configured
weight, but change its signed value from `weight / (1 + error)` to
`-weight * error / (1 + error)`. It is exactly zero at ideal readiness and
approaches `-weight` for large error. Existing strict dwell and terminal success
rewards remain unchanged. Evaluator revision 7 and validation-cache v7 prevent
reuse of held-out reward diagnostics computed with the positive form; the
comparison and metric-registry revisions remain 5 because ranking metrics and
their ordering do not change. Re-run host gates and a bounded 3x diagnostic
before the procedural smoke.

### 56. Let training recover from the strict range boundary

Date: 2026-08-13.

Plain-language view: Correcting readiness removed the positive-reward exploit,
but the drone still lost most lessons at the first small overshoot. Let early
training retreat from that overshoot while keeping both the target pose and all
held-out acceptance checks unchanged.

Technical view: Ten updates of
`20260813T005031Z-push-ready-readiness-cost-diagnostic` kept training success at
zero and ended 97.9% of completed episodes in flight failure. The one-update
cause-isolation run
`20260813T005622Z-push-ready-failure-cause-diagnostic` then showed that every
early flight failure was range undershoot: cumulative flight-failure and
range-undershoot rates were both 60.7978%, while crash, altitude, and relative-
envelope failures were zero. Timeout covered the remaining 39.2022%.
Alignment occupied 37.497% of samples, linear speed 19.21%, and the complete
3x goal 13.14%, so immediate termination below 0.400 m prevented the retreat
command and predictive barrier from teaching recovery.

Add a training-only `range_undershoot_min_m` phase field and configure 0.38,
0.39, and 0.40 m across the three phases. The training success range retains
its 0.40 m lower bound in every phase; the existing tolerance curriculum still
widens only its upper margin. Production termination, validation, final test,
and the final training phase retain the 0.400 m boundary. Configuration
requires the limits to be positive, no greater than the strict goal minimum,
nondecreasing, and exactly strict in the final phase. TensorBoard reports the
active boundary as `train/07_range_undershoot_min_m` and splits cumulative crash, altitude,
range-undershoot, and relative-envelope failure causes. Before another
procedural smoke or campaign, run one bounded diagnostic and require a material
drop from the 60.7978% undershoot baseline without replacement by excessive
collision or another immediate failure mode.

### 57. Restart from a minimal pose-and-hold objective

Date: 2026-08-13.

The preceding reward and curriculum series accumulated overlapping incentives
without establishing strict held-out dwell. Replace it with one normalized pose
error over range-center, combined lateral/vertical, and quaternion-geodesic
attitude errors. PPO receives only ten times pose-error progress, one point per
complete strict-goal sample, and a +150/-300 terminal outcome. Speed and contact
remain strict predicates rather than shaping terms.

Remove settling resets, difficult-reset mixing, shortened holds, and tolerance
scaling. Every phase uses the full measured handoff envelope and the strict
100-sample goal. Restore the 200/100/100 fixed schedule at `1e-4`, `3e-5`, and
`1e-5`, with 0.38/0.39/0.40 m recovery boundaries and normalized exploration
noise 0.03. Rename TensorBoard metrics by semantic group and restart all
comparison, evaluator, report, registry, and cache revisions before collecting
new evidence.

### 58. Add only the evidenced predictive undershoot barrier

Date: 2026-08-13.

Campaign `20260813T043749Z.32951` was the first scratch run using the minimal
pose-and-hold objective. Strict held-out success and dwell remained zero. Mean
held-out normalized pose error improved from 11.324 at model zero to 6.013 at
update 40, so the update-40 no-progress stop did not apply. At that checkpoint,
38.67% of validation episodes failed and every one was a range undershoot; the
remaining 61.33% timed out. By update 50, all validation episodes ended in a
flight failure, attitude pass rate fell from 91.1% at model zero to 21.1%, and
mean normalized pose error regressed to 7.355. The campaign was stopped for
material flight-instability growth.

Add no speed, action-history, readiness, reset-mixture, or tolerance factor.
Add only a predictive range barrier: project positive wall-normal closing speed
over the existing 0.4-second velocity-reference horizon, measure projected
undershoot below the strict 0.40 m minimum, normalize it by the 0.02 m strict
range width, square it, and apply weight -10. Report it as
`train/reward/predictive_barrier`. Start a new scratch campaign with the same
200/100/100 schedule, learning rates, full handoff resets, and recovery
boundaries. Report schema 6, evaluator revision 9, metric-registry revision 7,
comparison revision 7, and validation-cache schema 8 isolate the new reward
diagnostics and metric contract.

### 59. Strengthen the same barrier after undershoot remains dominant

Date: 2026-08-13.

Scratch campaign `20260813T050015Z.53940` kept the initial schedule and added
only the weight-10 predictive barrier. Mean held-out normalized pose error
improved from 11.324 at model zero to a best 6.245 at update 50, but strict
success and dwell remained zero. At update 50, 33.0% of episodes failed and
every failure was range undershoot. At update 80, normalized pose error had
regressed for three validations to 6.990 and flight failure rose to 42.67%,
including 28.33% range undershoot and 14.33% other flight-envelope failures.
The campaign stopped immediately for that material new instability mode.

Range undershoot therefore remains the dominant evidenced failure and the
weight-10 barrier is insufficient. Change only its weight from 10 to 50, making
a 0.01 m projected deficit cost 12.5 points instead of 2.5 points. Keep the
barrier definition, all other reward terms, full handoff reset, 200/100/100
schedule, learning rates, and recovery boundaries unchanged. Start from a new
network and optimizer. Report schema 7, evaluator revision 10, comparison
revision 8, and validation-cache schema 9 prevent the changed barrier reward
diagnostics from reusing weight-10 evaluations.

### 60. End the unstable high-rate phase at its evidenced frontier

Date: 2026-08-13.

The fresh weight-50 isolation campaign `20260813T052618Z.75257` was stopped at
update 10. It reduced range undershoot from the weight-10 run's 26.67% to
24.67%, but full-attitude pass rate collapsed from 93.53% to 53.48%, mean
normalized pose error improved only from 11.324 to 11.013, and 4.33% of
episodes acquired a non-undershoot flight failure. Reject weight 50 and restore
the weight-10 barrier.

The two weight-10 campaigns independently located the useful `1e-4` frontier
at approximately update 50: the minimal run became completely unstable there,
while the barrier run reached its best normalized pose error of 6.245 there and
then regressed for three validations before acquiring a new 14.33% non-range
flight-failure mode at update 80. This is consistent within-phase saturation.
Change only phase 1 from 200 to 50 updates. Retain its `1e-4` learning rate and
the unchanged 100-update `3e-5` and `1e-5` refinement phases, strict goal, full
handoff resets, and 0.38/0.39/0.40 m recovery boundaries. The next campaign is
again scratch; no checkpoint from either stopped run is resumed. Returning the
barrier to weight 10 also advances report schema 8, evaluator revision 11,
comparison revision 9, and validation-cache schema 10 so the weight-50 model-zero
reward record cannot be reused.

### 61. Select demonstrated pose progress instead of timeout tradeoffs

Date: 2026-08-13.

Scratch campaign `20260813T053650Z.93412` reproduced phase-1 pose learning and
reached mean normalized pose error 6.245 at update 50, versus 11.324 at model
zero. Strict success and dwell were still zero, while aggregate task cost never
beat model zero because changing flight failures into timeouts dominated its
ranking. The phase boundary therefore restored model zero. Five consecutive
phase-2 validations through phase update 25 did not beat the run-wide pose best;
the campaign stopped under the explicit monitoring rule.

Change only the configured plateau and selection scalar from
`held_out_task_cost` to `mean_normalized_pose_error`, keeping direction
`minimize`, minimum delta 0.003, and patience 10. Aggregate cost and every cost
component remain reported diagnostics, and acceptance still reads only the
fresh reserved-seed gates. This makes phase restore preserve demonstrated pose
learning instead of preferring a different mixture of unsuccessful outcomes.
Comparison revision 10 separates the new selection contract. Start the next
campaign from scratch.

### 62. Increase the barrier only within the stable interval

Date: 2026-08-13.

Campaign `20260813T060231Z.109766` used pose-error selection and correctly
restored phase-1 model 50 into phase 2. Phase-2 update 30 reached a new best
mean normalized pose error of 5.903 and the first strict held-out dwell, with
maximum dwell fraction 0.14, but success remained zero. All 52.67% flight
failures at that checkpoint were range undershoots. Five following scheduled
validations produced no new best success, dwell, or pose error, so the campaign
stopped at phase update 55.

The evidenced factor remains the predictive range barrier. Weight 10 permits
useful pose and dwell learning but does not prevent dominant undershoot; weight
50 caused immediate attitude instability. Change only its weight to 20 and
retain the pose-selection contract, 50/100/100 schedule, learning rates, strict
goal, full handoff resets, and recovery boundaries. Report schema 9, evaluator
revision 12, comparison revision 11, and cache schema 11 prevent stale reward
diagnostics. Start a new scratch campaign.

### 63. Advance phase 2 at its repeated update-30 frontier

Date: 2026-08-13.

Weight-20 campaign `20260813T063800Z.127791` reached its best phase-2 pose
error, 5.910, at phase update 30 and its first 0.14 maximum dwell at update 45.
Five validations after that dwell did not improve success, dwell, or pose. It
therefore matched rather than improved the weight-10 campaign, which reached
pose error 5.903 and 0.14 dwell at phase update 30. Restore weight 10.

At the best weight-10 checkpoint, strict alignment pass rate was only 0.96%,
while linear-speed pass was 25.36%, angular-speed pass was 96.98%, and contact
clear was 100%. Pose does not yet pass, so speed or action-history shaping is
not evidenced. Both barrier runs consistently saturated phase 2 at update 30.
Change only phase 2 from 100 to 30 updates, preserving its `3e-5` rate and the
three-phase structure, so phase 3 restores the demonstrated pose/dwell best and
tests `1e-5` refinement. Report schema 10, evaluator revision 13, comparison
revision 12, and cache schema 12 isolate the restored weight-10 diagnostics.
Start a new scratch campaign.

### 64. Stop after the strict phase also saturates without success

Date: 2026-08-13.

Campaign `20260813T071443Z.143071` used the final evidence-backed configuration:
weight-10 predictive barrier, pose-error selection, and 50/30/100 updates at
`1e-4`, `3e-5`, and `1e-5`. Phase 3 correctly restored phase-2 model 80. Its
update-10 validation improved mean normalized pose error from 5.903 to 5.826 and
retained 0.04 maximum dwell, but success remained zero. The best dwell remained
phase-2 model 80 at 0.14. Five subsequent strict-phase validations through
phase update 35 produced no new best success, dwell, or pose error, so the
campaign stopped under the monitoring rule.

At the pose-best checkpoint, all traces were finite and collision remained
zero, but success was 0%, range undershoot was 59%, minimum closest range was
0.3970 m, strict alignment pass was 0.92%, linear-speed pass was 26.80%, and
angular-speed pass was 96.87%. The task remains pose-blocked, not speed-only or
action-oscillation-blocked. No further reward factor or difficult-reset mixture
is supported by this campaign. Preserve this configuration and evidence; do
not run artifact preflight, reserved-seed acceptance, policy transfer, or
routine integration because no candidate passed validation.

### 65. Increase initial exploration before changing the objective

Date: 2026-08-13.

Campaign `20260813T071443Z.143071` reduced held-out normalized pose error but
converged to a pose-blocked policy with zero success. The learned normalized
action standard deviation also fell from 0.03 toward 0.025 during that run, so
the existing stochastic policy did not explore substantially beyond its narrow
initial neighborhood.

Change only the training experiment's initial normalized action standard
deviation from 0.03 to 0.05. At the retained 0.5 m/s translation scale this is
0.025 m/s per component, still far below the previously rejected 0.12 setting.
Keep entropy zero so PPO may reduce the learned variance when deterministic
task return favors a quiet hold. Preserve the RMS pose objective, reward,
50/30/100 schedule, learning rates, full handoff reset, recovery boundaries,
selection rule, held-out workloads, and acceptance gates. Start a fresh
campaign and revert to 0.03 if the higher initial variance produces no pose or
dwell improvement, prevents variance contraction, or materially increases
collision or flight instability.

### 66. Optimize the worst strict pose component

Date: 2026-08-13.

Exploration campaign `20260813T163312Z-explore005` isolated initial normalized
noise 0.05. It improved held-out pose error much faster than noise 0.03,
reaching 6.299 at update 30 instead of 7.004, but produced no strict dwell. At
update 40, pose regressed to 6.661, full-attitude pass collapsed to 46.06%, and
22.67% of episodes acquired non-range flight failures. The campaign stopped
immediately for material new instability. Reject noise 0.05 and restore 0.03.

The best exploratory checkpoint still passed alignment on only 0.74% of
samples while attitude passed 92.12% and angular speed passed 97.24%. Replace
only the RMS aggregation of the three normalized pose components with their
maximum. Pose progress then follows the currently worst range, alignment, or
attitude component and cannot improve by trading one strict predicate against
another. Preserve every tolerance, reward weight, barrier, reset, PPO value,
phase, workload, and acceptance gate. Advance report schema 11, evaluator
revision 14, metric-registry revision 8, comparison revision 13, and cache
schema 13 because the selection metric and reward diagnostics changed. Start a
fresh campaign from noise 0.03.

### 67. Split approach and stabilization reward stages

Date: 2026-08-13.

Worst-component campaign `20260813T164901Z-maxpose` improved its held-out pose
error from 18.862 at model zero to 9.719 at total update 60 and produced 0.14
maximum strict dwell. It then remained at zero success through update 70. The
matching training metrics showed the same oscillation: complete-goal occupancy
was at most 0.0032%, maximum dwell was 0.055, and pose progress alternated sign
as alignment and attitude traded places. This rules out validation mismatch and
shows that one simultaneous pose objective asks the policy to approach and
settle before it has reliably acquired the position.

Represent the current task as two episode-level stages derived from the current
sample. Outside the strict range and lateral/vertical envelope, reward only the
decrease in worst normalized position error. Inside that envelope, reward
bounded stabilization quality over position, full attitude, linear speed, and
angular speed. Leaving the envelope returns to approach, so no hidden latch is
added to the observation or policy schema. Preserve the evidenced predictive
barrier, strict hold, terminal outcomes, noise 0.03, schedule, reset, PPO,
workloads, and acceptance gates.

Campaign `20260813T171302Z-proximity1` was abandoned after stale interrupted
Isaac containers exhausted GPU memory before its first PPO update; it supplies
no reward evidence. The stale containers were stopped, and its speculative
all-episode proximity reward is replaced rather than carried forward. Advance
report schema 13, evaluator revision 16, metric-registry revision 10,
comparison revision 15, and cache schema 15. Start a fresh campaign.

### 68. Smooth every staged reward component

Date: 2026-08-13.

Campaign `20260813T172426Z-staged` was stopped after update 10. Its validation
pose error improved from 18.862 to 16.691, but remained slightly worse than the
16.267 reached by the preceding worst-component campaign at the same update.
More importantly, its stabilization reward reused a maximum over pose and
motion errors even though campaign 67 had already shown oscillation when the
identity of the maximum component changed. Continuing would not isolate the
intended two-stage design from this known gradient-switching problem.

Keep the two directly derived stages, but use smooth aggregation. Approach
progress follows the RMS of normalized range and alignment error.
Stabilization is the bounded inverse RMS of normalized range, alignment, full
attitude, linear speed, and angular speed. Every component therefore supplies a
gradient simultaneously; strict hold, terminal success, and the strict stage
predicate remain binding so RMS aggregation cannot qualify a policy that fails
one component. Preserve every other reward factor, weight, schedule, reset,
PPO setting, workload, and acceptance gate. Advance report schema 14,
evaluator revision 17, metric-registry revision 11, comparison revision 16,
and cache schema 16. Start a fresh campaign.

### 69. Make stabilization reachable before the final strict goal

Date: 2026-08-13.

Smooth staged campaign `20260813T173430Z-staged-rms` reduced validation pose
error from 18.862 to 10.000 by update 30, outperforming the comparable
worst-component campaign's 12.322. Alignment pass reached 0.90%, range pass
reached 14.17%, failures remained exclusively range undershoots, and collisions
remained zero. Despite that useful approach learning, stabilization reward was
exactly zero at updates 10, 20, and 30 across all scheduled validation traces.
Requiring the final strict range and 0.01 m alignment conjunction to enter stage
2 makes the next learning signal unreachable.

Change only the stage-transition threshold. Enter stabilization within one
fixed 0.03 m Euclidean capture radius around the target position; leaving that
radius returns to approach. The radius matches the existing wall-normal
alignment capture scale and is fixed across all training phases. It does not
relax the strict 0.40-0.42 m range, 0.01 m alignment, pose, motion, 100-sample
success, or acceptance predicates. Preserve both reward formulas, every reward
weight, schedule, reset, PPO setting, workload, and acceptance gate. Advance
report schema 15, evaluator revision 18, metric-registry revision 12,
comparison revision 17, and cache schema 17. Start a fresh campaign.

### 70. Train hover directly against the measured pose envelope

Date: 2026-08-23.

Campaign `20260822T100450Z` used the newly qualified controller and exact
flight-parameters run but retained the staged approach reward during the hover
stage. All three scratch seeds selected zero-success checkpoints. Seeds 42, 43,
and 44 had other-flight-failure rates 0.2667, 0.2733, and 0.2067, and minimum
closest ranges 0.39782, 0.39774, and 0.39822 m. The predictive undershoot term
dominated the hover reward even though hover starts from the pinned measured
envelope and does not need to learn an approach. No actor passed validation and
no test workload or publication followed.

Keep the 5/15/30-second training holds and strict 30-second acceptance hold.
For hover only, remove position progress and the predictive barrier and use one
bounded measured-pose cost, `-utilization / (1 + utilization)`, plus strict hold
and terminal outcome. The first two hover phases recover at 0.38 and 0.39 m;
the final phase and acceptance retain 0.40 m. Approach behavior remains
unchanged. Remove the unused stabilization-capture configuration field. Advance
report schema 17, evaluator revision 20, metric-registry revision 14,
comparison revision 19, and validation-cache schema 19 so no prior reward
diagnostic or evaluation can be reused.

Deployment completion now evaluates the full flight-parameters envelope
embedded in the policy artifact for the configured 600 continuous samples,
matching training and held-out evaluation. The stage-neutral three-range
transfer routine derives the artifact label from the active curriculum stage;
the hover and approach top-level configs select their stage explicitly and give
each policy 45 seconds to complete.

### 71. Keep unavailable hover statistics explicit in every evaluation

Date: 2026-08-23.

Campaign `20260823T055548Z.36337` stopped during seed 42 at learner iteration
35 because the first actor to exceed the six-second statistics warmup appended
40 flight-parameter metrics that earlier zero-dwell evaluations had omitted.
The strict TensorBoard registry contained only the 31 always-available metrics,
so it rejected the expanded tuple. This is a reporting-contract defect, not a
policy failure. The preserved checkpoint had improved mean normalized pose
error from 5.232 at model zero to 2.960, maximum dwell fraction from 0.004 to
0.24, zero collision, and 34% other flight failures. The attempt never reached
validation selection or test.

Make the evaluator return one fixed 71-metric tuple on every checkpoint. The 40
flight-parameter maximum, RMSE, p95, and p99 fields are JSON null until a policy
produces the required post-warmup samples, then become finite values in the same
positions. Register all 40 fields for validation and test TensorBoard output.
Advance report schema 18, evaluator revision 21, metric-registry revision 15,
comparison revision 20, and validation-cache schema 20 so the incomplete cached
tuples cannot be reused. Preserve the objective, training schedule, seeds,
thresholds, and scratch-only campaign contract for the clean rerun.

### 72. Reward exact pose recovery during hover

Date: 2026-08-23.

Campaign `20260823T063936Z.51981` completed all three independent scratch seeds
after the fixed 71-metric reporting contract. Seeds 42, 43, and 44 selected
zero-success actors with normalized pose errors 2.9607, 3.2826, and 3.1158.
Their mean maximum dwell fractions remained about 0.004 to 0.005, and none ran
the held-out test workload. Seed 44 exhausted all 180 updates. The actors
quickly learned collision-free survival, while intermittent pose improvements
reset the plateau without producing sustained occupancy of the exact hover
envelope.

Pinned-flight-parameter hover resets already start around the exact target, so
the missing signal is retention and recovery after the neutral actor leaves the
envelope. For hover only, use the existing `position_progress` contribution for
the per-step reduction in exact measured pose utilization. Keep the bounded
pose cost, strict hold, terminal outcome, zero predictive barrier, reset
distribution, curriculum, PPO settings, workloads, and every acceptance gate
unchanged. Approach retains its target-position progress definition. Advance
evaluator revision 22, comparison revision 21, and validation-cache schema 21;
the report schema and fixed metric registry do not change.

### 73. Align hover reward, resets, and metrics with continuous hold progress

Date: 2026-08-26.

Campaign `press-hover-20260825T2027Z` was interrupted after 50 phase-one updates
and two refinement updates. Its best validation checkpoint was update 20: zero
success, 5.87% measured-envelope occupancy, 0.715 seconds mean episode-longest
hover, 3.4 seconds maximum episode-longest hover, and 3.344 mean measured-pose
envelope utilization. Updates 30 through 50 reduced the mean longest hover to
0.14-0.26 seconds. The apparent 0.105 training completion spike after the
refinement reset was transition occupancy during one rollout, not episode
completion. The critic value loss ranged from about 400 to 5200 while the PPO
surrogate remained near zero.

The full-envelope reset sampler placed 56% of starts at or above 0.9 envelope
utilization, so the first phase mostly learned edge recovery. Expand training
reset errors through 0.25, 0.50, and 1.00 of the pinned envelope while keeping
all validation and test resets at 1.00. End phase one at update 20, where the
previous run peaked. Reward only increases in the episode's longest continuous
hold; a shorter repeated hold cannot collect the same reward again. Uniformly
scale all PPO reward weights by 0.01 to reduce critic targets without changing
their relative weights or normalized policy advantages.

Replace completion-like occupancy names and phase-dependent dwell fractions
with measured-envelope occupancy, utilization, and continuous-hover seconds.
Keep task-target occupancy separate from measured flight-parameter acceptance.
Name PPO losses by their optimizer role and record actor and critic relative
parameter changes after every update. Advance report schema 21, metric-registry
revision 17, and comparison revision 23. Older metric and report shapes are
unsupported.

### 74. Stop after full-envelope hover refinement erases progress

Date: 2026-08-26.

Campaign `20260826T170226Z.47212` validated the scaled reward, progressive
resets, continuous-hover metrics, and actor/critic movement in the Isaac/RSL
runtime. Seed 42 improved from 0.108 seconds mean episode-longest hover at the
neutral actor to 1.109 seconds at update 30, with 8.92% measured-envelope
occupancy and zero collisions. The selected actor still had zero success, 100%
other flight failures, 34% range undershoots, and 3.136 mean measured-pose
envelope utilization. Its maximum episode-longest hover was 2.6 seconds.

The final phase restored that exact update-30 checkpoint before changing the
training reset scale from 0.50 to 1.00. Mean episode-longest hover then fell to
0.814, 0.367, and 0.520 seconds at its three validation checkpoints. Each had
zero success and 100% other flight failures, so the phase plateaued after 15
updates. Stop the campaign before a second scratch attempt: the first attempt
proved that learning occurs, but also showed that the final reset expansion
reliably erases the selected behavior. Preserve the interrupted run and its
selected checkpoint as diagnostic evidence. Do not start another campaign
until the transition from half-envelope to full-envelope resets is revised.

### 75. Preserve the measured-reset learning peak with smaller refinement steps

Date: 2026-09-02.

Campaign `press-hover-trace-resets-20260902` replaced synthetic reset scaling
with 14,400 paired position and attitude samples from the pinned flight-parameter
policy windows. Seed 42 improved from 0.119 seconds mean episode-longest hover
at model zero to 2.400 seconds at update 10, with a 6.550-second maximum, 7.16%
measured-envelope occupancy, zero collisions, and 90% timeouts. Update 20 then
collapsed to 0.122 seconds mean hover and 100% other flight failures. Refinement
from the selected update-10 learner at `3e-5` reached 1.938 seconds after five
updates and collapsed to 0.131 seconds after ten. The run was interrupted before
another seed because repeating the same update path had no credible route to
acceptance.

The reset conversion matches the producer convention: flight traces store
actual-minus-target ENU position error, and the environment reconstructs the
corresponding target-relative pose. Training reward improved while validation
collapsed. Actor relative parameter movement was 0.00379 at update 10, grew to
0.0171 at update 20, and reached 0.0156 in refinement despite the lower learning
rate. This isolates destructive policy updates rather than a reset sign error.

Keep the measured reset population, reward, PPO batch, exploration, workloads,
and acceptance criteria unchanged. End the `1e-4` phase at update 10 and validate
at update 5 as well. Refine at `1e-5`, then `3e-6`, evaluating every five updates.
The later phases continue to restore the complete validation-selected learner.
Stop if the lower-rate path repeatedly loses the update-10 hover duration or
flight stability.

### 76. Penalize persistent hover error directly

Date: 2026-09-02.

Campaign `press-hover-small-refinement-20260902` reproduced the measured-reset
learning peak at update 10 with 2.378 seconds mean episode-longest hover, zero
collisions, 10% other flight failures, and 90% timeouts. Refinement at `1e-5`
then produced 1.947, 0.132, and 1.877 seconds and plateaued. A final branch at
`3e-6` fell to 1.152 seconds after five updates. The campaign was interrupted
during its next validation because both lower-rate branches had moved away from
the same selected actor; no second seed was started.

At a nominal target-centered, motionless observation, the selected update-10
actor outputs about +0.0198 normalized vertical velocity, or +0.0099 m/s. Later
actors increase that command to about +0.018 to +0.024 m/s physical velocity.
This matches the validation pattern: most episodes survive initially, then fail
together after the persistent command has accumulated vertical error.

The hover reward allows this behavior. Its discounted pose-progress difference
can prefer an early correction followed by later drift, and its bounded pose
cost differs little between utilization 3 and utilization 7. For hover only,
set `position_progress` to zero and use the direct per-step stabilization cost
`-0.01 * utilization`. Keep hold progress, terminal outcomes, reset samples,
PPO settings, schedule, workloads, and acceptance criteria unchanged. Approach
retains its existing staged reward. Advance comparison revision 25 and start a
fresh campaign. Stop if direct pose cost does not reduce the persistent action
bias and improve held-out continuous hover.

### 77. Train against the complete hover hold from scratch

Date: 2026-09-02.

Campaign `press-hover-direct-pose-cost-20260902` improved the update-10 mean
episode-longest hover from 2.378 to 3.333 seconds and its maximum from 6.55 to
8.40 seconds. Measured-envelope occupancy rose to 9.53%, collision remained
zero, 89.33% of episodes timed out, and mean pose utilization was 3.303. At a
nominal motionless observation, vertical command fell from +0.0198 to +0.0157
normalized, confirming that direct pose cost reduced the persistent drift.

The phase-2 checkpoints at `1e-5` reached 2.949, 2.992, and 0.124 seconds. The
phase plateaued, and the `3e-6` branch then regressed on its first validation
seed. The campaign was interrupted without starting another attempt. The shared
failure is now the phase objective change: the first phase terminates and pays
success after five seconds, while later phases restore its optimizer and
suddenly require 15 and 30 seconds. The selected actor learned the available
five-second shortcut but not stable hover.

Use the exact 30-second acceptance hold during every hover training phase. Keep
the direct pose cost, measured reset population, PPO settings, evaluation
workloads, and acceptance criteria unchanged. Allow the `1e-4` phase up to 30
updates with five-update validation and the existing three-check plateau, so it
can continue only while continuous hover improves. Keep the `1e-5` and `3e-6`
refinement branches unchanged. Advance comparison revision 26. Stop if the
single hold objective cannot improve past the current 3.333-second reference.
