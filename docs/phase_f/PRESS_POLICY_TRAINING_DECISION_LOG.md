# Press policy training decision log

Status: FOURTH SCRATCH SELECTION REJECTED; 40% FINE SWEEP PENDING
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
