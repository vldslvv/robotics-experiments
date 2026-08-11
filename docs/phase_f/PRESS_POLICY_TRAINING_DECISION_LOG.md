# Press policy training decision log

Status: SECOND SCRATCH SELECTION REJECTED; TARGETED FINE SWEEP PENDING
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
| `20260811T034512Z-scratch-hard30-coarse` | `20260810T-accepted-reproduction` | 121 updates at `1e-4`, then 51 at `3e-5` | Iteration 150 selected; score -0.566693, 94.7% success, advisory rejected | Fine-sweep updates 146-154 from scratch |

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
whether updates 146-149 or 151-154 pass. Reproduce the same scratch path, keep
coarse checkpoints through 145, and save every update from 146 through 155.
Do not change controller, task, reward, PPO values, reset mixture, acceptance
gates, or initialization. This is the targeted precision stage justified by
the broad sweep.
