# Flight-parameter stability decisions

This log records the qualification attempts requested on 2026-09-27. Raw
evidence and execution details remain in `.build/flight-parameters-stability/`.
Entries for the first two attempts summarize work already performed when this
log was requested. Neither attempt has established a baseline.

## Qualification rules

- Qualify statistical repeatability across fresh same-seed and disjoint-seed
  batches. Exact sample replay is diagnostic, not the target.
- Keep capture, controller logic, and stability analysis separate. Analyze only
  completed campaigns. Freeze the capture source, environment, and criteria for
  each attempt; retain failed evidence without replacing selected flights.
- Apply `configs/flight/stability/default.yaml`: whole-flight bootstrap intervals
  for RMS and p99, plus peak and worst-flight checks. Retain drift and excursion
  diagnostics. Do not relax criteria merely to turn observed failures into passes.
- Establish a baseline only after qualification and a further fresh confirmation,
  as specified in `docs/FLIGHT_PARAMETERS.md`.
- There is no attempt limit. Continue investigating failures or inconclusive
  results. Per the latest user instruction, shut down the host only after
  stability is achieved and results are saved. Commit changes for each attempt.

## Attempt 1: fixed measurement window, original capture completion

- Evidence: `20260927-statistical-qualification/`.
- Decision: derive measurement start from transition allowance plus settling
  time, rather than actual PX4 mode-publication time. Require verified loiter by
  the deadline; retain the full measurement window.
- Result: 32 hover flights verified before trial 32, seed 2002, timed out during
  capture. The separate speed-based hold timer had reset during settling, even
  though the later scheduled measurement interval was quiet. The failed capture
  did not publish a raw trace, so its diagnostic interval is inferred, not a
  verified measurement window.
- Decision: retain this failed attempt and start fresh qualification after a
  narrow capture-completion correction. Do not merge its successful flights into
  the corrected attempt.

## Attempt 2: completion at the end of the fixed capture schedule

- Evidence: pilots in `20260927-fixed-capture-end/`; qualification in
  `20260927-qualification-fixed-end/`.
- Change: use the existing scenario stop predicate after transition allowance,
  settling, and measurement have elapsed. Preserve driver-failure recovery and
  completed-ULog verification. No RL controller tuning or PX4 synchronization
  changes are included.
- Validation: three fresh seed-2002 pilots passed full capture and ULog
  verification. Focused existing tests and the complete pre-commit suite passed.
- Design: three fresh batches, each with 50 hover and 12 complete response
  flights. Batch 2 repeats batch 1's seeds; batch 3 uses disjoint seeds. The
  additional response repetitions support whole-flight statistical estimates.
- Progress when this log was created: batch 1 completed all 62 flights without
  failure; batch 2 was collecting hover flights. Batch 1 horizontal error was
  26.686 mm RMS, 50.468 mm p99, 60.781 mm peak, and 32.534 mm worst-flight RMS.
  These descriptive values alone do not prove stability.
- Source provenance: `source-digests-at-launch.json` pins the source bytes used
  for this attempt. The first commit requested during this attempt checkpoints
  the existing validated flight-parameter work, including response-state
  reconstruction and its consumers. Committing does not change capture bytes;
  the final audit checks the pinned digests across all batches.
- Status: in progress. No baseline or shutdown is authorized by a partial pass.

## Resource decision: stagger a fourth independent capture

- Checkpoint commit: `643e335` contains the current qualification implementation.
- Evidence: `20260927-capacity-four/decision.json`, the pilot capture and ULog
  verification, and the early batch-3 capture logs. The main qualification
  directory continues to record GPU utilization every five seconds.
- Observation: three workers averaged 72.5% GPU utilization over ten minutes.
  A separately started fourth worker passed full hover verification; GPU usage
  reached 82-89% with 14.8 GB of 24.6 GB device memory in use.
- Timing: representative three-worker captures took about 131 seconds; four
  simultaneous captures took about 157 seconds each. This suggests roughly 11%
  more aggregate throughput, subject to confirmation over the remaining work.
- Decision: collect batch 3's first ten scheduled hover trials with the extra
  worker while batch 2 continues. Use unchanged seeds, configuration, recorder,
  verification, and criteria. Keep each producer's files in its own run directory;
  the ordinary campaign resume path will retain completed verified hovers.
- No production code or threshold changed. Record this host-load variation when
  interpreting the comparison; do not discard a failure caused by higher load.
- A fifth-worker pilot in `20260927-capacity-five/` failed during ground settling:
  Isaac remained at zero simulation time for about 52 seconds and produced no
  ULog. Existing qualification captures continued successfully. Retain this
  diagnostic failure and keep four total workers; higher GPU utilization alone
  does not justify unreliable startup. The underlying startup cause is not yet
  established, and no runtime workaround is applied to the qualification source.
- All first ten early batch-3 hovers passed verification. Continue the fourth
  worker with the next twenty preselected trials, 11-30, while batch 2 completes
  its response captures. Stop starting extra flights once batch 2 has nine
  response recordings, before its coordinator can start batch 3. The supplemental
  execution script and decision are retained in `20260927-capacity-four/`.

## Attempt 2 completed: capture passed, statistical qualification failed

- All 150 hover and 36 full response flights completed without capture failure.
  All 150 measurement windows matched the derived fixed start and 600 samples.
  The final audit verified every manifest and all 276 pinned source files.
- Evidence: `20260927-qualification-fixed-end/results/summary.json`,
  `metrics.csv`, `nonpassing.json`, and `hover-distributions.svg`.
- Across the six ordered comparisons, 1,560 metric checks passed, 160 were
  inconclusive, and 32 failed. These include correlated checks and both
  directions of each comparison; they are not independent failures.

| Batch | Horizontal RMS, mm | Horizontal p99, mm | Horizontal peak, mm | Worst-flight RMS, mm |
| --- | ---: | ---: | ---: | ---: |
| Q1 | 26.69 | 50.47 | 60.78 | 32.53 |
| Q2, same seeds | 27.11 | 50.97 | 62.98 | 31.36 |
| Q3, disjoint seeds | 25.95 | 46.54 | 65.74 | 31.79 |

- The fitting-set peak in Q3 exceeded Q1 by 14.6%. Small hover subsets left
  wide p99 intervals. Response failures concentrated in extreme values and
  displacement during yaw profiles. Reversing the comparison also exposed
  response failures between the same-seed batches.
- A sample-size diagnostic shows that merely tightening all intervals through
  more flights could require hundreds of response repetitions per batch. This
  is approximate planning evidence, not a power guarantee or a qualification.
- Decision: investigate response capture timing before another large campaign.
  Do not relax the criteria, designate a baseline, or shut down the host.

## Attempt 3: fixed response preconditioning deadline

- Evidence: `20260927-fixed-response-deadline/`.
- Finding: adaptive preconditioning shifted the same-seed response profiles by
  seconds. For seed 1104, the CTBR yaw-envelope profile began at 246.084 seconds
  in Q1 and 251.684 seconds in Q2. The response experiment uses deterministic
  simulation-time sensor excitation, so these shifts change the excitation
  phase. This is a candidate explanation for response variability, not yet a
  demonstrated cause. Timing evidence is in attempt 2's
  `results/profile-start-timing.json`.
- Change: reserve the entire existing 12-second preconditioning allowance for
  every profile. Require the continuous stability hold at that deadline; fail
  if it is unmet. Preserve the command sequences, physical model, and controller
  logic. Increase the response routine's timeout to cover the full schedule:
  276 seconds of preparation plus 132.75 seconds of commands and transition
  overhead.
- Validation plan: run three fresh complete response flights using seed 1104,
  which exposed the large timing shift. Preserve all pilot outcomes, verify all
  profile starts and ULog augmentation, then decide whether the change merits
  full statistical qualification. The stability criteria remain unchanged.
- The 13 existing focused capture tests pass after updating their expected
  deadline behavior. No new test was added.
- Result: all three pilots failed before `fit-pitch-steps` at the same
  simulation timestamp, 337.364 seconds, then landed through recovery. All
  pinned source files remained unchanged. No complete response trace was
  published; the original ULogs, bridge logs, and failed execution logs remain
  available. `pilot-failure.json` records the evidence digests and measurements.
- The speed at the deadline was 22.5-25.4 mm/s, above the unchanged 20 mm/s
  limit. Angular speed and tilt passed. The preceding roll profile left a
  decaying return-to-position oscillation; in pilot 1 its speed rose again from
  a 7.1 mm/s mean in preparation second 10 to 26.2 mm/s in second 12.
- Decision: the 12-second deadline is insufficient for these pilots. Preserve
  the fixed schedule and stability limits; investigate additional settling time.

## Attempt 4: longer fixed response preparation

- Evidence: `20260927-fixed-response-20s/`.
- Change: reserve 20 seconds per profile in the PX4 response experiment and
  allow 700 seconds for its policy step. This covers 460 seconds of preparation
  plus 132.75 seconds of commands and transition overhead. No stability limit,
  command sequence, sensor, or controller changes.
- Validation plan: four concurrent complete pilots, two each for seeds 1104
  and 2102. Both seeds previously showed seconds-long adaptive start shifts.
  Preserve all results. Additional settling is a hypothesis to test, not a
  guarantee that the fixed deadline will satisfy every profile.
- Statistical qualification remains pending. These pilots cannot establish a
  baseline or authorize shutdown.
- Result: all four complete captures passed in 1,477-1,495 wall seconds each.
  All 23 profiles per flight passed the existing command, cadence, completion,
  controller-boundary, and bridge-stream validation. All 276 pinned source
  files remained unchanged. Corresponding profile start times were identical
  across all four flights. Evidence: `pilot-results.json` and
  `completion-summary.json` in this attempt's directory.
- Error variation remains substantial: yaw-reversal displacement RMS was
  15.5-36.9 mm; the low-speed lateral profile ranged from 71.6 to 119.9 mm.
  Some profiles improved relative to the four matching old recordings and
  others worsened. These small descriptive samples do not prove either a
  causal improvement or statistical stability. Do not start a full campaign
  on the assumption that matching timestamps solved trajectory variability.
- Correction to attempt 3's excitation explanation: the deterministic sensor
  excitation period is 20 milliseconds. All 46 old profile pairs checked for
  seeds 1104 and 2102 had matching instantaneous excitation phase despite their
  different start times. Their settling histories differed, but the stated
  start-phase explanation does not apply to these pairs. The phase calculation
  is retained in `excitation-phase-diagnostic.json`.
- Four workers averaged 62.5% GPU utilization over the entire recorded period,
  reaching 88%. The final measurement includes startup and completion; the
  earlier five-minute steady-work sample averaged 77.5%. Do not claim 95%
  utilization or a measured whole-campaign speedup.
- The complete pre-commit suite passed. Two existing timeout tests now derive
  their expectations from the configured duration; no tests were added.
- Criteria diagnostic: independently resampling two 12-flight batches from the
  same fixed empirical population of attempt 2's 36 response flights rejected
  96.4% of 10,000 pairs on at least one peak/worst-flight rule. With 24 flights,
  76.9% were rejected. Hover peak/worst-flight rejection was 51.8% for ten-flight
  groups and 20.1% for fifty-flight groups. This diagnostic holds the sampled
  population constant; it does not establish real-process stationarity, model
  unseen extremes, or replace any qualification result. Evidence:
  `peak-rule-diagnostic.json` and its run-scoped script.
- Decision: review the sampling sensitivity of the statistical criteria before
  committing to another large campaign. The current criteria remain unchanged.
  The user has been asked to choose that review or a larger campaign under the
  existing rules. No baseline is established; the host remains running.

## Attempt 5: prepare larger samples without increasing counts

- Decision: keep the physical flight definitions and current statistical gates.
  Prepare per-flight lossless compression, validated resume for both behaviors,
  bounded worker storage and an explicit functional campaign flow. Statistical
  analysis remains a final stage. A new baseline is still conditional on later
  qualification and fresh confirmation.
- Raw logs are retained because actuator audit and controller replay use them.
  Gzip preserves original-content hashes; completed checkpoints allow removal
  of uncompressed copies and matching private harness duplicates. Failed attempts
  remain evidence. Source files (including dirty working-tree contents), runtime
  images and capture definitions are pinned across resume.
- Scope: all production seed lists and counts remain unchanged. Verification
  uses the existing 50 hover and three response seeds, with four workers in a
  run-scoped config. This verifies the pipeline, not statistical stability.
- Storage feasibility: a retained response pilot's 387 MiB of raw logs compress
  to 97 MiB (about 75% smaller), with verified original SHA-256 hashes. Raw and
  compressed actuator-audit reports are identical and pass. Controller replay
  results are also identical; both report the same existing controller failures.
- Evidence and execution reports:
  `.build/flight-parameters-stability/20260927-pipeline-verification/`.
- The sampling/criteria proposal is in
  `docs/backlog/drone_platform_followups.md#flight-parameter-distribution-qualification-proposal`.
  Its synthetic diagnostic illustrates why larger samples reduce uncertainty
  but do not make 300 a universal sufficient count. No proposed rule is active.
- The first live pipeline check stopped after four hover captures: bridge logs
  still went to the shared project log directory, so the new private-evidence
  check rejected them. No checkpoint or bundle was published. The failed run's
  available evidence is retained as `pipeline-20260927` and is not qualification
  evidence. The harness now mounts each flight's private logs at the existing
  bridge path inside Isaac. This also prevents concurrent filename collisions
  without changing authored config identity. Verification restarts with a new
  source-pinned run ID.
- The second check confirmed isolated log routing, then exposed container-owned
  files preventing host postprocessing. `pipeline-20260927-a2` retains those four
  captures with no completed checkpoint. The harness now returns ownership of
  private output directories to the host after container writers stop. This
  changes storage ownership only; flight commands and simulation logic are
  unchanged.
- Verification result: `pipeline-20260927-a3` completed all 50 hover flights
  and all three response flights with no capture failures, retaining 30,000
  hover samples and 69 response profiles. All 106 archives passed independent
  original-content hash checks; verified raw duplicates were removed from both
  producer and private harness directories. Logs fell from 2,543.3 MiB to
  714.5 MiB (71.9% smaller). The complete producer bundle is 761.2 MiB.
- Final analysis, manifest validation and verification-root `latest` publication
  passed. Alignment reference preparation and actuator audits passed for all
  three sets using compressed producer symlinks. Stability is
  `insufficient-evidence`; no baseline is designated and no production selector
  was advanced. Default seed lists and counts are unchanged.
- The capture run took 98.9 minutes. Total observed GPU utilization averaged
  68.8%, peaked at 99%, and included other host applications; peak memory was
  22,506 MiB. The mixed final hover/response batch left workers idle and delayed
  the third response. After this source-pinned capture completed, dispatch was
  grouped into hover batches followed by response batches. A parallel host test
  verifies that response capture starts after all hover checkpoints complete.
  The measured runtime belongs to the preceding mixed-batch implementation;
  no speedup is claimed for the grouping change without a new measurement.
- Live checks also confirmed that a second campaign writer is rejected and
  that completed compressed checkpoints reload without recapture. Full results
  are in `completed-pipeline-verification.json`, `campaign-execution.json` and
  the audit reports in this attempt's evidence directory.
- The general Docker integration sweep passed seven targets. Contact-chain
  initially selected mismatched existing `best`/`latest` artifacts and passed
  when rerun with the matching September 12 controller and parameter bundle.
  The separate injected-instability test detected its fault but timed out
  waiting for landing/disarm, including on an isolated rerun. That runtime
  recovery failure remains outside the flight-evidence changes; its logs are
  retained with the verification artifacts.
- Final host tests, type checks and quality hooks passed, including the parallel
  batching case. The full sweep reformatted one test parameter list; the lint
  and formatting reruns then passed. New unstaged files were checked explicitly.
  Changes remain unstaged, and the host remains running.
