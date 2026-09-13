# Flight-to-policy improvements

The flight-to-policy flow is complete. Controller run `20260912T211745Z` is
qualified against the fixed flight bundle `20260912T102640Z`. Press-hover campaign
11, `20260913T095657Z`, is accepted and published as `latest` and `best` after
300/300 successful reserved-test episodes with every acceptance check passing.
Test position error is 17.24 mm RMSE, 19.08 mm P95, and 19.62 mm P99.

This document separates implementation expectations from measured execution
results and retains unsuccessful iterations. The final report and exact campaign
ledger are under `.build/flight-to-policy/20260912T102640Z/`.

| Change | Implemented behavior | Expected impact |
| --- | --- | --- |
| Record experiment inputs | Existing campaign and attempt reports retain the full resolved configuration, pinned input paths and identities, and runtime versions alongside seeds, checkpoint references and results. Attempt initialization is recorded. | Two policy runs can be compared using the recorded training and evaluation inputs, including failed or interrupted runs. |
| Document pooled acceptance metrics | The workflow documents sample pooling, successful-hold windows, each metric's population, and conservative per-flight reference limits. | Acceptance results are easier to interpret without changing the statistical calculation or thresholds. |
| Reject NaN and infinity | Evaluation checks active observations, raw actions before clipping, transition metric inputs and aggregate results; invalid values raise an error with context. | Invalid values cannot silently become clipped actions or valid-looking acceptance evidence. Failed evaluation stops the workflow before publication. |
| Author press duration once (stage 3) | The selected Hydra curriculum stage owns episode duration. Press experiment overrides are removed, and both the environment and comparison record read the stage value. The standalone drone-hover task retains its generic duration. | A duration change requires one edit. Current hover/approach durations remain 40/45 seconds, with unchanged comparison identities. |
| Train one candidate policy (stage 3) | Shipped Hydra configs use one training attempt, seed 42. Checkpoints are selected on validation seeds 101/202/303, then the frozen actor is tested once on acceptance seeds 404/505/606. Each workload has 100 episodes per seed. | Campaigns have one training-attempt budget while retaining evaluation across multiple seeds. Results qualify the selected actor; they do not measure training repeatability across initializations. |

Report schema 26 adds provenance; the policy schema and comparison fingerprint
remain unchanged. Tests cover provenance serialization and retention without
launching subprocesses, pooled statistics, NaN and both infinities, completed episode
masking, and unavailable successful-hold statistics.

Finite checks add evaluation work and device synchronization. Their isolated GPU
overhead has not been measured; end-to-end results appear below. Training does not
collect Git commits, working-tree state or container identity. Keep the checkout
and configuration fixed during a campaign.

Stage 3 checks cover changing a stage's duration through Hydra without a second
edit, comparison identity following that duration, single-attempt configuration,
and the existing validation-selection and final-test publication rules.
The change does not alter the campaign driver or its stopping rules. PPO phases,
reset mixtures, acceptance thresholds, comparison mechanics, and sensor noise
retain their existing behavior. Controller fitting and policy training remain
separate sequential activities. Stage 5 adds independent controller qualification
and hover feasibility; the live execution record appears below.
The detailed current contract is in [RL workflow](RL_WORKFLOW.md).

Stage 4 implements recording and dataset separation:

| Change | Implemented behavior | Expected impact |
| --- | --- | --- |
| Record complete hover motion | Raw and verified traces retain pose plus directly measured ENU linear/angular velocity from the same snapshot. Reset loading reads those values directly. | Removes velocity estimation from pose differences and preserves the measured motion at reset. PPO observations and rollout storage retain their existing format. |
| Assign complete flights to three sets | YAML lists define 30 fitting, 10 validation, and 10 acceptance hover flights, plus one complete general controller-response flight per set. Each recorder uses the same behavior and cadence for all sets, with distinct recorded seeds. | Provides held-out evidence without changing how a set is captured. Hover capture grows from 30 to 50 flights and response capture from one to three complete suites. Measured runtime appears in the execution record. |
| Keep held-out recordings out of derivation | Bounds and acceptance limits use the 30 fitting hover flights; resets use only their policy windows. Alignment preparation pins all three response flights. All sets retain their original evidence and statistics. | Preserves the reference sample count and prevents held-out measurements from influencing current training inputs or thresholds. Stage 5 uses independent response sets for controller selection and final qualification. |
| Validate motion and set metadata | Strict loaders reject missing/misaligned/non-finite motion arrays. Campaign derivation requires every configured flight and set/seed assignment; bundle validation checks recorded identities and file hashes. | Invalid or misassigned evidence raises an error before publication or consumption. Old artifacts require a fresh capture; they are never migrated or overwritten. |

Host checks cover recording and angular-frame conversion across all three sets,
verified-window sample pairing, direct reset velocities, held-out exclusion,
set/seed validation, and immutable bundle publication. These host checks establish
contracts, not measured controller or policy performance. Live capture and GPU
validation results appear in the execution record below.

## Stage 5: independent controller qualification and hover feasibility

Implemented changes:

- Validate and reference all three immutable response recordings before fitting.
  Search uses fitting recordings only; fitting response and hover checks determine
  validation eligibility. Validation uses the existing ranking and regularization.
- Freeze validation-selected settings before acceptance. Acceptance failure keeps
  the frozen selection as diagnostic evidence. No thresholds were changed.
- Isolate evaluator caches and target files by recording set and pin the source
  reference digest before worker launch. Cached results with a mismatched source
  digest are rejected. Reports distinguish
  recording membership from the internal fit/validation/stability profile labels.
- Require actuator audit and controller replay on all three response flights,
  plus a settings-bound integration report. A provisional controller is explicitly
  quality-failed until finalization completes these checks. Training retains its
  existing quality-passing controller boundary.
- Add `eval-press --hover-feasibility`: deterministic observed position/yaw
  feedback through the normal press-hover velocity action, measured fitting resets,
  validation seeds, configured duration, and unchanged evaluation criteria.
  Feedback gains are position 1.25/s, velocity damping 0.2, and yaw 2.0/s;
  their measured feasibility results appear below.

Expected impact: validation now measures transfer to an independent PX4 response
flight before selection; fixed-benchmark acceptance can reject the frozen
controller. Feasibility can expose controller/action/reset problems before PPO
spends a training budget. Measured controller qualification, policy acceptance,
and GPU utilization appear in the execution record.

## Execution record

The operator confirmed driver resolution and authorized GPU execution on
2026-09-12. The host reports an NVIDIA GeForce RTX 4090 with driver 580.178.04
and 24,564 MiB memory. Capture run `20260912T102640Z` started at 10:27 UTC with
the configured single worker. Its console and one-second GPU samples are under
`.build/flight-to-policy/20260912T102640Z/`; the published flight bundle is
under `.build/flight-parameters/holybro-x500-v2-hover-v1/20260912T102640Z/`.
Asset-generation prerequisites are suppressed with top-level
`make --assume-old=generate-drone-usd` and recursive
`MAKEFLAGS=--assume-old=generate-drone-usd`; starting asset digests are in
`source-assets.sha256`.
Capture completed at approximately 12:01 UTC. All 50 hover flights passed ULog
verification, preserving 600 samples each (30,000 total). Fitting seeds are
1001-1030, validation seeds 2001-2010, and acceptance seeds 3001-3010. The three
response flights use seeds 1101, 2101, and 3101; each completed all 23 profiles
and ULog controller-state augmentation. No flight retries were needed.
The full manifest validates, and both existing drone asset hashes are unchanged.

The pinned flight fingerprint is
`cc8da34a4e36a06891b4f96b48986c350a3681bb7b6a23f5bbfee25549ebc0bc`;
`flight_parameters.json` SHA-256 is
`65f02c5e5bfe6d74291329967033510839acad8b5d6254961b10e39e36c73661`.
The exact capture inputs are `configs/flight/holybro_x500_v2_hover.yaml`,
`configs/flight_parameters_x500_hover.yaml`,
`configs/alignment_px4_x500_velocity.yaml`, and
`configs/alignment/controllers/px4_sitl_x500_velocity.yaml`, with the existing
Hydra recording settings unchanged. Their fingerprints and source image are
retained by the producer artifact.

Capture took about 94 minutes: hover launches averaged 84.81 seconds, and response
launches took 445.87, 445.79, and 451.94 seconds. Across 5,658 one-second samples,
GPU utilization averaged 28.77% (32.38% excluding idle samples), peaking at 59%.
A representative hover sample showed Isaac using approximately five CPU cores
and PX4 approximately one core. Single-flight lockstep capture did not reach the
95% GPU target; the configured recording workload was retained. Raw measurements
and exact summary values are in the operator directory's `capture-gpu.csv` and
`capture-summary.json`.

Controller qualification run `20260912T120210Z` failed fitting hover stability.
All three actuator audits passed. Four of five candidates passed response
quality, but none passed the hover bounds; validation and acceptance were not
run. The diagnostic selection's peak position utilization was 1.134571, with
Y error -0.055088 m at 29.96 seconds against a -0.048554 m lower bound. Its
positive X peak was 0.057629 m against 0.052156 m. The reference response's
maximum axis utilization was 0.413994. The growing horizontal oscillation
motivates testing stronger low-command velocity feedback: iteration two expands
the declared `velocity_observation_hold_scale` grid to X 1.10/1.30/1.50 crossed
with Y 1.06/1.25/1.45, with Z 1.0. Motion scales retain their original grid.
Thresholds, flight evidence, and controller implementation remain unchanged.
The failed controller and raw candidates are preserved under the alignment run;
the next definition is recorded as `alignment-grid-02.yaml` in the operator
directory. Feasibility and training await accepted controller qualification.

Iteration two (`20260912T120710Z`) evaluated 45 fitting candidates and failed
before validation. Four candidates passed response checks, all with the original
hold scale. The smallest hover utilization was 1.097541 at hold scale
[1.10, 1.25, 1.0], but Y braking error rose to 0.008912 m above its 0.005 m
limit. Larger damping therefore did not produce a qualified candidate.
`alignment-02-diagnosis.json` retains all candidates' hover and response summaries.
The stability replay follows the recorded commands and shows signed drift from
the captured estimator offset. Iteration three restores the original hold scale
and tests faster existing `velocity_estimator_bias_hold_decay_hz`: X
0.014/0.030/0.060/0.120 Hz crossed with Y 0.075/0.15/0.30/0.60 Hz, keeping Z
0.04 Hz and the original motion-scale grid. The exact definition is
`alignment-grid-03.yaml`; the intended effect is to reduce persistent position
offset without changing the motion-command bias model or any acceptance limit.

Iteration three (`20260912T121300Z`) evaluated 80 fitting candidates and failed
before validation. Faster Y decay brought both Y extrema inside the bounds, but
X 0.014 Hz retained a positive peak near 0.057 m while X 0.030 Hz produced a
negative peak near -0.052 m. Neither meets the unchanged asymmetric X limits.
`alignment-03-hover.json` records every candidate's extrema. Iteration four
tests X 0.016/0.017/0.018/0.019/0.020 Hz crossed with Y 0.10/0.12/0.15/0.20 Hz,
keeping Z 0.04 Hz and the original scales. This narrows the fit between the
measured opposing X failures; `alignment-grid-04.yaml` records the exact inputs.

Iteration four (`20260912T122220Z`) evaluated 105 fitting candidates. Thirty-six
passed fitting response and hover checks, and all 36 passed independent
validation. Validation selected hold decay [0.016, 0.10, 0.04] Hz and motion
scale [1.00, 1.04, 1.0], with position utilization 0.777135 and attitude
utilization 0.394694. That exact frozen setting failed acceptance: combined
endpoint error 0.029751 m exceeded 0.010 m and hover position utilization was
1.810328. It remains unpublished; acceptance did not select a replacement.
The operator directory retains `alignment-04-hover.json`,
`alignment-04-validation.json`, and `alignment-04-acceptance.json`.

ULog diagnosis found the combined-turn initial X estimator offset was about
+0.0093/+0.0096 m/s in fitting/validation but -0.0101 m/s in acceptance.
The mean offsets were +0.0013/+0.0017/-0.0027 m/s, respectively. The existing
steady -0.010 m/s correction and captured-offset decay therefore do not
generalize across these boundary states. Acceptance hover X error peaked at
0.094419 m; its reference remained between -0.023409 and +0.021295 m.
Iteration five jointly tests the existing declared X steady bias
-0.010/-0.006/-0.002/0.0 m/s, motion decay 0.075/0.15 Hz, X hold scale
1.10/1.30/1.40, and X hold decay 0.002/0.006/0.010 Hz with Y hold decay 0.10 Hz.
Other parameter grids are retained. The stronger X scale changes how each
captured physical/estimated boundary initializes its offset; slower hold decay
tests whether that correction can retain fitting hover while accommodating the
different acceptance boundary. This is repeated qualification against the same
fixed benchmark; `alignment-grid-05.yaml` records the complete definition.

Iteration five (`20260912T124410Z`) evaluated 480 fitting candidates; none met
all position bounds. The closest position utilization was 1.008821. Forty
candidates below 1.04 utilization all failed response quality. One nearby
candidate missed only X braking: it traveled 0.042078 m versus PX4's 0.048389 m,
stopping at 3.4 versus 3.5 seconds. The diagnostic tables are
`alignment-05-hover.json` and `alignment-05-nearby-response.json`.
The coordinator was stopped after preserving every GPU trace because repeated
full-bundle validation dominated its CPU qualification pass. The existing host
`alignment-fit` entrypoint then completed the failure report from those exact
results after validation was moved outside candidate iteration. Each evaluation
stage still validates the entire immutable bundle before using its loaded bounds;
selection, metrics, and thresholds are unchanged. Tests verify that fitting,
validation, and frozen acceptance each validate once regardless of candidate count.

Iteration six uses the nearby X hold scale 1.30 and X hold decay
0.004/0.005/0.006 Hz, with Y decay 0.12 Hz to remove the small Y bound excess.
It tests braking-lobe scale 0.20/0.35/0.50 to lengthen the observed short stop,
motion decay 0.15/0.30 Hz, X steady bias -0.010/-0.006/-0.002/0.0 m/s, and Y
motion scale 1.04/1.06/1.08. The base setting is centered in this declared grid
to avoid replaying irrelevant implicit combinations. Its exact definition is
`alignment-grid-06.yaml`. The complete host pre-commit check passed after the
qualification validation fix; the focused alignment tests also cover this grid.

Iteration six (`20260912T130500Z`) evaluated 216 fitting candidates. None passed
response quality. The diagnostic selection missed X braking by 0.000752 m over
the 0.005 m limit and had hover utilization 1.01285. At X bias -0.006 m/s,
doubling motion decay from 0.15 to 0.30 Hz moved signed combined-turn X endpoint
error from -0.010114 to -0.004522 m. Iteration seven therefore tests motion
decay 0.30/0.60/1.20 Hz and X bias -0.006/-0.004/-0.002/0.0 m/s. It adds X
motion scale 0.94/0.97/1.00 to increase arrival speed and recover the short
braking distance, crossed with Y scale 1.04/1.06. Hold scales and decay grid
remain unchanged; braking scale is fixed at 0.35.
`alignment-grid-07.yaml` records all 216 combinations through the existing
`search.initial_settings` capability. This batches independent fitting work
upfront; the normal parameter search, validation ranking, and frozen acceptance
still run. The production YAML retains the concise declarative parameter grid.

Iteration seven (`20260912T131745Z`) evaluated all 216 candidates in approximately
three minutes but failed fitting response quality. Thirty candidates met the
axis bounds. The diagnostic selection's X braking error was 0.005883 m and
combined endpoint error 0.006450 m. Lower X motion scale improved braking while
X envelope velocity NRMSE remained 0.0431 against 0.10. Iteration eight extends
X motion scale to 0.85/0.88/0.91 and concentrates X hold decay at
0.0035/0.0040/0.0045 Hz around the passing region. It retains motion decay
0.30/0.60/1.20 Hz and X steady bias -0.006/-0.004/-0.002 m/s, fixes Y motion
scale at 1.06, and retains all other settings. `alignment-grid-08.yaml` contains
the complete 81-candidate initial grid for the existing batched search.

Iteration eight (`20260912T132330Z`) completed 81 candidates in about 93 seconds
and failed fitting. Its diagnostic selection had braking error 0.005028 m and
position utilization 1.00527. Independent controller-boundary replay then showed
why the reduced X motion scale cannot be pursued: physical-state normalized
motor RMSE was approximately 0.0158-0.0161 on the X envelope and 0.0200-0.0207
on the moderate stability profile across all three recordings, above 0.0100.
The core replay passed; the physical-state mapping failed. The failed reports
remain in that alignment run and `alignment-08-boundary.log`.
Iteration nine restores X motion scale 1.0 and tests the existing X velocity
proportional gain 1.6/1.7/1.8 and derivative gain 0.10/0.15/0.20. It retains the
motion decay and steady-bias grid, with X hold decay 0.0035/0.0040 Hz. This tests
transient response without distorting high-command physical velocity. Y/Z gains
and every qualification threshold remain unchanged. `alignment-grid-09.yaml`
contains 162 explicit initial candidates.

Iteration nine (`20260912T133200Z`) produced one fitting-qualified candidate,
which also passed validation: X proportional/derivative gains 1.6/0.10, X bias
-0.006 m/s, motion decay 1.20 Hz, and hold decay [0.004, 0.12, 0.04] Hz.
Frozen acceptance failed with position utilization 1.17977, X braking error
0.007212 m, and combined endpoint error 0.014883 m. The signed endpoint errors
were [-0.008537, -0.010535, -0.006135] m; X braking remained short.
Hover positive X/Y peaks were 0.061532/0.063659 m. Controller-boundary replay
passed fitting and validation, but acceptance's moderate profile had physical
motor RMSE 0.011668 above 0.010000.

Iteration ten tests steady biases X -0.008/-0.010 and Y -0.018/-0.022 m/s to
correct the negative endpoints. Hold scales [1.35, 1.15, 1.0], X decay
0.0015/0.0020/0.0025 Hz, and Y decay 0.10 Hz test the remaining hover peaks while
retaining the captured-offset correction. The braking lobe begins at 0.025 m/s
and peaks at 0.055 m/s, with scales 0.25/0.35, to extend low-speed stopping.
Original X gains are restored as candidates alongside 1.6/0.10 so actuator
accuracy can improve; motion decay remains 0.60/1.20 Hz. The 192-candidate
definition is `alignment-grid-10.yaml`. No limits or recording settings changed.

Iteration ten (`20260912T134135Z`) failed fitting after 192 candidates. Its
diagnostic selection stopped short by 0.008708 m in X and 0.007046 m in Y;
the wider low-error band slowed the approach before braking. Hover utilization
was 1.04963. Iteration eleven restores the original 0.050/0.070 m/s braking
band and adds the existing `command_delay_s` field to the declarative fit at
0/0.008/0.016 seconds. This tests the observed early stopping without changing
the large-command velocity mapping; both velocity and CTBR quality remain
binding. Y steady bias is fixed at -0.018 m/s, while X remains
-0.008/-0.010 m/s. Other iteration-ten candidates are retained.
`alignment-grid-11.yaml` contains 288 explicit initial candidates.

Iteration eleven (`20260912T134830Z`) failed fitting. The selected 8 ms delay
passed translation checks (X braking error 0.004511 m; endpoint 0.006581 m),
but CTBR yaw error was 3.051115 degrees and hover position utilization 1.03407.
The failing hover bound was negative X (-0.044994 m versus -0.043512 m).
Command refresh at 50 Hz explains the nearly identical 8/16 ms CTBR responses.
Iteration twelve tests 4/8 ms delays, existing yaw-torque filter values
2.0/2.5/3.0 Hz, and slower X hold-offset decay 0.00025/0.00050/0.00075/0.00100 Hz.
It fixes X gains at 1.6/0.10 and braking scale at 0.35 while retaining the two
motion decays and X biases. `alignment-grid-12.yaml` contains 96 initial
candidates; every field is declared and quality limits are unchanged.

Iteration twelve (`20260912T140009Z`) evaluated 624 settings after expansion
from its 96 initial candidates and out-of-grid base settings. It failed fitting:
160 candidates passed response checks, but their best hover position utilization
was 1.020832. Yaw filter 2.5 Hz resolved the CTBR failure. X gain 1.8 reduced
hover utilization to 0.94442 but failed braking; X gain 1.6 passed response
but retained excess negative drift. Iteration thirteen tests intermediate X
gains 1.650/1.675/1.700/1.725 with hold decay 0.0005/0.0010/0.0015 Hz,
braking scales 0.25/0.35, and the retained bias/motion-decay pairs. Delay is
4 ms and yaw filter 2.5 Hz. The base now lies inside the explicit 96-candidate
grid in `alignment-grid-13.yaml`, avoiding out-of-grid search expansion.

Iteration thirteen (`20260912T141600Z`) evaluated 96 settings; 44 passed fitting
and were independently validated. The frozen selection (X gains 1.725/0.4/0.1,
braking scale 0.25, motion decay 1.2 Hz, hold decay 0.0015 Hz) passed acceptance
hover but failed X braking (0.007039 m) and combined endpoint (0.011466 m).
Signed endpoint error was [-0.007351, -0.006335, -0.006108] m. Boundary replay
also rejected the added delay: Z-envelope collective RMSE reached 0.91 N,
versus the unchanged 0.05 N limit. Reports are retained in each set directory.
At the acceptance braking transition, target X velocity was 0.07069 m/s versus
reference 0.07723 m/s; the target was already slow before braking.

Iteration fourteen restores zero delay and the original 2 Hz yaw filter. It
tests existing X integral gains 0.4/0.5/0.6 to correct the deficient approach
response, with proportional gains 1.7/1.8, derivative gains 0.1/0.2, braking
scales 0.10/0.35/0.50, and hold decays 0.001/0.002/0.003 Hz. Bias is
[-0.012, -0.020, 0] m/s to address the negative endpoint residual; motion decay
is 1.2 Hz. `alignment-grid-14.yaml` contains 108 candidates. Controller equations,
reference flight settings, and all quality thresholds remain unchanged.

Iteration fourteen (`20260912T142343Z`) failed fitting after 108 settings.
Its diagnostic selection passed hover and X braking (0.004992 m) but failed
combined endpoint (0.010945 m), with signed residual [0.004683, 0.009049,
-0.003998] m. X integral gain 0.6 reduced braking error further to 0.003836 m.
Iteration fifteen tests X integral gains 0.6/0.7/0.8 and X biases
-0.008/-0.010/-0.012 m/s, restoring Y bias -0.018 m/s to reduce the positive
endpoint residual. X hold decays are 0.002/0.003/0.004 Hz; proportional and
derivative pairs remain 1.7/1.8 and 0.1/0.2. Braking scale is fixed at 0.35.
`alignment-grid-15.yaml` contains 108 explicit initial candidates.

Iteration fifteen (`20260912T142737Z`) failed fitting on Y braking alone:
0.005424 m versus 0.005 m. Its diagnostic selection used X gains 1.7/0.8/0.2,
bias -0.008 m/s, and hold decay 0.002 Hz; X braking was 0.002453 m and endpoint
0.008966 m. Iteration sixteen retains the X integral range 0.7/0.8, adds Y
integral gains 0.25/0.30, and reduces Y bias magnitude to -0.016 m/s. X biases
-0.006/-0.008 and motion decays 0.3/0.6/1.2 Hz test whether less steady
correction improves physical-state replay while retaining endpoint accuracy.
Derivative gains are restored to [0.2, 0.2, 0], X proportional gains remain
1.7/1.8, and X hold decay is 0.002/0.003 Hz. The 96-candidate definition is
`alignment-grid-16.yaml`. No controller equations or thresholds changed.

Iteration sixteen (`20260912T143155Z`) evaluated 96 settings. Fifteen passed
response checks, but none passed hover; the best response-passing position
utilization was 1.044299, dominated by Y minimum -0.050705 m versus the
-0.048554 m bound. Further parameter grids are paused for fixed-controller
and estimator-state diagnosis. No controller is accepted and no PPO campaign
has started.

A separate capture audit re-decoded all 50 completed ULogs and exactly reproduced
every published verification record and trace without modifying producer files.
The bundle manifest and hashes passed again. All 30,000 samples were inside the
fitting-derived pose bounds, including both held-out sets. Full-window pooled
3D position RMSE was 2.837/2.744/2.837 cm for fitting/validation/acceptance;
maximum 3D error was 6.091/6.096/5.436 cm and maximum tilt was
0.2493/0.2189/0.2282 degrees. Global maximum speed was 0.045062 m/s and
maximum full orientation error was 0.9639 degrees. These results support stable
hover capture; they do not qualify the lightweight controller. The operator
files `capture-hover-audit.json`, `capture-ulog-reverification.json`, and
`capture-hover-audit.png` retain the statistics, verification outcome, and all
50 full-window trajectories.

The earlier accepted controller `20260822T092045Z` passed its recorded actuator
audit and controller replay, but had endpoint error 9.753 mm against 10 mm and
Y braking error 4.665 mm against 5 mm. Its exact settings are present in the
first new fitting run. On the old/new recorded hover initial states, target
maximum X error was 29.474/57.415 mm and minimum Y error was
-19.183/-55.977 mm. The new measured positive-X bound also narrowed from
61.720 to 52.156 mm using the same 5 mm margin. Prior decision-log entry 392
already records a fixed accepted controller failing fresh-reference endpoint
qualification. This supports limited margin across captures rather than a claim
that the old acceptance was invalid. Through iteration seventeen, controller
and actuator equations stayed fixed. Recent gain tuning sometimes improved motion while
worsening motor replay, which motivates diagnosing the estimator mapping before
further fitting.

Fixed-controller diagnostic `20260912T144427Z-baseline-diagnostic` pins the
original `20260822T092045Z/controller.json` through a relative symlink and
references the same three new response files without altering either producer.
It is not a qualification or publication. Replay passes on fitting and
validation. Acceptance passes core PX4-estimated-state replay, but physical-state
motor RMSE fails on moderate stability (0.010608) and hover (0.015394), against
0.010000. This narrows the discrepancy to physical-to-estimated state behavior.
The working controller definition was restored to the earlier baseline; all
sixteen experimental definitions and results remain preserved. PID compensation
is not retained as an accepted improvement.

The response hover capture uses feedback on PX4's measured position to generate
velocity commands. Target alignment replays those recorded commands, so it tests
long-window response reproduction rather than independently closing the target's
position loop. This existing behavior makes accumulated estimator-model error
visible; the actual press-hover feasibility mode separately tests closed-loop
position feedback. Neither contract has been changed during this diagnosis.

The existing replay report now includes per-axis ENU velocity-input RMSE and
signed mean error. All 24 focused replay tests pass; full pre-commit also passes.
On acceptance hover, substituting only recorded PX4 velocity for the estimator
approximation reduces motor RMSE from 0.015394 to 0.002120. Substituting only
attitude yields 0.014755; only angular rate yields 0.015089. This diagnostic
uses recorded state as an oracle, is not deployable behavior, and does not count
as qualification. `recorded-state-substitution.json` preserves every result.

Estimator identification then used fitting response data only: captured physical
velocity, recorded PX4 velocity, held commands, initial velocity state, the
existing command blend, and a 125 Hz approximation of the existing filter/decay
model. Two bounded least-squares starts converged in 11 evaluations to nearly
the same parameters. Motion scales were [1.011564, 1.012211], hold scales
[1.015552, 1.008749], bias [-0.002195, -0.003795] m/s, shared motion decay
0.367224 Hz, and hold decay [0.030412, 0.075104] Hz. Both filter cutoffs reached
the diagnostic 30 Hz upper bound, supporting tests of 30 Hz and unfiltered
inputs. The method, bounds, fitting-reference digest, and per-profile errors
are retained in `fitting-velocity-estimator-identification.json`; this is a
candidate proposal, not a replacement selection metric.

Iteration seventeen tests 108 declared estimator combinations around that
proposal, with baseline PID gains, braking lobe, and actuators fixed. Motion
decays are 0.2/0.367/0.6 Hz; hold decays X 0.02/0.03/0.04 and Y 0.05/0.075/0.1 Hz;
X/Y filter cutoffs are independently 0/30 Hz. The rounded scale/bias centers
and complete definition are in `alignment-grid-17.yaml`. The production default
controller definition remains at the earlier baseline until qualification passes.

Iteration seventeen (`20260912T145835Z`) failed fitting qualification: selected
X braking error was 0.007203 m against 0.005 m and hover position utilization
was 1.44907. Closely fitting the observed velocity inputs did not establish a
passing simulated response.

Iteration eighteen tests the first estimator mechanism change in this task.
The observation scale previously classified zero-command braking as hold even
while physical speed remained above the motion threshold. This couples the
larger hover scale to braking and can exaggerate the velocity input there.
The scale transition now uses the larger absolute commanded or physical speed
per world axis, with the existing transition thresholds. Steady-bias and
seeded-offset decay rules, PID gains, actuators, and quality limits are fixed.
Both NumPy updates/seeding and their Torch twins use the same mapping. All 38
actuation tests pass, including directional/componentwise transition checks
and sequential state parity. The controller implementation digest starts a
new comparison fingerprint.

The declared 96-candidate `alignment-grid-18.yaml` tests hold scales
X 1.1/1.3/1.5 and Y 1.06/1.15, hold decay X 0.003/0.014/0.02/0.03 Hz and
Y 0.075/0.12 Hz, and motion decay 0.075/0.3 Hz. Every other setting is at the
baseline. This is an experimental correction pending full qualification;
the captured evidence is unchanged.

The operator directory also contains `controller-qualification-summary.json`,
which consolidates run IDs, qualification failures, frozen settings fingerprints,
and measured GPU throughput. Iterations eight through eleven averaged
31.3-33.4% GPU utilization, with 47% p95 and 48-51% peaks. Their 128-slot
worker processed approximately 0.97-1.16 candidates per second. The existing
measured capacity limit and simulation/command loops prevent 95% utilization;
no duplicate physical work is added to inflate utilization.

Pin the captured bundle's canonical run ID and fingerprint across controller
iterations. Reuse acceptance recordings as repeated qualification against a fixed
benchmark; recapture only after PX4 setup, flight profiles, or recording settings
change. Preserve failed controller results and diagnose them before changing
only declared parameters or evidenced implementation errors.

After controller qualification and hover feasibility pass, pin both accepted
producer paths for all hover campaigns. Campaign one uses
`configs/train_press_hover_quad.yaml`: scratch seed 42, 15 broad updates at
3e-5 then up to 100 refinement updates at 1e-6, evaluation every five updates,
and the existing plateau rule. Validation is 100 episodes each at 101/202/303;
the frozen actor's acceptance workload is 100 episodes each at 404/505/606.
Reset mixtures, sensor noise, pooled metrics, and acceptance limits stay fixed.

Each subsequent campaign starts from scratch at seed 42 and requires a recorded
diagnosis and justified training-configuration change. Record its exact resolved
inputs, best validation mean longest-hover duration, acceptance outcome, Trackio
URL/directory, zero-smoothing monitoring, raw metric names, and measured GPU
utilization here. No retry coordinator or comparison framework was added.

Stop immediately on policy acceptance. Otherwise complete at least five training
campaigns, continuing while progress remains credible. Compare each campaign's
best validation mean longest-hover duration with the previous overall best; an
improvement must be at least 0.01 seconds. After at least five campaigns, stop
when five consecutive campaigns fail to improve by that amount. Startup and
driver failures do not count as campaigns. Driver errors pause execution for the
operator and do not trigger shutdown.

At either policy acceptance or the five-stale-campaign stopping condition,
save the final evidence and diagnosis, stop task-owned workers and monitoring,
flush outputs, report the outcome, and shut down the host as requested.

Host verification on 2026-09-12 covers fitting-only search, validation selection,
frozen acceptance settings on both pass and failure, cache/set/digest isolation,
publication rejection for failed or mismatched qualification, feedback signs,
body/world frames, action saturation, and reuse of the production environment
inputs and evaluator. `uv run pre-commit run --all-files` is the complete host
check, including pytest, lint, formatting, type checks, and repository policies.
GPU integration checks and measured runtime results are recorded as the live
workflow completes each stage.

Resume measurements on 2026-09-12: full pre-commit passed before continuing GPU
execution. Iteration eighteen (`20260912T203255Z`) evaluated 96 candidates in
86.8 seconds, averaging 32.6% GPU utilization. No fitting candidate qualified.
The selected hover utilization was 1.04082 and X braking error was 0.005500 m;
the signed trace shows the target travels too far after the stop command.
Fitting and validation motor replay pass. Acceptance hover motor RMSE improves
to 0.008426, below 0.010000, while moderate-motion replay remains just outside
at 0.010828. These diagnostic improvements do not establish acceptance.

Iteration nineteen (`20260912T203725Z`, `alignment-grid-19.yaml`) narrows the
hold scale to X 1.30/1.32/1.34, hold decay to X 0.001/0.003/0.005 Hz, and tests
X motion scale 1.00/1.02/1.04 to reduce excessive stopping distance. It also tests
Y velocity filtering at 1.5/3.0 Hz to reduce the evidenced input lag during
moderate motion. Motion decay is 0.075/0.3 Hz. Y hold scale is 1.15 and its decay
is 0.12 Hz; PID gains and other baseline settings remain fixed. The 108-candidate
grid uses fitting data for search and the unchanged independent qualification
sequence. All prior failed results and the immutable capture remain preserved.

Iteration nineteen evaluated 108 candidates in 88.2 seconds at 34.6% average GPU
utilization. Thirty-seven pass response checks, but none pass hover. The closest
response-passing hover trace reaches X -0.044025 m against -0.043512 m, with
utilization 1.01180. Its braking error is 0.002650 m and combined endpoint error
is 0.004655 m. The failed excursion occurs around 8.7 seconds, rather than being
an initial-state discontinuity. Higher hold scale and faster seeded-bias decay
trade the positive and negative X extremes against one another.

Iteration twenty (`20260912T204055Z`, `alignment-grid-20.yaml`) tests 72
combinations around that tradeoff: X hold scale 1.29/1.30/1.31, X hold decay
0.0035/0.004/0.0045 Hz, X motion scale 1.02/1.04, X steady bias -0.008/-0.010 m/s,
and Y velocity filter 1.5/3.0 Hz. Motion decay is 0.3 Hz; Y hold scale/decay
remain 1.15/0.12 Hz. Baseline PID gains and all quality limits stay unchanged.

Iteration twenty passes ten fitting candidates into validation and freezes
settings fingerprint `7059d3d62e9892720c700f6debac1c5fef689c3845834437274a9187de8a6486`.
Acceptance response passes: X/Y braking errors are 0.002336/0.001648 m and
combined endpoint error is 0.007465 m. Acceptance hover fails with utilization
1.20317: X reaches 0.062753 m and Y 0.059354 m. Moderate physical-state motor
replay also fails at 0.010068 fitting, 0.010070 validation, and 0.012216 acceptance.
The same profile's Y velocity-input RMSE is about 0.041 m/s across all sets.

A fitting-only moderate-profile diagnostic tests X proportional gain
1.8/1.85/1.9, Y filter 3/6 Hz, and X motion scale 1.02/1.04 without changing
producer artifacts. `proportional-filter-diagnosis.json` retains its inputs and
raw metrics. Increasing Y filter bandwidth reduces motor error; X scale 1.04
can exceed 0.010000. At X scale 1.02 and Y filter 6 Hz, X gains 1.85/1.9 give
motor RMSE 0.009253/0.009458, so these gains remain candidates for full replay.

Iteration twenty-one (`20260912T204835Z`, `alignment-grid-21.yaml`) tests 54
combinations: X proportional gain 1.8/1.85/1.9, X hold scale 1.32/1.35/1.38,
Y hold scale 1.2/1.25, and X hold decay 0.0005/0.002/0.0035 Hz. X motion scale
is 1.02, Y filter is 6 Hz, steady bias is [-0.008, -0.014, 0] m/s, motion decay
is 0.3 Hz, and Y hold decay is 0.12 Hz. This iteration changes a declared PID
gain to test stronger recovery while increasing the hold scales; its diagnostic
motor results are not qualification, and all response/hover/replay limits remain
fixed. No controller has been published and no PPO campaign has started.

Iteration twenty-one passes seven fitting candidates into validation, all with
X gain 1.85 or 1.9. Validation selects X gain 1.85, hold scales [1.32, 1.25, 1],
and hold decays [0.0035, 0.12, 0.04] Hz. Acceptance response passes: X/Y braking
errors 0.000878/0.001148 m, endpoint error 0.006532 m. Hover fails only positive
X, reaching 0.055920 m against 0.052156 m (utilization 1.07218). Y maximum is
0.055138 m, within its bound. Acceptance 3D position error is 0.034044 m RMSE,
0.057692 m P95, 0.060284 m P99, and 0.060399 m maximum over all 600 samples.
The matched PX4 response-hover recording has 0.017258 m RMSE and 0.026452 m
maximum. `alignment-21-acceptance-statistics.json` includes each axis and attitude.
Fitting/validation motor replay pass; acceptance hover replay passes at 0.008117,
but moderate-motion replay fails at 0.011679 against 0.010000.

Iteration twenty-two (`20260912T205615Z`, `alignment-grid-22.yaml`) tests 72
combinations. X gain is 1.9/1.95, hold scale 1.35/1.36, and hold decay
0.002/0.003 Hz, extending the fitting-passing recovery region to reduce the
remaining late positive excursion. Y motion scale 1.02/1.04/1.06 and steady bias
-0.006/-0.010/-0.014 m/s revisit the earlier fitting-only state identification
for that axis, whose estimate favored approximately 1.012 and -0.0038 m/s.
This targets the remaining moderate-motion velocity/motor discrepancy without
changing the Y hold mapping. X motion scale/bias remain 1.02/-0.008 m/s;
Y hold scale/decay remain 1.25/0.12 Hz, Y filter 6 Hz, and motion decay 0.3 Hz.
All quality limits remain unchanged. The estimator implementation patch and
both source SHA-256 digests are preserved in the operator directory.

Iteration twenty-two passes 29 fitting candidates into validation. Its frozen
candidate uses X gain 1.95, hold scales [1.36, 1.25, 1], hold decay
[0.002, 0.12, 0.04] Hz, motion scales [1.02, 1.02, 1], and steady bias
[-0.008, -0.014, 0] m/s. Response and hover acceptance both pass. Acceptance
3D position RMSE/P95/P99/maximum are 0.032196/0.054243/0.056665/0.056775 m;
X maximum is 0.045886 m and Y maximum 0.054921 m. Braking X/Y errors are
0.000825/0.000874 m and combined endpoint error is 0.007358 m. The live
integration check passes, with its exact settings and flight fingerprints
recorded in `integration.json`.

Publication correctly rejects iteration twenty-two because acceptance moderate
physical-state motor replay remains 0.011698 against 0.010000; fitting and
validation replay pass. On that acceptance profile, mean velocity-input errors
are [-0.001050, -0.011088, 0.000951] m/s and normalized roll/pitch torque RMSE
is [0.014278, 0.007593]. The moderate profile includes a -0.6 m/s Y command,
so the residual signed Y mapping deserves separate fitting from the now-passing
hover behavior. A fitting-only command-refresh diagnostic is preserved in
`command-refresh-diagnosis.json`; the expected one-tick transmission phase
changes moderate motor RMSE from 0.009294 to 0.008977, without establishing a
complete qualification. The production replay implementation remains unchanged.

Iteration twenty-three (`20260912T210335Z`, `alignment-grid-23.yaml`) retains
all selected iteration-twenty-two parameters except the three declared Y
estimator fields. Its 48 candidates test Y motion scale 0.98/1.00/1.02/1.04,
steady bias -0.002/-0.006/-0.010/-0.014 m/s, and filter bandwidth 6/12/30 Hz.
This isolates the remaining motion-state discrepancy while retaining the
response/hover/integration-passing X settings. Search remains fitting-only,
selection remains validation-only, and frozen acceptance/publication checks
retain every existing limit.

Iteration twenty-three sends 18 fitting candidates to validation but selects
exactly the iteration-twenty-two settings. Response, hover, and live integration
pass again; finalization again rejects moderate acceptance motor RMSE 0.011698.
Lower Y bias (-0.010 m/s) fails validation combined endpoints: errors range
0.011343-0.012966 m across the evaluated scales and filter bandwidths. Their
signed endpoint Y residuals are -0.009302 to -0.011398 m. The Y-only search did
not improve the selected controller.

A fitting-only mechanism diagnostic then suppresses the authored steady velocity
bias while actual velocity is far from the command. It uses the existing smooth
transition on tracking error in addition to the existing command activation.
For the frozen iteration-twenty-two settings, fitting moderate motor RMSE changes
from 0.009509 to 0.007995; approach/braking changes from 0.005423 to 0.005418 and
combined turns from 0.002501 to 0.003355, both below 0.010000. The isolated
in-memory proposal and raw results are preserved in `steady-bias-diagnosis.json`
and `estimator-steady-bias-proposal.patch`; producer artifacts are unchanged.

This second estimator mechanism change is now implemented in NumPy and Torch,
including captured-state seeding. Steady-bias activation is the product of
command activation and one minus the same transition applied to the magnitude
of commanded minus physical velocity, per ENU axis. Hold scaling and seeded-bias
decay retain their previous behavior. No controller fields or artifact schema
were added, and the original PX4 recording definition remains unchanged.
All 39 actuation tests pass, covering matched steady motion, large tracking
errors, signs, componentwise transitions, seeded decay, and sequential parity.
The full check was rerun after formatting and updating the old seed expectation.

Iteration twenty-four (`20260912T211430Z`, `alignment-grid-24.yaml`) tests 108
combinations under this new implementation fingerprint: X bias
-0.008/-0.012/-0.016 m/s, Y bias -0.014/-0.018/-0.022 m/s, Y motion scale
1.00/1.02/1.04, Y filter 6/12 Hz, and full-transition command/tracking-error
threshold 0.06/0.10 m/s. The wider transition is evaluated because tracking-error
activation now also depends on actual velocity. Other settings retain the
iteration-twenty-two selection. Source patch and SHA-256 digests are preserved
in `estimator-steady-motion-bias.patch` and `estimator-steady-motion-implementation.sha256`.

Full pre-commit passes with the steady-motion estimator behavior. Iteration
twenty-four passes hover for all 108 candidates but none passes response. The
selected combined endpoint error is 0.010776 m against 0.010000 m, with signed
residual [-0.009822, -0.001880, -0.004013] m. The narrower 0.06 m/s full-transition
threshold preserves braking behavior; the tested 0.10 m/s band worsens the
response and is not retained for the next grid. Other quality limits are fixed.

Iteration twenty-five (`20260912T211745Z`, `alignment-grid-25.yaml`) tests 54
combinations to correct the remaining X endpoint deficit: X steady bias
-0.020/-0.024/-0.028 m/s, Y bias -0.012/-0.014/-0.016 m/s, Y motion scale
1.00/1.02/1.04, and Y filtering 6/12 Hz. The command/tracking-error transition
is 0.04-0.06 m/s. Gains, hold scales, hold decay, X motion scale, and other
settings remain fixed at the preceding hover-passing values.

Iteration twenty-five is fully accepted and published as
`.build/alignment/x500-px4-v1/20260912T211745Z`. Forty-seven fitting candidates
reach validation; the frozen selection has steady bias [-0.020, -0.014, 0] m/s,
motion scales [1.02, 1.04, 1], and velocity filters [0, 12, 0] Hz. Other settings
are recorded in the immutable controller artifact. All response, hover,
actuator-audit, per-set controller replay, and live integration checks pass.
Moderate physical-state motor RMSE is 0.007656 fitting, 0.007867 validation,
and 0.009357 acceptance, each below 0.010000. Acceptance braking X/Y errors are
0.000021/0.002299 m and combined endpoint error is 0.008213 m.

Accepted full-window acceptance-hover 3D position RMSE/P95/P99/maximum are
0.032203/0.054232/0.056671/0.056788 m. X maximum is 0.045834 m and Y maximum
is 0.054887 m; all pose bounds pass. The complete per-axis statistics are in
`alignment-25-acceptance-statistics.json`. These full-window alignment statistics
are distinct from policy acceptance's successful-hold pooled statistics.

The accepted controller SHA-256 is
`f293d818ef9925d6f6f89bc070e84614dd91f9d9073ad4f50aa2958d7e871c12`.
`configs/train_press_hover_quad.yaml` now pins its exact controller file and
flight bundle `20260912T102640Z`, whose flight-parameter digest remains
`65f02c5e5bfe6d74291329967033510839acad8b5d6254961b10e39e36c73661`.
Host policy-input validation confirms matching lineage and passing controller
quality before the actual press-hover feasibility run. Feasibility uses the
configured gains 1.25/0.2/2.0, seeds 101/202/303, 100 episodes per seed, and the
production 40-second episode / 30-second hold criteria. Its operator log is
`hover-feasibility-01.log`. PPO has not yet started at this checkpoint.

Hover feasibility passes in
`.build/hover-feasibility/20260912T212409565743Z/feasibility.json`: all 300
validation episodes achieve a continuous 30-second hold, with zero collisions,
flight failures, or timeouts. All pooled acceptance statistics pass. Successful-
hold 3D position RMSE is 0.002290 m, P99 0.005477 m, and maximum 0.006177 m;
minimum closest range is 0.425378 m against the 0.4 m minimum. This demonstrates
feasible feedback through the actual press action/reset/evaluation path before
PPO; it is not trained-policy acceptance.

Hover campaign one is `20260912T212725Z`, launched from scratch with seed 42
using the pinned `configs/train_press_hover_quad.yaml` and unchanged initial
15-update broad / up-to-100-update refinement schedule. Its operator log is
`press-hover-campaign-01.log`. Every policy campaign retains the same accepted
controller and immutable flight bundle.

Campaign-one monitoring is available at
`http://127.0.0.1:7860/?project=press-hover&runs=attempt_01&smoothing=0&accordion=hidden`.
Its Trackio database is `.build/training/press-hover/20260912T212725Z/monitoring/press-hover.db`,
served by task-owned container `vv-rl-trackio.20260912T212725Z`. The server's API
confirms the campaign's reset metrics are loaded, with complete metric names and
zero smoothing. The live root training report identifies the active attempt's
report until the completed attempt is folded into the campaign report.

The initial untrained actor records mean longest-hover duration 0.160667 s,
0% success, 79% other flight failure, and 21% timeout over the 300 validation
episodes. This baseline is not a completed training attempt or acceptance.
PPO then starts phase one at learning rate 3e-5, with its first checkpoint after
five updates.

Direct `eval-press` and `train-rl` invocations must pass Make's
`--assume-old=generate-drone-usd` option itself, in addition to the recursive
`MAKEFLAGS` override, to suppress the root target's asset-generation prerequisite.
The initial direct calls reran the text generator; both asset SHA-256 values
still match `source-assets.sha256` and Git reports no asset changes. Subsequent
direct workload calls use the explicit option. No binary assets were modified.

Campaign-one validation after updates 5/10/15 records mean longest hover
0.163833/0.166167/5.406333 s. All three checkpoints have 0% success, 2% other
flight failure, and 98% timeout, compared with 79% flight failure initially.
The first two checkpoints do not meet the 0.01 s minimum improvement, but the
completed broad phase makes substantial hover progress. Refinement therefore
continues from the complete update-15 learner checkpoint at the configured
1e-6 learning rate. Update 20 improves mean longest hover to 6.202667 s, with
the same outcome rates. No final test has run and no trained actor is accepted.

At update 20, rollout collection takes 20.166 s and PPO optimization 0.102 s
per update. Mean GPU utilization is 52.85% during training and approximately
46% during validation, below the 95% diagnostic target. Physics rollout and
the serial fixed-seed evaluation workload dominate measured time; optimizer
work is a small fraction. These measurements do not justify changing the
deterministic evaluation workload or adding redundant work. Trackio retains
the complete metric names and raw values without smoothing.

Refinement checkpoints 25 and 30 improve validation mean longest hover to
7.494833 s and 7.693000 s, respectively. Both retain 0% success and 2% other
flight failure. Update 25 reaches 10.5 s in its longest single episode. These
are qualifying progress improvements, but no checkpoint has completed the
required hold or produced successful-hold acceptance statistics yet.

The pinned policy acceptance position limits, in meters, are:

| Signal | RMSE | P95 absolute | P99 absolute | Maximum absolute |
| --- | ---: | ---: | ---: | ---: |
| ENU X | 0.028193 | 0.048343 | 0.051984 | 0.052156 |
| ENU Y | 0.032319 | 0.054606 | 0.056991 | 0.057216 |
| ENU Z | 0.016957 | 0.030570 | 0.032967 | 0.034952 |
| Horizontal | 0.036456 | 0.060793 | 0.065518 | 0.065811 |
| 3D position | 0.038041 | 0.060956 | 0.065673 | 0.065905 |

These rounded values summarize the exact criteria in `training_report.json`.
All ten pose signals retain their four independently binding statistical
limits. Policy statistics pool the final 24 seconds of successful 30-second
holds; failed or incomplete holds do not supply these statistics. The other
requirements remain at least 95% success, zero collisions and other flight
failures, finite traces, and at least 0.40 m closest range.

Campaign one stops after update 45. The last three validation checkpoints
regress to 7.210333/6.758333/6.417000 s at updates 35/40/45, with 0% success
and 2% other flight failure throughout. Its best is update 30 at 7.693000 s;
no checkpoint qualifies for the reserved test workload. Three successive
regressions, unchanged failures, and no complete hold provide no credible
current path to acceptance under this refinement schedule. The operator
interrupts the campaign before exhausting its update budget.

The campaign report records `interrupted` and retains the active attempt
pointer. The worker enters teardown but exceeds Docker's 60-second stop grace
period, after which the container exits 137. Its last atomic attempt report
still says `training`; the root report and operator record identify the actual
interruption. All ten completed validation records, the update-30 selected
checkpoint, and its actor digest
`8baf571a61d1a487cd097a014a31dfeac3c0263d49ec8f5b404385661e4baa33`
remain preserved. This is an operator stop, not a driver failure. The campaign's
monitoring server is stopped and its database retained.

`policy-campaigns.json` in the operator directory records the completed
campaign, raw validation history, selected actor/checkpoint, interruption, and
cross-campaign progress count. Campaign one establishes the first overall best
at 7.693000 s; consecutive campaigns without a 0.01 s improvement are zero.

Campaign two changes only `rl.training.num_learning_epochs` from 1 to 4 in
`configs/train_press_hover_quad.yaml`. It starts from scratch with seed 42.
The hypothesis is that more optimizer passes over each million-transition
rollout can learn a stronger stabilizing policy during the existing 15 broad
updates before the 1e-6 refinement phase. Campaign one's optimizer consumed
approximately 0.1 s of each roughly 20-second update, so this uses the existing
rollouts more fully at small expected wall-time cost. This is a training
configuration experiment, not an established diagnosis. The controller,
flight bundle, reward, reset mixture, sensor noise, validation/test workloads,
phase schedule, and acceptance thresholds remain fixed.

Campaign two is `20260912T221915Z`, with operator log
`press-hover-campaign-02.log` and monitoring directory
`.build/training/press-hover/20260912T221915Z/monitoring`. Its separate Trackio
database is served by `vv-rl-trackio.20260912T221915Z` at the same local URL
with project `press-hover`, run `attempt_01`, and smoothing zero. The explicit
Make `--assume-old=generate-drone-usd` option suppresses asset generation.
Input validation passes with the same pinned digests. Full pre-commit passes
after updating the existing configuration/report assertions and workflow
documentation to four epochs.

Campaign two reproduces the exact initial actor SHA-256 and every initial
validation metric from campaign one. At update 5 its optimizer takes 0.373 s
against 20.802 s rollout collection; the added optimization has small measured
runtime cost. Update-5 validation remains at 0.159333 s mean longest hover,
but update 10 reaches 7.999500 s, exceeding campaign one's overall best by
0.306500 s. Success remains 0% and other flight failure 2%. This supports
continuing the campaign; it does not establish policy acceptance.

Campaign two's update-15 validation regresses to 4.692500 s, so refinement
restores the complete update-10 learner. Its first five/ten refinement updates
(total attempted updates 20/25) regress to 7.007500/5.943667 s. Both retain
0% success and 2% other flight failure. At the best checkpoint, mean position
utilization is 3.545133 and envelope occupancy 0.207621. After ten refinement
updates these become 3.366493 and 0.156218: average position error improves
while time inside the envelope and continuous hold duration worsen. Attitude
utilization is much smaller than position utilization. The operator stops this
campaign after the repeated regression rather than using its remaining budget.
No checkpoint passes validation and no reserved test runs. The overall best
advances by 0.306500 s to 7.999500 s; the consecutive no-improvement count
remains zero after two completed campaigns.

The next training hypothesis uses the already implemented `boundary_weighted`
hover reward. It retains pose cost weight 0.01 inside the complete measured
envelope and uses the existing 0.03 weight outside it, instead of 0.01
everywhere. This tests whether emphasizing envelope occupancy addresses the
observed mismatch between lower average pose error and shorter continuous
holds. It retains four PPO epochs, scratch seed 42, and the existing phase
schedule. Controller settings, flight evidence, resets, noise, hold duration,
validation/test workloads, and every acceptance threshold remain unchanged.

Campaign two's worker teardown also exceeds a 300-second Docker stop grace;
the container exits 137, while the root report is correctly interrupted and
all six completed validation records and the selected checkpoint are retained.
The worker's last durable report still carries its prior training status.
Its monitor is stopped and its separate database preserved. Longer stop grace
did not resolve this teardown limitation.

Campaign three is `20260912T225516Z`, with operator log
`press-hover-campaign-03.log`. It starts from scratch with seed 42, four PPO
epochs, and `boundary_weighted` hover stabilization. Existing tests are updated
for the selected configuration and all pre-commit checks pass. Host input
validation confirms unchanged controller and flight-parameter digests.

The original local diagnostics below use the positive world-quaternion
representation. They are superseded by the production-quaternion correction
recorded with campaign six below; retained values explain the original
experiment decisions and must not be treated as production-input derivatives.

`policy-local-feedback-diagnosis.json` records an additional read-only actor
diagnostic for each completed campaign's best and last evaluated actor. It uses
the production observation packer at the stationary goal, identity world/body
and contact frames, valid sensors, zero previous action, and no noise. Autograd
computes the exported actor's local velocity-command derivatives; this is not
a trajectory evaluation or acceptance evidence. No actor bytes are modified.

At campaign two's best checkpoint, diagonal position-to-velocity derivatives
are approximately [0.000442, -0.000040, -0.011846] per second, versus -1.25
on each axis for the feasibility feedback. Its velocity feedback derivatives
are also near zero. Nominal normalized actions are approximately
[-0.001297, -0.002656, 0.014316, -0.000341]. Campaign one's best is similarly
weak near the goal. The actors appear to have learned small command offsets
without an effective local stabilizing feedback law. This narrows the training
diagnosis beyond aggregate reward/hold mismatch; stronger policy optimization
or a longer effective learning phase may be needed if the reward experiment
does not improve that behavior. This does not implicate or change the accepted
controller.

Campaign three's first trained validation, after five PPO updates, ends with
300/300 other flight failures, zero successes, and mean longest hover 0.160167 s.
Each seed finishes in approximately 14 wall-clock seconds because every episode
terminates early. The operator stops the campaign immediately. Its initial
actor remains its best validation checkpoint at 0.160667 s. This is the first
completed campaign without a 0.01 s improvement over the 7.999500 s overall
best; three campaigns have completed. No test workload runs.

The existing failure penalty has remained 3.0 while the outside pose cost
increased threefold. For illustration, position plus attitude utilization 3.75
costs 0.1125 per step outside the envelope. At gamma 0.995, maintaining that
cost has an infinite-horizon discounted value of 22.5, much larger than the
terminal failure penalty of 3.0. This is an illustrative reward-scale
calculation, not an estimate of actual policy returns. Together with universal
early failure, it supports testing the incentive to terminate rather than
continue costly flight. Campaign four will increase only the existing
`terminal_failure_weight` from 3.0 to 30.0 in the hover training configuration.
It retains boundary-weighted pose cost, four PPO epochs, scratch seed 42, the
phase schedule, and all controller, evidence, reset, noise, and acceptance
requirements. The approach-stage reward configuration is unchanged.

Campaign three's root report is interrupted and its last completed worker
report/checkpoints remain preserved; worker teardown again exceeds the
60-second Docker stop grace. Its monitoring database is retained and its
server stopped. Campaign four is `20260912T230536Z`, with operator log
`press-hover-campaign-04.log`. Full pre-commit and pinned-input validation pass
with the stronger hover failure penalty before launch.

Campaign four reduces update-5 validation flight failures from campaign three's
100% to 2%, with mean longest hover 0.168000 s. At updates 10 and 15, failures
fall to 2/300 episodes (0.666667%), and mean longest hover reaches 0.180167 s
and 1.891333 s. Success remains zero. The stronger penalty prevents the prior
universal early termination and improves survival, but has not yet produced
sustained holding or surpassed the 7.999500 s overall best. Refinement restores
the complete update-15 learner at the configured 1e-6 learning rate.

Campaign-four refinement checkpoints at total updates 20/25 improve mean
longest hover to 4.657167/5.842833 s, with the same 2/300 flight failures and
zero successes. Both exceed the within-campaign 0.01 s progress minimum,
supporting continued refinement. The overall best across completed campaigns
remains 7.999500 s at this checkpoint.

Campaign-four updates 30/35 reach 6.128333/6.203167 s. Its successive refinement
gains shrink from 2.765833 s to 1.185667 s, 0.285500 s, and 0.074833 s.
There are still zero successes and 2/300 flight failures. The saved update-35
actor's local position-to-velocity diagonal derivatives are approximately
[-0.011269, -0.000143, -0.032640] per second; corrective feedback remains weak,
especially on Y. These diminishing gains and weak feedback do not support a
credible current path to the 30-second hold under the 1e-6 refinement schedule.
The operator stops the campaign, preserving its eight completed evaluations
and update-35 selected actor. A subsequent validation was in progress when
interrupted and is excluded from selection. The container exits 143 without
using the full stop grace; the root report is interrupted, while the last
durable worker report retains its prior training status.

Four campaigns have completed, with two consecutive campaigns without a
qualifying improvement over the 7.999500 s overall best. Campaign five tests
a stronger and longer broad phase: maximum updates 15 to 50 and learning rate
3e-5 to 3e-4. This gives the actor more opportunity to learn state-dependent
feedback before the low-rate refinement phase. Refinement remains up to 100
updates at 1e-6, evaluation remains every five updates, and the plateau minimum
and patience remain 0.01 s and five evaluations. Reward, four PPO epochs,
scratch seed 42, controller, flight bundle, resets, noise, and all acceptance
criteria retain the campaign-four settings. The hover phase lives in the
existing Hydra curriculum; approach-stage phases are unchanged.

Campaign five is `20260912T234752Z`, with operator log
`press-hover-campaign-05.log`. Full pre-commit passes and host input validation
confirms the accepted controller and flight-parameter digests before launch.
The exact configured phase pairs are 50 updates at 0.0003 and 100 at 0.000001,
both evaluated every five updates.

Campaign five's validation at updates 5/10/15/20 records mean longest hover
0.174500/0.182167/0.181500/0.205500 s, with 3/1/2/1 other flight failures
out of 300 and zero successes. Updates 10 and 15 are stale against the
0.174500 s reference; update 20 resets the plateau counter with a qualifying
improvement. These short holds remain far below acceptance or the prior
overall best. The update-15 actor still has weak local corrective feedback.
Trackio shows exploration standard deviation 0.030032/0.029749/0.029270/0.028728
at updates 1/5/10/15, so exploration has not collapsed under the higher learning
rate. The current approximately 0.015 m/s velocity exploration is much smaller
than the 0.0625 m/s position-feedback correction for a 5 cm error in the
feasibility mode. A wider action exploration range is a possible subsequent
training experiment if this longer broad phase does not produce useful holding.

Campaign-five updates 25/30/35/40/45/50 produce mean longest hover
0.966167/2.707333/2.413667/3.067000/0.197000/1.980333 s. All have zero
successes; flight failures are 1/300 except update 50 at 2/300. Refinement
restores the complete update-40 learner, then records 2.123333/2.499500 s after
five/ten refinement updates (total attempted updates 55/60). Both remain below
the selected broad checkpoint and retain 1/300 flight failure. The operator
stops the campaign; its root report is interrupted, and all 13 completed
evaluations and the selected checkpoint are retained. The container exits 143
without exhausting its stop grace; the last durable worker report still has
its prior training status. Five campaigns have completed and three consecutive
campaigns have failed to improve the 7.999500 s overall best by 0.01 s.

The best campaign-five actor has local position feedback diagonal derivatives
approximately [-0.046273, -0.015986, -0.155413] per second. Its local yaw-rate
derivative with respect to yaw is approximately -0.001086 per second, versus
-2.0 for feasibility feedback, while its nominal normalized yaw action is
-0.008928. This uses the same noiseless goal observation and exact saved actor;
it is diagnostic evidence, not a trajectory qualification. Weak yaw feedback
and nonzero command offset can move the actor outside the tight orientation
envelope. Raw observation inputs include small quaternion errors alongside
much larger position-utilization and availability values.

The next experiment enables RSL-RL's existing empirical observation
normalization through a new Hydra-owned `policy.normalize_observations` flag.
The pinned runtime RSL-RL 3.1.2 implementation was inspected directly; the host
test environment uses 3.3.0. The flag controls actor and critic normalization,
defaults to false for other task configurations, and is true for press-hover.
Training statistics remain in complete learner checkpoints. Deterministic
inference and TorchScript export now include the actor normalizer followed by
the actor network; export deep-copies both so later training cannot alter the
selected actor's input mapping. Artifact files, observation/action schema,
controller, flight bundle, resets, sensor noise, and qualification limits are
unchanged. The changed policy configuration starts a new comparison fingerprint.

Host tests cover enabled/disabled configuration, inference/export equivalence,
no normalization updates in evaluation mode, and export independence from later
training-statistic updates. The focused tests and full pre-commit pass. Before
the next full campaign, the existing GPU procedural check verifies the phase
restore and exported-actor evaluation path with normalization enabled; its
results are workflow evidence rather than a training campaign or policy
acceptance, and it does not use reserved test seeds.

The normalization procedural check `20260913T005727Z` failed after initial
validation and before PPO: the GPU telemetry `nvidia-smi` query returned exit
status 255. Its root report records `attempt_failed`; the operator log is
`normalization-gpu-smoke.log`. No normalization learning or phase-restore
result was obtained. The subsequent read-only host GPU query succeeded, but
that does not establish that the runtime failure is resolved. GPU execution
is paused for the user to handle this driver-related error. The failed check
is retained and does not count as a campaign: the totals remain five campaigns,
three consecutive campaigns without a qualifying improvement, and a best
validation mean longest hover of 7.999500 s. Campaign six has not started.
The task-owned monitoring process is stopped and its run-scoped database is
retained. This driver pause does not meet the workflow's shutdown condition.

The user authorized a GPU recheck and training resumption. The host query
succeeded on the RTX 4090 with driver 580.178.04, and procedural check
`20260913T021616Z` starts with the same normalization configuration, accepted
controller, and immutable flight bundle. It retains both phase boundaries with
one update per phase and does not count as a training campaign. Its operator
log is `normalization-gpu-smoke-02.log`; its Trackio database is under that
training run's `monitoring` directory, served locally on port 7860 with zero
smoothing and complete metric names. The earlier failed check is preserved.

Procedural check `20260913T021616Z` completes initial validation, one PPO
update, and exported-actor validation without a GPU telemetry error. It then
fails when restoring the selected initial learner for refinement: RSL-RL 3.1.2
replaces actor and critic normalization standard-deviation buffers under
`torch.inference_mode()` during rollouts, and checkpoint loading outside that
mode cannot overwrite those inference tensors. The complete error is retained
in the run's integration `isaac/logs/train_rl.log`. This is an implementation
failure, not a controller or driver failure, and the procedural check remains
excluded from campaign counts.

Before checkpoint loading, the training adapter clones inference tensors in
the actor and critic normalization buffers into writable tensors. It then uses
the existing complete runner load, including the optimizer, outside inference
mode. The regression test exercises enabled and disabled normalization, learns
normalization statistics inside inference mode, restores exact prior network
and statistics values plus optimizer state and iteration, and performs another
optimizer update. No producer artifact is modified.

Procedural check `20260913T022616Z` confirms the restore fix in Isaac Lab:
initial validation, the first PPO update, exported-actor validation, complete
checkpoint restore, refinement reset, and the next PPO update succeed. The
user asks to stop during final refinement validation. The saved checkpoint
`attempt_01_seed_42/checkpoints/phase_02_model_1.pt` and exported
`evaluated/candidate_002/policy.pt` are preserved; their final validation is
incomplete and cannot establish a completed procedural check or acceptance.
The worker and monitor are stopped at the user's request, and the host remains
on. Focused tests and full pre-commit pass for the checkpoint fix. AGENTS.md
now requires plain URLs for Trackio and similar monitoring links.

Resume with a new immutable procedural-check run using the current Hydra
configuration to finish verification, then launch campaign six from scratch
with training seed 42 and its own Trackio database. Do not resume or externally
initialize from this interrupted learner. The controller and flight bundle stay
pinned. The campaign ledger remains five completed campaigns, three consecutive
campaigns without a qualifying improvement, and best validation mean longest
hover 7.999500 s. No policy has passed acceptance and no reserved acceptance
seeds have been used in these procedural checks.

After commit `c180a12`, the user resumes execution. The RTX 4090 responds
normally with driver 580.178.04, the working tree is clean before launch, and
input validation passes for the pinned controller and flight bundle.
Procedural check `20260913T042458Z` starts from scratch to complete the
normalization verification interrupted in the preceding run. Its operator log
is `normalization-gpu-smoke-04.log`. The worker crashes during Kit startup
before environment setup or PPO with SIGSEGV; the recorded native stack runs
through `getenv`, `XOpenDisplay`, and `omni.platforminfo`. The monitor cannot
start because the worker has not yet created its monitoring directory. This
startup failure is excluded from campaign counts. A fresh procedural check
`20260913T042617Z` starts with the same inputs and configuration, preserving
the failed run and its integration log. Campaign six remains pending
completion of this check.

Procedural check `20260913T042617Z` passes completely in 639.3 seconds:
three full validation workloads, both PPO updates, checkpoint restore,
refinement reset, exported-actor evaluation, and worker teardown complete.
The report status is `procedural_smoke_complete`; no final acceptance test is
run. The normalization integration issue is resolved, while policy learning
performance remains to be established by a full campaign. No GPU telemetry
error recurs in this run. Its monitor is stopped and its database retained.

Campaign six is `20260913T043730Z`, with operator log
`press-hover-campaign-06.log`, scratch seed 42, and empirical actor/critic
observation normalization enabled. Other settings retain campaign five:
4096 environments, 256 rollout steps, four PPO epochs, 50 broad updates at
3e-4 followed by up to 100 refinement updates at 1e-6, evaluation every five
updates, boundary-weighted pose cost, failure penalty 30, and initial action
noise standard deviation 0.03. Controller, flight bundle, recorded resets,
sensor noise, validation/test workloads, and acceptance limits remain pinned.
The run has its own Trackio database under its `monitoring` directory.

Campaign-six validation at updates 5/10/15 gives mean longest hover
0.169500/0.183500/0.195000 s, with 4/1/1 flight failures out of 300 and
zero successes. Update five falls short of the 0.01 s plateau improvement;
updates ten and fifteen each qualify against the campaign reference. Training
at update five averages 55.5% GPU utilization: rollout collection takes
21.23 seconds per update and PPO optimization 0.393 seconds. Simulation
collection remains the measured bottleneck below the 95% utilization target.

The local actor diagnostic requires a correction. Production canonicalizes the
relative quaternion to positive scalar, but its world quaternion near this
reset is [0, 0, 0, -1] in xyzw order. The original diagnostic supplied
[0, 0, 0, 1] for both. These represent the same physical orientation but are
different neural-network inputs, especially after empirical normalization.
The update-15 checkpoint's world-quaternion scalar mean is -0.999929 with
standard deviation 0.000138, confirming the production representation. The
original JSON remains preserved; corrected results for the same saved actors
are in `policy-local-feedback-production-quaternion.json`. No actor or producer
artifact is changed, and this is still a local diagnostic, not an evaluation.

With the production quaternion, campaign five's best actor has local yaw
feedback -0.148658 per second, rather than the original diagnostic's
-0.001086. Campaign six at update 15 reaches -1.984649 per second, near the
feasibility feedback's -2.0. Its position feedback diagonal remains weak at
[-0.022737, -0.053686, -0.047178] per second versus -1.25 per second for
feasibility. The corrected evidence supports a substantial improvement in yaw
response with normalization while position holding remains unresolved.

Campaign-six updates 20/25/30 record mean longest hover
0.206667/0.209667/0.225833 s. Flight failures are 1/1/2 out of 300;
all checkpoints have zero successes, and the maximum individual hold at update
30 is 0.60 s. Despite occasional 0.01 s improvements, this trajectory remains
far from a credible 30-second hold within the campaign. The operator stops it
after update 30 and retains all seven completed validation records and actors.
Its best 0.225833 s does not improve the previous 7.999500 s overall best.
Six campaigns are counted, with four consecutive campaigns without a qualifying
overall improvement. No acceptance test is run.

The next campaign increases only press-hover initial normalized action
exploration standard deviation from 0.03 to 0.10. Translation exploration
therefore rises from 0.015 to 0.05 m/s at the unchanged 0.5 m/s action scale;
yaw-rate exploration changes by the same factor. The position-feedback
feasibility controller requests 0.0625 m/s for a 5 cm error before damping, so
the wider exploration can sample corrections that are rare under the previous
noise. Normalization, phase schedule, reward, seed 42, controller, flight bundle,
resets, sensor noise, validation/test workload, and quality thresholds remain
unchanged. This tests the weak position-learning hypothesis; it does not
establish a learning improvement in advance.

Campaign six's stop exhausts the 60-second container grace and exits 137.
The root report is interrupted; the last durable worker report retains its
training status. Its seven completed evaluations and checkpoint references are
preserved, and its monitor is stopped. Campaign seven is `20260913T051339Z`,
with operator log `press-hover-campaign-07.log` and a separate Trackio database
in that run's `monitoring` directory. Full pre-commit passes before launch.

Campaign-seven validation at updates 5/10/15 gives mean longest hover
0.228000/0.734333/1.992667 s, with zero flight failures and collisions across
all 300 episodes at each checkpoint, and zero successful 30-second holds.
The update-ten maximum individual hold is 2.30 s. Wider exploration produces
substantial early holding progress over campaign six, so the campaign continues.

Campaign-seven updates 20/25/30/35/40/45 record mean longest hover
2.190000/2.586833/3.140167/3.959500/4.457667/5.860833 s. All have zero
successes; flight failures remain zero through update 40, then 1/300 at update
45. Update 50 produces 298/300 successful holds and mean longest hover
29.803333 s. It fails validation on two flight failures, minimum closest range
0.398951530 m against 0.400000000 m, and vertical position RMSE
0.019271377 m against 0.016957235 m. All other checks pass, including pooled
3D position RMSE 0.023463658 m, P95 0.027722006 m, P99 0.029335901 m,
and maximum 0.033703472 m. These are validation measurements, not final-test
acceptance. The complete update-50 learner is restored for refinement at
learning rate 1e-6. The actor is not published and reserved test seeds remain
unused. This campaign has already exceeded the previous 7.999500 s overall
best, so its eventual completed-campaign progress accounting will reset the
consecutive-no-improvement count.

After five refinement updates (total update 55), campaign seven records
295/300 successes and mean longest hover 29.507167 s. Vertical RMSE and all
other pose statistics pass, but five flight failures and minimum closest range
0.399021149 m fail validation. After ten refinement updates (total 60),
success falls to 291/300 and mean longest hover 29.111000 s. Minimum range
0.399353027 m and nine flight failures fail, as do X RMSE 0.039090470 m,
horizontal RMSE 0.040435027 m, and total position RMSE 0.041154815 m.
The operator stops this second consecutive refinement regression and preserves
all 13 completed validation records. No reserved acceptance workload runs.
The best update-50 validation result raises the overall mean-longest-hover best
to 29.803333 s, an improvement of 21.803833 s, so the completed-campaign stale
counter resets to zero after seven campaigns. Policy acceptance is still false.

A read-only diagnostic isolates normalization-statistic drift from weight
updates using the saved update-50 and update-60 actors. At 27 stationary
positions around the goal with the production quaternion, total normalized
X-action change RMS is 0.015882; substituting only update-60 normalization
statistics into a temporary in-memory copy of the update-50 actor produces
0.014971 RMS change. The remaining weight-change contribution measured with
update-60 statistics is 0.001046 RMS. At the exact goal, X action moves from
0.010952 to 0.027057; the statistics-only substitution gives 0.026177.
The diagnostic file is `campaign-07-normalization-drift-diagnosis.json`.
It is not a trajectory qualification, and no saved actor is changed or used
as external training initialization.

The next campaign changes only the hover refinement learning rate from 1e-6
to 3e-5. The evidence suggests that the previous rate cannot compensate for
the moving input normalization during refinement. The higher rate gives the
actor more capacity to adjust while remaining ten times below the broad-phase
3e-4 rate. Broad-phase length, exploration 0.10, normalization, reward,
seed 42, controller, flight bundle, resets, sensor noise, validation and test
workloads, and all acceptance limits retain campaign-seven values.

Campaign seven's root report is interrupted; the container exits 143 promptly,
and its last durable worker report retains training status with all 13
completed evaluations. Its monitor is stopped. Campaign eight is
`20260913T061756Z`, with operator log `press-hover-campaign-08.log` and its own
Trackio database under the run's `monitoring` directory. Full pre-commit passes
before launch. It starts from scratch with seed 42; no campaign-seven learner
or temporary diagnostic actor initializes it.

Campaign eight reproduces campaign seven exactly through broad update 50: the
update-50 actor digest and every validation metric match. It records 298/300
successes and mean longest hover 29.803333 s, with the same range, flight-failure,
and vertical-RMSE failures listed above. The complete update-50 learner is
restored and refinement starts at 3e-5. This verifies that the experiment changes
refinement behavior only; the broad-phase starting point is identical.

Campaign-eight update 55 regresses to 1/300 successes, six flight failures,
and mean longest hover 17.027167 s, with several horizontal position failures.
Update 60 recovers to 298/300 successes and 29.803000 s. Its remaining failed
checks are minimum range 0.399680614 m, two flight failures, X RMSE
0.033113616 m against 0.028192865 m, and horizontal RMSE 0.036984307 m
against 0.036455555 m. Vertical RMSE 0.007213045 m and total position RMSE
0.037681122 m pass. Refinement continues because the second checkpoint
recovers long holds; neither checkpoint is validation-eligible.

Update 65 regresses again to 53/300 successes, four flight failures, and
mean longest hover 16.033167 s. X/Y/horizontal/total position RMSE and total
position P95 fail. The operator stops campaign eight after these three
refinement checkpoints: the best remains broad update 50 at 29.803333 s,
with no qualifying overall improvement. Eight campaigns are counted, and the
consecutive-no-improvement count becomes one. No acceptance test runs.

The read-only `campaign-08-normalization-drift-diagnosis.json` uses the same
27 production-quaternion goal-neighborhood observations as the campaign-seven
diagnostic. At update 65, statistics-only X-action change RMS is 0.020732;
weight changes measured with current statistics contribute Y/Z RMS
0.028137/0.054807, compared with 0.000228/0.004625 at campaign-seven
update 60. These local probes support both continuing normalization drift and
substantially larger weight changes at 3e-5; they are not trajectory evidence.
No producer actor is modified.

Campaign nine will change only refinement learning rate from 3e-5 to 1e-5,
between the weak adaptation at 1e-6 and oscillatory refinement at 3e-5. The
expected impact is smaller refinement weight changes while allowing more
adaptation than campaign seven. Seed 42, broad schedule, exploration 0.10,
normalization, reward, controller, flight bundle, resets, sensor noise, and all
validation/test acceptance rules remain unchanged. It starts from scratch.

Campaign eight exhausts the 60-second Docker stop grace and exits 137; its
root report is interrupted and all 14 completed worker evaluations remain
preserved. Its monitor is stopped. Full pre-commit passes before campaign
nine, `20260913T072658Z`, whose operator log is `press-hover-campaign-09.log`
and whose run-scoped Trackio database is in its `monitoring` directory.
The driver check succeeds immediately before launch.

Campaign nine reproduces every broad-phase actor digest and every validation
metric through update 50 exactly. The starting refinement result is again
298/300 successes, mean longest hover 29.803333 s, and the same minimum-range,
flight-failure, and vertical-RMSE failures. The complete update-50 learner is
restored and refinement begins at 1e-5. Representative Trackio readings near
update 20 show training GPU utilization 53.2%, validation 46.5%, rollout
collection 20.77 s/update, and PPO optimization 0.37 s/update. Simulation
collection remains the limiting factor below the 95% utilization target.

Campaign-nine update 55 regresses to zero successes, four flight failures,
and mean longest hover 17.414000 s; successful-hold pose statistics are
unavailable. Update 60 recovers to 295/300 successes, two flight failures,
three timeouts, and mean longest hover 29.751333 s. Failed checks are minimum
range 0.399686337 m, flight failures, X RMSE 0.043377382 m, X P95
0.050601614 m, horizontal RMSE 0.045146378 m, and total position RMSE
0.048119309 m. The recovery justifies observing one more checkpoint, but the
intermediate learning rate has not resolved horizontal drift or qualification.

Campaign-nine update 65 regresses to zero successes, four flight failures,
and mean longest hover 18.169333 s. Successful-hold pose statistics are
unavailable. The operator stops it after three refinement checkpoints; all
14 completed evaluations are retained. Docker exhausts the 60-second grace
and exits 137, the root report is interrupted, and the monitor is stopped.
The best remains 29.803333 s at broad update 50. Nine campaigns are counted,
with two consecutive campaigns without a qualifying improvement and no
reserved acceptance test.

`campaign-09-normalization-drift-diagnosis.json` repeats the read-only
27-position diagnostic with production quaternion. At update 65, total
X-action change RMS is 0.017833 and the statistics-only change is 0.021323;
weight changes with current statistics contribute 0.005063 in X and
0.068715 in Z. Across refinement rates 1e-6, 3e-5, and 1e-5, input-statistic
drift persists while larger weight updates do not consistently compensate.
These local diagnostics support an experiment freezing the input mapping;
they do not qualify any actor.

The next experiment adds Hydra phase setting
`update_observation_normalization`. The hover broad phase keeps it true and
refinement sets it false. After restoring the complete selected learner, the
adapter sets RSL-RL empirical actor/critic normalizers to stop updating via
their existing sample-limit control. The means, variances, standard deviations,
counts, weights, and optimizer state are preserved at the boundary. Unlike
setting evaluation mode, the sample limit persists when RSL-RL re-enters
training mode for each chunk. Other phases explicitly keep updates enabled.
The setting is reapplied from the declared phase after checkpoint restore.

Expected impact: refinement optimizes weights against a fixed observation
mapping, removing the measured statistics-driven command shift. Learning
rate remains 1e-5 and every other campaign-nine setting remains unchanged.
This changes policy training only; controller settings, immutable flight
evidence, reset sampling, sensor noise, action interface, evaluation criteria,
and thresholds remain fixed. A focused test covers enabled/disabled
normalization, frozen buffers after training-mode re-entry, continuing optimizer
updates, and subsequent phase re-enabling. A fresh GPU procedural check will
verify the complete restore/refinement flow before the next counted campaign.

Full pre-commit passes for the phase-controlled normalization change. GPU
procedural run `20260913T083641Z` completes successfully in 621.3 s with
three complete validation workloads and two PPO updates. It restores
`phase_01_model_1.pt`, applies frozen normalization, and saves
`phase_02_model_2.pt`. All eight actor/critic normalization buffers are
exactly equal across the refinement update, while actor and critic weights
change and optimizer steps advance from 32 to 64. Exact evidence is in
`normalization-freeze-gpu-verification.json`; the log is
`normalization-freeze-gpu-smoke.log`. Teardown succeeds, its monitor is stopped,
and no reserved test runs. This procedural check is excluded from campaign
and no-progress counts. Campaign ten will start from scratch using the same
seed 42 and campaign-nine settings, with only refinement normalization updates
disabled.

Campaign ten is `20260913T084837Z`, with operator log
`press-hover-campaign-10.log` and a separate Trackio database in its run
`monitoring` directory. Full pre-commit and the driver check pass immediately
before launch. No prior campaign learner initializes this scratch run.

Campaign ten starts successfully. Its initial actor tensors and executable
TorchScript code are exactly equal to campaign nine, and every initial
validation metric matches. The serialized actor digest differs: ZIP-member
comparison isolates the differences to debug metadata and the serialization
identifier after the adapter source change. Raw artifact digests remain
recorded; no producer bytes are rewritten to force a match.

Campaign ten reproduces every completed broad-phase validation metric through
update 50 exactly. Direct comparison of the initial, update-25, and update-50
actors confirms equal tensors; serialization debug metadata accounts for the
different raw digests. Update 50 again records 298/300 successes and mean
longest hover 29.803333 s with the same three qualification failures. The
complete update-50 learner is restored and actor/critic normalization updates
are disabled before the refinement reset. Representative readings near update
20 are training GPU utilization 54.3%, rollout collection 20.86 s/update,
and PPO optimization 0.37 s/update.

Campaign-ten update 55 records 154/300 successes, two flight failures, and
mean longest hover 26.852333 s. This improves on campaign-nine update 55,
but success rate, minimum range 0.399165154 m, flight failures, X RMSE
0.042058837 m, X P95 0.051360238 m, horizontal RMSE 0.048651443 m,
and total position RMSE 0.048967229 m fail. Direct learner-checkpoint
comparison confirms all eight normalization buffers remain exactly fixed
from update 50 to 55, retaining 52,428,800 samples per normalizer; evidence
is in `campaign-10-normalization-freeze-verification.json`.

`campaign-10-local-feedback-diagnosis.json` shows that even with fixed
normalization, nominal X action grows from 0.010952 to 0.022908 and local
X position feedback weakens from -0.314977 to -0.264607 per second. This
limits the previous diagnosis: normalization drift contributed to changes, but
refinement weight updates independently produce command bias. The local probe
uses the production quaternion and remains separate from trajectory evidence.

Update 60 improves holding to 227/300 successes and 29.573667 s, with three
flight failures and 70 timeouts. Minimum range 0.399518013 m, success rate,
flight failures, X RMSE 0.042349803 m, X P95 0.050430745 m, horizontal
RMSE 0.044700310 m, and total position RMSE 0.046914212 m fail. The
improving hold duration justifies continuing to the next validation checkpoint.

Campaign-ten update 65 regresses to zero successes, five flight failures,
and mean longest hover 18.311000 s. Successful-hold statistics are unavailable.
The operator stops after the third refinement checkpoint and retains all
14 completed evaluations. The best remains broad update 50 at 29.803333 s,
so ten completed campaigns have three consecutive campaigns without a
qualifying overall improvement. No reserved acceptance workload runs.

The deterministic validation action-effort penalty decreases in magnitude
from 0.000593 at update 50 to 0.000392 at update 65, while stabilization cost
increases from 0.007791 to 0.018110 and total cost worsens from 0.008701
to 0.019139. Fixed normalization has not prevented weak horizontal correction
or command bias. These observations motivate testing delayed credit assignment;
they do not prove that the action penalty or GAE setting caused failure. Exact
reward means and decay calculations are in
`campaign-10-credit-assignment-diagnosis.json`.

Campaign eleven changes only GAE lambda from 0.95 to 0.99. The pinned RSL-RL
3.1.2 return implementation propagates advantages with gamma times lambda.
At gamma 0.995 and 20 Hz, the nonterminal decay time to 1/e increases from
0.888 to 3.319 s; the weight at three seconds increases from 0.034104 to
0.405039. This gives delayed position effects more weight when estimating
action advantages. Critic bootstrapping also contributes, so the decay is not
a hard planning horizon. The expected benefit is stronger credit for delayed
position correction; the tradeoff is higher advantage-estimate variance.
Refinement normalization remains frozen, learning rates remain 3e-4/1e-5,
and reward weights, seed 42, controller, flight bundle, reset distribution,
sensor noise, action limits, validation/test seeds, and quality thresholds
remain unchanged. The next campaign starts from scratch.

Campaign ten exhausts the 60-second Docker stop grace and exits 137; its
root report is interrupted and all 14 worker evaluations remain preserved.
Its monitor is stopped. Campaign eleven is `20260913T095657Z`, with log
`press-hover-campaign-11.log` and a separate Trackio database in its run
`monitoring` directory. Full pre-commit and the driver check pass immediately
before this scratch launch.

Campaign eleven reproduces the initial actor digest and every baseline metric
exactly. With lambda 0.99, update 5 reaches mean longest hover 0.448167 s
with one flight failure out of 300, versus 0.228000 s and zero failures
previously. Update 10 reaches 3.481667 s with zero failures, versus
0.734333 s previously. Both checkpoints have zero successful 30-second holds.
The early increase supports continuing this configuration; it does not qualify
the policy or establish the proposed causal explanation.

Campaign-eleven updates 15/20 reach mean longest hover 3.785667/7.059500 s
with zero flight failures and zero successes. Update 25 passes every validation
check: 300/300 successful 30-second holds, zero failures/collisions/timeouts,
and mean longest hover 30.000000 s. Minimum closest range is 0.410164833 m.
Pooled total position error is RMSE 0.017226349 m, P95 0.019054704 m,
P99 0.019596363 m, and maximum 0.020412661 m. Exact checks and the frozen
actor digest are preserved in `campaign-11-first-passing-validation.json`.
This is validation evidence only; reserved test seeds remain unused. The
existing schedule and plateau stopping continue, while selection retains this
validation-eligible checkpoint. Representative update-20 telemetry is training
GPU utilization 53.6%, rollout collection 20.47 s/update, and PPO optimization
0.37 s/update.

Campaign-eleven updates 30/35/40/45/50 each pass every validation check,
with 300/300 successes, zero failures/collisions/timeouts, and mean longest
hover 30.000000 s. The broad phase reaches its five-evaluation plateau at
update 50. Selection preserves the first exact tie, so refinement restores
the complete update-25 learner and freezes its normalization statistics.
Subsequent monitoring steps count all attempted updates, while checkpoint
iteration numbers follow the restored branch.

Campaign-eleven refinement at total updates 55/60 (restored learner
iterations 30/35) passes every validation check, with 300/300 successes and
mean longest hover 30 s. At total 65 all episodes still succeed, but X RMSE
0.030017259 m fails the 0.028192865 m limit. At total 70 all episodes
succeed, but X RMSE 0.035726169 m, horizontal RMSE 0.041721162 m, and
total position RMSE 0.043037370 m fail. These later checkpoints cannot
replace the earlier validation-passing actor. The configured plateau reaches
four stale refinement evaluations; the first passing update-25 actor remains
selected under the existing exact-tie rule.

At total update 75, every validation episode succeeds, but X RMSE
0.035375462 m, horizontal RMSE 0.041650537 m, and total position RMSE
0.043747771 m fail. Refinement reaches its five-evaluation plateau and
training ends after 75 attempted updates. Selection freezes broad update 25,
`phase_01_model_25.pt`, actor SHA-256
`14e8651141b58172b53b80193916fd11c72c5651fe922797bca5cda99dbf37ac`.
The worker tears down cleanly, and the first reserved acceptance workload
starts with seeds 404/505/606, 100 episodes each. No later failed checkpoint
is substituted for the selected actor.

## Final accepted outcome

Campaign eleven completes successfully in 4324.5 s. The selected update-25
actor passes its first fresh reserved test on seeds 404/505/606 with 100 episodes
each: 300/300 successful 30-second holds, zero collisions, zero flight failures,
zero timeouts, and all 45 acceptance checks passing. Test minimum closest range
is 0.413337708 m. Pooled 3D position error is RMSE 0.017243223 m,
P95 0.019075497 m, P99 0.019622781 m, and maximum 0.020354448 m.
Vertical RMSE is 0.015679585 m against 0.016957235 m, the closest position
limit. These are the unchanged successful-hold pooled metrics and thresholds.

The exact selected, tested, and published actor digests match. Both
`.build/training/press-hover/latest` and `best` resolve to `20260913T095657Z`.
The accepted controller and flight-parameter payload digests are rechecked and
unchanged. Eleven training campaigns are counted; startup/procedural checks are
excluded. The overall best mean longest hover reaches 30 s, an improvement of
0.196667 s over the previous best, and the stale-campaign counter resets to zero.
The workflow stops on policy acceptance without another training campaign.

The decisive campaign-eleven change is GAE lambda 0.99: every other setting is
retained from campaign ten, and the initial actor and validation metrics match.
The first passing actor occurs in the broad phase, before normalization freezing.
This establishes the measured result for training seed 42, not repeatability
across training initializations. Later refinement can degrade position accuracy;
the unchanged validation-selection rule protects the earlier passing actor.

Final report files are `final_report.json` and `final_report.md` under
`.build/flight-to-policy/20260912T102640Z/`. They reference all exact producer
inputs, the campaign ledger, acceptance metrics, and remaining diagnosis.
The training worker and coordinator exit normally, and the task-owned Trackio
monitor is stopped after flushing results. Its database remains under
`.build/training/press-hover/20260913T095657Z/monitoring`. The monitor URL is
inactive after shutdown:

http://127.0.0.1:7860/?project=press-hover&runs=attempt_01&smoothing=0&accordion=hidden

Full pre-commit checks and GPU workflow checks pass for the implementation;
final report verification and host-shutdown status are retained in
`final_report.json`.
