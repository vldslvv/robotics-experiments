# Flight parameters

The flight-parameters workflow measures task-neutral vehicle behavior and
publishes the one strict evidence contract shared by controller alignment and
policy training. It has no switch, wall, reward, policy-observation, or
consumer-specific fields.

## Schema-1 contract

`robotics_experiments.flight_parameters` owns this flow:

`flight-parameters config -> complete behavior traces -> FlightParameters`

Schema 1 is the only supported schema. There are no legacy envelope schemas,
migrations, aliases, or compatibility loaders. Its configured behavior kinds
are:

- `hover`: repeated verified PX4 position holds used for general hover bounds
  and distribution statistics;
- `controller-response`: lossless command/response evidence covering the fit,
  validation, stability, velocity, braking, yaw, and CTBR profiles currently
  required by controller alignment.

The production config is `configs/flight/holybro_x500_v2_hover.yaml`. Its
`sets.fit`, `sets.validation`, and `sets.acceptance` sections each contain
positive integer `hover_count` and `response_count` values. `starting_seed` owns
the first seed. Each flight starts fresh Isaac and PX4 processes. The default
configuration is:

```yaml
starting_seed: 1001
sets:
  fit: {hover_count: 30, response_count: 1}
  validation: {hover_count: 10, response_count: 1}
  acceptance: {hover_count: 10, response_count: 1}
```

Seeds follow `starting_seed + flight_index`, with a zero-based global index.
All hover flights precede all response flights; each kind uses fit, validation,
then acceptance order. The default hover seeds are 1001 through 1050, followed
by response seeds 1051 through 1053. Assignments happen before dispatch and stay
the same on retry or resume, regardless of worker completion order. Changing
counts can shift later seeds and requires a new campaign.

| Set | Hover flights | Complete controller-response flights |
| --- | ---: | ---: |
| Fitting (`fit`) | 30 | 1 |
| Validation | 10 | 1 |
| Acceptance | 10 | 1 |

The same recorder, behavior configuration, sampling cadence, and verification
run for every set. Each response flight includes the entire configured suite:
translation, braking, yaw, collective thrust, and body rates, including its
existing fit/validation/stability profile labels. Dataset membership is recorded
as `set_name`, separate from those internal profile labels. Membership is
assigned before capture to whole flights, never to randomly split samples.
The seed initializes the simulator's existing NumPy random stream before fleet
setup; no new noise or randomization model is introduced. Seeds record the
requested random stream, not a guarantee of bitwise simulator repeatability.

## Campaign flow

The host runner is a sequence of functions with explicit inputs and results:

```text
campaign config + experiment/response definitions
  -> resolve immutable per-flight definitions and pin capture inputs
  -> validate completed-flight checkpoints
  -> dispatch unfinished hover batches, then response batches

for each flight:
  definition -> capture -> verify and extract complete traces
  -> metadata and hashes -> gzip and verify -> completion checkpoint
  -> remove verified raw duplicates -> return completed-flight record

after all configured flights:
  validate completeness -> derive parameters -> aggregate statistics
  -> final stability analysis -> campaign/report metadata
  -> validate manifest -> finalize immutable bundle -> publish latest
```

A flight definition identifies its kind, index, set and seed. The campaign
configuration references the physical experiment (vehicle, scene, PX4 and
sensor settings), hover schedule, and complete response command suite.
Derived-output margins and policy warmup remain campaign settings. Processing
and statistical settings are separate from the flight instructions. No second
authored flight-definition format is maintained.

The runner pins actual source-file hashes, including uncommitted capture code,
container image IDs, host package versions, configuration and resolved flight
identities in `capture_inputs.json`. The configuration fingerprint includes
`starting_seed` and every flight count. It checks those inputs before each batch
and after collection. Source/runtime changes require a new capture campaign;
completed evidence is never silently combined with a changed method.

Each worker finishes verification, extraction, compression and checkpointing
before accepting another flight. Statistical analysis is never called from the
per-flight pipeline. Worker count changes scheduling, not the command sequence.
Each batch contains one flight kind, keeping the long response flights together
instead of leaving response work queued behind a mixed batch.

### Lossless storage and resume

`configs/flight/processing/default.yaml` owns gzip level and disk budgets;
`FLIGHT_PARAMETERS_PROCESSING_CONFIG` selects it. Gzip stores the exact original
ULog and bridge bytes. JSON traces remain complete and directly readable.
The closed harness bridge log is retained, including teardown records after the
standalone recorder's initial copy; its recorded prefix must match that copy.

Compression writes a temporary file, verifies the original-content SHA-256 by
streaming decompression, then atomically installs the archive. A flight's
`complete.json` records its identity, input digest, retained-file sizes, stored
and original-content hashes, and capture/processing times. Only then are the
uncompressed logs and matching copies in that flight's private harness directory
removed. Other logs and failed attempts remain available. Archive readers stream
decompression, including through alignment's relative symlinks; no persistent
uncompressed cache is needed. The manifest hashes stored bytes, while ULog and
bridge provenance hashes identify original contents. Plain and gzip are storage
encodings of the same evidence contract; existing bundles are not rewritten.

`capture.json` checkpoints successful raw capture separately. An interrupted
postprocessing stage reuses that capture. A verified `complete.json` skips both
capture and postprocessing for hover and response. A capture without a success
receipt is preserved under `failed/` before retrying on explicit campaign
resume; the campaign reports these failed/incomplete capture attempts. A corrupt
completed checkpoint fails validation instead of silently recapturing its flight.

Before each batch the runner estimates space for all remaining archives, raw
files and compression staging for active workers, and the configured free-space
reserve. Completed-flight sizes refine the initial estimate upward. Insufficient
space stops dispatch without deleting evidence; resume is possible after space
is available. This is a capacity estimate, not a bound on unexpectedly large
failure logs. Console and per-flight metadata expose real storage and timings.

## Hover collection and statistics

Each hover repetition starts fresh PX4 SITL and Isaac processes. The collector
sends only arm and takeoff, decodes the completed ULog inside
`flight_parameters`, and proves the external commands, `AUTO_TAKEOFF ->
AUTO_LOITER` transition, automatic position control, armed and airborne state,
fresh reports, and unchanged held position/yaw goal. A routine label, elapsed
time, or low velocity is not accepted as proof of hover.

The raw hold has a fixed schedule: the configured transition allowance, then
the settling interval, then exactly 30 seconds at 20 Hz. Verification requires
AUTO_LOITER entry before that allowance expires and selects the scheduled
measurement window regardless of the actual transition time. With the production
configuration, measurement starts 21 seconds after raw capture begins and has
at least six seconds of verified loiter settling. A late transition fails;
verification never shifts the window to accommodate it. All 600 samples in each
trace are preserved. The isolated capture stops when this complete fixed schedule
has elapsed, rather than waiting for the routine's separate speed-based hover
hold. An existing driver failure still requires recovery and fails the capture;
the completed ULog remains the authority for hover verification. The published
hover bounds use every sample from the 30 fitting repetitions, expanded by the authored
position and attitude margins. This full window is the controller-alignment
qualification input and must not be shortened or resampled.

For every trace, schema 1 also records raw minimum, raw maximum, maximum
absolute error, RMSE, absolute p95, and absolute p99 for:

- ENU X/Y/Z position error;
- horizontal and three-dimensional position-error norms;
- relative roll/pitch/yaw error;
- tilt and quaternion-geodesic orientation error.

Both full-window statistics and policy-window statistics are retained. Policy
qualification discards the configured first six seconds and evaluates the
final 24 seconds. `hover.acceptance_limits` is the conservative per-statistic
limit across the 30 fitting policy windows plus the authored unit-appropriate margin.
Maximum absolute error, RMSE, p95, and p99 are all binding policy gates.

Raw and verified hover traces retain `linear_velocity_enu_mps` and
`angular_velocity_enu_radps` alongside pose and timestamps. All four state
components come from one simulator snapshot. The recorder rotates measured
body-FLU angular velocity into world ENU using that snapshot's attitude.
Verification selects the same sample indices for every array. Missing fields,
unequal lengths, NaN, or infinity raise an error; velocity is never reconstructed
from pose differences. Existing artifacts without these fields and set/seed
metadata are unsupported; capture a new bundle without modifying old bundles.

Statistics are retained for all 50 flights. Bounds, acceptance limits, and the
policy training reset population use fitting flights only. Validation and
acceptance recordings do not influence those derived values.

## Controller-response collection

Controller-response capture executes the complete configured profile suite at
the exact command cadence and retains the original
`flight_parameters_response.json`, completed
ULog, and bridge log for each repetition. The trace is augmented from its ULog
with the complete controller-boundary state before the flight-parameters bundle
is finalized. No field, sample, command, timestamp, fit/validation split, or
controller-boundary value is reduced in the artifact.
Each response profile reserves the full configured preconditioning timeout
before recording begins. The configured continuous speed/rate/tilt stability
hold must be satisfied at that fixed deadline; otherwise capture fails. Early
settling never advances the next profile, and late settling never shifts its
measurement window. The response routine timeout must cover every profile's
complete preconditioning interval and command sequence.
The response header records `controller_state_method` as
`px4-ulog-controller-boundary-v2`. This method reconstructs the velocity
controller's filtered derivative from full-rate estimator velocity and its
logged filter parameters. It does not use EKF acceleration as the controller
derivative. Alignment requires this capture method; older recordings must be
recaptured without modifying their producer artifacts.

Alignment preparation resolves one immutable `FlightParametersBundle`, pins it
inside the alignment run, selects one complete response flight from each of the three sets, and
validates before fitting that the response has:

- every required profile exactly once and in authored order;
- exact family, control mode, metric, split, and axis identity;
- the exact command sequence, duration-derived sample count, and cadence;
- complete samples and valid controller-boundary state;
- a matching definition fingerprint.

Cadence validation compares every retained timestamp with the ideal command
schedule while allowing up to half a command period for bridge-tick
quantization and one adjacent-tick sample at a command-duration boundary. It
derives the expected command at each actual timestamp, admitting the runtime's
alternating adjacent-tick schedule while rejecting accumulated drift or an
incorrect command sequence. The timestamps and response samples are not
resampled or reduced.

It then exposes the reference JSON, ULog, and bridge log through relative
symlinks with the filenames expected by the existing alignment workflow. The
immutable producer files remain in their timestamped flight-parameters bundle;
alignment neither copies nor mutates them. Fitting retains the root filenames;
validation and acceptance each have their own subdirectory with the same names.
All three inputs are validated before creating the run directory.

Alignment searches only on the fitting response recording, selects fitting-
qualified candidates on validation, then evaluates the frozen settings on
acceptance. Existing internal fit/validation/stability profile labels are
preserved within every recording. All three recordings receive actuator audit
and controller-boundary replay. Response and hover thresholds stay unchanged;
independent hover recordings remain retained statistics, while bounds and policy
reset samples still use fitting hover recordings only. Subsequent controller
iterations reuse this immutable bundle as a fixed qualification benchmark.

## Artifact publication

Run the complete campaign from the repository root:

```bash
make measure-flight-parameters
```

Each run is immutable and timestamped:

```text
.build/flight-parameters/<flight-parameters-id>/<run-id>/
  flight_parameters.json
  campaign.json
  capture_inputs.json
  stability.json
  evidence/trial-001/raw_trace.json
  evidence/trial-001/hover_verification.json
  evidence/trial-001/trace.json
  evidence/trial-001/capture.json
  evidence/trial-001/complete.json
  evidence/trial-001/bridge.log.gz
  evidence/trial-001/flight.ulg.gz
  ...
  response/repetition-001/capture.json
  response/repetition-001/complete.json
  response/repetition-001/raw_response.json
  response/repetition-001/flight_parameters_response.json
  response/repetition-001/flight_parameters.ulg.gz
  response/repetition-001/flight_parameters_bridge.log.gz
  manifest.json
```

An explicit `FLIGHT_PARAMETERS_RUN_ID` may continue an unpublished partial run;
complete verified hover and response flights are retained and must match their
checkpointed inputs and hashes. Once
`flight_parameters.json` exists, resuming retries only final analysis and
publication. A finalized run cannot be resumed or modified. Parameter and manifest writes
are atomic; a failed manifest validation cannot mark a partial run finalized.

`flight_parameters.json` has artifact kind
`robotics-experiments-flight-parameters`, schema version 1, `created_at_utc`,
the complete hover and response records, source/configuration fingerprints, an
ordered evidence fingerprint, and a canonical artifact fingerprint. Nested
JSON evidence kinds use the same `robotics-experiments-flight-parameters-*`
prefix. The exact manifest inventories every file by relative path, size, and
SHA-256.

Publication succeeds only after every configured hover and response repetition
matches its configured set and seed, and every manifest hash validates. Raw
and verified hover metadata must agree with the statistics record; response
metadata must agree with its recorded JSON. `execution.worker_count` limits
concurrent independent flights for both behavior kinds. It atomically advances the relative symlink
`.build/flight-parameters/<flight-parameters-id>/latest` to that immutable run.
There is no `best` label because these are measured inputs, not ranked
candidates. `latest` means complete evidence, not statistical qualification.
A qualified baseline is a separate explicit designation.

Alignment and RL record the exact artifact fingerprint. They never silently
switch to a newer `latest`: alignment pins the resolved run during preparation,
records its canonical `run_id`, and training rejects a flight-parameters
artifact whose `run_id` or fingerprint does not match its controller
qualification.

## Stability analysis

Stability is a read-only final stage inside `flight_parameters`. The campaign
collects and verifies all hover and response flights, derives and writes
`flight_parameters.json`, analyzes that completed evidence into `stability.json`,
then finalizes the manifest and publishes. Capture callbacks, sensor updates,
controller execution, and parameter derivation do not call the analyzer.
The primary schema and downstream alignment/training checks are unchanged.
A reported stability failure does not prevent ordinary evidence publication.

`configs/flight/stability/default.yaml` owns analysis settings separately from
capture settings. Its digest is recorded in the report, without changing the
capture configuration fingerprint or measured outputs. The immutable report
pins each input artifact and analyzed evidence by digest. New bundles inventory
`stability.json` in their existing manifest; historical bundles remain unchanged.

The report retains full-window statistics for each flight, each dataset, and the
whole campaign: signed extrema, RMS, p95, p99, absolute peak, worst-flight RMS
and identity, excursion counts/frequency/duration, endpoint displacement, linear
position trend, attitude, linear speed, and angular speed. Endpoint displacement
compares the first and last configured windows; it can include oscillation and
is not by itself proof of unbounded drift. The excursion threshold is a reporting
threshold, not a flight or controller acceptance limit.

Comparisons retain raw sample counts, timestamps, held goals, physical and
controller initial states, commands, per-field differences, and the first
unequal sample. Full-rate response bridge logs additionally identify the first
simulation timestamp with different actuator timing, rotor commands, or physical
state. No interpolation, fitted time shift, or trajectory alignment is applied.
The exact-repeatability result is diagnostic; it does not decide statistical
stability. Different controller-state reconstruction methods or response
definitions make response comparisons incompatible.
Every recorded hover and response bridge stream must contain dynamics with strictly
increasing simulation timestamps. Empty or mixed timelines raise an evidence error,
including when no comparison run was supplied. All analyzed bridge logs are pinned
by digest in the report.

Both repeated-seed and disjoint-seed batches receive statistical comparisons.
Whole flights, rather than correlated time samples, are resampled for bootstrap
confidence intervals of horizontal RMS and p99 relative changes. The campaign,
fit, validation, and acceptance groups must each qualify; good pooled statistics
cannot hide an inconclusive subset. Peak and worst-flight RMS increases are
checked separately. Drift and excursions remain descriptive diagnostics.

Response profiles are compared separately across complete response flights from
all three sets, which execute the same profile suite. Each profile retains RMS,
p99, peak, and worst-flight RMS for displacement from its initial position and
for velocity and body yaw-rate tracking error (velocity mode), or body-rate
tracking error and speed (CTBR mode). Each signal uses its original samples;
there is no interpolation or fitted time shift. A flight's many profiles do not
count as independent repetitions. These are repeatability metrics, not controller
acceptance limits, and they do not replace the retained trajectory diagnostics.

Settings own the confidence level, relative change limit, and minimum flight
count. An interval spanning the allowed boundary is inconclusive; missing
comparisons, a zero reference scale, or too few independent flights cannot pass.
The ordinary three-response-flight campaign is insufficient for the default
statistical minimum; qualification campaigns need additional response seeds.
Such expanded campaigns qualify the gathering mechanism. They are not alignment
inputs: alignment currently requires exactly one response recording per set.
Both repeated-seed and disjoint-seed comparisons must pass hover and response
statistics for the combined report to pass. An individual comparison records its
own result even when the combined evidence is incomplete. No baseline is
designated by this function.

Pass explicit completed comparison directories with
`FLIGHT_PARAMETERS_COMPARE_RUNS='<run-a> <run-b>'` when measuring a campaign.
An analysis exception leaves completed recordings and derived parameters in place;
resuming that unpublished run retries analysis without recapturing flights.
Analysis of an existing run writes outside every producer bundle:

```sh
PYTHONPATH=src uv run --with pyulog python make/scripts/run_flight_parameters_campaign.py \
  --analyze-run .build/flight-parameters/<id>/<current-run> \
  --compare-run .build/flight-parameters/<id>/<comparison-run> \
  --analysis-output .build/flight-parameters-stability/<analysis-run>/stability.json
```

Establish a baseline only after freezing the capture implementation/environment,
qualifying fresh repeated-seed and disjoint-seed campaigns, and checking a further
fresh campaign against those qualification runs. Keep the qualification inputs
and report as immutable evidence. Subsequent routine runs compare to this saved
baseline; capture-affecting changes require requalification before replacement.
