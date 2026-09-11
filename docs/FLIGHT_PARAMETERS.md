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
explicit `hover_seeds` and `response_seeds` lists. Each seed starts one complete
flight in fresh Isaac and PX4 processes. All lists must be non-empty and seeds
must be distinct across the campaign. List lengths own the counts:

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

## Hover collection and statistics

Each hover repetition starts fresh PX4 SITL and Isaac processes. The collector
sends only arm and takeoff, decodes the completed ULog inside
`flight_parameters`, and proves the external commands, `AUTO_TAKEOFF ->
AUTO_LOITER` transition, automatic position control, armed and airborne state,
fresh reports, and unchanged held position/yaw goal. A routine label, elapsed
time, or low velocity is not accepted as proof of hover.

Six seconds after verified loiter entry, the collector retains exactly 30
seconds at 20 Hz. All 600 samples in each trace are preserved. The published
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

Alignment preparation resolves one immutable `FlightParametersBundle`, pins it
inside the alignment run, selects its single fitting response flight, and
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
alignment neither copies nor mutates them. Alignment search, metrics, ranking,
fit/validation behavior, actuator audit, controller-boundary replay, and
publication checks are unchanged. Independent validation and acceptance
recordings remain available in the bundle; using them to select and qualify
controllers is deferred to Stage 5. The current alignment workflow still uses
the internal fit/validation/stability profiles in the fitting flight. Merely
capturing three sets does not establish independent controller acceptance.

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
  evidence/trial-001/raw_trace.json
  evidence/trial-001/hover_verification.json
  evidence/trial-001/trace.json
  evidence/trial-001/bridge.log
  evidence/trial-001/flight.ulg
  ...
  response/repetition-001/flight_parameters_response.json
  response/repetition-001/flight_parameters.ulg
  response/repetition-001/flight_parameters_bridge.log
  manifest.json
```

An explicit `FLIGHT_PARAMETERS_RUN_ID` may continue an unpublished partial run;
complete verified hover trials are retained and must match the configured set,
seed, and configuration fingerprint. Response flights are captured again. A finalized run cannot be resumed
or modified.

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
candidates.

Alignment and RL record the exact artifact fingerprint. They never silently
switch to a newer `latest`: alignment pins the resolved run during preparation,
records its canonical `run_id`, and training rejects a flight-parameters
artifact whose `run_id` or fingerprint does not match its controller
qualification.
