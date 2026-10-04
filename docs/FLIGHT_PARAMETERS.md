# Flight parameters

Flight parameters capture task-neutral physical evidence. Each independent
flight contains native PX4 hover followed by all 23 response stages. Capture
owns neither dataset splits nor fitted controller or training bounds.

## Flight definition

`configs/flight/holybro_x500_v2_hover.yaml` owns the complete physical suite:
vehicle and experiment, `start_seed`, `flight_count`, takeoff height threshold,
ordered stages, response preparation, and worker count. The first stage defines
hover timing, sampling cadence, requested pose and PX4 report freshness. Each
response stage defines its name, control mode, axis, optional position hold, and
ordered commands with durations. Velocity commands use body-FLU velocity and
yaw rate; CTBR commands use normalized collective thrust and body-FLU rates.
Names containing `fit` or `validation` are physical stage identifiers; they do
not assign flights or samples to a consumer dataset.

The default is ten complete flights. Each starts fresh Isaac and PX4 processes,
with seed `start_seed + flight_index` for zero-based indices. Seed assignment
happens before dispatch and stays unchanged on retry or resume. Stages share the
flight's physical state, PX4 controller history and random stream. There is no
stage reset or reseeding. A seed identifies a random stream; it does not promise
bitwise replay of PX4 command activation.

`configs/flight_parameters_x500.yaml` owns the physical environment. Calibration
uses simulator ground truth and disables artificial random sensor noise. Small
deterministic pressure and magnetic excitation keeps PX4 sensor initialization
healthy. Sensor noise and domain randomization during training are separate
consumer decisions. The recorder rejects a noise-enabled calibration experiment.

## Running and flow

```sh
make flight-parameters
make flight-parameters FLIGHT_COUNT=100
```

The optional count overrides the YAML default and changes campaign identity.
GNU Make's `-n` prints commands; it is not a flight count. Processing and stability
settings live separately in `configs/flight/processing/default.yaml` and
`configs/flight/stability/default.yaml`.

```text
flight definition -> assign independent flights and seeds

for each independent flight:
  definition -> capture the complete timeline
  -> verify PX4 state and extract stage metadata
  -> reconstruct response controller states from ULog
  -> lossless compression and hashes -> completion checkpoint
  -> remove verified raw log duplicates

after all flights:
  validate completeness -> write unsplit dataset metadata
  -> final stability analysis in a separate report
  -> validate file integrity -> finalize immutable run -> publish latest
```

Capture callbacks never invoke stability analysis. Each worker completes
postprocessing before accepting another flight. Worker count changes scheduling,
not stage commands or seed assignment. `definition.json` records the flight
configuration, including the requested count and starting seed. Resume requires
the same definition. Validation checks that the requested flights and stages
completed and that retained files are intact. Git state and unrelated repository
changes do not affect capture validity.

## Timing and measurements

Ground-truth position, attitude, linear velocity, angular velocity and simulation
time come from the same physical-state snapshot. Recording continues through
startup, takeoff, hover, OFFBOARD priming and every response preparation and
measurement. No transition trajectory is discarded. Full-rate bridge dynamics
and ULog remain available alongside the sampled continuous `motion.jsonl`.
Requested commands, their generation time, successful transmissions and their
simulation timestamps are separate fields; a requested command is not proof
that PX4 has already applied it.

The shell derives the routine from the stage definition. The first hover hold
lasts `mode_transition_timeout_s + settle_s + measure_s`, and its measurement
starts after `mode_transition_timeout_s + settle_s`. With the default schedule
these are 51 and 21 seconds after hover entry. Completion uses elapsed simulation
time from step entry. Offline verification must still prove accepted arm and
takeoff, AUTO_TAKEOFF followed by AUTO_LOITER before the transition deadline,
armed airborne automatic position control, fresh reports and an unchanged held
pose throughout the measurement. A low speed or elapsed timer is insufficient.
External OFFBOARD commands after the hover window are allowed; earlier unexpected
commands fail verification. Verification never moves a window to accommodate a
late transition.

Each response stage first recenters using the existing velocity feedback. It
waits the entire configured preparation interval and requires the minimum neutral
period plus continuous speed, body-rate and tilt readiness at that deadline.
Early readiness does not advance the stage; late readiness fails the flight.
Measurement advances when every authored command duration has elapsed. Metadata
records actual preparation, measurement and end timestamps, initial physical
state, and the measurement's half-open sample range. ULog postprocessing proves
armed airborne OFFBOARD state and reconstructs controller state at each response
measurement boundary using `px4-ulog-controller-boundary-v2`.

## Evidence, storage and resume

```text
.build/flight-parameters/<id>/<run-id>/
  definition.json
  flight_parameters.json
  flights/flight-000/
    raw_flight.json
    motion.jsonl
    flight.json
    capture.json
    complete.json
    flight.ulg.gz
    bridge.log.gz
  ...
  manifest.json
```

`flight.json` is postprocessed metadata, not a second motion recording. Stage
views select the original sample ranges from `motion.jsonl`, without shifting,
interpolation, smoothing, decimation or trajectory alignment. Every profile's
initial physical and reconstructed controller states remain attached to its
measurement boundary. Full-rate evidence is compressed losslessly after each
flight; the runner does not accumulate an entire campaign of uncompressed logs.

Compression writes a temporary archive, verifies its decompressed SHA-256 and
atomically installs it. `complete.json` records the flight identity, completion
result and retained file inventory. File hashes protect archive integrity and
resumption; they do not classify capture behavior. Only then are matching
uncompressed logs and private harness copies removed. The closed harness bridge log must match the recorder's prefix; teardown
records are retained. Failed captures remain under `failed/`.

A successful `capture.json` lets interrupted processing reuse raw capture.
A validated `complete.json` skips capture and processing. Corrupt checkpoints
fail rather than silently recapturing. Resume with the original explicit
`FLIGHT_PARAMETERS_RUN_ID`; the flight definition, count and seeds must still
match. After the dataset header exists, resume retries final analysis and
publication. Finalized runs cannot be modified or resumed.

Before dispatch the runner reserves space for all remaining archives, active
raw flights and compression staging, plus the configured free-space reserve.
Completed-flight sizes refine the estimate upward. Insufficient space preserves
completed evidence and stops dispatch. This estimate does not bound failure logs.

## Dataset contract and consumers

`FlightDataset`, read through `load_flight_dataset(path)`, is an immutable unsplit
schema-1 artifact with kind `robotics-experiments-flight-dataset`. The reader
resolves selectors once, freezes metadata and validates completed-flight files,
input identity, hashes and an exact final manifest. The header contains the producer ID, run, a `completed` result and ordered
flight checkpoint references. `definition.json` contains the authored flight
parameters; per-flight metadata contains seeds, stage boundaries, measured
initial states and sample ranges. Successful completion requires every requested
flight and stage, hover verification and response controller-state extraction.
Source, environment and capture-method fingerprints are not part of the dataset.

Publication advances the relative `<id>/latest` symlink only after all flights,
postprocessing and the manifest validate. `latest` means complete evidence,
not stable evidence or a qualified baseline. There is no automatic `best` or
baseline designation.

The distinct `FlightParameters` contract represents consumer-qualified hover
bounds used by existing alignment and RL. Those consumers reject the new raw
dataset until their follow-up implementation; there is no compatibility adapter
or evidence migration. Alignment must assign whole independent flights to
training, validation and test sets and derive fitting-only bounds. See
[the alignment handoff](backlog/flight_parameters_alignment_handoff.md).

## Final stability analysis

Stability is read-only and runs only after collection and per-flight processing.
The report provides campaign and individual hover RMS, signed extrema, p95, p99,
absolute peak, worst-flight RMS and identity, excursion frequency/duration,
endpoint displacement, position trend, attitude, linear and angular speed.
Every response stage also reports distributions and worst flights for displacement
and velocity/yaw-rate tracking errors, or body-rate errors and speed for CTBR.

Comparisons bootstrap complete independent flights, not correlated time samples.
The existing confidence level, relative change limit and minimum flight count
remain unchanged. RMS and p99 confidence intervals must fit inside the allowed
relative-change interval; peak and worst-flight RMS increases are checked
separately. Drift and excursions are descriptive diagnostics. Exact sample,
initial-state, command and full-rate bridge differences remain diagnostic and do
not decide statistical acceptance.

Reports are written outside the producer bundle, at
`<output-root>/<id>/stability/<run-id>.json`. Comparisons check the recorded flight
definitions (excluding count and starting seed), stage commands and measurement
cadence. This checks that the same measurements are being compared; it does not
assert that software or hardware environments were identical.

Qualification requires passing both repeated-seed and disjoint-seed batch
comparisons for hover and every response profile. Missing comparisons, too few
flights, zero reference scales or unresolved intervals cannot pass. A single
ten-flight campaign therefore reports `insufficient-evidence`; its successful
publication does not prove stability. Changes to capture behavior require fresh qualification before a baseline is
established or replaced.

Pass completed comparison directories with
`FLIGHT_PARAMETERS_COMPARE_RUNS='<run-a> <run-b>'`. To analyze existing evidence
without modifying producer files:

```sh
PYTHONPATH=src uv run --with pyulog python make/scripts/run_flight_parameters_campaign.py \
  --analyze-run .build/flight-parameters/<id>/<current-run> \
  --compare-run .build/flight-parameters/<id>/<comparison-run> \
  --analysis-output .build/flight-parameters-stability/<analysis-run>/stability.json
```
