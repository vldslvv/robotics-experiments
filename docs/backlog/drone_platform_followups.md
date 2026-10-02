# Drone platform follow-ups

## Flight validation

- Define what vehicle-side state verification is required in addition to the
  simulator ground-truth routine gates, including when the simulator may stop a
  routine whose onboard estimate disagrees.
- Add small-motor/no-liftoff sanity cases and measure expected hover throttle for
  each supported airframe.
- Reproduce and isolate the historical hexacopter failsafe/wild-flight report
  beyond the current quad/hex navigation integration coverage.
- Periodically validate the fresh-checkout sequence documented in the root
  README on a clean host and empty Docker cache.
- Compare Isaac upgrades using the existing core benchmark and matched hardware;
  retain bridge/physics attribution and comparable baseline/candidate images.
- Verify the battery SKU, pusher spring/guide/pad/mount, generated prop-guard
  geometry, and mass accounting against physical hardware. Correlate controller
  dynamics with bench or flight evidence before claiming hardware fidelity.
- Complete the [harness prerequisites](safety_harness.md) before real contact.

## Airframes and physical variation

- Add an octocopter config only with a matching generated PX4 allocation and
  integration gate.
- Decide whether URDF adds value over the current cited airframe YAML plus
  generated USD/PX4 workflow before introducing a second source format.
- Model per-build physical variation/noise in a seeded, configurable way rather
  than changing authored nominal component values.
- Add prop-guard geometry and reconcile its mass with the component model.

These are durable ideas, not current behavior or accepted implementation
requirements. Promote an item into a scoped design document or issue before
implementation.

## Flight-parameter distribution qualification proposal

This is a design for review, not active qualification criteria or authorization
to increase flight counts. The current capture schedules, seed lists and
statistical gates remain unchanged.

### Population and sampling

Freeze the physical experiment, controller definition, simulator/PX4 images and
capture implementation first. One independent unit is a complete flight with
fresh processes. Keep all profiles of that flight together. Disjoint seeds
provide requested random-stream coverage; a seed does not control PX4 scheduling.
The current response experiment disables stochastic IMU/GPS noise, so additional
seeds do not introduce wind, mass, hardware or sensor variation absent from the
experiment. Claims must stay within those configured conditions.

Distribute each proposed cohort across at least three separately started capture
sessions. Preassign fit/validation/acceptance membership to whole flights and
balance that membership across sessions. Use disjoint seeds for the cohorts
being compared. Collect a separately budgeted repeated-seed diagnostic group
across sessions to distinguish repeatability from broader coverage. Do not count
its correlated conditions as additional independent population coverage. Measure
session effects and use session-aware resampling; three sessions are an initial
coverage design, not sufficient evidence of small between-session uncertainty.
[NIST's data-collection guidance](https://www.itl.nist.gov/div898/handbook/mpc/section2/mpc232.htm)
emphasizes covering relevant conditions and longer-term measurement variation.

Keep unsuccessful/incomplete capture attempts and their reasons. Separate
infrastructure interruptions from physical capture failures in the review, and
report both denominators. A successful retry must not erase a physical failure
or become an extra independent flight. Do not exclude outliers or select seeds
based on observed performance.

### Metrics and candidate decision rules

Retain full measurement traces and report both pooled-sample and per-flight
distributions. The primary measurements are RMS, p95/p99, maximum error,
worst-flight RMS, excursions and drift. A percentile of flight maxima is a
different quantity from the percentile of all time samples; label both explicitly.
Never calculate a pooled percentile by averaging per-flight percentiles.

Propose two separate conclusions:

- Distribution equivalence: define meaningful absolute/relative change margins
  before qualification, then require simultaneous confidence intervals to lie
  inside them. Start the sizing study with the current 10% relative margin, but
  do not declare it an engineering requirement. Use an authored absolute margin
  for signals close to zero instead of dividing by an arbitrary numerical floor.
  Bootstrap whole flights jointly across metrics, preserving their correlations,
  and account for session effects. Evaluate a joint maximum-standardized-deviation
  interval against a conservative multiplicity correction before choosing the
  production method. Report individual and overall pass/fail/inconclusive results.
- Engineering acceptability: specify absolute limits and a flight-level event
  such as any prescribed excursion or capture failure. Estimate its probability
  with a one-sided upper confidence bound. Keep maxima and worst-flight identities
  visible, but replace the sample-maximum ratio as a distribution-equivalence
  gate only after that replacement has been reviewed. Define drift/excursion
  limits if they are to become binding; their current descriptive thresholds
  are not engineering acceptance limits.

The review must select margins, event definitions, coverage/confidence targets,
metric families and treatment of session effects before new production gates are
implemented. No source permits those project-specific limits to be inferred just
from a desired passing result. See the discussion of predefined equivalence
bounds and power in [Lakens' equivalence-testing chapter](https://lakens.github.io/statistical_inferences/09-equivalencetest.html).

### Sample size, stopping and confirmation

Size hover and response populations separately. Use a pilot from the frozen
capture method to evaluate confidence-interval precision and the ability to
establish equivalence, including realistic session variation. Retain a power
curve over candidate sizes rather than one extrapolated number. Define a fixed
qualification size and stopping rule in advance; an inconclusive result requires
an explicit revised study, not repeated collection until a favorable pass.

For one predefined flight-level violation event, independent flights from the
claimed population, and zero violations, the one-sided 95% upper risk bound is
`1 - 0.05**(1/n)`. Consequently, 59, 299 and 2,995 flights support respective
95%, 99% and 99.9% success-coverage claims under those assumptions. A size of 300
is therefore meaningful for a narrowly defined rare-event claim. It does not
provide precise p99 estimation, prove equivalence of two batches, or automatically
cover many separately tested requirements. Joint events need to be defined before
capture. [NASA's Monte Carlo verification paper](https://ntrs.nasa.gov/api/citations/20100035661/downloads/20100035661.pdf)
discusses order statistics, sample size and the distinction between consumer and
producer risk.

A sizing scenario of two 300-response cohorts means 600 response flights, before
repeated-seed diagnostics, hover collection or fresh confirmation. The completed
pipeline will accept longer explicit seed lists; there is no new count scalar.
Estimate cost using measured throughput and compressed sizes, including temporary
storage and failed attempts. The earlier four-worker response pilot suggests
about 31 hours for 300 response flights; that is a planning estimate, not a
throughput guarantee under another workload.

The run-scoped synthetic diagnostic at
`.build/flight-parameters-stability/20260927-pipeline-verification/` evaluates a
candidate simultaneous RMS comparison on unchanged populations, a 5% change,
and a 20% change in one signal. It uses six correlated lognormal flight metrics,
50 independent comparisons per scenario/size and 400 whole-flight bootstrap
resamples. At 300 flights per batch, 35/50 unchanged comparisons pass and 46/50
20%-shift comparisons fail; the remainder are inconclusive. At 600, both counts
are 50/50. A 5% change still frequently remains inconclusive even at 600 because
its uncertainty must fit inside the 10% margin. These are coarse synthetic
implementation/sizing results, not a PX4 sample-size recommendation, p99/tail
calibration or evidence that the gathering mechanism is stable. The script,
model assumptions and complete outcomes are retained in the run directory.

After qualification passes, require a fresh, preregistered confirmation cohort
before establishing the baseline. Pin all qualification and confirmation inputs,
reports and settings. Routine monitoring may use smaller batches but cannot
independently reprove rare-event coverage. Capture-affecting changes require
requalification before replacing the baseline.

### Alignment follow-up

Expanded response cohorts currently qualify the gathering mechanism; alignment
still requires exactly one response recording per set. Specify a separate
alignment evaluation design before consuming the larger distribution. Do not
silently average trajectories or choose a favorable representative flight.
