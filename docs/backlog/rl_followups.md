# Learning and transfer follow-ups

These are unfinished outcomes, not current runtime contracts or permission to
launch training. Use [RL workflow](../RL_WORKFLOW.md) for execution and
[ADRs 0016-0019](../adr/README.md#learning-and-qualification) for the decisions
behind the current design.

## Qualification and transfer

- Qualify the complete nominal contact curriculum with the current generated
  switch models. Physical push tests and deterministic task feedback must agree
  before PPO; each learned stage needs its own fresh acceptance test. Tighten
  the provisional contact requirements after the curriculum becomes trainable.
- The old nominal blockage in
  `.build/press-curriculum-20260925/nominal-blocker.json` predates shared generated
  switches. Treat it as historical geometry evidence, not a current failure.
- Finish the tighter-hover transfer objective: fresh Lab acceptance followed by
  three consecutive complete warehouse executions of the same actor digest,
  completing all policy segments without contact, lamp changes, timeout, or
  flight failure. Keep PX4 qualification separate from Lab-ranked publication.
  The prior reward/horizon trials did not meet this objective; diagnosis is in
  `.build/hover-correction/20260914T135916Z.qualification.json`. Do not infer
  current deployment labels from that historical report.
- Refresh measured navigation-handoff evidence for the production route before
  approach training. Require matched actor-entry/reset distributions and retain
  handoff failure separately from actor failure.
- Establish repeated fresh controller qualification before claiming scratch
  reproducibility. Preserve exact flight evidence, compare capture variability,
  and measure search throughput instead of carrying forward old sweep recipes.

## Complete known-switch behavior

- Implement a scripted contact baseline behind the generic policy adapter and
  compare it with learned behavior on the same seeds, resets, physical limits,
  success, contact stability, retreat, time, and effort measurements.
- PX4 contact/press deployment needs explicit support and qualification before
  the full approach -> handoff -> press -> lamp confirmation -> release ->
  retreat -> land mission can be claimed. Reuse the existing commanding driver.
- Add bounded mission recovery for failed-but-recoverable policy execution,
  including re-approach from a waypoint, retreat, and exhausted-retry landing.
  Exercise entry rejection, lost detection, recovery, and abort paths before
  hardware contact. Keep retry policy explicit in config.

## Physical randomization and perception

After nominal contact acceptance, use complete stage experiments to randomize
mass/inertia, motor response, thrust, pusher compliance, contact friction,
switch geometry/mechanics/pose, wall tilt, and start conditions. Declare ranges,
units, sampling laws, and fixed independent evaluation distributions in Hydra.
Reuse the existing curriculum runner and respect certified-controller ownership.
Include the [tether disturbance](safety_harness.md) when qualifying hardware use.
The recorded mature-contact targets are 85% press success, 95% stable first
contacts, and 99% retreat; they remain future requirements, not current acceptance.

Then introduce synthetic pose degradation behind `AbstractPoseProvider`:
Gaussian position/axis-angle noise, fixed/jittered latency, independent and burst
dropouts, correlated bias drift, rare outliers, and age-dependent confidence.
Keep estimator and RNG progress frozen and transitions reproducible; the adapter
returns the same `PoseSample` and actor observation shape. Test deterministic
sequences, stale/low-confidence handoff rejection, and bounded retry/abort.
Qualify press-and-retreat under a fixed noise workload, including false handoffs
and stale-observation failures, before replacing synthetic data with sensors.

Real RGB-D/LiDAR/fusion providers belong to the later sensor milestone. Preserve
confidence, age, and validity through the provider boundary; do not expose raw
sensor implementation details to the actor or add another observation layout
merely to swap providers.
