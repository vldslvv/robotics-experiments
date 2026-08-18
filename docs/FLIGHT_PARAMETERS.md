# Flight parameters

The flight-parameters workflow measures task-neutral vehicle behavior and
publishes a strict artifact for controller alignment, policy training, and
future simulator-to-hardware comparison. It does not know about switches,
walls, press tasks, rewards, or policy observations.

## Contract

`robotics_experiments.flight_parameters` is a host-safe module with one flow:

`flight config -> independent flight traces -> flight envelope`

Schema 1 supports only the `hover` scenario. The config and artifact carry a
scenario discriminator so later flight parameters can be added without putting
task logic into the module. The production config is
`configs/flight/holybro_x500_v2_hover.yaml`.

The configured hover procedure is:

1. Start fresh PX4 SITL and Isaac processes for one trial.
2. Send only arm and takeoff commands to the configured target pose.
3. Capture a 51-second raw simulator ground-truth pose candidate at 20 Hz from
   the start of the routine's hover step. The routine label is not accepted as
   proof that the aircraft is hovering.
4. Decode the completed PX4 ULog inside `flight_parameters`. Require the exact
   external arm/takeoff command sequence and the PX4 transition
   `AUTO_TAKEOFF -> AUTO_LOITER` within 15 seconds.
5. Reject the trial if PX4 leaves `AUTO_LOITER`, enters OFFBOARD, is not armed
   in automatic multicopter position control, reports ground contact, receives
   a goto/reposition/landing/other external command, or changes its active
   position or yaw goal before the measurement completes.
6. Starting six seconds after the verified `AUTO_LOITER` transition, select
   exactly 30 seconds of pose samples at 20 Hz. PX4 mode, control-mode, and
   land-state reports must remain fresh throughout the selected interval.
7. Complete the routine, tear down the ephemeral simulator, and retain the raw
   trace, decoded verification record, selected neutral trace, ULog, and bridge
   log.
8. Repeat the complete trial 30 times, then derive one envelope from all 18,000
   verified samples.

This is 30 independent 30-second measurements, not one long hover split into
windows. Independent process, estimator, takeoff, and settling histories are
part of the evidence distribution. `execution.worker_count` controls safe
campaign concurrency without changing trial identities or aggregation order.
The six-second settling interval is measured from PX4's verified
`AUTO_LOITER` transition. It excludes the repeatable vertical convergence seen
during the first five seconds after loiter while retaining the steady hover
variation that follows.

Run the campaign from the repository root:

```bash
make measure-flight-envelope
```

## Hover envelope

The hover envelope contains only error relative to the PX4-verified held pose:

- ENU position-error observed minimum, observed maximum, lower bound, and upper
  bound for X, Y, and Z;
- observed maximum and upper bound for horizontal and three-dimensional
  position-error norms;
- relative roll, pitch, and yaw observed minimum, observed maximum, lower
  bound, and upper bound;
- observed maximum and upper bound for tilt and full quaternion geodesic
  orientation error.

The effective position target is reconstructed from PX4's accepted global
takeoff goal and the trial's PX4 home anchor. The effective yaw target combines
PX4's held NED yaw with the simultaneous simulator attitude and PX4 local
heading, so the selected trace and target share one world frame. The authored
config target remains in the raw and verification records for diagnosis; it is
not assumed to be the target PX4 actually held.

The published bounds are the extrema across every complete verified trace
expanded by the configured 0.005 m position and 0.25 degree attitude margins.
The artifact also records trial/sample counts, timing, vehicle and experiment
fingerprints, source image/commit, the ordered evidence fingerprint, and its
canonical payload fingerprint. Velocity, body rate, actuator output,
wall-relative position, and switch state are not hover-envelope dimensions.

## Artifacts and consumers

Each immutable run is stored under:

```text
.build/flight-envelopes/<flight-id>/<run-id>/
  flight_envelope.json
  campaign.json
  evidence/trial-001/raw_trace.json
  evidence/trial-001/hover_verification.json
  evidence/trial-001/trace.json
  evidence/trial-001/bridge.log
  evidence/trial-001/flight.ulg
  ...
  manifest.json
```

The verification record contains the accepted commands and acknowledgements,
PX4 mode transitions, active position goals, selected time interval, control
and land-state report counts, effective target, PX4 version, hashes of its raw
trace and ULog, and a canonical verification fingerprint. The envelope binds
the ordered set of those fingerprints. The exact manifest inventories every
payload by relative path, size, and SHA-256.

Only a complete valid bundle is published. Publication atomically advances the
relative symlink `.build/flight-envelopes/<flight-id>/latest` to the immutable
run. There is no `best` label because an envelope is measured evidence, not a
ranked candidate.

Controller alignment requires the artifact and replays a dedicated
`stability-hover-pose` probe through the selected Isaac Lab settings. A
controller cannot pass quality when its simulated position or attitude leaves
the measured bounds. The accepted controller records the exact envelope
fingerprint.

RL training independently resolves and validates the configured `latest`
bundle before constructing Isaac Lab. It requires the envelope fingerprint to
match the qualified controller, and embeds the complete envelope plus its file
digest in every policy artifact. Reward and termination design can therefore
consume the task-neutral pose bounds without importing alignment or the flight
collector.

For hardware alignment, add a hardware collection backend that emits the same
neutral trace contract and preserve the scenario dimensions. Simulator and
hardware envelopes can then be compared at the position/pose boundary while
ULogs and bridge/device logs diagnose controller, estimator, or plant causes.
