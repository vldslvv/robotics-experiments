# ADR 0003: Mitigate PX4 takeoff instability and classify infra flakes

- Status: Accepted
- Date: 2026-06-16
- Related: ADR 0007 (test semantics)
- Distills: `docs/PX4_TAKEOFF_TEST_INSTABILITY.md`,
  `docs/flight_analysis_00_26_50.md`

## Context

`test-integration-px4-takeoff` had two different instability signatures that
looked similar at the make target level:

- A real flight instability: the drone flipped shortly after liftoff in the
  `00_26_50` run.
- Infrastructure flakes: PX4 lockstep disconnects, mostly actuator polling
  timeouts during slow boot or mid-flight process stalls under host load.

The flight flip initially looked like a possible bridge-force, frame-conversion,
or rotor-position bug. Log analysis ruled those out: rotor forces were symmetric,
net torque was effectively zero, HIL sensor time and frame conversions were
correct, and PX4's first asymmetric actuator response happened after the attitude
perturbation had already appeared.

## Decision

Treat these as two separate failure classes and fix each at the correct layer:

- The real flip was attributed to a PhysX ground-contact-release impulse at
  liftoff. The drone now spawns with more stationary ground clearance
  (`stationary_ground_clearance_m=0.05`) and settles before PX4 connects, avoiding
  the asymmetric contact release that started the flip.
- The bridge keeps a tilt safety gate (`max_tilt_angle_deg=60.0`) that zeros
  rotor forces once the vehicle is beyond a safe attitude, preventing inverted
  post-crash thrust from driving the vehicle into the ground.
- Bootstrap actuator timeouts are treated as readiness/host-load flakes, not
  dynamics regressions. `first_actuator_timeout_s` was raised to `30.0` so PX4
  can finish controller boot after Isaac connects without building a backlog.
- PX4 failsafe is a distinct `Px4FailsafeDetected` failure, not an
  `ExpectedInstabilityDetected` envelope breach. The negative instability test
  must pass only on a deliberate envelope breach, never on a failsafe flake.

Do not relax the steady-state `lockstep_timeout_s=1.0`: a mid-flight PX4 stall
should still disconnect promptly. Only the first-actuator boot wait is longer.

## Consequences

- A contact-release perturbation no longer causes the normal takeoff test to flip
  at liftoff.
- If a crash or injected fault does invert the vehicle, the safety gate removes
  thrust rather than amplifying the crash.
- Slow PX4 startup under host contention is less likely to fail a healthy run.
- Test outcomes are better classified: infra flakes, PX4 failsafes, and true
  flight-envelope breaches are not collapsed into the same exception path.
