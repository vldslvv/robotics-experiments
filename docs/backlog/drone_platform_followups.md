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
