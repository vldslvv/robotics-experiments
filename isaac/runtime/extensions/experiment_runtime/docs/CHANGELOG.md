# Changelog

## [1.2.0]
- Extend the immutable runtime snapshot with routine, motion, navigation,
  switch-relative handoff, policy, and transmitted-setpoint telemetry built
  only from values already consumed by the control tick.

## [1.1.0]
- Expose the active extension and its immutable runtime snapshot to dependent
  Kit plugins while preserving the forgiving live-runtime behavior.

## [1.0.0]
- Initial release. Live Kit shell hosting the host-agnostic
  `FleetOrchestrator`: opens the scene after Kit reports app readiness, applies sim
  settings, and on Play spawns the fleet, starts the PX4 lockstep bridge group,
  and ticks each drone's routine driver. Replaces the `drone_spawner` and
  `sim_settings` extensions (whose roles moved into
  `robotics_experiments.runtime`).
