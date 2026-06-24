# Changelog

## [1.0.0]
- Initial release. Live Kit shell hosting the host-agnostic
  `FleetOrchestrator`: opens the scene after Kit reports app readiness, applies sim
  settings, and on Play spawns the fleet, starts the PX4 lockstep bridge group,
  and ticks each drone's routine driver. Replaces the `drone_spawner` and
  `sim_settings` extensions (whose roles moved into
  `robotics_experiments.runtime`).
