# Changelog

## [Unreleased]

### Changed

- Removed dead shim modules (`articulation_io.py`) and collapsed the per-module
  re-export shims (`ports.py`, `px4_bridge.py`, `thrust_model.py`,
  `vehicle_state.py`, `frames.py`) into `__init__.py`, which now re-exports the
  public API directly from `robotics_experiments.px4_bridge`.
- The no-op extension shell now logs its load/unload with the `[Px4SitlBridge]`
  title for parity with the other extensions; it still owns no scene, drone, or
  physics callback.

## [0.1.0] - 2026-04-17

### Added

- Initial version of Px4SitlBridge extension
