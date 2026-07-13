# ADR 0006: Command/routine pure-function contract layer

- Status: Accepted
- Date: 2026-06-22
- Related: ADR 0002 (layering)
- Living contract reference: `docs/COMMAND_ROUTINE_CONTRACT.md`

## Context

The only way a drone received high-level commands (arm, takeoff) was an ad-hoc
~11-stage procedure buried in the integration test
`_px4_takeoff_scenario.py`: it hand-opened a pymavlink GCS link, sent
`command_long_send`, waited for `COMMAND_ACK`, and sequenced phases inline. There
was no reusable command/plan abstraction, and no home for future autonomy
(navigate-to-point, the switch-pressing task).

## Decision

Add `robotics_experiments.commanding` as the high-level control layer, built as a
**pure-function contract that mirrors the PX4 bridge contract part-for-part**:

- `RoutineObservation` (pure input) -> `decide(routine, state, observation)`
  (pure stepper) -> `RoutineDecision` (`SEND`/`STREAM`/`WAIT`/`ADVANCE`/`DONE`/
  `FAIL`) across the PX4-specific `MavlinkCommandPort` seam, with
  `VehicleCommander` and the per-drone `RoutineDriver` as orchestrators
  (analogues of `commands_to_dynamics`
  / the `Px4MavlinkPort` lockstep seam / `BridgeHost`).
- It is a **domain leaf on the core import path**: no module imports Isaac or
  pymavlink at module scope (the concrete `MavlinkCommandPort` imports `pymavlink`
  only inside its methods), so `import robotics_experiments` succeeds with both
  absent. `MAV_CMD_*` / `MAV_RESULT_*` are named int constants.
- **Per drone, like the bridge**: one commander + one driver per drone, no fleet
  object; cross-drone sync deferred.
- The single side effect is the port; `decide` and every completion/invariant
  predicate are pure and unit-tested with plain dataclasses. Step timeouts are
  measured in sim time to keep the stepper pure.
- Public API is the production runtime surface only (`RoutineDriver`,
  `DriverParams`, `DriverEnvironment`, `RoutineObservation`, `CommandAck`,
  `MavlinkCommandPort`, `FailureKind`); steppers/builders stay private.
- The command layer talks only the per-drone `gcs_port`, never the bridge's
  simulator MAVLink port or lockstep -- the symmetric half of the bridge rule.

## Consequences

- The standalone takeoff scenario was refactored onto this layer (no hand-rolled
  MAVLink), and the live `experiment_runtime` extension drives the same
  `RoutineDriver`; routines became config-driven via the Hydra `routine` group.
- New autonomy primitives (`goto`, etc.) are added as pure routine steps.
