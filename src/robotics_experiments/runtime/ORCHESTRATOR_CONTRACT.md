# Fleet runtime contract

`robotics_experiments.runtime` is the single composition layer above config,
drone spawning, simulation settings, the PX4 bridge, and commanding. It uses a
functional core with small Isaac/MAVLink resource boundaries.

## State and resource boundary

- `FleetState`, `SpawnedDrone`, and `FlyingDrone` are frozen values. Lifecycle
  functions return successor state; callers must explicitly rebind it.
- `RoutineDriverState`, `Px4BridgeState`, and `SimClockState` follow the same
  rule in their owning packages.
- `BridgeHost` and `BridgeHostGroup` remain mutable only because PhysX callback
  return values are ignored and Kit subscriptions are lifetime-managed handles.
  Each host retains one latest frozen bridge-host state and replaces it after a
  callback transition.
- MAVLink sockets and Isaac articulation handles remain mutable adapters.
  Consumers read them through `read_command_port_status`, `read_bridge_status`,
  `read_articulation_io_status`, or `read_bridge_host_status`; runtime code does
  not use state-reading properties.

## Ordered lifecycle

The two shells drive the same sequence:

1. `create_fleet(config=...)`
2. `open_scene(state=...)`
3. after stage load, `adopt_current_stage(state=...)`
4. `apply_settings(state=...)`
5. on Play, `spawn_fleet(state=...)`
6. `start_bridges(state=...)`
7. `build_drivers(state=..., make_env=...)`
8. repeatedly `tick_drivers(state=...)` until `fleet_is_done(state=...)`
9. `teardown(state=...)`

Teardown stops the bridge group first, disconnects command ports, shuts down
observers, and only then despawns vehicles. This order is load-bearing.

`open_scene` requests an asynchronous stage open and leaves state unchanged.
`adopt_current_stage` binds the completed stage. `spawn_fleet` rejects a missing
stage. An empty vehicle mapping and an empty bridge group are valid.

## Per-vehicle composition

`spawn_fleet` iterates `ExperimentConfig.vehicles` in authored order. Every
vehicle is spawned. A bridge-enabled vehicle also receives one `BridgeHost` and
one state-only `IsaacArticulationIO` observer sharing the same resolved
articulation path. A non-empty routine receives one GCS `MavlinkCommandPort` and
one frozen driver state; a routine on a PX4-disabled vehicle is warned and left
manual.

The one `BridgeHostGroup` owns the PhysX subscription. Each tick runs every
host's sensor-send leg before any actuator-poll leg. This ordering preserves
multi-vehicle isolation and bounds lockstep latency by the slowest PX4 rather
than serially starving later sends.

## Shell responsibilities

The live `experiment_runtime` extension owns Kit stage/timeline/update event
wiring and forgiving per-driver error logging. The standalone runner owns the
blocking `SimulationApp.update()` loop, strict-versus-forgiving failure
reaction, and integration-only checks such as ground settling, first-heartbeat
waiting, pre-arm tilt, fault injection, and bridge-log validation.

Neither shell owns scene, flight, bridge, or routine logic. Timing helpers take
a sim-time callable so they do not depend on resource properties.

## Failure contract

Invalid lifecycle order or configuration raises. Recoverable runtime bridge
errors are logged by `BridgeHost` without taking down other vehicles. Driver
failure reaction is injected through `FleetDriverEnv`: strict environments
raise, while forgiving environments log and leave manual control available.

The runtime modules are Isaac-only lazy leaves. `robotics_experiments.runtime`
must remain importable without Isaac; importing a lazy runtime symbol may load
the Isaac dependencies.
