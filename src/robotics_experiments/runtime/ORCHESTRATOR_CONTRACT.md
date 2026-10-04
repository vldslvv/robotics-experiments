# Fleet runtime contract

`robotics_experiments.runtime` is the single composition layer above config,
drone spawning, simulation settings, the PX4 bridge, and commanding. It uses a
functional core with small Isaac/MAVLink resource boundaries.

## State and resource boundary

- `FleetState`, `SpawnedDrone`, and `FlyingDrone` are frozen values. Lifecycle
  functions return successor state; callers must explicitly rebind it.
- `DriverContext`, `RoutineDriverState`, `Px4BridgeState`, and `SimClockState`
  follow the same rule in their owning packages. The driver context carries
  immutable parameters and resources; the driver state carries progress only.
- `BridgeHost` and `BridgeHostGroup` remain mutable only because PhysX callback
  return values are ignored and Kit subscriptions are lifetime-managed handles.
  Each host retains one latest frozen bridge-host state and replaces it after a
  callback transition.
- MAVLink sockets and Isaac articulation handles remain mutable adapters.
  Consumers read them through immutable values returned by public status methods
  (`read_status()` or `is_initialized()`). The PX4 simulator port additionally
  exposes the narrow `is_connected()` hot-path probe so the physics send leg does
  not construct a full `BridgeStatus`. The project-owned `BridgeHost` keeps the
  module-level `read_bridge_host_status` boundary. Runtime code does not use live
  state-reading properties.

## Ordered lifecycle

The two shells drive the same sequence:

1. `create_fleet(config=...)`
2. `open_scene(state=...)`
3. after stage load, `adopt_current_stage(state=...)`
4. `apply_settings(state=...)`
5. on Play, `spawn_fleet(state=...)`
6. `start_bridges(state=...)`
7. `build_drivers(state=..., make_env=...)`
8. feed each physics `dt` to `tick_drivers(state=..., physics_dt_s=...)` until
   `fleet_is_done(state=...)`
9. `teardown(state=...)`

Teardown stops the bridge group first, disconnects command ports, shuts down
observers, and only then despawns vehicles. This order is load-bearing.

`open_scene` requests an asynchronous stage open and leaves state unchanged.
`adopt_current_stage` binds the completed stage. `spawn_fleet` rejects a missing
stage. An empty vehicle mapping and an empty bridge group are valid.

The live Kit shell re-adopts the context's current stage at every `PLAY` event
before teardown or spawn. Kit can replace the Python `Usd.Stage` wrapper while
finishing a scene or timeline transition; retaining the wrapper captured by the
earlier `OPENED` event is therefore not a valid cross-event identity guarantee.

Policy descriptors are metadata-preflighted before `spawn_fleet` mutates the
stage. If later switch, drone, or policy-observer composition raises,
`spawn_fleet` tears down every locally accumulated resource before propagating
the error. The live shell records setup failures without installing its PhysX
control callback, and a later `PLAY` starts from the cleaned state.

## Per-vehicle composition

`spawn_fleet` iterates `ExperimentConfig.vehicles` in authored order. Every
vehicle is spawned. A bridge-enabled vehicle also receives one `BridgeHost` and
one state-only `IsaacArticulationIO` observer sharing the same resolved
articulation path, one frozen `DriverContext` containing the GCS
`MavlinkCommandPort` and environment, and one frozen progress-only driver state.
Empty routine steps use that same composition and complete without commands.
PX4-disabled vehicles have no bridge or driver composition.

The one `BridgeHostGroup` owns the PhysX subscription. Each tick runs every
host's sensor-send leg before any actuator-poll leg. This ordering preserves
multi-vehicle isolation and bounds lockstep latency by the slowest PX4 rather
than serially starving later sends.

Policy observers use the selected vehicle's pusher limits. Exported policy
artifacts continue to own their execution rate; shared experiment rate defaults
only author new policies and calibration commands.

## Shell responsibilities

The live `experiment_runtime` extension owns Kit stage/timeline/PhysX event
wiring and forgiving per-driver error logging. `FleetState` owns the shared
driver cadence, uses its timestamp as logical control time, and resets it on
rebuild/teardown. Driver creation and due cycles each receive one shared
`DriverTick` for the whole fleet. The standalone runner owns the
blocking `SimulationApp.update()` loop plus a matching PhysX subscription,
strict-versus-forgiving failure
reaction, and integration-only checks such as ground settling, first-heartbeat
waiting, pre-arm tilt, fault injection, and bridge-log validation.

Neither shell owns scene, flight, bridge, or routine logic. Timing helpers take
a sim-time callable so they do not depend on resource properties.

## Failure contract

Invalid lifecycle order or configuration raises. Recoverable runtime bridge
errors are logged by `BridgeHost` without taking down other vehicles. Driver
failure notification is injected through `FleetDriverEnv` and never raises;
strict shells report the preserved error after terminal recovery, while
forgiving shells log recovery progress and keep the simulator running.
If a bridge send or receive leg raises, `BridgeHost` abandons both its host-level
poll marker and the frozen bridge state's pending tick before the next physics
step.

The runtime modules are Isaac-only lazy leaves. `robotics_experiments.runtime`
must remain importable without Isaac; importing a lazy runtime symbol may load
the Isaac dependencies.

Flight capture may pass resolved `routine_steps` to `build_drivers`, alongside
its injected `policy_factory`. The capture shell derives these steps from its
canonical physical stage schedule and records them in producer evidence. Ordinary
runtime callers use each vehicle's authored routine steps. The strict scenario's
optional `physics_tick_hook` receives the rebound frozen fleet immediately after
each physics/driver tick; it is a recording hook, not a second driver clock.
