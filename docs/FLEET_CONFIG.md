# Fleet configuration

`ExperimentConfig.vehicles` is the only spawn path. It is an insertion-ordered
mapping from an operator-facing label to a complete immutable `VehicleConfig`.
An empty mapping is a valid empty fleet.

## Authoring and inheritance

Hydra composes one explicitly named model profile per vehicle. The
`holybro_x500_v2` and `hexacopter_s550` profiles each own their USD asset path
and inherit shared spawn, bridge, and manual routine settings from an incomplete
internal base. The experiment overrides identity, pose, prim, or routine at that
same node. There is no implicit default vehicle model.

```yaml
defaults:
  - base
  - experiment/vehicle@vehicles.x500_01: holybro_x500_v2
  - override experiment/routine@vehicles.x500_01.routine: takeoff
  - _self_

vehicles:
  x500_01:
    id: 0
    ros: {namespace: /x500_01}
    spawn: {prim_name: Drone0, position_xy: [0.0, 0.0]}
```

The mapping label is used in runtime logs. `id` remains the PX4 numeric identity;
`ros.namespace` is the independent operator-facing ROS identity.

## Resolution and validation

The loader enriches each `VehicleConfig` once:

- the spawn asset selects `configs/airframes/<asset>.yaml` bridge values;
- `rotor_constant` is computed from cited thrust and maximum rotor velocity;
- MAVLink port, GCS port, bridge log path, PX4 simulator model, and PX4 autostart
  id are derived from the vehicle identity, spawn asset, and nested configs.

Repo-generated PX4 model airframes use PX4's reserved custom model range
`22000-22999`. The current bindings are `none_x500_v2`/`22000` and
`none_hexa_x`/`22001`; the airframe generator rejects project configs outside
that range.

Airframe physical values are authoritative over the generic bridge fallback.
Unknown enabled assets fail configuration loading instead of selecting the wrong
airframe. PX4 instance identity and ROS frame prefixes are derived at their
consuming boundaries from `vehicle.id` and `vehicle.ros.namespace`; they are not
stored as parallel fields.

Validation rejects invalid or duplicate IDs, ROS identities/frame prefixes,
spawn prim names, MAVLink endpoints, and rendered log paths. Consumers use the
derived fields and never recompute ports.

## Runtime

The `FleetState` lifecycle functions iterate the mapping in authored order.
`spawn_fleet` spawns each `vehicle.spawn` and creates a bridge resource for each
`vehicle.bridge.enabled`; `build_drivers` creates one frozen driver context and
one frozen progress state for each bridge-enabled vehicle, including those with
empty `vehicle.routine.steps`.
Shells explicitly rebind every returned state.
`FleetState` also owns the shared physics-to-driver cadence and logical control
time. Every due cycle passes one tick snapshot to every vehicle in authored
order, so a multi-vehicle fleet stays synchronized. Driver build, teardown, and
restart reset that cadence.

One `BridgeHostGroup` runs every send leg before every poll leg. Each PX4 instance
therefore remains isolated while lockstep latency tracks the slowest instance,
not the sum of all instances. Host-side PX4 launch uses the same ordered vehicle
mapping and airframe-derived SITL model.
