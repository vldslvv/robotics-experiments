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
- MAVLink port, GCS port, PX4 instance, ROS frame prefix, and bridge log path are
  derived from the vehicle identity and nested configs.

Airframe physical values are authoritative over the generic bridge fallback.
Unknown enabled assets fail PX4 launch resolution instead of selecting the wrong
airframe.

Validation rejects invalid or duplicate IDs, ROS identities/frame prefixes,
spawn prim names, MAVLink endpoints, and rendered log paths. Consumers use the
derived fields and never recompute ports.

## Runtime

`FleetOrchestrator` iterates the mapping in authored order. It spawns each
`vehicle.spawn`, creates a bridge for each `vehicle.bridge.enabled`, and creates
a driver for each non-empty `vehicle.routine.steps`.

One `BridgeHostGroup` runs every send leg before every poll leg. Each PX4 instance
therefore remains isolated while lockstep latency tracks the slowest instance,
not the sum of all instances. Host-side PX4 launch uses the same ordered vehicle
mapping and airframe-derived SITL model.
