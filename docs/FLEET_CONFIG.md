# Fleet configuration and PX4 airframe generation

This is the single source for the multi-vehicle fleet configuration detail:
the `vehicles:` list, the `ResolvedVehicle` derivation/validation, the
config-generated PX4 SITL airframes, and the fleet runtime
(`FleetOrchestrator` + `BridgeHostGroup`). `AGENTS.md` keeps only the short
summary and points here. The bridge's own interface contract lives in
`isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md`;
related ADRs are 0008 through 0012.

## The `vehicles:` list

Multi-vehicle scenes are driven by a top-level `vehicles:` list. It is the
single source of truth and the only spawn path: one vehicle = a one-element
list (the default config ships exactly that, reproducing the single-drone
behavior), and an empty list `vehicles: []` is a valid empty fleet (0..N is
handled uniformly -- the extension spawns nothing, runs an empty
`BridgeHostGroup`, and `run-px4` starts zero containers). Each entry has an
`id`, a mandatory `ros.namespace`, a `spawn_xy`, and partial `drone`/`px4`
overrides. `id` is the PX4 numeric identity: there is no `px4.vehicle_id` knob
-- the MAVLink port, the PX4 `-i` instance index, the per-vehicle log
`{vehicle}` token, and the GCS link offset are all *derived* from `id`.
`ros.namespace` is the explicit operator-facing ROS identity for sim and real
onboard stacks (for example `/x500_01`); it is not derived from `id`.

## Resolution and validation

One private enrichment step (`config.load._resolve_fleet`) merges each entry's
`drone`/`px4` partials over the global `drone`/`px4` defaults
(`OmegaConf.merge`) into a pure `VehicleConfig`, then constructs an immutable
`ResolvedVehicle` carrying that config plus the derived runtime fields:
`mavlink_port` (`= px4.connection_baseport + id`), `px4_instance_index`
(`= id`), `gcs_port` (`= px4.gcs_baseport + id`, the per-drone GCS MAVLink
link used by control/test code; the bridge itself does not use it),
`ros_namespace` (the authored absolute ROS namespace), `ros_frame_prefix`
(derived from `ros_namespace` by stripping the leading slash and replacing
nested `/` with `_`, for future TF/message frame IDs), `debug_log_path` (the
`{vehicle}` token rendered to `id`; `{date}`/`{time}` stay for runtime), and
`prim_name`. Every consumer (Isaac orchestration, the bridge host, the
host-side PX4 launch, the standalone scenario's GCS endpoint) reads these
derived fields instead of recomputing `baseport + id`.

`_resolve_fleet` validates the resolved fleet (`_validate_fleet` --
missing/invalid/duplicate ROS namespace or derived frame prefix, duplicate
id/endpoint/`prim_name`, bad `spawn_xy`, negative id, or colliding
`debug_log_path` raise a `ValueError` at resolution time).
`get_config().vehicles` is `list[ResolvedVehicle]`. `EnvConfig.vehicles` is
annotated `List[Any]` for Hydra compose-time tolerance of partial entries; the
runtime value is `list[ResolvedVehicle]`.

## PX4 SITL launch configs and generated airframes

The host-side launch scripts read `get_enabled_px4_launch_configs()` (the
launch loop's source of `PX4_INSTANCE` / `PX4_SIM_MODEL` / `PX4_SYS_AUTOSTART`,
so non-contiguous ids keep their derived ports and each PX4 container launches
the SITL airframe matching its drone asset). Each entry is a frozen
`Px4SitlLaunchConfig` (`instance_index` / `sim_model` / `sys_autostart`)
derived by the config loader from the drone USD asset basename and the PX4
airframe defined in `configs/airframes` (`holybro_x500_v2.usda` ->
`none_x500_v2`/`4019`, `hexacopter_s550.usda` -> `none_hexa_x`/`6001`; unknown
PX4-flyable asset raises `ValueError`). The Dockerfile's px4 stage generates
these repo-managed
`none_*` posix airframe scripts from `configs/airframes`, registers them in
PX4's airframe CMake list, writes an image manifest, then runs
`make px4_sitl_default`, so USD geometry, bridge rotor arrays, and PX4 control
allocation share one source.

The airframe YAML authors only **cited per-component leaves** (each physical
value carries an inline vendor/BOM citation): the body
`total_mass`/`center_of_mass`/`diagonal_inertia` are **computed** by the USD
builder (`build_copter_usd_standalone.compute_body_mass_properties`, summing
the component box/cylinder inertias + parallel-axis, excluding the separate
pusher pad) and the bridge `rotor_constant` is **computed** by `config.load`
(`_compute_rotor_constants`) from the cited `max_thrust_per_rotor_g` +
`max_rotor_velocity` (standard-gravity gram-force -> Newton conversion, so
`T = rotor_constant * omega^2` reaches the cited static max at
`max_rotor_velocity`). Neither derived value is authored, so fixing a vendor
mismatch is a one-line leaf edit + regenerate. The default quad asset is
`holybro_x500_v2` -- a hardware-faithful Holybro X500 V2 Stage 1 X4 (MN4006 EVO
/ P134.4 13x4.4 / 0.25 m motor radius / ~2.251 kg AUW); the `hexacopter_s550`
remains a placeholder that shares the same computed-properties schema.

The `px4.sitl.sys_autostart` ids must be **free** in the posix airframe set:
posix `rcS` selects the airframe by globbing `airframes/<id>_*` and sourcing
the last match, so a stock `<id>_*` (e.g. `4001_gz_x500`) would shadow a
generated one. The quad therefore uses the free `4019` (X500 V2) slot, not
`4001`; the generator fails the build loudly on such a collision
(`generate_px4_airframes.py` `_conflicting_airframe_files`), with a
`--prune-conflicts` seam to remove a stock file and reclaim its id.
The PX4 container launcher
(`px4_sitl_common.sh::launch_px4_sitl`) reads `PX4_SIM_MODEL` (default
`none_x500_v2`/`4019` only for manual unset-env launches), so the
script-driven fleet path selects the generated per-asset airframe.

## Fleet runtime: orchestrator and bridge host group

The fleet orchestrator (`robotics_experiments.runtime.FleetOrchestrator`)
spawns each vehicle at its `spawn_xy` (yaw 0) and gives every `px4.enabled`
vehicle its own independent `BridgeHost` (constructed from a frozen
`BridgeHostParams` composing the vehicle's `px4` config with its derived
identity), all driven by a single `BridgeHostGroup` that owns the one
physics-step subscription. Each PhysX step runs in two phases -- every host's
send leg (`send_phase`: sensors out to its PX4, no block) first, then every
host's poll leg (`recv_phase`: block for that PX4's actuator). The PX4
instances therefore compute concurrently, so per-tick latency is ~the max
blocking wait across vehicles (not the sum) and the per-vehicle lockstep is
decoupled: one stalled PX4 disconnects only its own vehicle instead of
freezing the others. The port exposes the seam as `send(state)` / `poll()`,
with `exchange(state)` kept as the single-vehicle send-then-poll convenience.

Per-vehicle isolation: the MAVLink simulator port is the resolved
`mavlink_port` (`Px4MavlinkPort` takes a single resolved `port`, handed to the
vendored Pegasus backend with a zero `vehicle_id` offset), the USD prim is
`<default_prim>/<drone.prim_name>`, and bridge debug logs carry a `{vehicle}`
token so the filenames never collide. The token alone is not enough: every
bridge layer logs through the shared `robotics_experiments.px4_bridge` package
logger, so the `BridgeHost` wraps each vehicle's construction/teardown and
per-tick work in `px4_bridge.vehicle_log_context(id)`, and each per-vehicle
file handler carries a `VehicleLogFilter` (see `px4_bridge/logging.py`).
