# Multi-Vehicle PX4 Bridge - Implementation Plan

> Status: planned, not started. This document is a concrete, ordered
> implementation plan intended to be executed by an automated agent without
> further design refinement. Every decision below is fixed; do not re-open them.
> When a step changes project conventions, mirror it into `AGENTS.md` per the
> repo's hard rules.

## 1. Goal and intent

Lay the scaffolding for future multi-drone simulations. After this change, a
single Isaac Sim scene can host N independent drones, each connected to its own
PX4 SITL instance over its own MAVLink simulator port. Drones may have different
assets, airframes, and per-vehicle bridge configuration. The design is modular
and scalable: adding a vehicle is a config-only change, and each vehicle's bridge
is a fully independent instance reusing the existing single-vehicle bridge code
unchanged.

The single-vehicle bridge already works. The bridge component itself
(`Px4Bridge`, `Px4MavlinkPort`, thrust model, `IsaacArticulationIO`) is already
per-instance and needs **no changes** for multi-vehicle: the simulator port is
`connection_baseport + vehicle_id` (`mavlink.py` -> Pegasus
`px4_mavlink_backend.py:258-264`). What is missing is the layer above it:
per-vehicle configuration, per-vehicle spawning, and constructing/ticking one
bridge host per vehicle.

## 2. Fixed design decisions

These were decided with the requester and are not open for change:

1. **Config shape.** A new top-level `vehicles:` list. Each entry composes a
   `drone:` override and a `px4:` override over the global `drone`/`px4`
   defaults, plus an `id` and a `spawn_xy`. Everything in `drone`/`px4` is
   overridable per vehicle.
2. **The list is the only path.** There is no scalar single-drone code path
   after this change. One vehicle = a one-element `vehicles` list. The default
   config ships a one-element list reproducing today's single drone exactly.
3. **Scope: Isaac/bridge side only.** No PX4 Docker/launch changes in this plan.
   Running N PX4 SITL instances (instance `i` -> port `4560 + i`) is assumed to
   be provided by the operator out of band. See Out of Scope.
4. **N independent `BridgeHost`s.** Each vehicle gets its own `BridgeHost`, which
   already owns its own physics-step subscription and `SimClock`. The
   scene-owning extension holds a list of them. No central orchestrator.
5. **Serial blocking lockstep.** Each `BridgeHost`'s physics callback runs its
   own blocking `exchange()` in turn; PhysX invokes the N callbacks sequentially
   in registration order. Per-tick wall-clock latency is the sum across
   vehicles. This is accepted for v1. (Parallel two-phase exchange is future
   work.)
6. **Per-vehicle asset/airframe.** Each vehicle may use a different USD asset and
   airframe (different rotor count and motor mapping). Rotor discovery already
   runs per spawned prim from authored USD metadata, so this works as long as
   each asset carries its own `drone:rotorMarker` / `drone:px4MotorIndex` /
   `drone:spinDir` metadata.
7. **Spawn pose.** Per-vehicle explicit `spawn_xy` from config; stationary
   ground Z computed from asset bounds (as today); yaw fixed at 0. Deterministic
   and reproducible.
8. **Tests: unit only.** Host-side pytest with the existing fakes. No new
   integration test, no PX4 process required.

## 3. Architecture after the change

```
EnvConfig
  .drone   (global default DroneConfig)        \  merge bases for each
  .px4     (global default Px4BridgeConfig)     /  vehicle entry
  .vehicles: list[VehicleConfig]
       each = { id, spawn_xy, drone: DroneConfig, px4: Px4BridgeConfig }
                                  (fully resolved: global default + entry override)

DroneSpawner Extension (on PLAY)
  for v in env.vehicles:
      path = spawn_drone(stage, v.drone, position=v.spawn_xy, yaw=0.0)
      if v.px4.enabled:
          host = BridgeHost(articulation_path=path, config=v.px4,
                            physics_hz=env.sim.physics_hz)
          host.start()                # own physx sub + own SimClock + own port
          self._bridge_hosts.append(host)

PhysX step  ->  host_0._on_physics_step -> exchange() blocks on PX4 #0 (port 4560)
            ->  host_1._on_physics_step -> exchange() blocks on PX4 #1 (port 4561)
            ->  ...                                       (serial, registration order)
```

Per-vehicle isolation:
- MAVLink simulator port: `v.px4.connection_baseport + v.px4.vehicle_id`.
- USD prim: `<default_prim>/<v.drone.prim_name>` (must be unique per vehicle).
- Bridge debug log: `v.px4.debug_log_path` rendered with a new `{vehicle}` token
  so logs never collide.

## 4. Implementation steps (ordered)

Do these in order. Run `uv run pytest` and
`uv run pre-commit run --all-files` at the end (see Acceptance).

### Step 1 - Config types (`src/robotics_experiments/config/types.py`)

1. Add a `VehicleConfig` dataclass after `Px4BridgeConfig`:

   ```python
   @dataclass
   class VehicleConfig:
       """One drone instance: its asset/spawn and its PX4 bridge config.

       ``drone`` and ``px4`` are fully resolved (global defaults merged with the
       per-vehicle override). ``spawn_xy`` is the world X/Y; Z is computed from
       asset bounds at spawn; yaw is 0.
       """
       id: int
       spawn_xy: Tuple[float, float]
       drone: DroneConfig
       px4: Px4BridgeConfig
   ```

2. Add `vehicles` to `EnvConfig`. Annotate it `List[VehicleConfig]` for the
   public contract. Note the compose-time caveat handled in Step 3: partial YAML
   entries must not be validated against the full `VehicleConfig` schema during
   `compose()`. Keep `drone` and `px4` on `EnvConfig` unchanged - they remain
   the global default templates the entries merge over.

3. Add `VehicleConfig` to the module `__all__`.

### Step 2 - Config exports (`src/robotics_experiments/config/__init__.py`, `schemas.py`)

1. Re-export `VehicleConfig` from `config/__init__.py` and add it to `__all__`
   (mirrors how `Px4BridgeConfig` is exported).
2. In `schemas.py`, import `VehicleConfig` and add it to `__all__`. Do **not**
   register it as a Hydra group node (entries are partial and are merged in
   Python). `register_schemas()` is otherwise unchanged.

### Step 3 - Vehicle resolution in the loader (`src/robotics_experiments/config/load.py`)

The vehicle list entries are partial overrides; resolve them by merging each
entry's `drone`/`px4` partial over the structured global `cfg.drone`/`cfg.px4`
with `OmegaConf.merge` (which both fills defaults and rejects unknown keys),
then build typed `VehicleConfig` instances.

1. Add a private helper:

   ```python
   def _resolve_vehicles(cfg) -> list[VehicleConfig]:
       # cfg is the composed DictConfig (cfg.drone / cfg.px4 are structured).
       raw = list(cfg.get("vehicles") or [])
       if not raw:
           raise ValueError("config 'vehicles' must contain at least one entry")
       resolved: list[VehicleConfig] = []
       for entry in raw:
           drone = OmegaConf.merge(cfg.drone, entry.get("drone", {}))
           px4 = OmegaConf.merge(cfg.px4, entry.get("px4", {}))
           resolved.append(
               VehicleConfig(
                   id=int(entry["id"]),
                   spawn_xy=tuple(entry["spawn_xy"]),
                   drone=cast(DroneConfig, OmegaConf.to_object(drone)),
                   px4=cast(Px4BridgeConfig, OmegaConf.to_object(px4)),
               )
           )
       _validate_vehicles(resolved)
       return resolved
   ```

2. Add `_validate_vehicles(vehicles)` that raises `ValueError` (invalid config
   must fail loudly, never silently) on any of:
   - duplicate `id`;
   - duplicate spawn endpoint
     `(px4.connection_type, px4.connection_ip, px4.connection_baseport + px4.vehicle_id)`;
   - duplicate `drone.prim_name`;
   - `spawn_xy` not length 2;
   - more than one vehicle resolving to the **same** `px4.debug_log_path`
     template string (would overwrite each other's logs). With the default
     template carrying `{vehicle}` (Step 6) this only trips on a misconfigured
     override.

3. In `get_config()`, after `compose(...)` and before/around `to_object`,
   produce the resolved list and attach it. To avoid compose-time strict
   validation of partial entries, the `vehicles` key must be loosely typed when
   the structured `base_env` node validates it. Concretely: declare the
   compose-facing field as untyped (`List[Any]`) so `compose()` accepts partial
   dicts, build the typed list via `_resolve_vehicles(cfg)`, and set it on the
   returned `EnvConfig` object:

   ```python
   env = cast(EnvConfig, OmegaConf.to_object(cfg))   # env.vehicles = raw dicts here
   env.vehicles = _resolve_vehicles(cfg)             # typed, validated
   ```

   If the structured-config typing of `EnvConfig.vehicles` makes `compose()`
   reject partial entries, set the registered field annotation to `List[Any]`
   and keep the public/runtime value as `list[VehicleConfig]` (the runtime
   assignment above). Add a short comment in `types.py` documenting this split.

### Step 4 - Default YAML (`configs/experiment/config.yaml`)

Add a one-element `vehicles` list inline (reproduces today's single drone). The
empty `drone:` means "use the global drone defaults"; `px4` only pins
`vehicle_id: 0`.

```yaml
defaults:
  - base_env
  - sim: default
  - drone: default
  - switches: default
  - px4: default
  - isaac_lab: default
  - _self_

# Multi-vehicle scaffolding. Each entry merges its drone/px4 over the global
# drone/px4 defaults above. One entry = today's single-drone behavior. Add
# entries (distinct id, vehicle_id, prim_name, spawn_xy) to add drones.
vehicles:
  - id: 0
    spawn_xy: [0.0, 0.0]
    px4: {vehicle_id: 0}
```

Also add a commented two-vehicle example in the same file (or a sibling
`configs/experiment/` doc comment) showing a second entry with a different
`drone.usd_path`, `drone.prim_name`, `vehicle_id: 1`, and `spawn_xy`, so the
extension/override pattern is discoverable.

### Step 5 - Spawning (`src/robotics_experiments/drone/spawn.py`)

Generalize `spawn_drone` to support an explicit position and to return the prim
path. Keep it backward compatible (existing callers/tests that pass no position
keep the current random-pose behavior).

1. Change the signature to:

   ```python
   def spawn_drone(
       stage,
       config: DroneConfig | None = None,
       *,
       position: tuple[float, float] | None = None,
       yaw: float = 0.0,
   ) -> str:
   ```

   - When `position is not None`: skip random sampling and the guide cube; place
     the drone at `(position[0], position[1], 0)` with the given `yaw`; if
     `config.spawn_stationary`, compute the stationary Z from asset bounds (reuse
     `_computed_stationary_z`) and re-apply; clear velocities. This is the
     multi-vehicle path.
   - When `position is None`: current behavior unchanged.
   - Return the drone prim path (`<parent>/<config.prim_name>`) in both cases.

2. `despawn_drone` is unchanged; it already removes `<parent>/<config.prim_name>`
   and is called once per vehicle with that vehicle's drone config.

3. Update the module docstring/`__all__` only if signatures' public behavior
   notes need it; do not remove existing public exports.

### Step 6 - Per-vehicle log path token (`src/robotics_experiments/px4_bridge/mavlink.py`)

Prevent bridge-log collisions between vehicles.

1. In `_resolve_log_path_template`, add a `vehicle` substitution:

   ```python
   def _resolve_log_path_template(log_path, now, vehicle_id):
       return Path(log_path.format(
           date=now.strftime("%Y-%m-%d"),
           time=now.strftime("%H_%M_%S"),
           timestamp=now.strftime("%Y-%m-%d/%H_%M_%S"),
           vehicle=vehicle_id,
       )).expanduser()
   ```

2. Thread `vehicle_id` from `Px4MavlinkPort.__init__` through
   `_configure_file_logging` into `_resolve_log_path_template`.

3. Update the default `px4.debug_log_path` in
   `configs/experiment/px4/default.yaml` to include the token, e.g.:
   `/isaac-sim/proj/px4/logs/{date}/{time}_v{vehicle}_bridge.log`. The `HH_MM_SS`
   stem stays first, so flight-analysis pairing (by date dir + `HH_MM_SS`) still
   works; the `_v{vehicle}` segment disambiguates vehicles. Update the
   `Px4BridgeConfig.debug_log_path` docstring in `types.py` to list the new
   `{vehicle}` placeholder, and note the rename in `flight_analysis/README.md`
   (the documented bridge-log filename pattern).

### Step 7 - Extension wiring (`isaac/runtime/extensions/drone_spawner/DroneSpawner_python/extension.py`)

Replace the single-drone setup/teardown with iteration over `env.vehicles`.

1. In `on_startup`, initialize `self._bridge_hosts: list[BridgeHost] = []` and
   `self._spawned_drones: list[DroneConfig] = []` (used for teardown).

2. `_setup_drone`:

   ```python
   def _setup_drone(self):
       if self.stage is None:
           return
       self._teardown_drone()
       env = _get_experiment_config()
       for v in env.vehicles:
           path = spawn_drone(self.stage, v.drone, position=tuple(v.spawn_xy), yaw=0.0)
           self._spawned_drones.append(v.drone)
           if not v.px4.enabled:
               continue
           host = BridgeHost(
               articulation_path=path,
               config=v.px4,
               physics_hz=env.sim.physics_hz,
           )
           host.start()
           self._bridge_hosts.append(host)
   ```

   `BridgeHost` derives the articulation path from `config.articulation_path or
   articulation_path`; pass the spawned `path` so each host targets its own prim.
   (`v.px4.articulation_path` is normally empty per vehicle.)

3. `_teardown_drone`: stop every host, clear the list, then despawn every
   spawned drone:

   ```python
   def _teardown_drone(self):
       for host in self._bridge_hosts:
           host.stop()
       self._bridge_hosts = []
       for drone_cfg in self._spawned_drones:
           despawn_drone(self.stage, drone_cfg)
       self._spawned_drones = []
   ```

4. `BridgeHost` itself needs **no changes**: it is already a single-instance
   owner with its own physx subscription, `SimClock`, and port. N instances run
   independently and PhysX calls their callbacks serially (registration order),
   which is exactly the accepted serial-blocking model.

### Step 8 - Unit tests

Add/extend host-side pytest only (no PX4, no Docker). Use the existing fakes in
`tests/_fakes.py` (`_FakeFlightStack`, `_FakePegasusBackend`,
`_install_fake_pegasus`) and the fake-Isaac pattern in
`tests/test_bridge_host.py` / `tests/test_drone_spawner_extension.py`.

1. New `tests/test_vehicles_config.py`:
   - `_resolve_vehicles` merges an entry's partial `px4`/`drone` over the global
     defaults (vehicle 0 inherits all defaults; vehicle 1's overridden field
     wins, untouched fields inherit).
   - Resolved endpoints differ (`baseport + vehicle_id` => 4560, 4561).
   - `_validate_vehicles` raises on: duplicate `id`; duplicate endpoint;
     duplicate `prim_name`; bad `spawn_xy` length; colliding `debug_log_path`.
   - Default config composes to exactly one vehicle equivalent to today's drone
     (compose via `get_config()` with `reset_config()`; assert
     `len(env.vehicles) == 1` and the resolved `drone`/`px4` equal the global
     defaults).

2. Extend `tests/test_drone_spawner_extension.py`:
   - Monkeypatch `get_config` to return an `EnvConfig` with two `VehicleConfig`s
     (distinct `vehicle_id`, `prim_name`, `spawn_xy`); PLAY -> two `BridgeHost`s
     started with distinct articulation paths and ports; STOP -> both stopped and
     both prims despawned. Assert `spawn_drone` was called with each vehicle's
     `position`.

3. Extend `tests/test_bridge_host.py`:
   - Construct two `BridgeHost`s with different `vehicle_id`; assert independent
     lifecycle and distinct resolved endpoints / debug-log paths (the
     `{vehicle}` token renders distinct filenames).

4. `tests/test_core_imports.py`: confirm `VehicleConfig` lives in
   `config.types` and does not pull Isaac onto the core import path (it should
   not - it only references `DroneConfig`/`Px4BridgeConfig`).

### Step 9 - Documentation updates (required, same task)

1. `AGENTS.md`: add a short multi-vehicle subsection (under the PX4 bridge
   config notes) stating: `vehicles` list is the single source of truth; each
   entry merges `drone`/`px4` over globals; the list is the only path (one
   vehicle = one-element list); N independent `BridgeHost`s tick serially; port
   is `connection_baseport + vehicle_id`; bridge logs carry the `{vehicle}`
   token. Link this plan.
2. `isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md`:
   note that the scene-owning extension may construct N independent bridges (one
   per vehicle), each owning its own simulator port `4560 + vehicle_id`, ticked
   serially; the per-instance contract is unchanged. Add `{vehicle}` to the
   debug-log token list.
3. `flight_analysis/README.md`: update the bridge-log filename pattern to the
   `{time}_v{vehicle}_bridge.log` form.

## 5. Out of scope (explicit) and future work

These are intentionally not in this plan; they are the natural next scaffolding
layers:

- **PX4 multi-instance launch.** Running N PX4 SITL instances (per-instance
  EEPROM/param dirs, instance `i` -> port `4560 + i`) and the Docker/`px4-iris`
  changes to start them. Operator-provided for now.
- **Parallel two-phase lockstep.** A `send_sensors()` / `recv_actuators()` split
  on the port so all N sends fire before blocking on all N receives, to scale
  per-tick latency. v1 is serial.
- **Multi-vehicle integration test.** Needs N PX4 containers; out of the
  unit-only scope.
- **Per-vehicle ROS namespacing**, and the non-bridge peer connections (offboard
  `14540 + id`, uXRCE-DDS `8888 + id`) - owned by control code, not the bridge.

## 6. Acceptance criteria

- `uv run pytest` is green, including the new/extended tests in Step 8.
- `uv run pre-commit run --all-files` is clean.
- Default config composes to exactly one vehicle and reproduces today's
  single-drone behavior (no behavioral change for existing single-drone runs).
- A two-vehicle config composes to two resolved `VehicleConfig`s with endpoints
  `4560` and `4561`, two independent `BridgeHost`s, distinct prims, and distinct
  bridge-log filenames.
- Invalid multi-vehicle config (duplicate id/endpoint/prim_name, bad spawn_xy,
  colliding log path) raises a clear `ValueError` at config resolution time.
- `AGENTS.md`, `PX4_BRIDGE_CONTRACTS.md`, and `flight_analysis/README.md` reflect
  the changes.
