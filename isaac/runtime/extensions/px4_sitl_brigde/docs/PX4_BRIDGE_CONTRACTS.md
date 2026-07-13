# PX4 Bridge — Interface Contracts

## Purpose

This document specifies the two contracts implemented by the `Px4Bridge` component, which connects an Isaac Sim drone articulation to a network-attached PX4 SITL instance.

- **Contract 1** (Isaac Sim ↔ Bridge) is an in-process Python contract executed every physics step.
- **Contract 2** (Bridge ↔ PX4 SITL) is a network protocol contract (MAVLink over TCP) using PX4's Simulator MAVLink API in lockstep mode.

Both contracts execute synchronously per physics tick. Failure on either side stalls or corrupts the simulation loop.

## Invariants (MUST)

The load-bearing rules, each detailed in the section noted. Breaking any of
these silently corrupts the flight or the lockstep loop.

- **Monotonic sim-time.** `HIL_SENSOR.time_usec` is strictly increasing and
  derived from sim time, not wall clock; the bridge passes `sim_time_us`
  straight through. Enforced at the source by `SimClock` (Contract 1, *Required
  state inputs*; Contract 2, *Lockstep semantics*).
- **Bridge listens first.** The bridge must be listening on the simulator port
  before PX4 dials in (Contract 2, *Connection lifecycle*).
- **Rotor order.** `rotor_link_paths` are passed in PX4 motor-index order and
  validated at discovery (resolve, unique, plausible arm geometry, spin match)
  (Contract 1, *Motor index ordering*).
- **Clean socket teardown.** Close the TCP connection cleanly before USD removal;
  a half-closed lockstep socket hangs the next session (Contract 1, *Lifecycle*).
- **Post-settle clock origin.** When `startup_settle_time_s > 0`, the
  bridge-facing `sim_time_us` is the accumulated time minus the settle offset, so
  it starts near zero on the first post-settle tick (Contract 1, *Construction
  checklist*).
- **Send precedes poll; all sends precede all polls.** A tick is a send leg
  (`port.send(state)`, sensors out, no block) then a poll leg (`port.poll()`,
  block for the paired actuator). Within a vehicle the send strictly precedes its
  poll. Across vehicles the coordinator runs every vehicle's send before any
  vehicle's poll, so a stalled PX4 cannot starve another's sensor send.
  `port.exchange(state)` is the single-vehicle convenience that composes
  send-then-poll and stays byte-for-byte the old indivisible tick (Contract 1,
  *Public API shape*).
- **Non-finite state raises.** A NaN/Inf vehicle state is unrecoverable; the
  bridge logs it and raises rather than forwarding it into lockstep (Contract 1,
  *Failure semantics*).
- **Sensor cadence == physics cadence.** The IMU must refresh on every tick the
  bridge sends; if it does not, the bridge disconnects with "sensor cadence
  mismatch" rather than shipping `HIL_SENSOR` packets with cleared accel/gyro
  bits (Contract 2, *Sensor cadence and actuator freshness*).
- **Fresh actuators only.** Prime-then-block startup prevents a stale actuator
  backlog; after PX4's first actuator the queue is drained newest-last, and a
  persistently stale backlog trips `max_sustained_actuator_lag_s` (Contract 2,
  *Sensor cadence and actuator freshness*).
- **Gating preserves lockstep.** Safety gating zeros applied rotor forces only;
  the per-tick MAVLink exchange still runs so `HIL_SENSOR.time_usec` keeps
  advancing (Contract 2, *Sensor cadence and actuator freshness*).

## Scope and non-scope

The bridge has exactly one responsibility: orchestrate a physics tick by reading
the resolved Isaac body state, sending PX4-compatible HIL messages, receiving
PX4 actuator commands, converting them through the thrust model, and applying
the resulting forces through Isaac I/O.

The bridge does NOT:

- Own the drone USD or any scene primitive.
- Register physics step callbacks (the fleet runtime's `BridgeHostGroup` calls into the bridge).
- Discover or author rotor metadata; that is handled before bridge construction.
- Publish ROS 2 topics.
- Run navigation, RL policy, or state machine logic.
- Synthesize camera, LiDAR, or RealSense data.
- Know about switches, lamps, or environment objects.
- Handle the offboard MAVLink connection (UDP 14540) used by control code — that is a separate peer connection, not routed through the bridge.

---

## Contract 1 — Isaac Sim ↔ Bridge

### Channel

In-process Python method calls. No serialization, no network. The bridge is a Python object instantiated within the Isaac Sim process.

### Direction of data flow

| Direction | Frequency | Payload |
|-----------|-----------|---------|
| Isaac → Bridge | every physics tick | ground-truth state of drone articulation |
| Bridge → Isaac | every physics tick | per-rotor forces in Newtons, body reaction torque, body drag |

The bridge is reactive: it has no opinion about when ticks happen. When called, it returns commands derived from the most recent rotor outputs PX4 produced.

### Required state inputs (Isaac → Bridge each tick)

| Field | Frame | Units | Source on Isaac side |
|-------|-------|-------|---------------------|
| Position | ENU world | meters | resolved body prim pose when available; articulation root only as a logged fallback |
| Orientation | FLU body relative to ENU world, quaternion `(w, x, y, z)` | dimensionless | resolved body prim pose when available; articulation root only as a logged fallback |
| Linear velocity | ENU world | m/s | resolved body prim velocity when available; articulation root only as a logged fallback |
| Angular velocity | FLU body | rad/s | resolved body prim angular velocity when available; articulation root only as a logged fallback |
| Sim time | — | microseconds | `BridgeHost` sim-time counter, reset to zero after any startup settle delay |

### Frame conventions

These conventions MUST be documented in a single `frames.py` module and never crossed elsewhere in the codebase:

- **Stage up-axis:** Isaac Sim defaults vary. Many demos use Y-up; most robotics scenes (and Pegasus) use Z-up. Confirm the project's choice once and treat it as fixed. PX4 expects ENU world (X-east, Y-north, Z-up). The bridge converts at the boundary.
- **Quaternion ordering:** Isaac's `get_world_pose` returns `(w, x, y, z)`. Some other Isaac APIs return `(x, y, z, w)`. The bridge accepts ONE convention only and converts at the input boundary.
- **PX4 body frame:** FLU (X-forward, Y-left, Z-up).
- **Angular velocity is world-frame.** Isaac articulation angular velocity is
  world-frame data and MUST be rotated into body-FLU before MAVLink converts it
  to PX4 body-FRD gyro fields. Treating it as already body-frame was the root
  cause of a past takeoff attitude divergence.

### Returned commands (Bridge → Isaac each tick)

A flat numpy array of length `num_rotors`. Each entry is the **force in Newtons** to apply to the drone rigid body at the corresponding rotor thrust marker position, along the body-local thrust axis (typically +Z, matching USD authoring convention).

For PX4/Pegasus operation the thrust model also returns the Pegasus body-Z rotor reaction torque and body-frame linear drag. The Isaac adapter applies these at the body origin so yaw authority and damping match Pegasus Simulator.

### Separation of concerns: bridge vs thrust model

| Concern | Owned by |
|---------|----------|
| MAVLink protocol, lockstep, network | Bridge |
| PX4/Pegasus rotor reference → Newtons | Thrust model |
| Per-motor scaling (domain randomization) | Thrust model |
| Response lag / actuator delay | Thrust model |
| Voltage sag effects | Thrust model |

This separation is required because the thrust model is what gets randomized during RL training. Entangling it with MAVLink logic blocks domain randomization.

### Motor index ordering

The single most error-prone detail. The default runtime discovers rotor thrust
marker USD paths from the spawned asset and passes them to `Px4Bridge` in **PX4
motor-index order**, not in scene-graph order or alphabetical order. These
marker prims locate where thrust is applied on the drone body; they are not
simulated as separate rotor bodies.

For PX4's `Generic Quadrotor X` airframe:

- Index 0 → motor 1: front-right, CW
- Index 1 → motor 2: rear-left, CW
- Index 2 → motor 3: front-left, CCW
- Index 3 → motor 4: rear-right, CCW

For PX4's `Octocopter Coaxial X` (Stage 2 X8): 8 motors with airframe-specific upper/lower CW/CCW pairing — different list, must be re-derived per airframe.

A wrong list means the drone arms, spins up, and immediately flips. Rotor
discovery must validate that marker paths resolve, are unique, have plausible
arm geometry, and match authored spin direction metadata where present. The
default runtime discovers markers from USD metadata: prims with
`drone:rotorMarker = true`, ordered by `drone:px4MotorIndex`, with spin from
`drone:spinDir`. PX4 geometry is FRD while Isaac marker positions are FLU, so PX4
right maps to Isaac negative Y when resolving marker positions.

### Construction ownership

The local implementation intentionally splits Pegasus' monolithic `Vehicle`
lifecycle into small owners. Sync work must preserve these boundaries:

| Layer | Owns | Must not own |
|-------|------|--------------|
| `robotics_experiments.runtime.FleetOrchestrator` | The host-agnostic runtime pipeline: open scene + apply settings, spawn the fleet, build per-vehicle hosts and wrap them in one `BridgeHostGroup`, build per-drone routine drivers, teardown. Driven by a thin per-host shell (the live `experiment_runtime` Kit extension or the standalone runner) | Timeline/Kit/SimApp lifecycle (the shell owns it), PX4 sockets, MAVLink, thrust math |
| `robotics_experiments.drone` (`spawn_drone` / `despawn_drone`) | Public stage-authoring contract for USD reference creation, runtime pose placement, stationary spawn height, velocity reset, and despawn; backed by an Isaac-only private implementation | Physics callbacks, PX4 bridge construction |
| `BridgeHostGroup` (`robotics_experiments.runtime.BridgeHostGroup`) | The single fleet-wide physics-step subscription; running every host's send leg before every host's poll leg per step | Per-vehicle bridge/clock/rotor state |
| `BridgeHost` (`robotics_experiments.runtime.BridgeHost`) | PX4 enable gate, per-leg `send_phase` / `recv_phase`, optional startup settle delay, rotor discovery, bridge lifetime, bridge-facing sim-time counter | Physics-step subscription (the group owns it), drone USD authoring, MAVLink packet logic |
| `Px4Bridge` | Per-tick orchestration across Isaac I/O, MAVLink port, thrust model, and force application | Timeline events, PX4 process ownership, rotor discovery |
| `Px4MavlinkPort` | PX4 simulator MAVLink connection, heartbeat, lockstep transaction, Pegasus sensors/GPS, actuator parsing | Isaac scene access or force application |
| `IsaacArticulationIO` | Body and rotor prim handles, body-vs-root diagnostics, state sampling, force/torque application | MAVLink or thrust model policy |

### Construction checklist

Before a PX4 bridge tick is allowed to run:

- `px4.enabled` is true. Disabled PX4 must not construct `BridgeHost` or import
  the bridge internals.
- The drone root exists under the stage default prim, unless
  `px4.articulation_path` explicitly overrides it.
- The resolved body prim exists and is the same rigid body used for state
  sampling, body drag, and body reaction torque.
- Rotor marker paths resolve, are unique, are in PX4 motor-index order, and have
  nonzero XY arm length.
- Authored rotor spin direction metadata, when present, matches the discovered
  PX4/Pegasus `rot_dir`.
- The physics step rate is compatible with PX4 lockstep and the configured
  `lockstep_timeout_s`.
- If `startup_settle_time_s > 0`, Isaac physics may run before the PX4
  simulator MAVLink endpoint opens. The bridge-facing `sim_time_us` starts at
  `0` on the first post-settle bridge tick.

`startup_settle_time_s` is an intentional local extension. It lets the spawned
drone settle in Isaac physics before PX4 initializes its estimator from HIL
sensor data. Keep it unless PX4 estimator behavior shows that delayed startup is
harmful.

### Lifecycle

```
[Shell startup]           FleetOrchestrator.open_scene(); register timeline/stage
                            subscriptions (live) or run the phases in sequence
                            (standalone)
        │
        ▼
[Play pressed]            spawn runtime drone USD; construct BridgeHost only
                            when px4.enabled is true
        │
        ▼
[BridgeHost.start()]      subscribe to physics steps; optionally wait for
                            startup_settle_time_s before constructing Px4Bridge
        │
        ▼
[Bridge start]            discover rotor mapping, construct port/thrust model,
                            acquire body and rotor handles, open TCP connection,
                            perform PX4 handshake
        │
        ▼
[Each bridge tick]        bridge.on_physics_step(dt, sim_time_us)
                          = send_sensors(dt, sim_time_us) + recv_and_apply():
                            - send_sensors: read resolved body state;
                              port.send(state) (sensors to PX4, no block)
                            - recv_and_apply: port.poll() (block for the paired
                              actuator commands); apply_dynamics to drone body at
                              rotor markers; then update_rotor_spin (cosmetic-only
                              visual rotor spin, never feeds back into physics)
                          (multi-vehicle: BridgeHostGroup runs all send_phase
                           legs, then all recv_phase legs, across vehicles)
        │
        ▼
[Stop / pause / teardown] BridgeHost.stop(), bridge.stop(), then despawn drone:
                            - close TCP connection cleanly before USD removal
                            (critical — half-closed lockstep socket
                             will hang the next session)
```

The bridge host and bridge MUST be safe to start/stop multiple times within one
Isaac session — common during development when pressing Play repeatedly.

### Failure semantics

| Failure | Bridge behavior | Observable to Isaac side |
|---------|-----------------|--------------------------|
| PX4 not running when `start()` called | Retry connection for N seconds, then raise | `start()` raises; scene init fails fast |
| TCP disconnect mid-flight | Set `is_connected = False`, return zero commands | Drone loses thrust; state machine sees flag flip and triggers ABORT |
| Lockstep response timeout (PX4 not replying within deadline) | Same as above | Same as above |
| Articulation handle still `None` on `start()` | Defer one tick, then raise | Scene wiring bug — loud failure is correct |
| Non-finite (NaN/Inf) articulation state | Log the offending state and raise `RuntimeError` | `on_physics_step` raises; `BridgeHost`'s per-tick `try/except` keeps the sim loop alive; persistent corruption surfaces loudly |

**Deferred:** last-good-state caching (cache the last finite `VehicleState` and,
on a non-finite tick, send `HIL_SENSOR` from it with a bumped `sim_time_us`
while applying zero forces) is *not* implemented. The bridge currently raises on
the first non-finite tick. The breadcrumb design lives in the comment block in
`bridge.py::Px4Bridge.on_physics_step`.

**Principle:** the bridge never silently corrupts the simulation. It either
succeeds or surfaces a clean error the surrounding system can interpret — a flag
flip for recoverable failures (disconnect, timeout), or a raised exception for
unrecoverable state (non-finite vehicle state). The bridge itself does not
swallow the latter; `BridgeHost`'s per-tick `send_phase` / `recv_phase` try/except
contains the raise so one bad tick does not crash the simulator loop.

### Public API shape

`robotics_experiments.px4_bridge.__init__` is the only supported public import
surface for this package. Runtime, command, RL, tests, and extension shims must
import bridge concepts from the package root, for example
`from robotics_experiments.px4_bridge import Px4Bridge, VehicleState`. Internal
submodules remain implementation detail even when their names are stable inside
the package.

The contract is expressed as four explicit pieces: pure-data state objects, a
settings object, a single-call lockstep port, and a single Isaac sink call.

```python
# State objects (pure data; no Isaac / MAVLink / Pegasus deps)
# Per-tick data flow: VehicleState -> ActuatorCommands -> RotorDynamics
@dataclass(frozen=True)
class VehicleState:        # Isaac -> bridge each tick
    position_enu; orientation_flu_enu_wxyz; linear_velocity_enu
    angular_velocity_flu; sim_time_us

@dataclass(frozen=True)
class ActuatorCommands:    # flight stack -> bridge each tick (port output, thrust-model input)
    controls               # length num_rotors; PX4 actuator references, interpretation owned by thrust model

@dataclass(frozen=True)
class RotorDynamics:       # thrust model -> Isaac each tick
    forces_n; body_torque_n_m; body_drag_n
    rotor_omega_rad_s      # per-rotor angular velocity (rad/s); additive, cosmetic-only (drives visual spin), never re-enters physics or lockstep

@dataclass(frozen=True)
class BridgeStatus:        # read-only runtime snapshot for consumers
    is_connected; is_armed; first_heartbeat_received
    sensor_send_count; actuator_receive_count
    last_sensor_send_us; last_actuator_receive_us; actuator_lag_us
    last_disconnect_reason

# Settings object for the lockstep exchange
@dataclass(frozen=True)
class LockstepSettings:
    lockstep_timeout_s; first_heartbeat_timeout_s; bootstrap_prime_ticks
    first_actuator_timeout_s; max_sustained_actuator_lag_s
    heartbeat_rate_hz; physics_rate_hz

# The PX4 flight-stack port: the lockstep seam (Pegasus stays hidden behind it).
class Px4MavlinkPort:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def send(self, state: VehicleState) -> ActuatorCommands | None: ...  # sensors -> PX4, no block; None => poll owed
    def poll(self) -> ActuatorCommands: ...  # block, recv <- PX4 (only after send returned None)
    def exchange(self, state: VehicleState) -> ActuatorCommands: ...  # send-then-poll; single-vehicle convenience
    @property
    def status(self) -> BridgeStatus: ...
    @property
    def num_rotors(self) -> int: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_armed(self) -> bool: ...

# The Isaac seam behavior: read state in, apply a full rotor-model result out.
# Implemented by IsaacArticulationIO; the local structural protocol is private.
class _ArticulationIO(Protocol):
    def read_state(self, sim_time_us: int) -> VehicleState: ...    # read <- Isaac
    def apply_dynamics(self, dynamics: RotorDynamics) -> None: ...  # send -> Isaac (physics sink)
    def update_rotor_spin(self, rotor_omega_rad_s, dt) -> None: ...  # cosmetic-only visual rotor spin; never touches forces/lockstep
    # plus initialize / shutdown / is_initialized / num_rotors

# The orchestrator ties the four pieces into a per-tick pipeline.
class Px4Bridge:
    def __init__(
        self,
        *,
        articulation_path: str,         # USD path to drone root
        rotor_link_paths: list[str],    # discovered rotor marker paths in PX4 motor-index order
        thrust_model: RotorThrustModel,
        port: Px4MavlinkPort,
        articulation_io: _ArticulationIO | None = None,  # test seam; defaults to IsaacArticulationIO
        max_tilt_angle_deg: float = 180.0,              # zeros forces past this tilt; 180 = disabled
        rotor_spin_visual_scale: float = 0.03,          # cosmetic-only spin scale; passed to the default IsaacArticulationIO
    ): ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def on_physics_step(self, dt: float, sim_time_us: int) -> None: ...

    @property
    def status(self) -> BridgeStatus: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_armed(self) -> bool: ...
```

**Lockstep ordering contract:** a tick decomposes into `send(state)` -- which
sends `HIL_SENSOR` (advancing `time_usec`) without blocking and returns `None`
when an actuator is owed -- and `poll()`, which blocks for the paired
`HIL_ACTUATOR_CONTROLS` and returns the rotor references. Within a vehicle the
send strictly precedes its poll. `exchange(state)` composes the two back into
one send-then-poll call and is the single-vehicle convenience;
`Px4Bridge.on_physics_step` (itself `send_sensors` then `recv_and_apply`) is the
canonical single-vehicle caller. The multi-vehicle `BridgeHostGroup` instead
calls every vehicle's send leg before any vehicle's poll leg, so the flight
stacks compute concurrently and one stalled PX4 cannot freeze the others; the
causal per-vehicle order (send before its own poll) is preserved.

**Cosmetic rotor spin (separate seam):** the visual spinning of the propellers
is *not* part of the four-part contract above and must never feed back into it.
`commands_to_dynamics` stays pure -- it only adds the `rotor_omega_rad_s` field
(built from the omega it already computes) -- and `apply_dynamics` stays the
unchanged physics sink. The spin lives in its own cosmetic-only seam,
`IsaacArticulationIO.update_rotor_spin(rotor_omega_rad_s, dt)`, called from
`recv_and_apply` alongside `apply_dynamics` (the relative order is immaterial:
the seam never touches physics). Because `recv_and_apply` takes no `dt`, the
tick's frozen pending state left by `send_sensors` carries it. The seam only integrates a
`RotateZ` xformOp on each rotor marker prim (visual scale `rotor_spin_visual_scale`
slows the real ~1100 rad/s omega to a non-aliasing on-screen rate); it never
applies forces, mutates the body, or participates in the MAVLink lockstep. It
runs even on a tilt-cutoff tick (forces zeroed, rotors still visibly spinning at
the real commanded omega).

---

## Contract 2 — Bridge ↔ PX4 SITL

### Channel

MAVLink v2 over TCP. Standard PX4 Simulator MAVLink API.

| Property | Value |
|----------|-------|
| Transport | TCP |
| Default port | resolved `mavlink_port = connection_baseport + id` (4560 for vehicle id 0) |
| Framing | Standard MAVLink v2 |
| Connection direction | Configurable. `tcpin`: bridge listens, PX4 dials in (Pegasus default). `tcpout`: reverse. |
| Connections per vehicle | 1 |

**Multi-vehicle:** a scene may host N vehicles (the top-level `vehicles:` config
mapping). `id` is the sole numeric identity; one enrichment step
derives each vehicle's `mavlink_port`
(`= connection_baseport + id`), PX4 `-i` index (`= id`), and log token from it
on the immutable `VehicleConfig`. The fleet runtime (`FleetOrchestrator`) then
constructs N fully independent bridges -- one per vehicle -- each owning its own
simulator port (the resolved `mavlink_port = 4560 + id`) and its own `SimClock`.
The `Px4MavlinkPort` takes that resolved `port` directly (handed to the vendored
Pegasus backend with a zero `vehicle_id` offset), so no layer recomputes
`baseport + id`. A single
`BridgeHostGroup` owns the one physics-step subscription for the whole fleet and
drives the tick in two phases: every vehicle's send leg (sensors out, no block)
runs first, then every vehicle's poll leg (block for each actuator). The PX4
instances therefore compute concurrently, so per-tick latency is ~the max
blocking wait rather than the sum, and a single stalled PX4 disconnects only its
own vehicle instead of freezing the shared physics thread for the others. The
per-instance contract on this page is unchanged.

### Connection lifecycle

```
1. Bridge opens listening socket on port 4560.
2. PX4 (in Docker) starts and connects to 4560.
3. Bridge accepts connection. MAVLink version negotiation (handled by pymavlink).
4. PX4 begins sending HEARTBEAT at 1 Hz. Bridge responds with HEARTBEAT.
5. Once both sides see heartbeats, normal lockstep operation begins.
```

**Sequencing requirement:** bridge must be listening before PX4 starts. If PX4 boots first, it retries for a configurable interval, then either keeps retrying or gives up depending on its build configuration. Standard sequence is **bridge first, PX4 second**.

### Messages — Bridge → PX4

#### `HIL_SENSOR` (msg id 107)

Frequency: every physics tick (typically 250 Hz).

Carries simulated IMU + barometer + magnetometer:

| Field | Frame | Notes |
|-------|-------|-------|
| `xacc`, `yacc`, `zacc` | Body | Linear acceleration in m/s². **Includes gravity.** Real accelerometers measure +9.81 in body-up at rest. The bridge adds a gravity vector rotated into body frame. |
| `xgyro`, `ygyro`, `zgyro` | Body | Angular velocity in rad/s |
| `xmag`, `ymag`, `zmag` | Body | Magnetic field in Gauss. Synthesized from world declination + body orientation. |
| `abs_pressure`, `pressure_alt`, `temperature` | — | Barometer |
| `fields_updated` | — | Bitmask telling PX4 which fields are valid |

Rate higher than physics step is wasteful; lower starves PX4's EKF.

Runtime IMU, barometer, and magnetometer data are produced by the vendored
Pegasus sensor models. The Hydra field `px4.sensor_noise_enabled` controls
whether Pegasus noise/bias terms are active. The default is `true`, so
integration tests and normal simulator runs exercise noisy sensors; precision
unit tests may disable it.

#### `HIL_GPS` (msg id 113)

Frequency: typically 5–10 Hz.

| Field | Notes |
|-------|-------|
| `lat`, `lon` | degrees × 1e7. Computed from world position + configurable home lat/lon. |
| `alt` | millimeters |
| `eph`, `epv` | horizontal/vertical accuracy |
| `vel`, `vn`, `ve`, `vd` | velocity components |
| `cog`, `satellites_visible`, `fix_type` | nav fix metadata |

For indoor-only operation: either send synthesized "fake but consistent" GPS centered on a fixed home (e.g., Zurich `47.3977, 8.5456` — the PX4 default), or omit GPS entirely with `COM_ARM_WO_GPS=1` configured on PX4. Synthesized GPS is the cleaner path because it keeps PX4's EKF satisfied without parameter changes.

Runtime GPS data is produced by Pegasus `GPS`, using the configured home
position. Pegasus returns noisy `latitude`/`longitude` in degrees but
ground-truth `latitude_gt`/`longitude_gt` in radians; the bridge normalizes the
ground-truth fields to degrees before handing the payload to the Pegasus
MAVLink backend.

#### `HIL_STATE_QUATERNION` (msg id 115) — optional

Frequency: typically 50 Hz.

Carries ground-truth state for logging/comparison. PX4 does NOT fuse this into its EKF (the entire point of HIL is that PX4 estimates state from `HIL_SENSOR`). Useful for debugging EKF performance.

### Messages — PX4 → Bridge

#### `HIL_ACTUATOR_CONTROLS` (msg id 93)

Frequency: 250–400 Hz, set by PX4's mixer.

| Field | Notes |
|-------|-------|
| `controls[16]` | Array of 16 floats. First `num_rotors` entries are per-motor commands; remainder are zero or used for control surfaces. Range typically 0..1 after mixer normalization. Exact semantics vary by airframe. |
| `mode` | Actuator mode (PX4 safety) |
| `flags` | Armed state, lockstep flags |

The bridge consumes `controls[0:num_rotors]`, applies Pegasus offset/scaling (config: `input_offset`, `input_scaling`, `zero_position_armed`), and forwards rotor angular velocity references to the Pegasus-compatible quadratic thrust model.

#### `HEARTBEAT`

Frequency: 1 Hz from PX4.

Reports system state (`MAV_STATE_ACTIVE`, `MAV_STATE_STANDBY`, etc.) — used by the bridge to determine whether PX4 has armed.

#### Other messages

`STATUSTEXT` (debug strings), `SYSTEM_TIME` (PX4 internal clock), `COMMAND_ACK` (responses to commands the bridge might send). Mostly logged or ignored.

### Lockstep semantics

Lockstep is what makes simulation reproducible. Without it, PX4 runs at wall-clock and Isaac runs at physics-rate, and they drift.

**Algorithm:**

1. PX4 advances its internal clock based on `HIL_SENSOR.time_usec`, NOT wall clock.
2. PX4 will not produce a new `HIL_ACTUATOR_CONTROLS` until it has received a `HIL_SENSOR` with a more recent timestamp.
3. The bridge will not advance Isaac Sim past the next physics step until it has received the next `HIL_ACTUATOR_CONTROLS`.

This produces strict alternation:

```
Bridge sends HIL_SENSOR(t=N)
       │
       ▼
PX4 advances clock to N, runs EKF + controller + mixer
       │
       ▼
PX4 sends HIL_ACTUATOR_CONTROLS(t=N)
       │
       ▼
Bridge applies forces; Isaac integrates physics; sim time → N+dt
       │
       ▼
Bridge sends HIL_SENSOR(t=N+dt)   ── loop
```

### Implementation requirements derived from lockstep

- **`port.exchange()` is blocking.** After sending this step's sensors, it blocks until PX4 has responded with actuator commands for this step before returning. If PX4 is slow, Isaac Sim slows down.
- **Timeouts MUST exist.** Recommended default: 100 ms. If PX4 doesn't reply in 100 ms, the bridge marks itself disconnected, returns zero commands, and lets the surrounding system handle recovery. A frozen lockstep without a timeout means a frozen Isaac Sim.
- **Timestamps MUST be sim time, not wall clock.** PX4 uses `HIL_SENSOR.time_usec` to advance its own clock. Wall-clock timestamps break lockstep. The bridge passes `sim_time_us` straight through to `HIL_SENSOR.time_usec`.
- **Sim speed scaling works for free.** If Isaac runs at 0.5× real time, PX4's clock advances at 0.5× real time. Same for 2× speedup. PX4 has no notion of wall-clock.

### Sensor cadence and actuator freshness

These guards keep the lockstep loop fed with fresh data on both legs. They are
project-specific hardening beyond the bare lockstep algorithm above; each maps to
a `LockstepSettings` field and a distinct fatal disconnect reason. Behavior is
implemented in `src/robotics_experiments/px4_bridge/mavlink.py`.

**Sensor cadence is the physics cadence.** `sim.physics_hz` is the single source
of truth for both the Isaac physics rate and the PX4 sensor rate; the host plumbs
it into the port as `LockstepSettings.physics_rate_hz` (there is no separate
sensor-rate knob to drift). The integer-microsecond tick is derived in exactly
one place -- `_physics_tick_us(physics_hz)` in `px4_bridge/sim_clock.py`, shared by
the port and by `SimClock`. The port pins the Pegasus sensor `update_period` to
that integer-microsecond tick (rate `1e6 / tick_us`), not `1/physics_rate_hz`, so
the `update_at_rate` gate fires on every physics step at any rate;
`1/physics_rate_hz` can sit a float ULP above the per-tick dt (for example at
333 Hz) and refresh the IMU only every other tick, shipping half the `HIL_SENSOR`
packets with the accel/gyro bits cleared. If the IMU does not refresh on a tick
the bridge is about to send (Isaac stepping faster than the configured sensor
rate), that is a fatal misconfiguration, not a recoverable state: the bridge
disconnects with the distinct **"sensor cadence mismatch"** reason and returns
zero rotor commands.

**Prime-then-block startup (no actuator backlog).** PX4 needs a few priming
`HIL_SENSOR` packets while it boots before it emits its first actuator. If Isaac
free-runs during that window it floods PX4 and builds a FIFO backlog of
stale-timestamp actuators that may never drain (host-load dependent) and
destabilizes takeoff. The bridge therefore sends at most `bootstrap_prime_ticks`
priming sensor packets non-blocking, then blocks (up to `first_actuator_timeout_s`,
larger than `lockstep_timeout_s` because PX4 boot takes seconds of wallclock) for
the first actuator. `bootstrap_prime_ticks` must exceed PX4's boot sensor need
(~15) or the first-actuator wait times out and the bridge disconnects. This bounds
the backlog to ~`bootstrap_prime_ticks` ticks, so steady-state actuator lag stays
~1 tick regardless of host load -- a deliberate improvement over upstream Pegasus,
which also free-runs before the first actuator but blocks without a timeout or
drain.

**Fresh actuators (drain the backlog).** Once PX4 has produced its first
`HIL_ACTUATOR_CONTROLS`, the bridge drains queued actuator messages after each
sensor tick and applies the newest packet, so it never flies a stale queued
command.

**Sustained-lag tripwire.** `max_sustained_actuator_lag_s` is a safety tripwire on
steady-state actuator lag (current sim time minus the newest applied actuator
timestamp), not a happy-path control. Strict blocking lockstep already holds the
lag at ~1 tick, prime-then-block prevents the startup backlog, and a PX4 stall is
caught by `lockstep_timeout_s`. The ceiling catches the one remaining failure mode
-- a regression that reintroduces a stale-but-still-flowing backlog (PX4 keeps
answering, but with persistently old timestamps). On exceed the bridge disconnects
with the distinct **"actuator lag exceeded sustained limit"** reason and returns
zero rotor commands.

**Safety gating preserves lockstep.** Safety gating gates only the applied rotor
forces, never the per-tick exchange. The tilt cutoff (`max_tilt_angle_deg`) zeros
forces and skips body torque/drag but still calls the MAVLink `step` every tick,
so `HIL_SENSOR.time_usec` keeps advancing and lockstep is preserved.

### PX4-specific failure modes

| Failure | Likely cause | Diagnosis |
|---------|--------------|-----------|
| PX4 boots but never sends HEARTBEAT | Wrong port, network unreachable, wrong `connection_type` | Bridge connection-attempt timeout; check Docker network config |
| `HIL_SENSOR` accepted but no `HIL_ACTUATOR_CONTROLS` returned | Airframe mismatch — PX4 expects different actuator count (e.g., fixed-wing vs multirotor) | Check PX4 logs in container; verify `MAV_TYPE` and airframe selection |
| Lockstep deadlock | One side started before the other, or one side died mid-flight | Hard timeout in bridge; force-close socket; require explicit restart |
| PX4 refuses to arm | EKF preflight checks failing — typically GPS, magnetometer, or accelerometer health | Check `STATUSTEXT` messages; tune sensor noise or disable failing checks via PX4 params |
| Drone arms then immediately disarms | Inconsistent sensor data — common: forgetting gravity in accel, or zero magnetic field | Verify each sensor field individually with PX4 `listener sensor_combined` |
| Drone arms, takes off, drifts/spins | Motor index ordering wrong | Re-check `rotor_link_paths` against airframe marker positions |

### What the bridge does NOT expose to surrounding code

- MAVLink messages themselves (the API is `exchange(state) -> commands`, never raw MAVLink).
- The TCP socket.
- PX4 heartbeat or system-state details, beyond binary `is_connected` and `is_armed`.

### The two PX4 connections — do not conflate

| Connection | Port | Owner | Direction | Purpose |
|------------|------|-------|-----------|---------|
| Simulator API | 4560/tcp | Bridge | bidirectional | HIL sensors out, actuator commands in. Physics-rate. |
| Offboard / control | 14540/udp | Control code (MAVSDK, pymavlink, MAVROS) | bidirectional | Setpoints out, telemetry in. Offboard rate. |

These are independent peer-level connections. The bridge owns 4560 only. The state machine and RL policy talk to PX4 over 14540 directly, not through the bridge.

If using uXRCE-DDS instead of offboard MAVLink, that becomes a third connection (UDP 8888 by default), also independent of the bridge.

**Per-instance ports (multi-vehicle).** The PX4 instance index equals the vehicle `id` (`px4_instance_index = id`), and PX4 instance `i` shifts every port by `i`: the bridge-owned simulator API is `4560 + id` (the resolved `mavlink_port`), the offboard/GCS link is `14540 / 14550 + id`, and uXRCE-DDS is `8888 + id`. The bridge still owns only its own `mavlink_port`; the GCS/offboard ports are owned by PX4 and addressed by control/test code (the multi-drone integration test connects a separate GCS link to the resolved `gcs_port = px4.gcs_baseport + id` per drone, each resolving to that PX4 instance's system id).

---

## Concrete tick example

One physics step at simulation time `t = 4.000000 s`:

```
[Isaac Sim: physics step begins]

Isaac → Bridge:
    state = VehicleState(
        position_enu             = [1.2, 0.5, 1.4],
        orientation_flu_enu_wxyz = [0.99, 0.0, 0.0, 0.14],
        linear_velocity_enu      = [0.3, 0.0, 0.0],
        angular_velocity_flu     = [0.0, 0.0, 0.0],
        sim_time_us              = 4_000_000,
    )

Bridge → PX4 (TCP 4560):
    HIL_SENSOR{time_usec=4_000_000, xacc=0, yacc=0, zacc=9.81, xgyro=0, ygyro=0, zgyro=0, ...}
    HIL_GPS   {time_usec=4_000_000, lat=47.3977..., lon=8.5456..., alt=1400mm, ...}

[Bridge blocks — lockstep wait]

PX4: reads HIL_SENSOR → EKF → position controller → mixer → emits:
    HIL_ACTUATOR_CONTROLS{time_usec=4_000_000, controls=[0.52, 0.51, 0.52, 0.51, 0, 0, ...]}

Bridge → Isaac:
    commands = ActuatorCommands(controls=[0.52, 0.51, 0.52, 0.51])  # port.exchange(...) return
    forces = thrust_model.commands_to_forces(commands)
           = [4.0, 3.85, 4.0, 3.85] Newtons
    apply_force(rotor_0_link, [0, 0, 4.0])    # +Z in link's local frame
    apply_force(rotor_1_link, [0, 0, 3.85])
    apply_force(rotor_2_link, [0, 0, 4.0])
    apply_force(rotor_3_link, [0, 0, 3.85])

[Isaac: integrates physics; sim time → 4.004 s]

[Loop]
```

This sequence runs ~250 times per simulated second. Both contracts hold simultaneously. Either failing breaks the loop.

---

## Quick reference

### Per-tick data flow summary

```
Isaac articulation
        │
        │ position, orientation, velocity, ang vel, lin accel, sim time
        ▼
   ┌─────────┐       HIL_SENSOR + HIL_GPS (TCP 4560)        ┌───────────┐
   │ Bridge  │ ──────────────────────────────────────────► │ PX4 SITL  │
   │         │                                              │ (Docker)  │
   │         │ ◄────────────────────────────────────────── │           │
   └─────────┘            HIL_ACTUATOR_CONTROLS             └───────────┘
        │
        │ ActuatorCommands (port.exchange return; Pegasus rotor angular velocity references)
        ▼
   Thrust model
        │
        │ RotorDynamics (per-rotor forces, body reaction torque, body drag,
        │                plus cosmetic rotor_omega_rad_s)
        ▼
Isaac articulation (apply_dynamics: rotor forces and body effects;
                    then update_rotor_spin: cosmetic visual spin only)
```

### Owned by the bridge

- TCP socket on port 4560
- MAVLink message packing/unpacking
- Lockstep timing
- Frame conversions (Isaac world → PX4 ENU/FLU)
- Quaternion convention conversion
- Connection lifecycle (start/stop)
- `is_connected`, `is_armed` flags

### NOT owned by the bridge

- USD scene primitives
- Physics step callback registration
- Drone USD asset
- Switch / lamp / warehouse scene
- Camera, LiDAR, RealSense simulation
- ROS 2 publication
- Navigation, RL policy, state machine
- Offboard MAVLink connection (UDP 14540)
- uXRCE-DDS (if used)
- Thrust curve / motor response / domain randomization (thrust model owns these)
