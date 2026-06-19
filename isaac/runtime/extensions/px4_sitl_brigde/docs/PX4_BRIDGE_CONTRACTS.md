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
- **One indivisible exchange.** A tick is a single `port.exchange(state)`:
  publish-then-block-recv cannot be split or reordered (Contract 1, *Public API
  shape*).
- **Non-finite state raises.** A NaN/Inf vehicle state is unrecoverable; the
  bridge logs it and raises rather than forwarding it into lockstep (Contract 1,
  *Failure semantics*).

## Scope and non-scope

The bridge has exactly one responsibility: orchestrate a physics tick by reading
the resolved Isaac body state, sending PX4-compatible HIL messages, receiving
PX4 actuator commands, converting them through the thrust model, and applying
the resulting forces through Isaac I/O.

The bridge does NOT:

- Own the drone USD or any scene primitive.
- Register physics step callbacks (the scene-owning extension calls into the bridge).
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
arm geometry, and match authored spin direction metadata where present.

### Construction ownership

The local implementation intentionally splits Pegasus' monolithic `Vehicle`
lifecycle into small owners. Sync work must preserve these boundaries:

| Layer | Owns | Must not own |
|-------|------|--------------|
| `DroneSpawner_python.extension` | Timeline and stage event handling; spawn on PLAY; teardown on STOP, PAUSE, shutdown, or stage-open | PX4 sockets, MAVLink, thrust math |
| `robotics_experiments.drone.spawn` | USD reference creation, runtime pose placement, stationary spawn height, velocity reset, and despawn | Physics callbacks, PX4 bridge construction |
| `BridgeHost` | PX4 enable gate, physics-step subscription, optional startup settle delay, rotor discovery, bridge lifetime, bridge-facing sim-time counter | Drone USD authoring, MAVLink packet logic |
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
[Extension startup]       register timeline/stage subscriptions only
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
[Each bridge tick]        bridge.on_physics_step(dt, sim_time_us):
                            - read resolved body state
                            - port.exchange(state) (send sensors to PX4, then
                              block for the paired actuator commands)
                            - apply_dynamics to drone body at rotor markers
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
swallow the latter; `BridgeHost._on_physics_step` contains the raise so one bad
tick does not crash the simulator loop.

### Public API shape

The contract is expressed as four explicit pieces: pure-data state objects, a
settings object, a single-call lockstep port, and a single Isaac sink call.

```python
# State objects (pure data; no Isaac / MAVLink / Pegasus deps)
# Per-tick data flow: VehicleState -> ActuatorCommands -> RotorDynamics
@dataclass
class VehicleState:        # Isaac -> bridge each tick
    position_enu; orientation_flu_enu_wxyz; linear_velocity_enu
    angular_velocity_flu; sim_time_us

@dataclass(frozen=True)
class ActuatorCommands:    # flight stack -> bridge each tick (port output, thrust-model input)
    controls               # length num_rotors; PX4 actuator references, interpretation owned by thrust model

@dataclass(frozen=True)
class RotorDynamics:       # thrust model -> Isaac each tick
    forces_n; body_torque_n_m; body_drag_n

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

# The flight-stack port: the lockstep seam (Pegasus stays hidden behind it).
class AbstractFlightStackBridge:
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def exchange(self, state: VehicleState) -> ActuatorCommands: ...  # send -> PX4, block, recv <- PX4
    @property
    def status(self) -> BridgeStatus: ...
    @property
    def num_rotors(self) -> int: ...
    @property
    def is_connected(self) -> bool: ...
    @property
    def is_armed(self) -> bool: ...

# The Isaac seam: read state in, apply a full rotor-model result out.
class ArticulationIO(Protocol):
    def read_state(self, sim_time_us: int) -> VehicleState: ...    # read <- Isaac
    def apply_dynamics(self, dynamics: RotorDynamics) -> None: ...  # send -> Isaac
    # plus initialize / shutdown / is_initialized / num_rotors

# The orchestrator ties the four pieces into a per-tick pipeline.
class Px4Bridge:
    def __init__(
        self,
        *,
        articulation_path: str,         # USD path to drone root
        rotor_link_paths: list[str],    # discovered rotor marker paths in PX4 motor-index order
        thrust_model: RotorThrustModel,
        port: AbstractFlightStackBridge,
        articulation_io: ArticulationIO | None = None,  # test seam; defaults to IsaacArticulationIO
        max_tilt_angle_deg: float = 180.0,              # zeros forces past this tilt; 180 = disabled
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

**Lockstep ordering contract:** a tick is a single `exchange(state)` call. It
sends `HIL_SENSOR` (advancing `time_usec`), then blocks for the paired
`HIL_ACTUATOR_CONTROLS`, and returns the rotor references. The send and the
blocking receive are one indivisible operation, so the causal lockstep order
cannot be broken from the caller side — there is no way to receive before
sending, or twice. `Px4Bridge.on_physics_step` is the canonical caller.

---

## Contract 2 — Bridge ↔ PX4 SITL

### Channel

MAVLink v2 over TCP. Standard PX4 Simulator MAVLink API.

| Property | Value |
|----------|-------|
| Transport | TCP |
| Default port | 4560 |
| Framing | Standard MAVLink v2 |
| Connection direction | Configurable. `tcpin`: bridge listens, PX4 dials in (Pegasus default). `tcpout`: reverse. |
| Connections per vehicle | 1 |

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
        │ RotorDynamics (per-rotor forces, body reaction torque, body drag)
        ▼
Isaac articulation (apply rotor forces and body effects)
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
