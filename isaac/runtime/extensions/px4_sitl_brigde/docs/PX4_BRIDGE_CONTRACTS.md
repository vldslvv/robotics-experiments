# PX4 Bridge — Interface Contracts

## Purpose

This document specifies the two contracts implemented by the `Px4Bridge` component, which connects an Isaac Sim drone articulation to a network-attached PX4 SITL instance.

- **Contract 1** (Isaac Sim ↔ Bridge) is an in-process Python contract executed every physics step.
- **Contract 2** (Bridge ↔ PX4 SITL) is a network protocol contract (MAVLink over TCP) using PX4's Simulator MAVLink API in lockstep mode.

Both contracts execute synchronously per physics tick. Failure on either side stalls or corrupts the simulation loop.

## Scope and non-scope

The bridge has exactly one responsibility: convert articulation state to PX4-compatible HIL messages and apply PX4's actuator commands as forces on motor links.

The bridge does NOT:

- Own the drone USD or any scene primitive.
- Register physics step callbacks (the scene-owning extension calls into the bridge).
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
| Bridge → Isaac | every physics tick | per-rotor forces in Newtons |

The bridge is reactive: it has no opinion about when ticks happen. When called, it returns commands derived from the most recent rotor outputs PX4 produced.

### Required state inputs (Isaac → Bridge each tick)

| Field | Frame | Units | Source on Isaac side |
|-------|-------|-------|---------------------|
| Position | World | meters | `articulation.get_world_pose()[0]` |
| Orientation | World→Body, quaternion `(w, x, y, z)` | dimensionless | `articulation.get_world_pose()[1]` |
| Linear velocity | World | m/s | `articulation.get_linear_velocity()` |
| Angular velocity | Body | rad/s | `articulation.get_angular_velocity()` (verify body vs world per Isaac Sim version) |
| Linear acceleration | Body | m/s² | finite-difference of velocity, or direct API if available |
| Sim time | — | microseconds | `SimulationContext.current_time × 1e6` |

### Frame conventions

These conventions MUST be documented in a single `frames.py` module and never crossed elsewhere in the codebase:

- **Stage up-axis:** Isaac Sim defaults vary. Many demos use Y-up; most robotics scenes (and Pegasus) use Z-up. Confirm the project's choice once and treat it as fixed. PX4 expects ENU world (X-east, Y-north, Z-up). The bridge converts at the boundary.
- **Quaternion ordering:** Isaac's `get_world_pose` returns `(w, x, y, z)`. Some other Isaac APIs return `(x, y, z, w)`. The bridge accepts ONE convention only and converts at the input boundary.
- **PX4 body frame:** FLU (X-forward, Y-left, Z-up).

### Returned commands (Bridge → Isaac each tick)

A flat numpy array of length `num_rotors`. Each entry is the **force in Newtons** to apply to the corresponding rotor link in that link's local thrust axis (typically +Z, matching USD authoring convention).

The bridge does NOT return torques. Yaw torque emerges from differential rotor speeds via PX4's mixer; the bridge applies thrust forces only. Per-rotor reaction torque (from the spinning prop) is folded into the thrust model or simulated as a small constant body torque per rotor — handled by the thrust model, not by the bridge.

### Separation of concerns: bridge vs thrust model

| Concern | Owned by |
|---------|----------|
| MAVLink protocol, lockstep, network | Bridge |
| Normalized command (0..1) → Newtons | Thrust model |
| Per-motor scaling (domain randomization) | Thrust model |
| Response lag / actuator delay | Thrust model |
| Voltage sag effects | Thrust model |

This separation is required because the thrust model is what gets randomized during RL training. Entangling it with MAVLink logic blocks domain randomization.

### Motor index ordering

The single most error-prone detail. The bridge accepts a list of rotor link USD paths in **PX4 motor-index order**, not in scene-graph order or alphabetical order.

For PX4's `Generic Quadrotor X` airframe:

- Index 0 → motor 1: front-right, CW
- Index 1 → motor 2: rear-left, CW
- Index 2 → motor 3: front-left, CCW
- Index 3 → motor 4: rear-right, CCW

For PX4's `Octocopter Coaxial X` (Stage 2 X8): 8 motors with airframe-specific upper/lower CW/CCW pairing — different list, must be re-derived per airframe.

A wrong list means the drone arms, spins up, and immediately flips. Validate via a unit test that asserts geometric layout matches index ordering.

### Lifecycle

```
[Construction]            cheap; no sockets, no articulation handles
        │
        ▼
[Play pressed]            scene begins simulating
        │
        ▼
[First physics tick]      bridge.start() runs:
                            - acquire SingleArticulation handle (deferred init)
                            - open TCP connection to PX4
                            - perform connection handshake
        │
        ▼
[Each subsequent tick]    bridge.on_physics_step(dt, sim_time_us):
                            - read articulation state
                            - call port.step()
                            - apply forces to rotor links
        │
        ▼
[Stop / scene teardown]   bridge.stop():
                            - close TCP connection cleanly
                            (critical — half-closed lockstep socket
                             will hang the next session)
```

The bridge MUST be safe to start/stop multiple times within one Isaac session — common during development when pressing Play repeatedly.

### Failure semantics

| Failure | Bridge behavior | Observable to Isaac side |
|---------|-----------------|--------------------------|
| PX4 not running when `start()` called | Retry connection for N seconds, then raise | `start()` raises; scene init fails fast |
| TCP disconnect mid-flight | Set `is_connected = False`, return zero commands | Drone loses thrust; state machine sees flag flip and triggers ABORT |
| Lockstep response timeout (PX4 not replying within deadline) | Same as above | Same as above |
| Articulation handle still `None` on `start()` | Defer one tick, then raise | Scene wiring bug — loud failure is correct |
| NaN in articulation state | Log, send last good state | One bad tick is recoverable; persistent NaN indicates physics divergence |

**Principle:** the bridge never crashes the simulation. It either succeeds or surfaces a clean error state that the surrounding system can interpret.

### Public API shape

```python
class Px4Bridge:
    def __init__(
        self,
        *,
        articulation_path: str,         # USD path to drone root
        rotor_link_paths: list[str],    # USD paths in PX4 motor-index order
        thrust_model: RotorThrustModel,
        port: AbstractFlightStackBridge,
    ): ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def on_physics_step(self, dt: float, sim_time_us: int) -> None: ...

    @property
    def is_connected(self) -> bool: ...
    @property
    def is_armed(self) -> bool: ...
```

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

The bridge consumes `controls[0:num_rotors]`, applies offset/scaling (config: `input_offset`, `input_scaling`, `zero_position_armed`), and forwards to the thrust model.

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

- **`port.step()` is blocking.** When the bridge calls into the port to exchange messages, it blocks until PX4 has responded with actuator commands for this step. If PX4 is slow, Isaac Sim slows down.
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
| Drone arms, takes off, drifts/spins | Motor index ordering wrong | Re-check `rotor_link_paths` against airframe |

### What the bridge does NOT expose to surrounding code

- MAVLink messages themselves (the API is `step(state) -> commands`, never raw MAVLink).
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
        position_enu        = [1.2, 0.5, 1.4],
        orientation_wxyz    = [0.99, 0.0, 0.0, 0.14],
        velocity_enu        = [0.3, 0.0, 0.0],
        angular_velocity_flu = [0.0, 0.0, 0.0],
        linear_acceleration_flu = [0.0, 0.0, 9.81],   # at hover, gravity in body +Z
        timestamp_us        = 4_000_000,
    )

Bridge → PX4 (TCP 4560):
    HIL_SENSOR{time_usec=4_000_000, xacc=0, yacc=0, zacc=9.81, xgyro=0, ygyro=0, zgyro=0, ...}
    HIL_GPS   {time_usec=4_000_000, lat=47.3977..., lon=8.5456..., alt=1400mm, ...}

[Bridge blocks — lockstep wait]

PX4: reads HIL_SENSOR → EKF → position controller → mixer → emits:
    HIL_ACTUATOR_CONTROLS{time_usec=4_000_000, controls=[0.52, 0.51, 0.52, 0.51, 0, 0, ...]}

Bridge → Isaac:
    forces = thrust_model([0.52, 0.51, 0.52, 0.51])
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
        │ normalized commands [0..1] per rotor
        ▼
   Thrust model
        │
        │ forces in Newtons per rotor
        ▼
Isaac articulation (apply_force on rotor links)
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
