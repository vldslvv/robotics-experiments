# Flight Analysis: 00_26_50 (2026-06-11)

## Logs

- ULog: `px4/logs/2026-06-11/00_26_50.ulg` (7.2 MB)
- Bridge log: `px4/logs/2026-06-11/00_26_50_bridge.log` (1.2 MB, 10174 lines)

## Timeline

| Sim Time | Event |
|----------|-------|
| 0-9.6s | Boot, sensors streaming, PX4 disarmed |
| 9.6s | Armed (`vehicle_command_ack` cmd=400 result=0) |
| 11.4s | Takeoff detected (`vehicle_land_detected`: landed -> not-landed) |
| 12-16s | Thrust ramp: commands 100 -> 720, drone lifts off slowly |
| 16.0s | Last clean tick: all forces symmetric, attitude level, zero velocity |
| 16.2s | **Liftoff**: Z jumps 0.008 -> 0.026, pitch/roll appear simultaneously |
| 16.5s | Attitude divergence: pitch=8.3 deg, roll=2.2 deg |
| 17.0s | Rapid divergence: pitch=25 deg, roll=9.2 deg |
| 17.5s | Flip: pitch=72.7 deg, roll=71 deg, velocity 8.5 m/s lateral |
| 17.76s | `fd_pitch` failure detector triggers |
| 18.0s | Drone fully inverted, crashing |
| 18.58s | `fd_roll` failure detector triggers |
| 19-30s | Post-crash: drone on ground, inverted, PX4 fighting |
| 30s+ | Settled: landed=1, motors [0, 1, 0, 1], attitude ~165 deg roll |

## Likely Root Cause: Ground-Contact-Release Impulse

The evidence points to the drone flipping due to an asymmetric impulse from
PhysX's ground contact solver at the moment of liftoff. All attitude deviations
appeared simultaneously in a single 200ms interval (ticks 4000-4050) when the
drone separated from the ground.

### Evidence

1. **Simultaneous perturbation**: Pitch (quaternion x: 8.4e-8 -> -0.00954),
   roll (quaternion y: -4.2e-8 -> -0.00122), lateral velocity (5.8e-7 ->
   -0.0129 m/s), and vertical velocity (-8.8e-6 -> +0.220 m/s) all jumped in
   the same interval. This is characteristic of a contact-release impulse, not
   a gradual instability.

2. **Bridge forces were perfectly symmetric throughout**: Commands were
   [720, 720, 720, 720] at the divergence tick. Net torque from rotor forces
   was at machine epsilon (~1e-15). The bridge code path
   (`commands_to_dynamics` -> `apply_forces` -> `apply_body_torque` ->
   `apply_body_force`) has no path that could produce an asymmetric impulse.

3. **PX4 reacted after the perturbation**: The first asymmetric actuator command
   (727 vs 728) arrived at sim_time 13.864s, applied between ticks 4050 and
   4100 -- after the perturbation already appeared at tick 4050.

4. **Exponential growth after perturbation**: Roll gyro grew from 0.20 -> 0.43
   -> 0.68 -> 0.96 -> 1.27 rad/s across successive 50-tick intervals. Once
   tilted, body-Z thrust acquired a lateral component that compounded the
   divergence faster than PX4's attitude controller could correct.

### Rotor Z-Offset Ruled Out

The rotor markers are at Z=+0.04m in body frame (above body center). Initial
analysis blamed this as an "inverted pendulum" effect. However, the cross
product `[x, y, z] x [0, 0, F] = [yF, -xF, 0]` shows the Z component cancels
entirely for body-Z thrust. The torque about the body center is independent of
the rotor Z position. This was confirmed by the net_torque_world values being
at machine epsilon at every sampled tick.

## Secondary Issues

### Post-Crash Inverted Thrust

After flipping, PX4 kept commanding max thrust (`vehicle_thrust_setpoint` =
-1.0), pushing the drone into the ground while inverted. Motors settled to
[0, 1, 0, 1] -- only 2 of 4 motors active. The bridge had no tilt-angle safety
check to zero forces when the drone exceeds a safe attitude.

### Magnetometer Fusion Rejected

After t=17.07s: `test_ratio=inf`, `innovation > 1 rad`. This is a consequence
of the flip, not a cause.

### Estimator Mode Change

At t=17.6s, `control_mode_flags` changed from 1051287 to 9439975. EKF lost
position validity during the violent motion.

## Verified Correct

- Accelerometer FLU->FRD conversion: `sensor_combined` acc_z = -9.79 (correct
  specific force for level drone).
- Gyroscope FLU->FRD conversion: correct.
- Quaternion FLU-ENU -> FRD-NED: PX4 yaw=-103.6 deg matches Isaac-derived
  -104.4 deg (0.8 deg error).
- Rotor position mapping: all 4 motors correctly placed for Generic Quad X.
- HIL_SENSOR `time_usec`: strictly increasing, no duplicates.
- Sensor freshness: all `fresh={imu:True mag:True baro:True}` before send.
- GPS ENU->lat/lon conversion: correct (north->lat, east->lon, up->alt).
- No ENU/NED frame mismatch in any data path.

## Attitude Divergence Detail

```
t=14.460s  roll=  -0.0  pitch=   0.0  yaw=-103.6
t=16.012s  roll=  -0.0  pitch=   0.0  yaw=-103.7
t=16.548s  roll=   2.2  pitch=   8.3  yaw=-103.0
t=17.068s  roll=   9.2  pitch=  25.0  yaw=-100.4
t=17.592s  roll=  71.0  pitch=  72.7  yaw= 144.0
t=18.080s  roll= -43.4  pitch=   1.0  yaw= 136.3
t=18.572s  roll=-107.9  pitch= -21.9  yaw=-163.8
t=19.064s  roll=-110.9  pitch=   8.6  yaw= -68.3
t=19.524s  roll=-117.7  pitch= -63.9  yaw= -49.6
t=20.032s  roll= 115.6  pitch= -59.9  yaw= -32.5
```

## Fixes Applied

1. **Increased spawn height** (`stationary_ground_clearance_m`: 0.01 -> 0.05):
   The drone now starts ~5cm above ground instead of ~1cm. This mitigates the
   suspected PhysX contact-release perturbation at liftoff. The 1-second
   startup settle period allows the drone to drop and stabilize before PX4
   connects.

2. **Added tilt safety cutoff** (`max_tilt_angle_deg: 60.0`): When the drone's
   body-Z axis deviates from world-Z by more than 60 degrees, the bridge zeros
   all rotor forces. This prevents post-crash inverted thrust and runaway.
