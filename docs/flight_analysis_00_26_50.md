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
| 16.5s | Attitude divergence begins: pitch=8.3 deg, roll=2.2 deg |
| 17.0s | Rapid divergence: pitch=25 deg, roll=9.2 deg |
| 17.5s | Flip: pitch=72.7 deg, roll=71 deg, velocity 8.5 m/s lateral |
| 17.76s | `fd_pitch` failure detector triggers |
| 18.0s | Drone fully inverted, crashing |
| 18.58s | `fd_roll` failure detector triggers |
| 19-30s | Post-crash: drone on ground, inverted, PX4 fighting |
| 30s+ | Settled: landed=1, motors [0, 1, 0, 1], attitude ~165 deg roll |

## Root Cause: Rotor Z-Offset Destabilizing Torque

The rotor marker prims are at Z=+0.04m in body frame (above the body center/CG).
Thrust is applied along body +Z at these elevated positions. This creates an
inverted pendulum effect:

- Any small tilt causes the thrust vector to produce a torque that increases the
  tilt.
- Torque = 4 x 0.04m x sin(theta) x F_per_rotor.
- At just 10 deg tilt with ~5N/rotor: torque ~ 0.14 Nm, angular accel ~ 800 deg/s^2.

The divergence grew from 0 deg to 71 deg in 1 second, faster than PX4's attitude
controller could compensate. Evidence:

1. Commands stayed nearly symmetric during the divergence ([892, 892, 892, 892]
   at t=17s), meaning PX4's rate controller could not produce enough differential
   thrust.
2. PX4's attitude setpoint commanded correction (pitch_sp=-34 deg at t=17.5s) but
   actual pitch was already +73 deg.
3. Actuator motors at t=16.3s show [0.833, 0.97, 0.802, 1.0] -- rear motors
   higher than front, consistent with pitch-down command, but insufficient.

## Secondary Issues

### Post-Crash Inverted Thrust

After flipping, PX4 kept commanding max thrust (`vehicle_thrust_setpoint` = -1.0),
pushing the drone into the ground while inverted. Motors settled to [0, 1, 0, 1]
-- only 2 of 4 motors active.

### Magnetometer Fusion Rejected

After t=17.07s: `test_ratio=inf`, `innovation > 1 rad`. This is a consequence of
the flip, not a cause.

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

## Recommendations

1. Move rotor markers to Z=0 (body center plane) to eliminate the pendulum
   torque.
2. Increase attitude controller gains to respond faster than the divergence time
   constant (~0.5s).
3. Add rotor tilt compensation in the bridge thrust model to account for the Z
   offset.
