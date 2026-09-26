# PX4 Flight Log Analysis

This directory contains the repeatable process and small helper tools used to
debug Isaac Sim + PX4 SITL flight failures. The goal is to decide whether a bad
flight was commanded by PX4, caused by stale bridge output, or caused by a frame,
rotor, or force-application mismatch in the Isaac bridge.

## Tools

Run the helpers from the repository root.

```bash
uv run python flight_analysis/tools/list_pairs.py px4/logs
uv run python flight_analysis/tools/bridge_summary.py px4/logs/2026-06-11/05_11_40_v0_bridge.log
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with pyulog --with numpy python \
  flight_analysis/tools/ulog_timeline.py px4/logs/2026-06-11/05_11_40.ulg
```

`list_pairs.py` has no third-party dependencies. `ulog_timeline.py` and
`compare_logs.py` need `pyulog` and `numpy`; use the `uv --with` command above so
those tools do not become permanent project dependencies. Both share the ULog
loaders in `tools/_ulog_common.py`.

## Sim-to-real comparison

Task-neutral repeated-flight evidence and position/pose envelopes are produced
by `make measure-flight-parameters`; see `docs/FLIGHT_PARAMETERS.md`. Those
artifacts establish the stable comparison boundary. The tools below remain
diagnostics for explaining differences in the retained ULogs and bridge or
hardware logs.

`compare_logs.py` overlays two PX4 ULogs and prints a per-metric divergence table
for the hover/dynamics signals (horizontal drift, vertical range, speed, tilt,
yaw, body rates, and commanded motor effort). Use it to quantify the sim-to-real
gap once real flights exist, and to track sim-vs-sim drift in the meantime.

`controller_state_replay.py` is an independent timestamped state-replay
diagnostic. It replays PX4 ULog estimator state and velocity setpoints
through the stateful numpy reference for Isaac Lab's velocity cascade on the
configured 250 Hz physics grid. Setpoints use zero-order hold, velocities and
rates are linearly interpolated, and attitudes use normalized shortest-path
quaternion interpolation. The tool converts PX4 motor controls through the
configured Pegasus scaling and compares per-rotor angular velocity, collective
thrust, and body torque at logged actuator timestamps. It also regresses the
logged normalized PX4 torque setpoint against reconstructed physical body
torque, reporting the per-axis N-m conversion scale, intercept, and fit RMSE.
The arbitrary-window replay is diagnostic. The complete actuator audit and
per-profile controller replay are required by controller-alignment publication.

Alignment trace schema 1 records every profile's absolute PX4 simulation start
time, exact command cadence, physical and PX4-estimated initial velocity, and
controller acceleration, rate-integral, and normalized hover-thrust state at
that timestamp. Profile replay runs on all three independent response recordings
(fitting, validation, and acceptance), preserving each recording's internal
profile splits. It uses those absolute boot times directly and
checks both the core controller on PX4-estimated state and the complete
physical-state observation boundary seeded from the recorded estimator offset.
The core enforces motor and collective limits; the physical-state replay adds a
normalized motor limit. The replay includes the recorded command transitions,
PX4's velocity-only vertical-state blend, and timestamped hover-estimate
updates. It does not reconstruct transitions from rate-limited ULog
`trajectory_setpoint` output. The tool reports thrust and torque setpoints,
motor controls, bridge-mapped rotor speed, reconstructed wrench, hover-thrust
estimate, and timestamp errors per fit, validation, and stability profile:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=src uv run --with pyulog python \
  flight_analysis/tools/controller_state_replay.py \
  --ulog .build/alignment/<alignment-id>/<run-id>/reference.ulg \
  --config configs/alignment_px4_x500_velocity.yaml \
  --audit-reference .build/alignment/<alignment-id>/<run-id>/reference.json \
  --bridge-log .build/alignment/<alignment-id>/<run-id>/bridge.log \
  --json .build/alignment/<alignment-id>/<run-id>/actuator_audit.json
```

The audit fails on incomplete topic coverage, a join error above one 4 ms
physics tick, normalized motor RMSE above 0.01, or collective-thrust RMSE above
0.05 N.

After fitting, replay every recorded profile without selecting windows by hand:

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=src uv run --with pyulog python \
  flight_analysis/tools/controller_state_replay.py \
  --ulog .build/alignment/<alignment-id>/<run-id>/reference.ulg \
  --config configs/alignment_px4_x500_velocity.yaml \
  --controller-artifact .build/alignment/<alignment-id>/<run-id>/controller.json \
  --profile-reference .build/alignment/<alignment-id>/<run-id>/reference.json \
  --json .build/alignment/<alignment-id>/<run-id>/controller_boundary_replay.json
```

Profile replay inverts unsaturated normalized motor controls through the
generated allocator mix and reports target-versus-PX4 normalized collective
and torque before the existing motor, rotor-speed, and physical-wrench
metrics. Samples at either motor limit remain in output-boundary metrics but
are excluded from upstream allocator attribution.

```bash
env UV_CACHE_DIR=/tmp/uv-cache PYTHONPATH=src uv run --with pyulog python \
  flight_analysis/tools/controller_state_replay.py <flight.ulg> \
  --config configs/routines/policy_press_approach_quad.yaml \
  --controller-artifact .build/alignment/x500-px4-v1/latest \
  --window 28.3,30.7 --json /tmp/controller-state-replay.json
```

Use a window containing finite OFFBOARD velocity setpoints. The comparison
boundary is deliberately below the policy and above the plant: identical state
and velocity intent enter both controllers, and physical rotor speed/body wrench
leave both. Isaac Lab's `Thruster` is a lower actuator-realization layer; with
zero time constants its output must match the requested
per-rotor thrust independently of this controller comparison.

Closed-loop controller fitting belongs exclusively to
`robotics_experiments.alignment` and `make/alignment.mk`; see
`docs/SIM_TO_SIM_ALIGNMENT.md`. Trained press-policy logs are downstream
acceptance evidence and are not ingested by flight-analysis fitting code.

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with pyulog --with numpy python \
  flight_analysis/tools/compare_logs.py <log_a.ulg> <log_b.ulg> \
  --align takeoff --hover-last 2.0 --labels sim,real --csv /tmp/cmp.csv
```

The comparison is **symmetric and origin-agnostic**: both inputs are just PX4
ULogs, so any pairing works (sim-vs-real, sim-vs-sim, real-vs-real). Neither log
is a privileged reference; swapping the two positional inputs only flips the sign
of each `delta (B-A)`. Labels are display strings only, and the tool never
branches on whether a log came from SITL or hardware -- it reads only standard PX4
estimator/actuator topics present in every log.

- `--align {takeoff,arm,none}` zeroes each log's clock on a shared flight event
  (accepted `vehicle_command_ack` for takeoff cmd 22 / arm cmd 400), so two runs
  with unrelated wall clocks line up. `none` uses raw log time.
- `--window START,END` selects an anchor-relative window; otherwise `--hover-last`
  takes the last N seconds of each log (default 2.0).
- `--csv` writes both logs' position resampled onto a shared normalized grid for
  later plotting (no plotting dependency added).

Caveats when reading the table:

- `vehicle_local_position` is **NED**: +z is down, so altitude is `-z`.
- Tilt is the body-up vs world-up angle (frame-agnostic magnitude); yaw is the
  Euler-Z of `vehicle_attitude`.
- The sim currently aids PX4 with simulated **GPS** (`HIL_GPS`); an indoor real
  drone's aiding source is undecided (mocap, optical flow, etc.). The tool reads
  estimator *outputs* that exist regardless of aiding source, but position-noise
  differences between a GPS-sim log and a mocap/flow-real log are an
  **aiding-source artifact**, not necessarily a dynamics-model gap. Interpret
  them as such until the real position source is chosen and matched in sim.

### Required real-drone log topics

For a clean overlay, the real drone must record the same ULog topic set the sim
already logs. Lock this union (the `ulog_timeline.py` topic list plus the metric
topics `compare_logs.py` reads) into the airframe's SDLOG profile:

`vehicle_command`, `vehicle_command_ack`, `vehicle_status`, `takeoff_status`,
`vehicle_land_detected`, `trajectory_setpoint`, `vehicle_local_position`,
`vehicle_attitude`, `vehicle_attitude_setpoint`, `vehicle_rates_setpoint`,
`vehicle_thrust_setpoint`, `vehicle_torque_setpoint`, `vehicle_angular_velocity`,
`actuator_motors`, `actuator_outputs`, `failsafe_flags`,
`estimator_status_flags`.

## Analysis Process

1. Pair the logs by date directory and `HH_MM_SS` stem. A normal run has:
   - `px4/logs/<date>/HH_MM_SS.ulg`
   - `px4/logs/<date>/HH_MM_SS_v<vehicle>_bridge.log`

   The bridge-log filename carries a `_v<vehicle>` segment (the per-vehicle
   `id`) after the `HH_MM_SS` stem so multiple vehicles in one scene never
   overwrite each other's logs. The `HH_MM_SS` stem stays first, so pairing by
   date directory + `HH_MM_SS` is unchanged; the single-vehicle default is
   `_v0_bridge.log`.
2. Confirm the bridge connected and streamed valid sensors:
   - first heartbeat received
   - first `HIL_SENSOR` sent
   - `HIL_SENSOR.time_usec` strictly increases
   - `fresh={imu:True mag:True baro:True ...}` appears before sends
3. Check actuator lockstep in the bridge log:
   - before the first actuator packet, the bridge may continue sending sensors
   - after first `HIL_ACTUATOR_CONTROLS`, actuator timestamps should stay close
     to sensor simulation time
   - growing `actuator_lag_us` means stale PX4 motor output is being applied
4. Check PX4 accepted the requested commands in the ULog:
   - `vehicle_command_ack.command == 400` with `result == 0` for arm
   - `vehicle_command_ack.command == 22` with `result == 0` for takeoff
5. Compare intent and response:
   - `trajectory_setpoint` and `vehicle_local_position_setpoint` show desired
     motion
   - `vehicle_attitude_setpoint` and `vehicle_rates_setpoint` show desired
     attitude/rates
   - `vehicle_attitude` and `vehicle_local_position` show actual response
6. Compare PX4 actuator output with bridge-applied output:
   - ULog `actuator_motors` / `actuator_outputs`
   - bridge `input_reference`, `commands`, `forces_n`, `body_torque_n_m`
7. If PX4 commands a correction and the simulated vehicle moves farther in the
   wrong direction, suspect rotor order, thrust axis, body-frame sign, or sensor
   frame conversion.
8. Treat estimator/failsafe warnings after the attitude has already diverged as
   consequences unless estimator topics show rejection before the motion starts.

## Bridge Log Queries

Use `rg` first; bridge logs are text and usually contain the fastest answer.

```bash
LOG=px4/logs/2026-06-11/05_11_40_v0_bridge.log

rg -n "PX4 MAVLink port constructed|first heartbeat|first HIL_SENSOR|stopped" "$LOG"
rg -n "HIL_ACTUATOR_CONTROLS|actuator_lag_us|bootstrap lag|timestamp stale|timed out" "$LOG"
rg -n "sensor stream|HIL_SENSOR pre-send|fresh=|non-increasing" "$LOG"
rg -n "rotor mapping resolved|commands=|forces_n=|body_torque_n_m|rotor force application" "$LOG"
rg -n "tilt cutoff|disconnected|last_disconnect" "$LOG"
```

Important bridge fields:

- `sim_time_us`: Isaac simulation time for the current sensor tick.
- `last_actuator_us`: PX4 actuator timestamp last received by the bridge.
- `actuator_lag_us`: `sim_time_us - last_actuator_us`; should settle near one
  sensor period during healthy lockstep.
- `PX4 bridge dynamics` records both fields beside the rotor commands and
  realized wrench, so actuator audits do not infer the applied command's
  timestamp from log-line order.
- `input_reference`: scaled PX4 actuator controls from Pegasus' backend.
- `commands`: rotor angular velocity references sent to the thrust model.
- `forces_n`: per-rotor forces actually applied in Isaac.
- `net_torque_world`: world torque from force-at-position application; useful
  for checking rotor geometry and command asymmetry.

## Pyulog Queries

For quick topic discovery:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with pyulog --with numpy python -c '
from pyulog import ULog
u = ULog("px4/logs/2026-06-11/05_11_40.ulg")
print("\n".join(sorted(d.name for d in u.data_list)))
'
```

For a focused flight timeline, prefer the included helper:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv run --no-project --with pyulog --with numpy python \
  flight_analysis/tools/ulog_timeline.py px4/logs/2026-06-11/05_11_40.ulg \
  --samples 10,11,12,13,13.6,13.8,14,14.2,14.4,14.6,14.8,15
```

Minimum ULog topics for flight divergence:

- `vehicle_command`, `vehicle_command_ack`
- `vehicle_status`, `takeoff_status`, `vehicle_land_detected`
- `trajectory_setpoint`, `vehicle_local_position_setpoint`
- `vehicle_local_position`
- `vehicle_attitude`, `vehicle_attitude_setpoint`
- `vehicle_rates_setpoint`
- `vehicle_thrust_setpoint`, `vehicle_torque_setpoint`
- `actuator_motors`, `actuator_outputs`
- `estimator_status`, `estimator_status_flags`
- `failsafe_flags`

Useful one-off query shape:

```python
from pyulog import ULog
import numpy as np

ulog = ULog("px4/logs/2026-06-11/05_11_40.ulg")
topic = next(d for d in ulog.data_list if d.name == "vehicle_command_ack").data
for i, timestamp in enumerate(topic["timestamp"]):
    print(timestamp, topic["command"][i], topic["result"][i])
```

## Worked Example: Takeoff Instability

The unstable takeoff was debugged with the latest paired ULog and bridge log.
The key observations were:

- The old flyaway mode was gone: actuator lockstep stayed healthy after startup,
  with `actuator_lag_us` settling around one 250 Hz sensor period.
- PX4 accepted arm and takeoff. The ULog showed command `400` and command `22`
  acknowledgements with `result=0`.
- During lift, PX4 began commanding large asymmetric motor outputs. The bridge
  applied those outputs faithfully; this made a stale actuator backlog unlikely.
- The vehicle stayed nearly level until rotation began, then attitude control
  diverged quickly.

The root implementation bug was the gyro frame. Isaac articulation angular
velocity was treated as if it were already body-frame FLU data. Isaac reports it
in the world frame, while PX4 HIL gyro input is body-frame FRD. The fix rotates
world ENU angular velocity into body FLU in the Isaac adapter, then lets the
MAVLink adapter do the final FLU to FRD conversion.

Related robustness fixes:

- actuator polling does not block before PX4 has emitted the first actuator
  packet
- after the first actuator packet, queued actuator messages are drained and the
  newest packet is applied
- the first actuator timestamp lag is accepted as a startup baseline, but later
  materially stale actuator timestamps disconnect and zero commands
- tilt cutoff zeros rotor forces after excessive tilt so an inverted/crashed
  vehicle does not keep applying thrust into the ground

When a future flight is unstable, first decide which side commanded the bad
motion. If the ULog actuator topics and bridge `commands` agree, inspect PX4
setpoints and sensor frames. If they disagree, inspect bridge actuator freshness,
rotor ordering, and thrust application.

## Routine-stage filtering

Live and standalone routines write structured `ROUTINE_STAGE` JSON records to
both the runtime output and the per-vehicle bridge log. Each record contains
`event`, `index`, `kind`, `label`, and the exact `sim_time_us`. Select the
interval from the `start` record through the matching `complete` or `failed`
record before comparing a learned policy with Isaac Lab. In particular, a
`kind: policy` comparison excludes takeoff, the preceding `goto`/`hover`, and
the return/landing steps:

```text
rg 'ROUTINE_STAGE.*"kind":"policy"' px4/logs/<date>/<time>_v0_bridge.log
```

ULog message timestamps use the same simulated PX4 boot-time clock. Tools whose
windows are relative to `ULog.start_timestamp` convert a stage boundary with
`(sim_time_us - ULog.start_timestamp) / 1e6`.
