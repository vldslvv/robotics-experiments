# Controller alignment subsystem

This document is the contract and operational guide for aligning the
high-throughput Isaac Lab policy-execution controller to PX4 SITL. Alignment is
upstream of RL. It covers the body-FLU linear-velocity and yaw-rate controller
active after policy handoff; waypoint navigation, the pre-policy `goto`,
trained-policy behavior, and domain randomization are not fitting inputs.

## Package and generation boundary

`robotics_experiments.alignment` is host-safe and imports without Isaac Sim,
Kit, Isaac Lab, or a PX4 container. Its public API is exactly:

- `ControllerDefinition`
- `ControllerSettings`
- `AlignmentArtifact`
- `load_controller_definition(path)`
- `load_alignment_artifact(path)`
- `align(target, reference)`

The package direction is `config -> alignment -> rl -> workflows`. The concrete
backends are `isaac-lab` as target and `px4-sitl` as reference. Controller
definitions live under `configs/alignment/controllers/`.

`x500-velocity-px4-v5` is the current comparison generation and uses alignment
artifact schema 5. It carries PX4-normalized collective and torque through the
pinned pseudo-inverse allocator and sequential desaturation, then applies the
composed X500's `omega = 50 + 475*u` motor boundary and quadratic rotor curve.
The normalized hover-thrust setting is distinct from the legacy physical hover
fraction. Schema-4 artifacts retain their hover-scaled physical-force path;
schema-2 and schema-3 artifacts retain the exact physical
`mass * (acceleration + gravity)` feed-forward used when they were published.
Schema-2 artifacts also retain a unit rate-output conversion. Legacy artifacts
load with a zero normalized-hover sentinel and are not silently promoted to
schema 5. Policies remain tied to the complete controller record embedded when
they were trained.

## Controller contract

`ControllerSettings` owns the complete PX4-like cascade:

- command delay;
- velocity P/I/D gains, with D applied to a filtered acceleration estimate;
- velocity and velocity-derivative filter cutoffs, velocity-integral limits,
  normalized PX4 hover thrust, and the legacy physical hover-thrust fraction;
- attitude P gains and yaw weight;
- body-rate P/I/D/feed-forward gains, rate-derivative filtering, and
  rate-integral limits;
- the legacy per-axis conversion from normalized rate-controller output to
  physical body torque in N-m;
- tilt, minimum/maximum collective-thrust fractions, and horizontal-thrust
  margin used by saturation and anti-windup;
- thruster rise/fall constants and maximum thrust slew.

The pure controller transition carries filtered velocity, acceleration,
velocity integral, previous angular velocity, filtered angular acceleration,
rate integral, and initialization state. NumPy is the single-environment
reference and Torch is the batched Isaac Lab twin. Both advance once per
physics sample and must remain sequentially equivalent. Controller state,
command-delay buffers, and thruster state are cleared on every environment
reset and candidate change. Alignment trace schema 4 then restores the PX4
velocity acceleration/setpoint boundary and body-rate derivative/integral
boundary after each probe's kinematic reset. Because every captured profile
starts in ongoing flight, its replay also primes current rotor thrust from that
seeded neutral-command controller boundary. Profile reset preserves the
candidate's rise/fall constants instead of re-sampling the construction-time
Thruster ranges. Older trace schemas preserve the zero-controller-state replay
under which they were captured.

The attitude stage uses PX4's reduced-quaternion law: roll and pitch align the
thrust vector first, yaw error is weighted and recombined, and world-Z yaw-rate
feed-forward is transformed into the body frame. The rate PID applies the
previous integral to the current output, then advances the integral with PX4's
large-error reduction factor for the next cycle.

The schema-5 velocity path follows pinned `MPC_ACC_DECOUPLE=1`: horizontal
acceleration and gravity form desired body Z, vertical acceleration scales the
normalized hover thrust, and collective is divided by the tilt-limited body-Z
vertical projection. PX4's minimum thrust, horizontal margin, maximum thrust,
vertical anti-windup, horizontal tracking anti-windup, and integral update
ordering are preserved. The attitude-setpoint boundary passes the magnitude of
that desired thrust vector to the body-Z allocator input; current attitude
affects the rate setpoint, not normalized collective. The rate PID emits
normalized torque directly.
`CA_METHOD=2` selects multirotor sequential desaturation, and pinned
`MC_AIRMODE=0` prioritizes thrust, roll/pitch, then yaw before clipping motor
controls to `[0, 1]`. The composed bridge offset, scale, armed zero, and
quadratic curve turn those controls into physical rotor thrust before the
existing actuator response model.
`VehicleState` remains a per-tick kinematic snapshot and does not gain
acceleration; alignment derives acceleration from timestamped trace velocity.

The public policy contract is unchanged: `press-v1` still consumes 34
observations and emits four normalized body-velocity/yaw-rate actions.
`rl.action` continues to own only normalized scales.

## Probe and trace contract

Fit and validation use different policy-scale command shapes covering:

- forward `0 -> 0.10 -> 0 -> -0.04 -> 0` m/s;
- lateral `0 -> 0.10 -> 0 -> -0.08 -> 0` m/s;
- low-speed ascent, descent, and return to zero;
- positive and negative yaw-rate steps;
- forward-plus-yaw and lateral-plus-yaw turns.

Validation includes the Stage 1 `0.08 m/s` approach and `-0.04 m/s` braking
envelope with different ramps and amplitudes. A separate moderate-speed
combined probe is a flight-envelope check; it can invalidate a candidate but
does not enter the low-speed ranking.

PX4 capture uses the normal fleet driver and an injected `PolicyAdapter`.
The alignment experiment disables stochastic IMU and GPS terms while retaining
deterministic barometer pressure and magnetometer excitation. PX4 rejects a perfectly
constant pressure stream as stale, while the composed exact magnetic field is
outside its yaw-initialization innovation gate. Alignment scales the normal
pressure and magnetic amplitudes to 0.1 and evaluates fixed-frequency signals
from simulation time. The signals repeat every 20 ms policy interval while
changing across its five 4 ms physics samples, retaining health variation
without random estimator input or profile-relative phase shifts. The capture
still seeds NumPy from
`alignment.capture.sensor_seed` before the bridge hosts are constructed so any
unexpected stochastic consumer is repeatable. Normal integration runs retain
the stochastic Pegasus process models. Structural sensor changes require
replicate captures before another full sweep.
The alignment experiment also enables PhysX enhanced determinism; normal
runtime leaves it disabled to avoid its performance cost.
All flight configurations reapply external rotor forces during every TGS
position iteration. This is part of the plant contract shared by the PX4
reference and Isaac Lab target, not a controller tuning parameter.
Before each recorded profile, the adapter streams a neutral velocity/yaw
setpoint for at least the configured minimum and then requires a continuous
stable-speed/rate/tilt window. The first profile therefore starts after PX4 has
accepted OFFBOARD rather than while the driver is still priming the stream, and
later profiles begin from the same near-equilibrium controller boundary used by
the reset Lab replay. A failed stability timeout invalidates capture.
Every trace records the actual initial state, the profile's absolute PX4
simulation start time, exact command timestamps,
body-FLU velocity commands, yaw-rate commands, position, attitude, body/world
velocity, angular velocity, completion, and flight-envelope failure. Isaac Lab
restores the captured initial state and replays the exact command intervals at
the configured physics cadence. Schema-4 reference traces also record
`vehicle_local_position_setpoint` acceleration, estimated linear and angular
acceleration, and `rate_ctrl_status` integrals at each absolute profile start.
The target derives each candidate's initial velocity integral so its first
neutral desired-acceleration boundary equals PX4, then restores the logged
rate boundary in body-FLU. The alignment-only routine ends when the probe
adapter completes and teardown stops the ephemeral simulator; landing and
auto-disarm are outside the policy-execution controller target and cannot
invalidate an otherwise complete capture. The target runner assigns one Lab
environment to every local candidate/probe pair. It evaluates one complete
beam expansion concurrently across its workers, padding smaller local shards
to the per-worker capacity so each fixed Isaac scene remains occupied. The
alignment capacity is derived independently of the training `rl.num_envs`. Every
environment has independent gains, filters, integrals, command delay, and
controller state. The deterministic coordinator partitions each expansion
across persistent spawned Isaac workers, merges results by original candidate
index, and records active-batch GPU utilization and throughput in
`gpu_utilization.json`. Each worker pads only its own fixed-capacity scene.
Worker loss is an infrastructure failure; a candidate flight-envelope breach
remains an ordinary unusable candidate. Worker count and GPU telemetry settings
change throughput only, not candidate order, ranking, initial state, commands,
cadence, controller reset, or the comparison fingerprint.
The runner uses the experiment's authored training seed for deterministic Lab
initialization before restoring every captured state.

Lab reduces the per-rotor forces and origin-referenced moments to one body
wrench. PhysX applies a position-less force at the center of mass, so Lab
translates the reduced torque by `-com x force` before submission. This
preserves the collective-thrust moment created by the X500 body's nonzero COM
offset and makes the reduced Lab wrench equivalent to the bridge's individual
rotor forces at their body-frame positions without using PhysX's substantially
slower positioned-force path.

The velocity policy interface remains body-FLU. On each controller tick, Lab
interprets the retained body velocity against the current heading inside the
pure NumPy/Torch controller seam. Command delay remains a physics-rate action
buffer upstream of that interpretation.

Old manual missions and trained warehouse-policy traces are diagnostic design
inputs only. They must never be included in fit or validation samples.

## Deterministic staged search

The target definition declares each `ParameterSpec` in fixed order with a
positive stage. The current generation uses `staged-beam-v2`:

1. Start from the complete base setting.
2. For the current parameter, expand every retained setting with every declared
   candidate.
3. Deduplicate settings and cache every evaluation.
4. Replay all uncached settings in the expansion as one GPU batch, then score
   them in their original deterministic order.
5. Reserve a beam slot for the best candidate on each advisory-gate objective,
   then fill the remaining slots by the registered deterministic ranking.
6. Retain at most the declared beam width; the current 32-setting beam doubles
   useful candidate coverage while remaining within measured device memory.
7. Make two passes through every parameter in the stage.

Stage 1 fits delay, normalized hover thrust, velocity filtering, velocity
integral limits, and velocity P/I/D. Stage 2 fits attitude/yaw, rate
P/I/D/feed-forward, rate filtering, and tilt. The legacy physical hover
fraction and rate-output torque scale are not fit in schema 5. Candidate
centers come from pinned PX4 v1.17 values and use
deterministic 0.5x, 0.75x, 1.0x, 1.25x, and 1.5x values; zero is included for
delay, integral, derivative, and feed-forward terms where meaningful. The
Historical torque-scale diagnostics regress logged normalized
`vehicle_torque_setpoint` against the physical wrench reconstructed from the
same ULog's motor outputs and shared rotor model; schema 5 instead retains the
normalized torque through allocation.
Pinned `MPC_VEL_LP=0` disables the position controller's velocity low-pass; zero
therefore means pass-through rather than a zero filter coefficient. Velocity-D
and rate-D centers use the pinned 5 Hz `MPC_VELD_LP` and 20 Hz
`IMU_DGYRO_CUTOFF` values. Enabled first-order filters use PX4's
`dt / (tau + dt)` AlphaFilter coefficient. The base settings likewise retain
the pinned 45-degree `MPC_TILTMAX_AIR`, 0.12 `MPC_THR_MIN`, and 0.30
rate-integral limits.

Actuator response stays instantaneous (`tau_inc=tau_dec=0`, effectively
unlimited slew) while a fresh instantaneous replay remains below both
normalized rotor-speed RMSE 0.01 and collective-thrust RMSE 0.05 N. The
corrected 250 Hz replay crossed the collective-thrust threshold, so the current
definition includes an explicit third stage for rise/fall time constants and
thrust slew. These candidates are applied independently per Lab environment
and preserved across every probe reset; they are not absorbed into the
controller stages.

## Metrics, quality, and publication

`policy-execution-controller-quality-v3` first ranks candidates by:

1. number of failed advisory gates;
2. largest normalized advisory-gate excess;
3. total normalized advisory-gate excess.

Stop-time events remain defined by the first sample at or below the registered
speed threshold. Their timestamp difference is quantized to nanoseconds before
comparison so an exact trace-grid boundary is not rejected by floating-point
subtraction of two late timestamps.

It then uses the v2 ordering as deterministic tie-breakers:

1. worst held-out braking-distance error;
2. worst held-out residual-speed error;
3. worst-axis held-out velocity NRMSE;
4. yaw-rate NRMSE and heading error;
5. combined-path endpoint error;
6. aggregate validation and fit velocity RMSE.

The report also includes acceleration RMSE derived from timestamped velocity,
per-axis velocity metrics, stop-time error, overshoot, settling, and aggregate
fit/validation diagnostics. Fit and validation are never pooled. All
moderate-speed traces must still be finite and complete.

Advisory quality requires:

- braking-distance error at most 5 mm;
- residual-speed error at most 0.01 m/s;
- stop-time error at most 0.10 s;
- worst-axis velocity NRMSE at most 10%;
- yaw-rate NRMSE at most 10%;
- heading error at most 3 degrees;
- combined-path endpoint error at most 10 mm;
- finite completed traces without flight-envelope failure.

Poor quality remains publishable and visible but warns training. Completed runs
live under `.build/alignment/<alignment-id>/<run-id>/`; only
`alignment.json` is the training contract. `latest` always advances. `best`
advances only for a better ranking under the same comparison fingerprint, and
the first completed result in a new generation initializes `best`.

## Failure-source separation

Before retraining, replay identical initial state, cadence, and policy commands
through PX4 and Lab. This exact-action comparison isolates controller response.
Then run the same actor closed-loop and report observation, action, and state
divergence separately. If exact-action alignment passes but closed-loop actions
diverge, classify the remaining problem as observation/state distribution or
policy behavior. Do not continue changing gains to compensate for an actor
that still commands forward motion near the wall.

Plant-owned mass, geometry, thrust curve, and drag are outside controller
fitting. A verified mismatch there is fixed in the plant contract.

## Operational workflow

```console
make alignment-capture-reference
make alignment-audit-actuators
make alignment-sweep-target
make alignment-fit
make alignment-replay-controller

make align-controllers
make test-integration-alignment-target
```

The end-to-end wrapper preserves one run ID. Capture writes `reference.json`,
copies its finalized `reference.ulg` and `bridge.log` into the same run, and
writes a relative-path `manifest.json`. The actuator audit writes
`actuator_audit.json`, the live staged sweep writes `target_candidates.json`
and `gpu_utilization.json`, and fit reproduces the same deterministic search
from those cached traces before writing and publishing `alignment.json`.
Profile replay then writes `controller_boundary_replay.json`, using each
trace's recorded PX4 start time and duration to report normalized collective,
torque, motor, rotor-speed, and physical-wrench errors while separating
saturated samples from upstream allocator attribution. The
file-backed evaluator indexes the generated candidate file once and reads one
candidate trace set at a time. It does not retain the complete diagnostic file
or parsed target traces after their metrics are recorded.

Initial controller-state enrichment linearly interpolates the PX4 acceleration
setpoint and rate-integral vectors to the trace's absolute profile start,
matching the linear and angular acceleration treatment. The report retains the
age of the latest raw source sample so low-rate ULog coverage remains visible
even though the reconstructed boundary is evaluated at the exact start
timestamp.

Alignment capture and execution settings live under the top-level Hydra
`alignment` group. Capture preconditioning is behavior-affecting and is included
in the experiment fingerprint. Worker count, GPU target, and sample interval
are execution-only and deliberately excluded. The coordinator supports
persistent worker sharding, but the measured default is one worker: three
workers reached 95.49% average GPU utilization while reducing useful throughput
from approximately 3.47 to 1.10 candidates/s because the Isaac/PhysX processes
contended. Active evaluation still records the 95% utilization target. This
measured PhysX bottleneck is reported rather than masked with duplicate work.

Training remains pinned to `.build/alignment/x500-velocity-px4-v2/best` while
v3 is unaccepted. Only after v3 advisory quality and the exact-action diagnosis
pass is the training default advanced to the accepted v3 artifact and Stage 1
retrained from scratch for the existing 300 iterations without reward or PPO
changes. Run checkpoint selection, independent Lab evaluation, the three-range
physical transfer gate, and the complete approach mission in that order. Stop
and preserve diagnostics at the first failure.

An alignment campaign has no fixed attempt cap while a completed attempt makes
material progress or trace and ULog evidence identifies a new correction to
test. Every attempt is committed separately with its hypothesis and result. If
an attempt does not reduce the failed-gate count, or reduce the largest
normalized gate excess by at least 20% without regressing a passed gate, use
trace and ULog evidence to expand up to four meaningful controller parameters
or ranges before the next attempt. Candidate additions may cover
acceleration/jerk limits, saturation, integral limits, or further
estimator/controller filters when the evidence locates the mismatch there;
plant-owned quantities remain outside gain fitting. Two consecutive
no-progress attempts with no evidence-backed correction left end the campaign.

Vector-valued `ParameterSpec` candidates need not keep roll/pitch or
horizontal-axis values tied. When per-probe evidence shows conflicting axis
requirements, the candidate table carries asymmetric complete vectors so the
generic search can fit those axes independently without introducing
field-specific search logic.

Parameter uncertainty, domain randomization, real-flight capture, residual
models, and PX4-in-loop training remain deferred.
