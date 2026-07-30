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

`x500-velocity-px4-v2` is a new comparison generation and uses alignment
artifact schema 2. Schema-1 artifacts are not upgraded or reinterpreted by the
schema-2 loader. Policies trained against schema 1 likewise remain tied to
their old controller and are unsupported by the current training/runtime
preflight.

## Controller contract

`ControllerSettings` owns the complete PX4-like cascade:

- command delay;
- velocity P/I/D gains, with D applied to a filtered acceleration estimate;
- velocity and velocity-derivative filter cutoffs and velocity-integral limits;
- attitude P gains and yaw weight;
- body-rate P/I/D/feed-forward gains, rate-derivative filtering, and
  rate-integral limits;
- tilt, minimum/maximum collective-thrust fractions, and horizontal-thrust
  margin used by saturation and anti-windup;
- thruster rise/fall constants and maximum thrust slew.

The pure controller transition carries filtered velocity, acceleration,
velocity integral, previous angular velocity, filtered angular acceleration,
rate integral, and initialization state. NumPy is the single-environment
reference and Torch is the batched Isaac Lab twin. Both advance once per
physics sample and must remain sequentially equivalent. Controller state,
command-delay buffers, and thruster state are cleared on every environment
reset, probe reset, and candidate change.

Velocity saturation uses conditional integration: an axis does not integrate
when its error would push farther into a saturated thrust request. Integral
limits remain active independently. `VehicleState` remains a per-tick
kinematic snapshot and does not gain acceleration; alignment derives
acceleration from timestamped trace velocity.

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
Before each recorded profile, the adapter streams a neutral velocity/yaw
setpoint for at least the configured minimum and then requires a continuous
stable-speed/rate/tilt window. The first profile therefore starts after PX4 has
accepted OFFBOARD rather than while the driver is still priming the stream, and
later profiles begin from the same near-equilibrium controller boundary used by
the reset Lab replay. A failed stability timeout invalidates capture.
Every trace records the actual initial state, exact command timestamps,
body-FLU velocity commands, yaw-rate commands, position, attitude, body/world
velocity, angular velocity, completion, and flight-envelope failure. Isaac Lab
restores the captured initial state and replays the exact command intervals at
the configured physics cadence. The alignment-only routine ends when the probe
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

Old manual missions and trained warehouse-policy traces are diagnostic design
inputs only. They must never be included in fit or validation samples.

## Deterministic staged search

The target definition declares each `ParameterSpec` in fixed order with a
positive stage. Search uses `staged-beam-v1`:

1. Start from the complete base setting.
2. For the current parameter, expand every retained setting with every declared
   candidate.
3. Deduplicate settings and cache every evaluation.
4. Replay all uncached settings in the expansion as one GPU batch, then score
   them in their original deterministic order.
5. Rank deterministically and retain the best eight.
6. Make two passes through every parameter in the stage.

Stage 1 fits delay, velocity filtering, and velocity P/I/D. Stage 2 fits
attitude/yaw, rate P/I/D/feed-forward, rate filtering, and tilt. Candidate
centers come from pinned PX4 v1.17 values and use deterministic 0.5x, 0.75x,
1.0x, 1.25x, and 1.5x values; zero is included for delay, integral,
derivative, and feed-forward terms where meaningful.
The velocity-filter center is the first-order cutoff equivalent of the pinned
`EKF2_TAU_VEL=0.25 s` (`1 / (2*pi*tau)`, approximately 0.637 Hz); velocity-D
and rate-D centers use the pinned 5 Hz `MPC_VELD_LP` and 20 Hz
`IMU_DGYRO_CUTOFF` values. The base settings likewise retain the pinned
45-degree `MPC_TILTMAX_AIR`, 0.12 `MPC_THR_MIN`, and 0.30 rate-integral limits.

Actuator response stays instantaneous (`tau_inc=tau_dec=0`, effectively
unlimited slew) while a fresh instantaneous replay remains below both
normalized rotor-speed RMSE 0.01 and collective-thrust RMSE 0.05 N. The
corrected 250 Hz replay crossed the collective-thrust threshold, so the current
definition includes an explicit third stage for rise/fall time constants and
thrust slew. These candidates are applied independently per Lab environment
after every probe reset; they are not absorbed into the controller stages.

## Metrics, quality, and publication

`policy-execution-controller-quality-v2` ranks candidates by:

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

Advisory v2 quality requires:

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
make alignment-sweep-target
make alignment-fit

make align-controllers
make test-integration-alignment-target
```

The end-to-end wrapper preserves one run ID. Capture writes `reference.json`,
the live staged sweep writes `target_candidates.json`, and fit reproduces the
same deterministic search from those cached traces before writing and
publishing `alignment.json`. The file-backed evaluator loads each diagnostic
file once; it does not reparse the complete candidate file for every setting,
and it releases parsed target traces after their metrics are recorded.

Alignment capture and execution settings live under the top-level Hydra
`alignment` group. Capture preconditioning is behavior-affecting and is included
in the experiment fingerprint. Worker count, GPU target, and sample interval
are execution-only and deliberately excluded. The coordinator supports
persistent worker sharding, but the measured default is one worker: three
workers reached 95.49% average GPU utilization while reducing useful throughput
from approximately 3.47 to 1.10 candidates/s because the Isaac/PhysX processes
contended. Active evaluation still records the 95% utilization target. This
measured PhysX bottleneck is reported rather than masked with duplicate work.

Training defaults to `.build/alignment/x500-velocity-px4-v2/best`. Only after
v2 advisory quality and the exact-action diagnosis pass may Stage 1 be
retrained from scratch for the existing 300 iterations without reward or PPO
changes. Run checkpoint selection, independent Lab evaluation, the three-range
physical transfer gate, and the complete approach mission in that order. Stop
and preserve diagnostics at the first failure.

Parameter uncertainty, domain randomization, real-flight capture, residual
models, and PX4-in-loop training remain deferred.
