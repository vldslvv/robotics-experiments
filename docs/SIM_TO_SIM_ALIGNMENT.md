# Controller alignment

Controller alignment makes the high-throughput Isaac Lab controller reproduce
PX4 SITL for identical physical commands. It is upstream of reinforcement
learning. Waypoint navigation, policy rewards, policy actions, and domain
randomization are not fitting inputs.

## Architecture and contracts

`robotics_experiments.controller` owns the immutable `ControllerSettings`, the
strict controller artifact, and the pure NumPy controller/allocator/actuator
reference. `robotics_experiments.alignment` owns probe definitions, generic
candidate search, metrics, evidence bundles, and publication. RL depends only
on the controller artifact and never imports alignment.

The dependency direction is:

`config -> controller -> {alignment, rl} -> workflows`

The only supported controller and alignment artifact contract is schema 1.
There are no artifact migrations, compatibility sentinels, alternate plant
mappings, or physical-force controller paths. Older files under `.build/` are
inert evidence and cannot be loaded by current production code.

The X500 normalized controller carries velocity, attitude, and rate state into
PX4-style normalized thrust and torque. The generated X500 effectiveness matrix
uses `CA_METHOD=2` sequential desaturation with `MC_AIRMODE=0`. Motor controls
are clipped to `[0, 1]`, then pass through the plant-owned input offset, input
scale, armed-zero position, quadratic rotor curve, and actuator rise/fall model.
The composed Holybro X500 mapping is `omega = 50 + 475*u`; its physical hover
input is approximately 0.375. Controller hover is represented only by
`hover_thrust_normalized` at the PX4 controller boundary.

NumPy is the single-environment reference and the Isaac Lab Torch seam is its
batched twin. Both apply the same filter, saturation, anti-windup, reset, and
allocator ordering. The strict settings include the velocity-controller update
rate, PX4's dedicated yaw-torque low-pass, and the target-side estimator
adapter. That adapter seeds a velocity offset from the captured PX4 estimate
and decays it toward scaled articulation velocity; it also filters articulation
body rate from the captured PX4 angular-rate seed. Controller state carries
these estimator states, the held outer-loop setpoints, and the acceleration
setpoint needed for PX4's thrust-preserving hover-estimate integrator
correction. Sequential NumPy/Torch parity is a host-test requirement.

The latest body-FLU policy action is held across its 20 Hz inference interval.
The configured 50 Hz OFFBOARD stream converts that retained action with the
attitude at each wire transmission, then holds the resulting world-ENU velocity
setpoint while the 125 Hz velocity loop and 250 Hz rate, allocator, actuator,
and physics loops advance. This mirrors PX4's `MAV_FRAME_BODY_NED` receiver:
body-to-world conversion occurs for each received message, independently of the
slower policy cadence.

Alignment receives an ordered list of probe families. A family contract is
only a stable name, a control mode, a registered metric, and physical command
profiles. The current required families are `velocity`, covering body-FLU X/Y/Z
velocity plus velocity-path yaw, and `ctbr`, covering normalized collective
thrust plus body-FLU roll/pitch/yaw rates. Capture, Lab replay, search, evidence,
and publication iterate this list generically. Adding a control capability does
not require another alignment engine or a policy dependency.

## Search

Controller definitions live in `configs/alignment/controllers/`. Every tunable
field must be declared as a `ParameterSpec`; search never invents controller
fields. The `coarse-to-fine-beam` strategy:

1. evaluates the committed base setting;
2. performs one wide, large-step sweep for each declared field;
3. retains the global beam plus per-gate champions;
4. generates midpoints only around retained regions for two refinement levels;
5. deduplicates settings by their exact float32 controller representation.

A zero-centered fitted field may declare a positive
`selection_regularization_scale`. If the registered metric finds any
quality-passing candidates, the search selects the passing candidate with the
smallest dimensionless L2 correction across those declared fields, then uses
the response metrics as tie-breakers. If no candidate passes, gate proximity
continues to determine selection. This keeps an estimator-offset fit from
spending unnecessary correction authority merely to improve an already-passing
trace metric.

The default uses one persistent worker and 128 candidate slots. This is the
measured useful-throughput optimum for the current GPU; additional workers
increase PhysX contention. The target coordinator reports candidates per
second, padding, cache hits, batches, and sampled GPU utilization. The 95% GPU
target remains diagnostic; deterministic useful throughput is the optimization
objective.

Each worker owns one persistent Isaac Lab environment for the entire ordered
probe-family list. Families execute sequentially by switching the controller
mode and padding the smaller family's trace stride to the fixed scene capacity;
constructing multiple `DirectRLEnv` instances in one process is unsupported by
Isaac Lab's simulation-context ownership.

## Evidence and boundary audit

PX4 capture uses the ordinary fleet/bridge path with a deterministic policy
adapter. Its configured pusher is deployed after takeoff and before probe
preconditioning so the reference plant matches Lab training and press-policy
deployment. The Lab target also consumes the Hydra-owned rigid-body gyroscopic
force setting, which is disabled to match the standalone PX4 plant. Each
profile precondition uses a bounded velocity command to return to the routine
reference position before applying the configured stability hold. The recenter
gain and speed bound live in `alignment.capture`, so profile order cannot
accumulate displacement against the routine safety envelope. Each profile then
records its absolute PX4 simulation start time and the complete initial
controller state required for replay. One run directory keeps:

- the ULog and bridge log;
- reference and target traces;
- selected controller and compressed candidate selection;
- GPU/search telemetry;
- actuator audit and per-profile controller-boundary replay;
- a relative-path manifest with sizes and SHA-256 hashes.

The actuator audit joins `vehicle_thrust_setpoint`,
`vehicle_torque_setpoint`, `actuator_motors`, bridge-mapped rotor speed,
reconstructed collective/torque, and `hover_thrust_estimate` by timestamp. It
fails on incomplete topic coverage, a join error above one 4 ms physics tick,
normalized motor RMSE above 0.01, or collective RMSE above 0.05 N. The
controller-boundary replay applies the selected controller to every fit,
validation, and stability profile. Its core replay uses PX4-estimated state and
enforces the same motor and collective limits. A second replay starts from the
physics-rate physical articulation state in the bridge log, seeds the recorded
initial estimator state, and requires normalized motor RMSE below 0.01;
closed-loop motion gates cover the remaining estimator residual. It uses each
profile's absolute PX4 boot time, the pinned 125 Hz PX4 velocity-loop cadence,
and recorded command cadence. The seed phase is derived per profile from the
recorded age of its `vehicle_local_position` sample. Because the replay grid
includes the profile boundary, its initial accrued outer-loop time is the
sample age minus one 4 ms physics tick; this preserves the next 125 Hz update
whether the boundary lands on or between estimator publications. ULog
rate-limited `trajectory_setpoint` samples are not used to infer command
transitions. Replay builds relative command time from integer physics steps so
large absolute profile start times cannot delay an exact command boundary by a
floating-point tick. The replay also reproduces PX4's
velocity-only vertical state blending and applies logged hover-estimate updates
at their source timestamps. The target-only `velocity_observation_scale`,
steady world-ENU `velocity_observation_bias_mps`,
`velocity_estimator_bias_decay_hz`, per-axis `velocity_filter_hz`, and
`angular_velocity_filter_hz` model the evidenced physical-to-estimated state
boundary without changing the pinned PX4 controller gains. The core replay
bypasses these target-only adapters because its inputs are already
PX4-estimated state; the physical-state replay retains them and therefore
audits the deployment boundary.

## Quality and acceptance

The registered `controller-quality` aggregate resolves each family's registered
metric, keeps every axis visible, and ranks first by failed-gate count and
normalized excess. The translational family applies the following limits to
each of X, Y, and Z independently:

- direct velocity NRMSE <= 0.10;
- envelope velocity NRMSE <= 0.10;
- braking-distance error <= 0.005 m;
- residual-speed error <= 0.01 m/s;
- stop-time error <= 0.10 s.

Its combined endpoint error remains <= 0.01 m. The CTBR family applies these
limits to roll, pitch, and yaw independently:

- direct body-rate NRMSE <= 0.10;
- envelope body-rate NRMSE <= 0.10;
- attitude error <= 3 degrees.

The CTBR collective probe additionally requires vertical-velocity and
vertical-acceleration NRMSE <= 0.10. There is no worst-axis substitute: every
declared axis gate must pass.

Braking distance and stop time apply only when the PX4 reference has reached
at least half of the preceding nonzero command at the transition. This keeps
the exact-action gates tied to genuine braking responses; a short reversal
that never establishes the commanded motion remains covered by velocity,
acceleration, residual, settling, and endpoint metrics.

Each translational axis has an independent envelope profile through 0.8 m/s.
The Z profile specifically prevents a low-speed aggregate from hiding the wrong
climb/descend correction observed during policy execution.

A run cannot finalize or publish unless all metric gates pass, every stability
trace completes with finite values, the actuator audit passes, and the
controller-boundary replay passes. Policy training rejects a controller whose
quality record is failed. The authoritative live gate is:

```bash
make test-integration-alignment-target
```

Run the full campaign with:

```bash
make align-controllers
```

The campaign runs the authoritative integration gate against its run-local
`controller.json` after replay and before finalization. A failure therefore
preserves diagnostic evidence without advancing `latest` or `best`.

`controller.json` is the downstream product. It contains complete controller
settings and the control modes certified by the accepted probe-family list;
policy-owned normalized action scales are not alignment fields. Candidate traces and ULogs are
valuable reproducibility evidence, but policy training consumes only the
controller artifact. `latest` and comparable `best` links advance only for a
fully accepted bundle.

## Change policy

Changing a metric, probe suite, environment, image, controller interface,
controller definition, or resolved experiment starts a new comparison
fingerprint. The fingerprint includes hashes of the metric and normalized
controller implementation modules, so an equation change cannot remain in an
old comparison. Plant residuals must be fixed at the plant seam; they must not be
absorbed into controller gains. A selected parameter touching a range boundary
is not evidence by itself: the matching trace or ULog boundary must support an
expanded range or structural correction.
