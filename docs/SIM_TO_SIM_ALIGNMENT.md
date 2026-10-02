# Controller alignment

Controller alignment makes the high-throughput Isaac Lab controller reproduce
PX4 SITL for identical physical commands. It is upstream of reinforcement
learning. Waypoint navigation, policy rewards, policy actions, and domain
randomization are not fitting inputs.

Flight-parameter bundles contain independently captured fitting, validation, and
acceptance flights. Preparation pins all three complete response recordings;
their `set_name` remains separate from each recording's internal profile splits.
Hover bounds and policy reset samples still come only from fitting hover flights.
See [flight parameters](FLIGHT_PARAMETERS.md) for capture and set ownership.

## Architecture and contracts

`robotics_experiments.controller` owns the immutable `ControllerSettings`, the
strict controller artifact, and the pure NumPy controller/allocator/actuator
reference. `robotics_experiments.alignment` owns probe definitions, generic
candidate search, metrics, evidence bundles, and publication. RL depends only
on the controller artifact and never imports alignment.

`robotics_experiments.flight_parameters` owns the task-neutral schema-1 hover
and controller-response evidence described in `docs/FLIGHT_PARAMETERS.md`.
Alignment no longer launches a private PX4 reference capture. Its preparation
step resolves and pins one immutable flight-parameters run, validates that its
lossless response evidence exactly covers the authored fit, validation, and
stability profiles, then exposes its `reference.json`, ULog, and bridge log to
the existing alignment evaluator through relative symlinks. The selected target
settings are still replayed through the full 30-second
`stability-hover-pose` profile and must keep position and attitude within the
full-window hover bounds. The registered metrics, response thresholds, and hover bounds remain unchanged. The exact flight-parameters fingerprint enters
the comparison fingerprint and strict controller artifact. The capture manifest
and controller artifact also record the resolved producer `run_id`; an authored
`latest` selector is never retained as lineage.

Input validation completes before the alignment run directory is created. An
invalid or incompatible flight-parameters selector therefore exits with one
preflight error and leaves no empty or partially staged alignment run. The
consumer never copies those producer-owned inputs; final alignment evidence
retains relative references to the pinned timestamped bundle. ULogs and bridge
logs may be gzip-compressed in that producer; audit and replay readers stream
their original bytes without changing the reference names or copying inputs.
The fitting entrypoint uses that prepared bundle for both controller definitions,
preserving their authored definition fingerprints. Different authored selectors
such as `latest` and an exact run path cannot override the prepared input or
cause a false definition mismatch after simulation.

The dependency direction is:

`config -> controller -> {alignment, rl} -> workflows`, with the independent
`flight_parameters -> {alignment, rl}` evidence boundary.

The only supported controller and alignment artifact contract is schema 1.
There are no artifact migrations, compatibility sentinels, alternate plant
mappings, or physical-force controller paths. Older files under `.build/` are
inert evidence and cannot be loaded by current production code. If an existing
`latest` or `best` label names an unsupported artifact, the next accepted
publication replaces that label atomically without loading or migrating the
obsolete payload.

The X500 normalized controller carries velocity, attitude, and rate state into
PX4-style normalized thrust and torque. The generated X500 effectiveness matrix
uses `CA_METHOD=2` sequential desaturation with `MC_AIRMODE=0`. Motor controls
are clipped to `[0, 1]`, then pass through the plant-owned input offset, input
scale, armed-zero position, quadratic rotor curve, and actuator rise/fall model.
The composed Holybro X500 mapping is `omega = 50 + 475*u`; its physical hover
input is approximately 0.375. Controller hover is represented only by
`hover_thrust_normalized` at the PX4 controller boundary.
For CTBR, `ctbr_effective_collective_scale` maps the raw streamed collective
to the effective inner-loop collective observed at PX4's motor boundary; it
does not change the raw normalized-action or OFFBOARD command contract.

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

The velocity observation hold scale applies per world axis only when both the
command and physical velocity are small. The larger absolute value drives the
existing smooth command-threshold transition to the motion scale. A zero command
during braking therefore retains the motion scale until the vehicle slows down.
The steady velocity bias additionally fades out as the difference between
commanded and physical velocity crosses the same thresholds. It therefore
describes settled motion without applying the steady correction during large
acceleration transients. Seeded-offset decay retains its command-based
transition. Seeding and subsequent controller updates use the same mappings.

Horizontal velocity proportional gain has two bounded error schedules. The
`horizontal_velocity_braking_gain_*` lobe owns low-error braking attenuation;
the `horizontal_velocity_error_gain_*` lobe owns higher-error recovery
authority. Each leaves the base gain unchanged below its `start_mps`, moves
smoothly to its positive scale at `full_mps`, and returns smoothly to the base
gain at `end_mps`. The disjoint lobe scales multiply. Settings validation
requires each `scale > 0` and `start < full < end`; scales below one attenuate
their bounded range and scales above one boost it. This lets alignment correct
zero-command stopping without removing the independently evidenced live
recovery authority.

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
4. generates midpoints around retained regions for each configured refinement level;
5. deduplicates settings by their exact float32 controller representation.

A zero-centered fitted field may declare a positive
`selection_regularization_scale`. Among candidates that pass both the registered
response metric and the pinned flight-parameters qualification, final selection
uses the smallest dimensionless L2 correction across those declared fields,
then the response metrics as tie-breakers. If no candidate passes the response
metric, gate proximity continues to determine the diagnostic selection; if
response candidates pass but none fit the flight parameters, the best response
candidate remains diagnostic evidence and cannot publish. This keeps an
estimator-offset fit from spending unnecessary correction authority merely to
improve an already-passing trace metric without allowing that correction to
escape the measured hover envelope.

The default uses one persistent worker and 128 candidate slots. This is the
measured useful-throughput optimum for the current GPU; additional workers
increase PhysX contention. The target coordinator reports candidates per
second, padding, cache hits, batches, and sampled GPU utilization. The 95% GPU
target remains diagnostic; deterministic useful throughput is the optimization
objective.
Explicit initial candidate settings participate in the same fixed-scene
capacity calculation as staged beam expansions, so a focused seeded grid is
evaluated in GPU-sized batches rather than one environment at a time.
The current X500 definition uses an explicit joint boundary grid and disables
additional midpoint refinement; broad exploratory definitions may enable it.

Each worker owns one persistent Isaac Lab environment for the entire ordered
probe-family list. Families execute sequentially by switching the controller
mode and padding the smaller family's trace stride to the fixed scene capacity;
constructing multiple `DirectRLEnv` instances in one process is unsupported by
Isaac Lab's simulation-context ownership.

### Independent selection and qualification

Parameter expansion and beam retention use the fitting response flight only.
After search finishes, candidates passing both response and full-window hover
checks are replayed on the validation response flight. Validation selects among
those candidates using the existing response ranking and declared L2
regularization, with the same hover bounds. It never expands the parameter grid.
The selected settings are frozen before the acceptance recording is evaluated.
An acceptance failure preserves that selection and fails qualification; it does
not trigger a second selection on acceptance data.

Each qualification stage validates the immutable flight bundle once and reuses
the loaded hover bounds for its candidates. Candidate iteration does not reread
and hash all producer evidence files. Fitting eligibility, validation selection,
and frozen acceptance each perform their own validation.

This is repeated qualification against a fixed benchmark. Subsequent controller
iterations reuse the same immutable PX4 bundle, including acceptance recordings.
Recapture only when PX4 setup, flight profiles, or recording configuration changes.
This repeated use is not a claim of an untouched final test population.

Fitting inputs and target traces retain the existing root filenames. Independent
validation and acceptance inputs and target traces live under `validation/` and
`acceptance/`. Evaluators have separate caches per set and reject mismatched
`set_name` headers or reference SHA-256 digests. Digests are pinned before worker
launch and checked when target results are loaded. Selection rows identify the recording set that owns their
metrics; metric names containing `fit` or `validation` still identify internal
profile splits. `qualification.json` records the frozen settings fingerprint,
pinned flight fingerprint, selection set, final evaluation set, raw metric
result, and failure reasons. Each held-out GPU evaluation also records its
reference digest and exact settings list in `evaluation.json`.

## Evidence and boundary audit

Flight-parameters response capture uses the ordinary fleet/bridge path with a
deterministic policy adapter. Its experiment selects PX4 logging profile `17`, preserving
the default topic set plus full-rate attitude, rate, thrust, torque, and
actuator topics needed for exact 250 Hz replay. Its configured pusher is
deployed after takeoff and before probe
preconditioning so the reference plant matches Lab training and press-policy
deployment. The Lab target also consumes the Hydra-owned rigid-body gyroscopic
force setting, which is disabled to match the standalone PX4 plant. Each
profile precondition uses a bounded velocity command to return to the routine
reference position and reserves the full configured preconditioning timeout.
The continuous stability hold must be satisfied at that deadline; an early
stable state does not advance the profile and a late one fails capture. The recenter
gain and speed bound live in `alignment.capture`, so profile order cannot
accumulate displacement against the routine safety envelope. Each profile then
records its absolute PX4 simulation start time and the complete initial
controller state required for replay. That boundary includes both loop phases,
the held velocity setpoint, the desired attitude and yaw rate, the desired and
held collective setpoints, the held body-rate setpoint, the filtered yaw-torque
state, velocity and rate integrals, estimator filters, and the hover estimate.
Resetting any held state
to a neutral value makes a profile's result depend on where capture happened to
land in PX4's asynchronous controller schedule and is therefore invalid. The
flight-parameters bundle retains these response traces without reduction.
Alignment preparation validates exact profile order, commands, cadence, sample
counts, split identity, completion, and controller-boundary state before
pinning the resolved run. One alignment run directory keeps:

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
reports the per-axis body-rate setpoint and collective setpoint errors before
the rate controller and the per-axis angular-velocity and derivative errors at
the rate-controller input. It consumes PX4's independently published angular
derivative rather than attempting to derive it again from logged rate samples;
held-setpoint source ages and rate-sample interpolation errors remain visible.
For velocity profiles, replay also reports per-axis ENU velocity-input RMSE
and signed mean error against PX4's recorded velocity state, including its
vertical-state blending. These diagnostic values separate estimator mapping
error from downstream attitude and motor errors and do not change acceptance limits.
The core replay then enforces the same motor and collective limits. A second
replay starts from the physics-rate physical articulation state in the bridge
log, seeds the recorded
initial estimator state, and requires normalized motor RMSE below 0.01;
closed-loop motion gates cover the remaining estimator residual. It uses each
profile's absolute PX4 boot time, the pinned 125 Hz PX4 velocity-loop cadence,
the target's fitted fractional attitude-loop cadence, and recorded command
cadence. A recorded policy sample contains the setpoint
published after observing that timestamp; replay therefore applies a changed
setpoint on the following 4 ms controller tick. The controller command and
PX4's vertical velocity-state blending both consume that same delayed setpoint.
The collapsed target cascade consumes an attitude-loop publication in its rate
controller on the same physics tick. PX4 runs attitude, rate, and allocation on
independent work queues, so an adjacent subscriber can observe that publication
on either the same tick or the next one. Replay derives those source-age choices
from the recorded attitude, rate, and thrust topics, then joins the target motor
prediction by at most one 4 ms physics tick. It reports both adjacent handoffs,
the total actuator join offset distribution, and the peak collective window.
When the thrust topic has no publication at a motor timestamp, replay recovers
that collective by inverting unsaturated PX4 motor allocation. A neighboring
logged thrust value must not introduce a spurious handoff delay. Saturated
outputs retain the logged thrust value because allocation is not invertible.
The unshifted intermediate setpoint metrics remain authoritative, so this join
cannot absorb a controller-value error. CTBR remains the direct rate-controller
contract.
Replay holds the exact recorded 40/60 ms intervals and does not quantize them
through a second synthetic 20 Hz cadence. The seed phase is derived per profile from the
recorded age of its `vehicle_local_position` sample. Because the replay grid
includes the profile boundary, its initial accrued outer-loop time is the
sample age minus one 4 ms physics tick; this preserves the next 125 Hz update
whether the boundary lands on or between estimator publications. ULog
rate-limited `trajectory_setpoint` samples are not used to infer command
transitions. The velocity-integral seed reconstructs the current acceleration
setpoint from the recorded `vehicle_local_position_setpoint` output and reports
its source age. Its derivative seed comes from the velocity-controller low-pass
of estimator velocity differences, using the recorded sample times and
`MPC_VELD_LP`. EKF `ax/ay/az` are a different acceleration signal and cannot seed
that filter. Capture requires disabled velocity low-pass/notch filters and
unchanged velocity-filter parameters. Response recordings identify this method
as `px4-ulog-controller-boundary-v2`; preparation rejects missing or different
methods and requires a new capture. Existing producer artifacts remain immutable.
Replay builds relative command time from integer physics steps so
large absolute profile start times cannot delay an exact command boundary by a
floating-point tick. The replay also reproduces PX4's
velocity-only vertical state blending and applies logged hover-estimate updates
at their source timestamps. For velocity profiles it also reconstructs PX4's
exact vertical acceleration setpoint from the full-rate desired attitude,
collective thrust, and hover estimate, then reports acceleration-setpoint and
inferred velocity-integral error before the thrust conversion boundary. It
keeps desired-collective to attitude-setpoint, attitude-setpoint to
rate-setpoint, and rate-setpoint to allocator-thrust error separate so the
first divergent vertical boundary is explicit. It
scans that upstream vertical setpoint over zero through three physics ticks so
outer-loop phase error remains separate from controller gains. A
zero-through-three-tick delay scan compares the target attitude-loop output to
PX4's downstream rates setpoint so asynchronous module handoff error remains
separate from controller gains. The target-only motion-rate
`velocity_observation_scale`, low-command `velocity_observation_hold_scale`,
steady world-ENU `velocity_observation_bias_mps`,
the componentwise command-magnitude transition
`velocity_observation_bias_command_start_mps` /
`velocity_observation_bias_command_full_mps`, the motion-rate
`velocity_estimator_bias_decay_hz`, the componentwise low-command
`velocity_estimator_bias_hold_decay_hz`,
per-axis `velocity_filter_hz`, and
`angular_velocity_filter_hz` model the evidenced physical-to-estimated state
boundary without changing the pinned PX4 controller gains. The smooth command
transition suppresses steady correction during exact zero-command ringdown.
Outside that band, the authored bias applies when physical velocity follows
the command; it fades out during larger tracking errors. Observation
scale uses the same transition with the larger command or physical speed on
each axis. The command transition also blends
estimator-bias persistence from each axis's low-command hold rate to the motion
rate.
The core replay
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

A run cannot finalize or publish unless acceptance response and hover checks
pass, every stability trace completes with finite values, all three actuator
audits and controller-boundary replays pass, and the integration report passes.
The run-local controller stays quality-failed with an explicit pending reason
until these checks complete. Finalization verifies that qualification and
integration reports identify the selected settings and pinned flight inputs
before marking the controller quality-passed. Policy training rejects a controller whose
quality record is failed. The authoritative live gate is:

```bash
make test-integration-alignment-target
```

Run the full campaign with:

```bash
make measure-flight-parameters
make align-controllers
```

`align-controllers` begins with `alignment-prepare-reference`; it never captures
PX4 reference data. The campaign runs the authoritative integration check against its run-local
`controller.json` after replay and before finalization, retaining a settings-bound
`integration.json` report. A failure therefore
preserves diagnostic evidence without advancing `latest` or `best`.

`controller.json` is the downstream product. It contains complete controller
settings, the control modes certified by the accepted probe-family list, and
the exact flight-parameters fingerprint used by its hover-pose qualification;
policy-owned normalized action scales are not alignment fields. Candidate traces and ULogs are
valuable reproducibility evidence, but policy training consumes only the
controller artifact and the independently published matching flight parameters.
`latest` and comparable `best` links advance only for a fully accepted bundle.

## Change policy

Changing a metric, probe suite, environment, image, controller interface,
controller definition, or resolved experiment starts a new comparison
fingerprint. The fingerprint includes hashes of the metric and normalized
controller implementation modules, so an equation change cannot remain in an
old comparison. Plant residuals must be fixed at the plant seam; they must not be
absorbed into controller gains. A selected parameter touching a range boundary
is not evidence by itself: the matching trace or ULog boundary must support an
expanded range or structural correction.
