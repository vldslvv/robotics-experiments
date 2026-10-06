# Controller alignment

Controller alignment makes the high-throughput Isaac Lab controller reproduce
PX4 SITL for identical physical commands. It is upstream of reinforcement
learning. Waypoint navigation, policy rewards, policy actions, and domain
randomization are not fitting inputs.

Capture publishes independent complete 24-stage flights without dataset roles.
Alignment owns the following flow:

```python
membership = _split_flights(flight_indices=indices, train_percent=80,
                          validation_percent=10, seed=start_seed)
# GPU coordinator: train search -> validation selection -> freeze settings.
# Evaluate the frozen controller on test flights, then replay and live hover/goto.
finalize_alignment_bundle(run_dir=run_dir)
publish_controller(run_dir=run_dir)
```

## Architecture and contracts

`controller` owns controller settings, artifacts and actuation equations.
`flight_parameters` owns immutable physical evidence. `alignment` owns splitting,
fitting, numerical evaluation and publication. RL never imports alignment.
The host-safe `_evaluation.py` has explicit measurement and limit inputs and no
filesystem, simulator, search or publication dependencies. `_metrics.py` computes
response measurements. The evaluator alone converts performance measurements into
normalized scores and acceptance results; callers consume those results.

Preparation resolves the capture selector once and verifies every flight's stage
commands, recorded sample times and reconstructed controller boundary. It writes
one minimal `dataset.json`:

```json
{"source":"../../flight-parameters/suite/run", "source_digest":"<SHA256>",
 "train":[1], "validation":[2], "test":[0]}
```

Membership lists contain producer flight indices. There are no copied samples,
counts, percentages or seeds in this file. Reference descriptors contain the set
name, profile interpretation and relative dataset path. `flight_stage_views`
selects the original sample ranges in memory without interpolation or reduction.
Inactive command fields are zero for the selected control mode. ULogs and bridge
logs are read directly from each flight's original compressed producer files.
Target results reuse the producer digest recorded in `dataset.json`; dataset
roles identify their assigned flights.

`alignment.dataset_split` in the composed experiment owns train and validation
percentages; test gets the remainder. Splitting uses a separate RNG seeded by the
capture's `start_seed`. It shuffles whole flights before allocation, leaving every
stage and sample in order. Largest-remainder rounding uses train, validation,
test as its tie order. Empty roles receive one flight from the largest role with
more than one. At least three complete flights are required. Thus three flights
produce 1/1/1; ten produce 8/1/1; one hundred produce 80/10/10.

Every assigned flight contributes to the response measurements. Profile identity
is `(flight_index, profile)`. Workers replay one flight's profiles at a time in a
fixed scene, so increasing flight count does not increase the number of GPU
environments. RMS measurements pool the actual samples; peak checks retain the
worst event across flights. Native-hover bounds use only full training-flight
windows, with the configured position and attitude margins. Held-out flights
cannot enlarge these bounds. The target's complete `stability-hover-pose` traces
are evaluated against those same bounds.

The dependency direction is:

`config -> controller -> {alignment, rl} -> workflows`, with the
`config -> flight_parameters -> {alignment, rl}` evidence boundary.

Controller artifacts remain schema 1. Completed evaluations publish whether
accepted or rejected; incomplete or corrupt evidence cannot publish. Publication reads the current artifact format without migration or fallback. Input capture
bytes remain producer-owned and immutable.

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
slower policy timing.

Alignment receives an ordered list of probe families. A family contract is
only a stable name, a control mode, a registered metric, and physical command
profiles. The current required families are `velocity`, covering body-FLU X/Y/Z
velocity plus velocity-path yaw, and `ctbr`, covering normalized collective
thrust plus body-FLU roll/pitch/yaw rates. Capture, Lab replay, search, evidence,
and publication iterate this list generically. Adding a control capability does
not require another alignment engine or a policy dependency.

## Search

Controller definitions live in `configs/alignment/controllers/`. Their command
rate is resolved from the referenced experiment's `rates.command_rate_hz` and
included in the definition fingerprint; it is not authored again in each
definition. Capture and alignment share the X500 calibration environment.
Every tunable
field must be declared as a `ParameterSpec`; search never invents controller
fields. The `coarse-to-fine-beam` strategy:

1. evaluates the committed base setting;
2. performs one wide, large-step sweep for each declared field;
3. retains the global beam plus per-gate champions;
4. generates midpoints around retained regions for each configured refinement level;
5. deduplicates settings by their exact float32 controller representation.

Candidates are ordered by one scalar: the maximum measured error divided by its
allowed limit. There is no accepted-first branch or regularization preference.
The beam also preserves champions for individual checks to explore competing
objectives. An unusable or incomplete simulation has no valid score.

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

### Independent selection and evaluation

Only training flights drive parameter expansion. All usable training candidates
are scored on validation flights, including candidates above the acceptance
threshold. The lowest validation score wins; an exact tie retains the earlier
candidate. Its settings are frozen before test evaluation. Test failures never
trigger reselection or another search. The GPU coordinator performs selection
once and writes the provisional controller and compressed selection records.
The Isaac Lab image includes `zstd` for these records.

Training inputs and outputs use the run root; validation and test use
`validation/` and `test/`. Evaluators have isolated caches per role. The selected
controller is then replayed against original ULog/bridge evidence and run through
the live hover/goto integration. Finalization requires complete evidence bound
to the selected settings and capture digest. It compresses target outputs in
place, preserving relative input references, and inventories the completed run.

This is repeated qualification against a fixed benchmark. Selecting published
`best` by the final test score allows benchmark comparison; it is not a claim of
an untouched test population across repeated controller-development efforts.

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
Alignment preparation checks profile order, authored and recorded commands,
completion, and reconstructed controller boundaries before pinning the run.
The producer reader validates original sample ranges and strictly increasing
measurement timestamps; alignment does not impose a synthetic time-zero grid. One alignment run directory keeps:

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
controller-boundary replay applies the selected controller to every train,
validation, and test profile. Its core replay uses PX4-estimated state and
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
profile's absolute PX4 boot time, the pinned 125 Hz PX4 velocity-loop timing,
the target's fitted fractional attitude-loop timing, and recorded command
timing. A recorded command request follows its policy observation; boundary replay
therefore applies that request on the following 4 ms controller tick. Request
timestamps remain separate from sampled physical observation timestamps. The controller command and
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
Replay uses the original measurement times and separate command-request times.
A controller boundary can precede its first physical measurement. The target
advances from that boundary to each command event and each observation, preserving
the offset without shifting or interpolating producer samples. The seed phase is derived per profile from the
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

`alignment.evaluation.response_limits` owns the existing response limits.
The pure evaluator returns the existing check records (name, measured error,
allowed limit), one score and acceptance. The score is
`max(error / limit)`; acceptance is exactly `score <= 1`. This rule ranks training
and validation candidates and decides final acceptance. No average can hide a
failed component. Final scoring combines test response, test hover, test
controller-boundary replay and live hover/goto measurements. Reference actuator
audits and valid timestamp associations are input-integrity prerequisites.

The translational limits apply to each X, Y and Z axis independently:

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

A completed rejected controller has `advisory_quality_passed: false` with
component failure reasons. `qualification.json` retains its final score and all
measured values and limits. A passing result uses the same report and artifact
format. `latest` advances to every completed evaluation; comparable `best`
advances only for a strictly lower final score, with ties retaining the existing
best. Different comparison inputs start a new best; publication does not scan
older runs to recover a previous group. Existing run artifacts are preserved.
Invalid publication inputs raise an error.

Run capture and alignment with:

```sh
make flight-parameters FLIGHT_COUNT=3
make align-controllers FLIGHT_PARAMETERS_ARTIFACT=<capture-run> ALIGNMENT_RUN_ID=<run-id>
```

Three flights exercise the interface with one flight per role. They do not
establish statistical stability. Alignment never launches its own PX4 capture.
The authoritative live check remains `make test-integration-alignment-target`.
During alignment, it records both hover and goto errors even if one exceeds its
limit. Interface failures or incomplete executions still stop the workflow.

To follow local progress in the existing Isaac Lab TrackIO image:

```sh
make trackio-alignment ALIGNMENT_RUN_ID=<run-id>
```

The board reads `progress.json` and stores its own data outside the immutable run.
It does not affect selection or acceptance. Final reports remain authoritative.
The default address is `http://127.0.0.1:7861`; Make overrides select host and port.

RL's qualified flight-parameter/reset reader remains a separate handoff. This
workflow does not change RL controller equations, training inputs or domain
randomization; see [remaining work](backlog/flight_parameters_alignment_handoff.md).

## Change policy

Changing a metric, probe suite, environment, image, controller interface,
controller definition, or resolved experiment starts a new comparison
fingerprint. The fingerprint includes hashes of the metric and normalized
controller implementation modules, so an equation change cannot remain in an
old comparison. Plant residuals must be fixed at the plant seam; they must not be
absorbed into controller gains. A selected parameter touching a range boundary
is not evidence by itself: the matching trace or ULog boundary must support an
expanded range or structural correction.
