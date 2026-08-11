# Experiment configuration

Every top-level YAML sibling of `configs/base.yaml` is a runnable experiment.
`load_config(path)` composes and process-caches one explicitly, returning an
immutable `ExperimentConfig`. It never reads the environment. Runtime roots read
the required `ROBOTICS_EXPERIMENT_CONFIG_PATH` variable once and pass the
resolved configuration to downstream consumers.

## Root structure

The root contains only execution-wide configuration:

- `sim`, `switches`, `rl`, `scene`, `driver`, `flight_benchmark`, and
  `alignment`;
- `vehicles`, an ordered mapping of complete per-vehicle configs.

`configs/base.yaml` composes the global groups only. Vehicle-specific `spawn`,
`bridge`, `px4`, and `routine` configuration is composed beneath each
`vehicles.<label>` node.
See `docs/FLEET_CONFIG.md` for the authoring pattern.

Experiments are selected with `HYDRA_CONFIG`, for example:

```bash
make run-all HYDRA_CONFIG=configs/takeoff_quad_quad.yaml
```

The standalone `--scene` option overrides the configured scene; `@`-prefixed
scene paths are repository-relative.

The global `switches.spawn_seed` deterministically drives both switch position
sampling and lamp heights. Reusing the same resolved experiment configuration
reproduces the complete switch/lamp layout; changing the seed selects a new
layout.

Switch count, press confirmation steps, and spawn retry budget must be positive.
Switch/lamp bounds must be non-empty three-value corners with each minimum no
greater than its maximum. Invalid values fail during `load_config`, before any
stage mutation.

`flight_benchmark.airborne_duration_s` is used only by the opt-in full-GUI
benchmark. Keeping it in the resolved graph makes the capture duration part of
the same canonical configuration fingerprint as the flown routine.

`alignment.capture.sensor_seed` seeds NumPy before a PX4
controller-alignment capture starts. The alignment experiment disables
stochastic IMU and GPS terms and uses deterministic simulation-time
barometer/magnetometer excitation: PX4 rejects a constant pressure stream as
stale, and the composed exact magnetic field lies outside its yaw-initialization
innovation gate. Alignment uses one tenth of both normal excitation amplitudes
and enables `sim.enhanced_determinism`. Normal experiments retain stochastic
sensor models and the default lower-cost PhysX execution. Complete closed-loop
reproducibility must still be verified with replicate captures.

`driver.control_rate_hz` drives the fleet cyclic executive,
`driver.offboard_setpoint_rate_hz` drives OFFBOARD transmission, and
`rl.policy_rate_hz` authors training and supplies the switch-latch fallback
when no press artifact is deployed. Exported artifacts own evaluation and
deployment policy cadence. All rates must be finite and positive; authored or
artifact policy cadence and setpoint rates cannot exceed the driver rate, and
the driver rate cannot exceed `sim.physics_hz`. Defaults are 50, 50, and 20 Hz
respectively.

`sim.enhanced_determinism` remains disabled by default and is enabled by the
alignment experiment. Rotor wrenches retain PhysX's default one-application-
per-step behavior. The bridge and Lab controller each submit one complete
wrench per physics tick; enabling per-TGS-iteration reapplication prevents the
standalone X500 from lifting under saturated thrust.

## Routine behavior

Every vehicle owns a complete `RoutineConfig`. Every bridge-enabled vehicle gets
exactly one `RoutineDriverState`; empty `steps` follows the normal driver
lifecycle, sends no commands, and completes. Routine descriptors use the typed
`RoutineStepConfig` mapping and are validated by the routine builder.

A `goto` descriptor may author a finite positive `ground_speed_m_s`. That form
uses PX4's guided `MAV_CMD_DO_REPOSITION` speed parameter, so PX4 owns trajectory
generation and braking; it is a target speed rather than a hard velocity limit.
A `goto` without the field retains the original streamed OFFBOARD lifecycle.
Future motion controls remain typed sibling fields rather than an untyped
parameter map.

Strict integration runs execute the same routines as live runs; only failure
reporting differs. `routine.failure_recovery_timeout_s` bounds layered landing
recovery after an armed failure. Fleet-wide waits derive from the active
per-vehicle routines, so there is no root routine.

`vehicles.<label>.px4.failsafe` owns PX4 process behavior independently of the
Isaac `bridge` group. `offboard_loss_action` accepts PX4's non-destructive flight
actions (`position`, `altitude`, `stabilized`, `return`, `land`, or `hold`) and
`offboard_loss_timeout_s` must be between 0 and 60 seconds. The shared vehicle
default is `land` after 1 second. PX4 has no native `disabled` value for
`COM_OBL_RC_ACT`; use an explicit supported action instead of pretending the
failsafe is disabled.

`vehicles.<label>.px4.logging_profile` is the PX4 `SDLOG_PROFILE` bitmask and
must be between 0 and 4095. Normal experiments use the compact default profile
`1`; controller alignment uses `17` to add full-rate attitude, rate, thrust,
torque, and actuator topics required by exact controller-boundary replay.

## RL vehicle selection

`rl.training.vehicle_id` selects the fleet vehicle whose spawn asset and resolved
bridge dynamics configure the Isaac Lab task. The ID is checked when the Lab task
is constructed. PX4 `bridge.enabled` does not affect Lab because Lab bypasses PX4.

`rl.lab_vehicle` owns Lab-only spawn height, rigid-body velocity caps, and
pusher effort/velocity limits. Pusher stiffness and damping are inherited from
the generated vehicle USD. `rl.hover` owns the hover task definition and its
`policy`; `rl.press` composes the named `approach` curriculum option and owns
its separate `policy`.
Each policy contains a shared activation, actor hidden dimensions, initial
noise, output initialization, and critic hidden dimensions. Hidden dimensions
must be positive, the activation must be non-empty, initial noise must be
finite and non-negative, and the output initialization is
`scratch-neutral-output`.

`rl.press.curriculum_stage` is required and currently accepts only `approach`.
`handoff` owns the single actor-entry envelope used by both Lab reset and live
deployment. `navigation_noise` records the accepted, repo-relative production
waypoint measurement summary, independent sample count, explicit position and
rotation margins, observed switch-range extrema, observed maximum combined
lateral/vertical error, and maximum full quaternion-geodesic attitude error.
Configuration requires at least ten samples and the handoff pose bounds to
cover those observations plus their margins;
`make measure-press-navigation-handoff` refreshes the raw evidence after sensor,
estimator, controller, route, or scene changes. `goal` owns the 100-sample
range/alignment/full-attitude/linear-speed/angular-speed predicate and
three-axis velocity reference, including its positive-motion alignment gate;
`reward` owns the position-progress/stabilization stage transition, one
predictive range-undershoot barrier, strict hold, and the terminal outcome;
`reset` owns only the physical scene;
`termination` owns contact
sensing and production flight envelopes, including a range-undershoot limit
equal to the strict goal minimum; and `evaluation` owns disjoint validation/test
seeds, 100 episodes per seed, cost weights, and the 95% success and 1%
maximum-contact gates. Training-attempt, validation, and test seed sets must be
nonempty, internally unique, and mutually disjoint; the final test set contains
exactly three seeds. Independent reset
velocity/range tunables are intentionally absent so training and deployment
cannot resolve different entry bounds. Invalid or uncovered navigation
measurements, handoff/goal/velocity-reference bounds, negative or non-finite
reward weights, invalid rates, and unknown stages fail during `load_config`.

`rl.training` owns the ordered phases in each PPO attempt, rollout settings,
task selection, artifact namespace, ordered `attempt_seeds`, and
`controller_artifact`. Its `flight_envelope_artifact` resolves the matching
task-neutral position/pose bundle required before training. Each phase owns its iteration count, checkpoint cadence,
learning rate, schedule, entropy, and optional training-only range-undershoot
recovery limit. Press recovery limits are positive, no greater than the strict
range minimum, nondecreasing across phases, and equal to the strict range
minimum in the final phase. They change only episode termination; every phase
uses the same full handoff reset distribution and strict 100-sample goal.
The campaign
coordinator starts every attempt in a distinct Isaac process and waits for that
process to exit before considering the next seed. Each later phase restores the
attempt-wide validation-selected actor, critic, optimizer, and iteration before
applying its configuration and resetting the environments. External checkpoint
initialization and process resume are unsupported. The
controller artifact defaults to
`.build/alignment/<alignment-id>/best` and is resolved before Isaac Lab is
constructed. `CONTROLLER_ARTIFACT` or the training CLI override selects an exact
run for reproduction.

`rl.action` is task-neutral and contains only normalized action kind,
dimension, and scales. Controller gains, command delay, maximum tilt, and
thruster response are not Hydra RL tunables; they are complete
`ControllerSettings` loaded from the controller artifact and embedded in policy metadata.
New campaigns capture both the task contract and alignment provenance before
PPO, validate every checkpoint on seeds 101/202/303, and test the first eligible
attempt's immutable selected actor on seeds 404/505/606. `rl.press.evaluation`
owns the held-out cost weights, both workload seed sets, workload size, and gate
thresholds. Accepted campaigns automatically advance `latest`, and comparable
`best` advances only after the fresh test passes. An explicit operator deployment
may instead assign `latest` to a strict diagnostic candidate without changing
`best` or its recorded qualification status. Only an accepted actor is exported
as a campaign publication.
`configs/train_press_approach_quad.yaml` selects the press task, the
`press-approach` artifact namespace, 15-second episodes, and its PPO settings.
