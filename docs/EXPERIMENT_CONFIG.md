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
`policy`; `rl.press` composes the `approach` task settings group and owns
its separate `policy`.
Each policy contains a shared activation, actor hidden dimensions, initial
noise, output initialization, and critic hidden dimensions. Hidden dimensions
must be positive, the activation must be non-empty, initial noise must be
finite and non-negative, and the output initialization is
`scratch-neutral-output`.

`rl.press.task` selects explicit `hover` or `approach` behavior, independently of
pipeline names and order. `rl.press.reset.mix` contains measured sources and
positive weights summing to one. `rl.episode_length_s` is the shared episode
timeout; `rl.press.goal.stable_steps` is the common hold requirement.
`rl.training.phases` and `rl.training.plateau` apply to each complete experiment.

`handoff` owns the single actor-entry envelope used by both Lab reset and live
deployment. `navigation_noise` records the accepted, repo-relative production
waypoint measurement summary, independent sample count, explicit position and
rotation margins, observed switch-range extrema, observed maximum combined
lateral/vertical error, and maximum full quaternion-geodesic attitude error.
Configuration requires at least ten samples and the handoff pose bounds to
cover those observations plus their margins;
`make measure-press-navigation-handoff` refreshes the raw evidence after sensor,
estimator, controller, route, or scene changes. `goal` owns the target pose and
three-axis velocity reference; hover success uses the measured position and
attitude envelope for `goal.stable_steps`. `reward` owns position progress,
stabilization, the predictive range-undershoot barrier, hold progress, and terminal
outcomes;
`reset` owns the physical scene and reset mixture;
`termination` owns contact
sensing and production flight envelopes, including a range-undershoot limit
equal to the acceptance minimum range; `evaluation` owns disjoint validation/test
seeds and 100 episodes per seed; and `acceptance` owns the minimum success rate,
maximum collision rate, and minimum range. Training-attempt, validation, and test seed sets must be
nonempty, internally unique, and mutually disjoint; the final test set contains
exactly three seeds. Independent reset
velocity/range tunables are intentionally absent so training and deployment
cannot resolve different entry bounds. Invalid or uncovered navigation
measurements, handoff/goal/velocity-reference bounds, negative or non-finite
reward weights, invalid rates, and unknown tasks fail during `load_config`.

`rl.training` owns shared PPO parameters, rollout settings, task selection,
artifact namespace, and `attempt_seeds`. Shipped configs author `[42]`: one
training attempt whose checkpoints are evaluated across multiple validation
seeds. Its `phases` and `plateau` fields apply to every task. Each phase owns its maximum iteration
count, validation cadence, learning rate, schedule, and entropy coefficient.
The campaign coordinator starts each attempt in a distinct Isaac process.
Each later phase restores the
attempt-wide validation-selected actor, critic, optimizer, and iteration before
applying its configuration and resetting the environments.
`rl.inputs.initial_checkpoint` optionally selects a producer training report's
complete learner and associated policy artifact. Independent stages require
integrity and compatibility, regardless of producer acceptance. Paths, producer
run ID, and digests are pinned and recorded; producer files remain unchanged.
`rl.training.pipeline` is an optional ordered list of `{name, config_path}` entries.
Names must be unique and nonempty. Paths reference complete experiments relative
to the repository, with empty pipeline and null initial checkpoint settings.
The pipeline accepts only a qualified matching starting learner, then advances
only on each stage's acceptance boolean and forwards the exact selected checkpoint.
`policy_name` owns artifact namespaces; each stage execution gets a distinct run
ID. Empty pipeline means an independent stage. Process resume is unsupported.
See `RL_WORKFLOW.md` for stage results and pipeline reporting. The
controller artifact at `rl.inputs.controller_artifact` defaults to
`.build/alignment/<alignment-id>/best` and is resolved before Isaac Lab is
constructed. `CONTROLLER_ARTIFACT` or the training CLI override selects an exact
run for reproduction. `rl.inputs.flight_parameters_artifact` selects the matching
task-neutral schema-1 evidence bundle required before training.

`rl.action` is task-neutral and contains only normalized action kind,
dimension, and scales. Controller gains, command delay, maximum tilt, and
thruster response are not Hydra RL tunables; they are complete
`ControllerSettings` loaded from the controller artifact and embedded in policy metadata.
New campaigns capture both the task contract and alignment provenance before
PPO, validate every checkpoint on seeds 101/202/303, and test the first eligible
attempt's immutable selected actor on seeds 404/505/606. Acceptance pools the
final-test episodes and requires every configured acceptance criterion and
measured flight-parameter statistic to pass. A failed final test ends the
campaign without retrying another checkpoint. Accepted campaigns automatically advance `latest`, and comparable
`best` advances only after the fresh test passes. An explicit operator deployment
may instead assign `latest` to a strict diagnostic candidate without changing
`best` or its recorded qualification status. Only an accepted actor is exported
as a campaign publication.
`configs/train_press_hover_quad.yaml` selects the `hover` task and `press-hover`
artifact namespace; it authors 40-second episodes.
`configs/train_press_approach_quad.yaml` selects the `approach` task and
`press-approach` namespace; it authors 45-second episodes.
