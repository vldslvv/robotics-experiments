# Experiment configuration

Flight experiments live in the flat `configs/routines/` directory.
`load_config(path)` composes and process-caches one explicitly, returning an
immutable `ExperimentConfig`. It never reads the environment. Runtime roots read
the required `ROBOTICS_EXPERIMENT_CONFIG_PATH` variable once and pass the
resolved configuration to downstream consumers.

## Flight configurations

Filename prefixes describe purpose: `basic_` selects a single-drone scripted
flight, `fleet_` selects a multi-drone scripted flight, and `policy_` selects a
flight executing a learned policy. All filenames below are relative to
`configs/routines/`.

| Configuration | Flight |
| --- | --- |
| `basic_takeoff_quad.yaml` | One X500 takes off and holds stable hover for three seconds; ends airborne. |
| `basic_forward_quad.yaml` | One X500 takes off, hovers, travels five metres along world +X, and lands. |
| `fleet_takeoff_quad_quad.yaml` | Two X500s take off and hover independently, starting four metres apart. |
| `fleet_navigate_quad_hex.yaml` | An X500 and S550 fly parallel guided and OFFBOARD waypoint legs, then land. |
| `fleet_circle_land_quad_quad.yaml` | Two X500s fly twelve-waypoint circles with altitude separation, then land. |
| `policy_press_approach_quad.yaml` | Navigate to the warehouse switch, execute the approach actor, return and land. |
| `policy_transfer_press_hover_quad.yaml` | Execute the press-hover actor from three entry ranges; ends after the final policy hold. |
| `policy_transfer_press_approach_quad.yaml` | Fly the same three-range transfer route using the approach actor. |

The single-X500 takeoff config supplies shared vehicle setup for forward,
navigation, dual-takeoff and press flights. The circle config inherits the
dual-X500 fleet. The production approach config supplies the warehouse and
switch setup to approach-transfer, which hover-transfer specializes with its
hover task, episode duration and reset mixture. Both transfer configs select
one shared route; its artifact path uses the explicit `rl.press.task` value.
Flight steps, timeouts and acceptance requirements remain task-owned.

Nested experiment files declare `# @package _global_` and use absolute Hydra
defaults such as `/base` and `/routines/basic_takeoff_quad`. The configuration
root remains `configs/`; do not add a second `base.yaml` below it. Airframe
lookup and repository-relative artifact paths are independent of the selected
experiment's directory. Shared per-vehicle routine groups remain under
`configs/experiment/routine/`.

Specialized experiment entrypoints stay at the config root:
`flight_parameters_x500.yaml` captures a complete hover/response suite,
`alignment_px4_x500_velocity.yaml` defines the alignment diagnostic environment, and
`flight_benchmark.yaml` / `flight_benchmark_takeoff_quad.yaml` wrap the circle
and single-drone flights with longer cold-start bridge deadlines. The two calibration experiments inherit the common environment in
`experiment/calibration/x500.yaml`, including deterministic sensor settings and
high-rate logging. The flight entrypoint also contains the complete
`flight_parameters` block: identity, seed/count, preparation, stages, execution
and final stability settings. Other experiments use `flight_parameters: null`.
The core loader composes and freezes this block; the flight module owns physical
validation and final analysis. Training curricula and stage experiments remain
under `configs/train/`.

## Root structure

The root contains only execution-wide configuration:

- `sim`, `rates`, `switches`, `rl`, `scene`, `driver`, `flight_benchmark`,
  `alignment`, and optional `flight_parameters`;
- `vehicles`, an ordered mapping of complete per-vehicle configs.

`configs/base.yaml` composes the global groups only. Vehicle-specific `spawn`,
`bridge`, `px4`, and `routine` configuration is composed beneath each
`vehicles.<label>` node.
See `docs/FLEET_CONFIG.md` for the authoring pattern.

Experiments are selected with `HYDRA_CONFIG`, for example:

```bash
make run-all HYDRA_CONFIG=configs/routines/fleet_takeoff_quad_quad.yaml
```

The standalone `--scene` option overrides the configured scene; `@`-prefixed
scene paths are repository-relative.

The global `switches.spawn_seed` deterministically drives both switch position
sampling and lamp heights. Reusing the same resolved experiment configuration
reproduces the complete switch/lamp layout; changing the seed selects a new
layout.

Switch count and spawn retry budget must be positive. `switches.model` owns the
shared geometry, mechanics and confirmation duration; all numeric model values
are validated before use. See `ASSET_GENERATION.md` for profiles and preparation.
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
`rates.command_rate_hz` drives capture commands, alignment and new training.
`rates.sample_rate_hz` defaults to `${rates.command_rate_hz}` in `base.yaml` and
may be overridden independently for recording. Flight capture resolves these
values from its single experiment entrypoint; alignment definitions resolve them
from their referenced experiment. Switch activation runs every physics step
using `switches.model.confirmation_s`, independently of policy cadence. Exported artifacts own evaluation and
deployment policy rate. All rates must be finite and positive; authored or
artifact policy cadence and setpoint rates cannot exceed the driver rate, and
the driver rate cannot exceed `sim.physics_hz`. Recording cannot exceed the
physics rate. Defaults are 50, 50, and 20 Hz
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

`rl.press.task` selects explicit `hover`, `approach`, `contact`, or `press` behavior, independently of
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
`contact` owns contact/press progress rewards, stable-touch samples, compression
and conditional first-contact/retreat requirements, plus diagnostic feedback
settings. The shared `goal.stable_steps` remains the press retreat hold;
`termination` owns contact
sensing and production flight envelopes, including a range-undershoot limit
equal to the acceptance minimum range; `evaluation` owns disjoint validation/test
seeds and a positive episode count per seed (100 by default); and `acceptance` owns the minimum success rate,
maximum collision rate, and minimum range. Training-attempt, validation, and test seed sets must be
nonempty, internally unique, and mutually disjoint; the final test set contains
exactly three seeds. Independent reset
velocity/range tunables are intentionally absent so training and deployment
cannot resolve different entry bounds. Invalid or uncovered navigation
measurements, handoff/goal/velocity-reference bounds, negative or non-finite
reward weights, invalid rates, and unknown tasks fail during `load_config`.

Training curricula use a separate `CurriculumConfig` and
`load_curriculum_config(path)` boundary. Their YAML entrypoints live in
`configs/train/`, with complete `ExperimentConfig` stage files subordinate in
`configs/train/stage/`. Both loaders use the same isolated Hydra composition and
immutable caching. Nested files declare `# @package _global_` and reference root
defaults using `/base` or `/base_curriculum`. Airframe enrichment remains rooted
at `configs/airframes`, independent of the experiment's directory.

A curriculum has exactly `policy_name`, optional `initial_learner`, and a nonempty
`stages` list of unique `{name, config_path}` entries. Stage paths resolve from the
repository root. One-stage curricula use this same schema. Names are display
labels. The curriculum owns the namespace and initializer; stage experiments own
all task, PPO, phase, plateau, evaluation and acceptance settings. Sharing stage
settings uses normal Hydra defaults, not a second shared-training section.

`rl.training` owns PPO, rollout settings, task selection and attempt seeds. Each
phase owns its maximum updates, validation cadence, learning rate, schedule,
entropy coefficient and observation-normalization updates. Every attempt starts
in a fresh Isaac process. Later phases restore the attempt-wide selected complete
learner, apply their schedule, then reset environments.

`initial_learner` is a producer training report, accepted or unaccepted, whose
selected complete learner must be intact and compatible. The coordinator then
advances only on each stage's acceptance, publishing before forwarding its exact
selected learner. There are no stage-local initializers, implicit standalone
mode, nested curricula or process resume. See `RL_WORKFLOW.md` for checkpoints,
reports and examples.

`rl.inputs.controller_artifact` and `rl.inputs.flight_parameters_artifact` remain
stage-owned evidence selectors. Preparation resolves them to canonical producer
paths and digests before creating a training run. The controller's adjacent `dataset.json` supplies training membership and the
numeric hover summary. Workers revalidate those pinned files; they never resolve a newer label implicitly.

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
`configs/train/press_hover_quad.yaml` selects the `hover` task and `press-hover`
artifact namespace; it authors 40-second episodes.
`configs/train/press_approach_quad.yaml` selects the `approach` task and
`press-approach` namespace; it authors 45-second episodes.

Alignment's shared block includes `dataset_split` (train and validation
percentages, with test as the remainder), `evaluation` (response/live limits and
training-hover margins and policy warmup/statistics windows), and `execution` (worker and GPU batch sizing). See
[controller alignment](SIM_TO_SIM_ALIGNMENT.md) for ownership and scoring.
