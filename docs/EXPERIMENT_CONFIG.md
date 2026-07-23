# Experiment configuration

Every top-level YAML sibling of `configs/base.yaml` is a runnable experiment.
`load_config(path)` composes and process-caches one explicitly, returning an
immutable `ExperimentConfig`. It never reads the environment. Runtime roots read
the required `ROBOTICS_EXPERIMENT_CONFIG_PATH` variable once and pass the
resolved configuration to downstream consumers.

## Root structure

The root contains only execution-wide configuration:

- `sim`, `switches`, `rl`, `scene`, `driver`, and `flight_benchmark`;
- `vehicles`, an ordered mapping of complete per-vehicle configs.

`configs/base.yaml` composes the global groups only. Vehicle-specific `spawn`,
`bridge`, and `routine` groups are composed beneath each `vehicles.<label>` node.
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

`driver.control_rate_hz` drives the fleet cyclic executive,
`driver.offboard_setpoint_rate_hz` drives OFFBOARD transmission, and
`rl.policy_rate_hz` authors training and supplies the switch-latch fallback
when no press artifact is deployed. Exported artifacts own evaluation and
deployment policy cadence. All rates must be finite and positive; authored or
artifact policy cadence and setpoint rates cannot exceed the driver rate, and
the driver rate cannot exceed `sim.physics_hz`. Defaults are 50, 50, and 20 Hz
respectively.

## Routine behavior

Every vehicle owns a complete `RoutineConfig`. Every bridge-enabled vehicle gets
exactly one `RoutineDriverState`; empty `steps` follows the normal driver
lifecycle, sends no commands, and completes. Routine descriptors use the typed
`RoutineStepConfig` mapping and are validated by the routine builder.

Strict integration runs execute the same routines as live runs; only failure
reaction differs. Fleet-wide waits derive from the active per-vehicle routines,
so there is no root routine.

## RL vehicle selection

`rl.training.vehicle_id` selects the fleet vehicle whose spawn asset and resolved
bridge dynamics configure the Isaac Lab task. The ID is checked when the Lab task
is constructed. PX4 `bridge.enabled` does not affect Lab because Lab bypasses PX4.

`rl.lab_vehicle` owns Lab-only spawn height, rigid-body velocity caps, and
pusher effort/velocity limits. Pusher stiffness and damping are inherited from
the generated vehicle USD. `rl.press` owns the press reward, reset geometry,
contact threshold, termination envelopes, deterministic evaluation seeds/count,
and acceptance targets. `configs/train_press_quad.yaml` selects the press task,
the `press` artifact namespace, ten-second episodes, and its PPO settings.
