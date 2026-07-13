# Experiment configuration

Every top-level YAML sibling of `configs/base.yaml` is a runnable experiment.
`load_config(path)` composes one explicitly; cached production callers use
`get_config()`. Both return the same immutable `ExperimentConfig`.

## Root structure

The root contains only execution-wide configuration:

- `sim`, `switches`, `rl`, `scene`, and `driver`;
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

## Routine behavior

Every vehicle owns a complete `RoutineConfig`. Empty `steps` means manual
operation; non-empty steps create exactly one `RoutineDriver` for that vehicle.
Routine descriptors use the typed `RoutineStepConfig` mapping and are validated
by the routine builder.

Strict integration runs execute the same routines as live runs; only failure
reaction differs. Fleet-wide waits derive from the active per-vehicle routines,
so there is no root routine.

## RL vehicle selection

`rl.training.vehicle_id` selects the fleet vehicle whose spawn asset and resolved
bridge dynamics configure the Isaac Lab task. The ID is checked when the Lab task
is constructed. PX4 `bridge.enabled` does not affect Lab because Lab bypasses PX4.
