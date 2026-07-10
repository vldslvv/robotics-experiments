# Experiment configuration

This is the reference for top-level Hydra experiment configs. `AGENTS.md`
keeps only the short routing summary and points here. Fleet-specific resolution
details live in `docs/FLEET_CONFIG.md`; routine semantics live in
`docs/COMMAND_ROUTINE_CONTRACT.md`.

## Top-level experiments

Every top-level YAML sibling of `configs/base.yaml` is an experiment. There is
no system distinction between a runtime experiment and a test experiment: every
experiment is runnable through the standalone workflow and the extension
workflow (`make run-all`).

The host test suite discovers every one of these top-level experiment files and
composes it through the public config facade. Adding an experiment therefore
automatically adds a configuration-composition regression case.

Each experiment sets a top-level `scene` (the USD stage; `@`-prefixed paths are
repo-relative) and a top-level `vehicles:` list. `configs/base.yaml` is shared
defaults only and does not carry `scene` or `vehicles`.

Experiment variants are selected with `HYDRA_CONFIG=<path>`, for example:

```bash
make run-all HYDRA_CONFIG=configs/takeoff_quad_quad.yaml
```

Library callers use `robotics_experiments.config.load_config(path)` when they
need an explicit, uncached composition. That operation does not consult
`ROBOTICS_EXPERIMENT_CONFIG_PATH` or clear process-global Hydra state.
Production shells use the cached `get_config()` adapter, which resolves the
environment-selected path once and delegates to `load_config`.

Both entrypoints return the same immutable resolved graph. Dataclass attributes
cannot be reassigned, resolved sequences are tuples, and routine step mappings
are read-only; construction-time mutation is confined to the private loader.

The standalone runner's `--scene` is optional and overrides the experiment
scene. Integration targets that need the experiment's authored scene use the
`@config` sentinel.

## Composable groups

Experiments inherit `configs/base.yaml`, which composes the shared groups under
`configs/experiment/`:

- `sim`
- `drone`
- `switches`
- `px4`
- `rl`
- `routine`
- `driver`

The groups are referenced as `experiment/<group>@<group>`. The `routine` group
is composed by `configs/base.yaml` itself, so every experiment that inherits
`base` has `EnvConfig.routine` and the default `routine.steps: []` means manual
operation.

## Routine and strict mode

Flight routine configuration and verification live in the Hydra `routine`
group: command-flow timeouts, takeoff target, and flight-envelope gates. There
is no separate `integration_test` group; integration tests run routine
experiments in strict mode through the standalone runner's `--strict` flag
(`STRICT=1` in the Make target).

Per-vehicle routine overrides are authored under each `vehicles[]` entry and are
resolved as part of fleet resolution. A drone whose resolved `routine.steps` is
empty is manual; a drone with non-empty steps gets a `RoutineDriver`.

The GCS link baseport stays in the `px4` group (`px4.gcs_baseport`); each
vehicle's command port uses the derived `ResolvedVehicle.gcs_port`.

## Shipped experiment configs

- `configs/takeoff_quad.yaml` selects the shared `takeoff` routine for one
  drone on the minimal integration stage.
- `configs/takeoff_quad_quad.yaml` selects the same `takeoff` routine for two
  drones and declares its own two-vehicle fleet.
- `configs/navigate_quad_hex.yaml` runs the absolute-waypoint `goto` navigation
  routine for a quad and a hexacopter in parallel lanes.
- `configs/forward_quad.yaml` runs a single-drone forward-flight routine.
- `configs/circle_land_quad_quad.yaml` runs two independent circular waypoint
  routines with altitude separation.
- `configs/policy_hover_quad.yaml` runs the trained-policy hover routine; its
  strict integration target requires a trained artifact at
  `.build/training/hover/latest`.

The persistent runtime entrypoint
`isaac/runtime/standalone/runapp_custom.sh` reads its stage from the experiment
`scene`.
