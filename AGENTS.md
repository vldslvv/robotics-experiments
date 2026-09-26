# AGENTS.md - Robotics stack (Isaac Sim + ROS 2 + PX4)

This is the single source of project instructions. `CLAUDE.md` is a symlink;
always edit `AGENTS.md`, never the symlink separately.

The repository integrates Isaac Sim, ROS 2, PX4 SITL, and Docker. Prefer minimal,
scoped changes. Do not introduce tooling unless explicitly requested.

## Required context

Read the owning documents before changing the corresponding area, including its
configs, tests, Make targets, and evidence consumers. Their contracts are project
instructions. Read only the areas relevant to the task; ADRs explain decisions
and backlogs describe future work, not current implementation requirements.

| Area | Required reference |
| --- | --- |
| Host Python, tests, quality checks, Makefiles, extension design | `docs/DEVELOPMENT_WORKFLOW.md` |
| Docker runtime, persistent containers, tmux, runtime logs | `docs/RUNTIME_OPERATIONS.md` |
| Hydra experiments, scene, config loading | `docs/EXPERIMENT_CONFIG.md` |
| Fleet config and runtime behavior | `docs/FLEET_CONFIG.md` |
| Runtime composition and lifecycle | `src/robotics_experiments/runtime/ORCHESTRATOR_CONTRACT.md` |
| Commanding, routines, GCS commands, goto/policy steps, driver | `docs/COMMAND_ROUTINE_CONTRACT.md` |
| RL, Isaac Lab, training, policy artifacts and operation | `docs/RL_WORKFLOW.md` |
| Controller, alignment, controller definitions, training provenance | `docs/SIM_TO_SIM_ALIGNMENT.md` |
| Flight config, hover/response capture, flight evidence consumers | `docs/FLIGHT_PARAMETERS.md` |
| Generators, switch behavior, asset preparation | `docs/ASSET_GENERATION.md` |
| Integration harness, standalone integration scripts and targets | `docs/INTEGRATION_TEST_ARCHITECTURE.md` |
| Standalone Isaac scripts and import ordering | `isaac/runtime/standalone/README.md`; tests also use `isaac/runtime/standalone/tests/README.md` |
| PX4 bridge interfaces, frames, motors, lockstep and freshness | `isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md` |
| Vendored Pegasus changes or re-sync | `isaac/runtime/extensions/px4_sitl_brigde/pegasus/PROVENANCE.md` |
| Commanding performance | `docs/COMMANDING_BENCHMARK.md` |
| Isaac upgrade performance | `docs/ISAAC_CORE_BENCHMARK.md` |
| Full-GUI/PX4 flight benchmark | `docs/FLIGHT_BENCHMARK.md` |
| ULog and bridge-log analysis | `flight_analysis/README.md` |

## Repository layout

- `configs/routines/`: flat runnable flights; prefixes `basic_`, `fleet_`, and
  `policy_`. Shared per-vehicle routines live in `configs/experiment/routine/`;
  measurement, alignment, and benchmark entrypoints stay outside this directory.
- `src/robotics_experiments/`: reusable experiment logic. `config` owns schemas
  and loading; `controller` owns shared settings/artifacts and NumPy actuation;
  `alignment` owns fitting and qualification; `flight_parameters` owns task-neutral
  measured evidence; `switches` owns shared asset identity and activation.
- `src/robotics_experiments/runtime/`: the single frozen `FleetState` pipeline
  and bridge-host composition; an Isaac-only lazy leaf driven by thin shells.
- `isaac/runtime/extensions/`: live Kit shells. `templates/` holds copyable
  templates, `standalone/` scripts and launchers, and `lab/` runtime examples.
  Isaac Lab task implementations live in the core package's `workflows/isaac_lab/`.
- `isaac/docker/`, `ros/docker/`, `px4/docker/`: image/build/init inputs.
- `isaac/isaac-sim/`: container-populated persistent Isaac cache/config.
  `ros/ros2_ws/`: persistent colcon workspace; `ros/config/`: shared middleware.
- `ros/scripts/`, `px4/scripts/`: host helpers. `px4/logs/`: persisted flight
  logs; `px4/eeprom/`: simulated EEPROM cleared by the default launch wrapper.
- `docs/`: current contracts and usage; `docs/adr/`: concise decisions;
  `docs/backlog/`: unfinished work. `benchmarks/` contains committed summaries
  and comparison policy; raw benchmark evidence stays in `.build/benchmarks/`.

## Python APIs and dependencies

- List intentional public top-level functions, classes, and constants in module
  `__all__`. Export only concepts needed by production consumers outside the
  boundary; tests do not justify public API. Keep internal helpers/private-only
  modules underscore-prefixed. Do not rename, remove, or broaden public exports
  during cleanup unless explicitly requested.
- `robotics_experiments.config` is the sole external config import boundary.
  `load_config(path)` is cached, explicit, environment-independent, and returns
  an immutable complete experiment graph. Runtime roots resolve
  `ROBOTICS_EXPERIMENT_CONFIG_PATH` and pass config downward.
- Import stdlib, project modules, and ordinary dependencies eagerly. Lazy and
  `TYPE_CHECKING` dependency imports are permitted only for container-only
  Isaac/Kit/Lab modules or required app-startup ordering. Resolve cycles
  structurally. Vendored Pegasus and its consuming import behavior are exempt.
- Use absolute project imports. Relative-import workarounds are limited to
  Isaac-only lazy leaves for modules unavailable on the host.
- Keep `import robotics_experiments` and its packages working without Isaac.
  Wholly Isaac-only domain modules may import `carb`, `pxr`, or `isaacsim` at
  module level only with a docstring declaring the exception and a lazy package
  boundary shielding the core import path. The guard is
  `tests/src/robotics_experiments/test_core_imports.py`.
- Layering: `config -> controller -> {alignment, rl} -> workflows`, with
  independent `flight_parameters -> {alignment, rl}` and
  `config -> switches -> {environment, rl}` boundaries. Domain packages
  (`drone`, `environment`, `simulation`, `px4_bridge`) sit below runtime and Kit
  shells. Controller and flight-parameter packages must not depend on alignment,
  RL, Isaac, Kit, or PX4 containers; alignment must not depend on RL or containers.
- Keep private helpers before public entrypoints where practical. Construct
  dataclasses directly when construction only forwards fields; builders need
  actual derivation, validation, or branching. Merge one-use forwarding helpers.
- Excluding `self`/`cls`, use either all keyword-only or all positional-only
  parameters in a function. Never mix calling conventions.
- When a requested change replaces an API, config, command, or workflow, update
  all production consumers, tests, and docs. Do not add compatibility shims or
  aliases unless requested; obsolete calls should fail loudly.

## State and configuration

- In runtime, commanding, and PX4 bridge code, evolving project state uses frozen
  dataclasses replaced by stateless transitions. Mutable classes are external
  resource adapters; expose immutable status snapshots, not live state-reading
  properties. Owning contracts document narrow hot-path exceptions.
- Tunable thresholds, timeouts, rates, and other runtime parameters belong in
  Hydra `configs/` YAML. Add a bare annotation with no default to the owning
  private config dataclass and plumb it through. Library constructors may have
  neutral fallback defaults, but must not duplicate project values.
- Invalid config and unexpected state must not fail silently. Raise for invalid
  config; warn on recoverable runtime fallbacks. Add checks only to prevent
  silent incorrect results or represent distinct states. Prefer concrete values
  or domain-appropriate sentinels over unnecessary `Optional`, routine `None`
  checks, and `<= 0 disables` conventions. Use `None` for genuine absence.
- `VehicleState` contains pose, velocity, angular velocity, and simulation time
  only. Do not add acceleration for PX4 HIL; the Pegasus IMU derives it from
  velocity deltas.

## Artifacts and execution

- Input artifacts are immutable, producer-owned evidence. Resolve and validate
  exact paths; pin digests across process boundaries. Resolve `latest`/`best`
  once and record the canonical producer `run_id` and digest/fingerprint as
  lineage. Reference the producer files directly; if a consumer filename is
  required, use a relative symlink. Never copy, rewrite, migrate, or mutate input
  artifact bytes into a consumer run.
- Do not commit new binary files. The three LightSwitchWarehouse USD crates in
  `.binary-asset-allowlist` are grandfathered; do not modify/regenerate them or
  expand the allowlist without an explicit asset-policy request. Read
  `docs/BINARY_ASSETS.md` for that policy.
- Do not regenerate assets unless explicitly asked. `make generate-assets` is
  the shared runtime/training/integration preparation dependency.
- Run Isaac, ROS 2, and PX4 through Docker Make targets, not directly on the host.
  Core targets are `make build`, `run-isaac`, `run-ros`, `run-px4`, `stop`, and
  `destroy`. Standalone scripts start `SimulationApp`/`AppLauncher` before
  Kit-dependent imports.
- Containers commonly use host networking and require NVIDIA GPU support;
  headless execution must work by default. Design independent GPU work units
  for parallel execution and aim for 95% measured utilization. Measure it;
  report real bottlenecks without changing deterministic semantics or adding
  meaningless work to inflate utilization.
- Preserve load-bearing vendored Pegasus fixes and their tests. Update its
  `PROVENANCE.md` divergence index when the snapshot or a divergence changes.
- Do not hardcode absolute paths.

## Development and documentation

- Use `uv` for all host Python work; never `pip`, `python -m venv`, or a
  system/global Python directly. Follow the development workflow for setup.
- Run `uv run pre-commit run --all-files` after implementation. Fix every reported
  issue, including pre-existing failures. Verify analyzer findings before
  deleting code; if correct code needs suppression, make it narrow and deliberate.
- Always run pre-commit with network access. Hooks fetch dependencies and remote
  schemas, including `json.schemastore.org`. In a network-restricted sandbox,
  use network-enabled execution permissions from the first attempt.
- Do not add lint/type ignores to `pyproject.toml`. Use the narrowest inline
  suppression with a brief reason when necessary. Do not make clear valid code
  indirect or verbose just to appease a checker that cannot express it.
- Use `# fmt: off` / `# fmt: on` around the smallest intentionally aligned Python
  table so formatters preserve its spacing.
- Refactoring must reduce maintenance cost or leave it unchanged. Prefer modifying
  existing files and deleting redundant code, tests, wrappers, and docs over
  adding abstractions, entrypoints, or parallel models.
- Use simple, concrete language and ASCII; avoid vague jargon. Comments and
  living docs describe current behavior, invariants, and intent, not change history.
- Update the owning doc when behavior changes affect conventions, error handling,
  runtime assumptions, public behavior, or developer workflow. Correct stale
  descriptions when touching the area.
- Do not create analysis or verification documents unless requested. Execution
  details belong in run-scoped artifacts; durable decisions become short ADRs,
  and unfinished work belongs in the backlog. Do not append campaign transcripts
  or duplicate tunable values and contracts across documents.

## When uncertain

Don't do anything, ask for instructions.
