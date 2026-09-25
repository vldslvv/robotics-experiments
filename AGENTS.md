# AGENTS.md - Robotics stack (Isaac Sim + ROS 2 + PX4)

> This file is the single source of project instructions. `CLAUDE.md` is a
> symlink to `AGENTS.md`, so never edit `CLAUDE.md` directly (and do not try to
> "sync" the two) -- always make changes here in `AGENTS.md` and they apply to
> both.

## Project goal
This repository integrates:
- Isaac Sim (simulation)
- ROS 2 (control, perception)
- PX4 SITL (flight controller simulation)
- Docker-based development

Prefer minimal, scoped changes.
Do not introduce new tooling unless explicitly requested.

## Context map
Read area docs on demand when a task touches that area:
- Development workflow: `docs/DEVELOPMENT_WORKFLOW.md` for host Python, `uv`,
  tests, quality gates, and Makefile maintenance.
- Runtime operations: `docs/RUNTIME_OPERATIONS.md` for Docker runtime targets,
  persistent Isaac/PX4 containers, tmux behavior, runtime logs, and flight-log
  debugging.
- Experiment config: `docs/EXPERIMENT_CONFIG.md` for top-level Hydra
  experiments, `scene`, `HYDRA_CONFIG`, `configs/experiment/`, and strict-mode
  routine test relationships.
- Fleet config: `docs/FLEET_CONFIG.md` for `vehicles:`, `VehicleConfig`,
  per-vehicle PX4 launch, config-generated airframes, and multi-vehicle runtime
  behavior.
- Command/routine: `docs/COMMAND_ROUTINE_CONTRACT.md` for
  `src/robotics_experiments/commanding/`, routine config, GCS MAVLink commands,
  `{kind: goto}` / `{kind: policy}`, and driver orchestration.
- Commanding benchmark: `docs/COMMANDING_BENCHMARK.md` for the isolated host
  routine-stepper/driver performance gate, statistical comparison, artifacts,
  and baseline acceptance.
- RL workflow: `docs/RL_WORKFLOW.md` for `src/robotics_experiments/rl/`,
  `src/robotics_experiments/workflows/isaac_lab/`, Isaac Lab images, training,
  policy artifacts, and RL integration gates.
- Controller alignment: `docs/SIM_TO_SIM_ALIGNMENT.md` for the host-safe
  alignment package, task-neutral PX4/Isaac Lab probes, artifact publication,
  controller settings, metric registry, and training provenance.
- Flight parameters: `docs/FLIGHT_PARAMETERS.md` for task-neutral flight
  configs, repeated hover evidence, strict envelope artifacts, publication, and
  alignment/RL consumers.
- Integration tests: `docs/INTEGRATION_TEST_ARCHITECTURE.md` for integration
  target mechanics, harness scripts, caching, artifacts, concurrency, and
  multi-instance PX4.
- Isaac core benchmark: `docs/ISAAC_CORE_BENCHMARK.md` for the Isaac upgrade
  gate, per-contributor attribution, comparable-hardware contract, Tracy
  artifacts, and baseline acceptance.
- Flight benchmark: `docs/FLIGHT_BENCHMARK.md` for the opt-in Kit wrapper,
  production-runtime isolation, full-GUI/PX4 method, artifacts, and comparison.
- PX4 bridge contract:
  `isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md` for
  bridge interfaces, frame conventions, motor ordering, lockstep, and actuator
  freshness.
- Pegasus vendoring:
  `isaac/runtime/extensions/px4_sitl_brigde/pegasus/PROVENANCE.md` for the
  vendored snapshot, reachable modules, divergences, and re-sync procedure.
- Standalone Isaac: `isaac/runtime/standalone/README.md` for direct Isaac Sim
  scripts and import-order rules; standalone test details live in
  `isaac/runtime/standalone/tests/README.md`.
- Flight log analysis: `flight_analysis/README.md` for ULog/bridge-log pairing
  and repeatable instability or sim-to-real debugging.

## Comments and documentation
- Comments and docs should describe current behavior, invariants, contracts, and
  intent. Do not leave historical comments about how code used to be implemented
  before.

## Repository layout
Place new code in the directory matching its purpose; each entry below is both
the current contents and the intended home for that kind of code.
- `src/robotics_experiments/` -> Core Python package for reusable experiment logic.
- `src/robotics_experiments/controller/` -> Host-safe controller settings,
  strict controller artifacts, and the normalized NumPy actuation reference.
  Shared by alignment and RL; no Isaac, Kit, PX4-container, alignment, or RL
  dependency.
- `src/robotics_experiments/alignment/` -> Host-safe probe definitions, generic
  candidate fitting/ranking, alignment evidence, and controller publication.
  Upstream of RL; no Isaac, Kit, PX4-container, or RL dependency.
- `src/robotics_experiments/flight_parameters/` -> Host-safe task-neutral
  flight config, hover statistics, lossless response traces, and strict artifact
  publication. Shared by alignment and RL; no Isaac, Kit, PX4-container,
  alignment, or RL dependency.
- `src/robotics_experiments/runtime/` -> Host-agnostic fleet runtime: frozen
  `FleetState` lifecycle transitions (the one runtime pipeline) and the PX4
  `BridgeHost` /
  `BridgeHostGroup` lockstep wiring. Isaac-coupled domain leaf (lazy package
  `__init__`, off the core import path); driven by the thin per-host shells
  (the live `experiment_runtime` extension and the standalone runner).
- `isaac/isaac-sim/` -> Isaac Sim cache and config, populated automatically by
  container. Persists changes made inside Isaac Sim container.
- `isaac/docker/` -> Isaac Sim Docker build/init inputs.
- `isaac/runtime/extensions/` -> Isaac Kit extension workflow.
- `isaac/runtime/templates/` -> Copyable Isaac runtime templates, not loaded directly.
- `isaac/runtime/standalone/` -> Direct Isaac Sim scripts and launch helpers.
- `isaac/runtime/lab/` -> Isaac Lab configs, task definitions, and future
  environment code.
- `ros/ros2_ws/` -> ROS 2 workspace (colcon). Persists changes made inside ROS 2
  container.
- `ros/config/` -> ROS middleware config mounted into Isaac Sim and ROS 2 containers.
- `ros/docker/` -> ROS 2 Docker build/init area.
- `ros/scripts/` -> ROS 2 host-side helper scripts.
- `px4/docker/` -> PX4 Docker build/init area.
- `px4/scripts/` -> PX4 host-side helper scripts.
- `px4/logs/` -> PX4 SITL log persistence.
- `px4/eeprom/` -> PX4 SITL simulated EEPROM mount; cleared by the default PX4
  launch wrapper.
- `docs/` -> Design and usage docs.
- `benchmarks/` -> Committed benchmark summaries and comparison policy; raw
  benchmark artifacts stay under `.build/benchmarks/`.

## Python API organization
- Public top-level functions, classes, and intentional constants in Python
  modules must be listed in module-level `__all__`.
- `robotics_experiments.config` has a single public boundary at the package
  root. Code outside `src/robotics_experiments/config` must import config
  dataclasses and loader functions from `robotics_experiments.config`, not from
  config submodules.
- `load_config(path)` is the cached, explicit, environment-independent config
  boundary. It returns an immutable `ExperimentConfig` whose ordered `vehicles`
  mapping contains complete per-vehicle `spawn`, `bridge`, `px4`, and `routine`
  configs.
  Runtime roots resolve `ROBOTICS_EXPERIMENT_CONFIG_PATH` themselves and pass the
  resolved config downward.
- Keep public API minimal. Public exports are for production consumers outside
  the module or package boundary; tests alone do not justify public API. If a
  top-level symbol is only used by tests or internal implementation, make it
  private. If a whole module is only used internally, prefer an
  underscore-prefixed module name. Keep `__all__` limited to the intended
  production API rather than exporting helpers, scaffolding, constants, or
  seams for test convenience.
- Keep private top-level helper functions before public top-level APIs when
  possible, so declarations read from implementation helpers into public
  entrypoints.
- Do not add builder functions that only forward fields into a dataclass (for
  example a `make_params(vehicle, ...) -> Params` that maps attribute-for-attribute).
  Construct the dataclass directly at the call site; reserve a builder function
  for construction that carries real logic (nontrivial derivation, validation,
  or branching).
- Import standard-library modules, project modules, and ordinary installable
  dependencies eagerly. Lazy imports and `TYPE_CHECKING` dependency imports are
  permitted only for container-only Isaac Sim / Kit / Isaac Lab modules or
  where `SimulationApp` / `AppLauncher` must start before Kit-dependent imports.
  Resolve circular dependencies structurally instead of hiding them behind
  function-scoped or type-only imports. The vendored Pegasus tree and its
  consuming import behavior are outside this policy.
- Use absolute imports for project modules (`from robotics_experiments...`)
  instead of relative imports. The only exception is a narrowly scoped workaround
  for Isaac Sim / Isaac Lab modules that are unavailable outside the Isaac Sim
  Docker environment; keep those workarounds in Isaac-only lazy leaf modules so
  core imports still work on the host.
- Package layering runs `config -> controller -> {alignment, rl} -> workflows`
  with the independent `flight_parameters -> {alignment, rl}` evidence boundary
  for controller alignment and learning, alongside `config -> domain (drone / environment /
  simulation / px4_bridge) -> Isaac extensions`; `import robotics_experiments`
  (plus its subpackages) must succeed with Isaac absent. A domain leaf module that is
  wholly Isaac-only (every function operates on a live USD stage) may keep
  top-level `carb`/`pxr`/`isaacsim` imports rather than scattering function-scope
  imports, but it must (a) carry a module docstring declaring the exception and
  (b) stay off the core import path, shielded by a lazy package `__init__`
  (re-export-only, or `__getattr__` deferral). `drone/spawn.py`,
  `environment/_stage.py`, `environment/_step.py`, and
  `simulation/_isaac_settings.py` follow this pattern;
  `tests/test_core_imports.py` is the regression guard.
- Do not rename, remove, or broaden public exports as part of cleanup unless
  explicitly requested.
- When a requested change intentionally replaces a public API, config shape,
  command, or workflow, do not add redundant compatibility shims or aliases by
  default. Wire the new public API through every production consumer, update
  tests/docs for the new contract, and let obsolete call sites fail loudly
  unless compatibility is explicitly requested.
- `VehicleState` is a frozen per-tick snapshot carrying pose, velocity, angular
  velocity, and sim time only. Do not add a kinematic acceleration field for the
  PX4 HIL path; the Pegasus IMU derives acceleration from velocity deltas
  internally.

## Isaac Sim extension design
- Keep internal logic and Isaac Sim event wiring separate.
- Treat `on_startup`, `on_shutdown`, and Isaac/Kit event callbacks as wiring.
  They should subscribe, unsubscribe, decode events, update direct runtime
  handles such as the current stage, and delegate meaningful behavior.
- Keep event callbacks small. They should route to domain actions such as
  `setup_drone`, `teardown_drone`, or equivalent names.
- Domain action methods should own complete lifecycle behavior for the feature,
  including related prims, bridge hosts, ROS nodes, callbacks, publishers, and
  subscriptions that must start and stop together.
- Prefer larger meaningful helpers over chains of tiny one-use helpers. If a
  helper is used once and only forwards to another helper, merge it into the
  caller.
- Avoid overgeneralized instance members for wiring objects. Use local variables
  for Isaac/Kit context objects unless their lifetime must be retained, and store
  direct feature state in private attributes such as `self._stage` instead of
  broad handles such as `self._usd_context`.
- Use `experiment_runtime` as the extension pattern: a thin Kit shell that wires
  Isaac/Kit events to the host-agnostic `robotics_experiments.runtime.FleetState`
  and owns no flight/scene logic itself. Cross-component runtime composition
  belongs in the orchestrator (`src/robotics_experiments/runtime/`), a layer
  above the extensions; an extension stays single-responsibility (event wiring).
- `robotics_experiments_core` exposes the shared `src/robotics_experiments`
  package to Isaac Sim extensions. Extension code that needs the core package
  should depend on `robotics_experiments_core` instead of manipulating
  `sys.path`.
- Extension config comes from `robotics_experiments.config`, which composes the
  Hydra config lazily and caches it for the process.
- Feature extensions should depend on `robotics_experiments_core` and use a
  small local `config.py` adapter when they need feature-specific config access.
- RL invariants: read `docs/RL_WORKFLOW.md` before touching RL code; keep Isaac
  Lab task/env package imports lazy; keep the in-Lab actuation seam pure and
  aligned with its numpy reference and torch twin; use the pinned
  `isaaclab_contrib.actuators.Thruster` for in-Lab motor rise/fall and slew
  dynamics while the project retains the pusher-equipped articulation, rotor
  allocation, yaw reaction, drag, and wrench sink; keep RL tunables in the
  Hydra `rl` group; keep task IDs, policy artifacts, and RL gates aligned with
  that doc and the command/routine contract. `robotics_experiments.rl` is the public
  boundary for task-neutral actuation, hover observation, artifact, and
  normalized-action-to-OFFBOARD concepts; `robotics_experiments.rl.press`
  exports only press observation, handoff, task, provider, and policy-progress
  concepts. Runtime policy interface dispatch and adapter composition belong in
  the private `runtime/_policy.py`, not the fleet orchestrator or commanding.
  `rl.action` owns only normalized action interpretation and scales. Those
  scales remain task-owned and need not equal the alignment probe experiment's
  normalized command scales.
  Controller/actuator internals come from
  `robotics_experiments.controller.ControllerSettings`. Training must load and
  validate a quality-passing schema-1 controller artifact before building Isaac
  Lab, embed its complete
  settings, certified control modes, digest, and fingerprints before PPO, and selection/evaluation must reuse
  that embedded record. Policy artifacts without alignment provenance are
  unsupported. Controller and policy artifacts are strict v1 contracts; older
  schemas and compatibility migrations are unsupported. A training invocation loads a
  `CurriculumConfig` from `configs/train/`, even for one stage. Complete stage
  experiments live in `configs/train/stage/`. The root owns `policy_name`, optional
  `initial_learner` (a producer training report), and a nonempty ordered list of
  `{name, config_path}` stages. Labels never choose task behavior or compatibility.
  Stages own all experiment, task, reward, reset, PPO, phase and acceptance settings;
  they have no initializer or namespace. Every attempt starts a distinct Isaac
  process with fresh environments, rollout buffers and runner. Compatible selected
  learners from failed or interrupted runs may initialize any curriculum.
  `rl/training/checkpoint.py` alone saves, validates and restores the strict complete
  learner: actor, critic, learned exploration, both observation normalizers, optimizer
  moments/steps, branch progress and RNG evidence. New attempts reseed from configured
  seeds; RNG snapshots are not applied over them. No persistent return normalizer or
  separate scheduler exists. Phase configuration overrides LR/schedule/entropy and
  normalizer updates after restoration. Learner update ancestry may rewind; completed
  curriculum work never rewinds and starts at zero for each invocation.
  `workflows/isaac_lab/training.py` owns attempt campaigns, validation and fresh tests;
  `curriculum.py` owns sequencing and publication, without a training environment.
  Results retain selected diagnostic candidates with `accepted: false`. Accepted
  stages publish before their exact pinned learner advances. Stop on rejection,
  publication error, runtime error or interruption; no curriculum retries, branches,
  checkpoint merging or process resume. Training/attempt reports are strict schema 29;
  curriculum summaries are schema 1. Each level owns its terminal report. Publication
  never rewrites completed training reports. All stages share one policy namespace and
  curriculum Trackio database, with distinct stage/attempt identities. Supplemental
  policy metadata records the accepted prefix and exact historical actor/report
  digests, not retesting of a later actor on earlier tasks. `rl.press.task` owns
  hover/approach/contact/press behavior, `rl.press.reset.mix` owns resets, `rl.episode_length_s`
  owns duration, `rl.press.goal.stable_steps` owns the hold, and `rl.training` owns
  phases and plateau rules. Contact tasks read the real button joint at policy
  cadence, classify filtered pad/button contact, and require one toggle followed
  by release and the complete stable retreat hold. Contact limits 80% full
  success, 85% stable first contacts and 95% retreat among toggled episodes are
  provisional Lab milestones; tighten them after the full curriculum is
  trainable. PX4 contact deployment remains unsupported. Later phases restore the attempt's complete selected
  learner, apply their schedule, and reset environments before new rollouts.
  Every checkpoint uses one fixed validation workload. The evaluator returns raw
  named metrics; each experiment owns its Hydra plateau rule, which reads one named
  scalar, and validation gates establish
  selection eligibility only. A completed validation-passing attempt freezes
  its exact exported actor and runs one fresh, uncached, disjoint test workload.
  Acceptance reads only that test `PolicyEvaluation`; failure ends the campaign
  without retry or automatic publication. `best` advances only after acceptance
  and always points to exact tested actor bytes. `latest` is the active
  operator-selected deployment label: an accepted campaign advances it
  automatically, while an explicit operator deployment may point it at any
  strict policy artifact. Such an override does not create test evidence,
  acceptance, or a `best` candidate, and its failed or unavailable qualification
  evidence must remain visible.
- Alignment invariants: read `docs/SIM_TO_SIM_ALIGNMENT.md` before touching
  `src/robotics_experiments/controller/`,
  `src/robotics_experiments/alignment/`, controller definitions, alignment Make
  targets, or training alignment provenance. `ControllerSettings`
  fields are added to a fit only through declarative `ParameterSpec` candidates.
  Search and replacement stay generic. Selection metrics are stable registered
  names; changing a metric/schema, definition, probe suite, environment, image,
  or controller interface starts a new comparison fingerprint. Structurally
  valid poor-quality results remain diagnostic evidence but cannot finalize,
  publish, or enter policy training. Finalization also requires passing actuator
  audit and per-profile controller-boundary replay. `latest` and `best` advance
  only after those acceptance checks; `best` compares only within one
  fingerprint.
- Flight-parameter invariants: read `docs/FLIGHT_PARAMETERS.md` before touching
  `src/robotics_experiments/flight_parameters/`, flight configs, hover or
  controller-response capture, or evidence consumers. Schema 1 is the only
  schema and supports repeated hover plus lossless controller-response evidence.
  Flight parameters are task-neutral and must not acquire switch, wall, reward,
  policy, or consumer-specific fields. Complete bundles use immutable run directories,
  an exact manifest, and an atomically advanced relative `latest` symlink.
  PX4 hover evidence must be decoded from the completed ULog inside
  `flight_parameters` and prove the mode, control, airborne, and held-goal
  states for the selected interval. A routine label, elapsed time, or low
  velocity is not proof of hover, and this evidence-only decoding does not
  broaden the shared commanding GCS port.
  Hover traces retain the full alignment window and separate final-24-second
  policy statistics (maximum, RMSE, p95, and p99). Alignment preparation must
  validate complete exact-cadence fit/validation/stability response traces and
  stage them losslessly without changing alignment logic. Alignment and RL must
  match the exact flight-parameters fingerprint; a consumer must not resolve a
  newer artifact implicitly after recording provenance.
- When launching an RL training run for the user, provide a run-scoped
  Trackio database and report the local URL and monitoring directory. Always
  provide Trackio or similar monitoring links as plain URLs, not labeled
  Markdown links or inline code. Keep
  monitoring available while the run is active unless the user asks otherwise.
  Use zero smoothing and show complete metric names so raw validation,
  acceptance, and training values remain visible; `training_report.json` is the
  exact validation-value audit source.
- When operating an RL training run for the user, actively inspect validation
  and acceptance progress while it runs. Do not merely wait for the configured
  budget to expire. Interrupt a campaign once its validation checkpoints and
  independent attempts show no credible path toward acceptance, preserve its
  artifacts, report the evidence, and diagnose it before launching a fresh
  campaign.
- PX4 bridge invariants: read
  `isaac/runtime/extensions/px4_sitl_brigde/docs/PX4_BRIDGE_CONTRACTS.md` before
  touching bridge code. Keep the interface contract, frame conversions, motor
  ordering, lockstep sequencing, sensor-cadence/freshness guards, and visual
  rotor-spin non-physics seam aligned with that doc. `apply_dynamics` remains
  the sole physics sink.
- Fleet invariants: read `docs/FLEET_CONFIG.md` before touching fleet config or
  runtime behavior. `vehicles:` is an ordered mapping and the only spawn path;
  identity, ports, frame
  prefixes, log paths, and airframes are resolved once from config; multi-vehicle
  lockstep runs all sends before all polls while isolating each vehicle.

## Commanding (high-level command / routine layer)
- Read `docs/COMMAND_ROUTINE_CONTRACT.md` before touching
  `src/robotics_experiments/commanding/`, routine config, GCS MAVLink commands,
  `{kind: goto}` / `{kind: policy}` behavior, or driver orchestration.
- `commanding` mirrors the PX4 bridge as a pure contract
  (`RoutineObservation` -> `decide(...)` -> `RoutineDecision` over
  the PX4-specific `MavlinkCommandPort`) and must stay import-clean with
  Isaac absent. Its concrete port eagerly owns the ordinary `pymavlink`
  dependency.
- `decide` and completion/invariant predicates are pure. The port is the only
  MAVLink side-effect seam, and `gcs_port` is control-only: it must not touch the
  bridge simulator port.
- `DriverContext` plus `RoutineDriverState` and `tick_routine_driver` is the only
  executor composition, one per bridge-enabled drone. The frozen context owns
  resources and immutable parameters; the frozen state owns progress only.
  Empty `routine.steps` follows the same driver lifecycle and completes without
  commands. Driver failures enter explicit landing recovery before terminal
  `FAILED`; terminal success is distinct. `on_failure` is notification-only so
  shells cannot interrupt recovery, and strict shells raise the preserved
  failure only after recovery terminates.
- In `runtime`, `commanding`, and `px4_bridge`, project-owned evolving state is
  represented by frozen dataclasses and replaced by stateless transition
  functions. Mutable classes are limited to external resource adapters (MAVLink,
  Isaac, Kit, and PhysX); callers read them through immutable values returned by
  public status methods, or a module free function for a project-owned host. A
  full adapter snapshot uses one `read_status()`-style method;
  `Px4MavlinkPort.is_connected()` is the deliberate narrow hot-path probe that
  avoids constructing the full bridge snapshot every physics tick. Do not expose
  live state-reading properties.

## Pegasus vendoring policy
- The Pegasus Simulator under
  `isaac/runtime/extensions/px4_sitl_brigde/pegasus/` is a vendored, stripped,
  unpinned source snapshot (not a PyPI dependency), reduced to the modules the
  PX4 bridge transitively imports and excluded from ruff/mypy via
  `pyproject.toml`. The project's deviations from upstream are load-bearing
  fixes, each with a guarding unit test -- never "sync them away."
- `isaac/runtime/extensions/px4_sitl_brigde/pegasus/PROVENANCE.md` is the single
  source of truth: upstream repo/license and synced commit, the reachable module
  set, the dead-at-runtime modules, the full divergence index (with guarding
  tests), and the re-sync procedure. Read it before touching the vendored tree or
  re-syncing, and update it -- not just code comments -- when the snapshot or a
  divergence changes.

## Development workflow
- Use `uv` for all host-side Python work. Do not call `pip`,
  `python -m venv`, or a system/global Python directly.
- Use simple, concrete language in code, comments, documentation, logs, and
  commit messages. Avoid vague jargon such as `seam`, `preflight`, `gate`, or
  `robust` when a direct description is available. When touching existing text,
  make it easier to understand.
- Run `uv run pre-commit run --all-files` after implementing a task and fix
  every reported issue before considering the task complete.
- Development workflow details live in `docs/DEVELOPMENT_WORKFLOW.md`.

## Runtime and integration entrypoints
- Run ROS 2, Isaac Sim, and PX4 through the Docker Make targets, not directly on
  the host. Runtime command details live in `docs/RUNTIME_OPERATIONS.md`.
- Key persistent targets: `make build`, `make run-isaac`, `make run-ros`,
  `make run-px4`, `make stop`, and `make destroy`.
- General integration targets and the aggregate suite are defined in
  `make/integration-tests.mk`; controller-alignment targets are defined in
  `make/alignment.mk`; RL, Isaac Lab, and press workflow targets are defined in
  `make/rl.mk`. Read
  `docs/INTEGRATION_TEST_ARCHITECTURE.md` before changing the harness,
  standalone integration scripts, or targets such as `make test-integration`,
  `make test-integration-all`, `make test-integration-press-hover-goto`,
  `make test-integration-alignment-target`,
  `make test-integration-press-policy-transfer`,
  `make test-integration-press-policy`, `make test-integration-px4-takeoff`,
  `make test-integration-px4-navigation`,
  `make test-integration-px4-takeoff-instability`, or
  `make test-integration-px4-multi-drone-takeoff`.

## Hard rules
- Do NOT commit new binary files. The three pre-existing LightSwitchWarehouse
  USD crates listed in `.binary-asset-allowlist` are grandfathered source
  assets; do not modify/regenerate them or expand the allowlist unless a task
  explicitly requests an asset-policy change. See `docs/BINARY_ASSETS.md`.
- Do not mix positional and keyword-only parameters in a Python function
  signature. Excluding an implicit `self` or `cls` receiver, choose one calling
  convention for all parameters: either make every parameter keyword-only so
  every argument is specified by name, or make every parameter positional-only
  so no argument can be specified by name.
- Do NOT regenerate assets unless explicitly asked.
- Do NOT hardcode absolute paths.
- Input artifacts are immutable, producer-owned evidence. Consumers must
  resolve and pin exact validated paths, pin content digests when work crosses
  process boundaries, and reference the producer files directly. Operator
  configuration may select `latest` or `best`, but consumers must immediately
  resolve that selector and persist the producer's canonical `run_id` plus its
  digest or fingerprint as lineage; selector names are never lineage. When an
  existing consumer filename is required, use a relative symlink to the
  resolved immutable run; never copy, rewrite, migrate, or otherwise mutate
  input artifact bytes into a consumer run.
- A refactor that adds no new functionality must reduce implementation bloat or,
  at minimum, leave it unchanged. Prefer deleting or merging redundant code,
  abstractions, tests, and documentation while preserving behavior and coverage;
  do not add wrappers, compatibility layers, helpers, or parallel models that
  increase the maintenance surface. Judge bloat by maintenance cost, not line
  count alone.
- Tunable thresholds, timeouts, rates, and similar runtime parameters belong in
  the Hydra `configs/` YAML as the single source of truth, not hardcoded in code.
  Add a matching field (a bare annotation with no default) to the relevant
  `*Config` dataclass in the owning private module under
  `src/robotics_experiments/config/` and plumb it through. A library constructor
  may keep a neutral fallback default, but it must
  not duplicate the project value.
- Prefer modifying existing files over adding new entrypoints.
- Avoid using Unicode symbols.
- Invalid configuration or unexpected runtime state must not fail silently.
  Raise an error for wrong configuration; log a warning for recoverable runtime
  fallbacks.
- Avoid overly defensive code. Use the minimal checks needed for correctness,
  and otherwise let code fail naturally. Add explicit checks only when the code
  would otherwise fail silently, continue with an incorrect result without
  notice, or when a genuinely distinct state must be represented. Do not wrap a
  value in `Optional`/`None`, add routine `is not None` checks, or carry a
  `<= 0 disables` special case when a plain concrete default works. Prefer a
  domain-appropriate concrete sentinel such as `NaN` for floating-point values
  (and the equivalent concrete sentinel for other applicable types) over
  `None`; use `None` only when absence is genuinely necessary and no concrete
  value represents the state correctly.
- Every behavior change must be reflected in `AGENTS.md` or the relevant
  referenced doc when it changes project conventions, error handling policy,
  runtime assumptions, public behavior, or developer workflow expectations.
- Per-area README and doc files are on-demand context: read a task's relevant
  doc only when the task actually touches that area, not preemptively. If a doc
  has diverged from the actual code, correct the doc as part of the same task.
- Do NOT ignore pre-existing issues. When a quality gate (`pre-commit`, lint,
  type check, test) reports a failure -- even in a file the current task did not
  touch, or one that predates it -- fix it along with the current task rather
  than dismissing it as out of scope or a false positive. If a flagged line is
  genuinely correct, silence it narrowly and on purpose.
- Do NOT add new lint/type ignores to `pyproject.toml` to get past quality
  gates. If a suppression is genuinely necessary, use the narrowest inline
  ignore at the offending line with a brief reason.
- Do not make clear, correct code more indirect or verbose solely to appease a
  static type checker. When an elegant expression is valid at runtime but the
  checker cannot represent its behavior accurately, keep the expression and
  use the narrowest per-case inline suppression with a brief reason. Do not
  broaden this into a global ignore unless explicitly requested.
- For intentionally aligned Python data tables, use Ruff/Black formatter pragmas
  (`# fmt: off` / `# fmt: on`) around the smallest useful block so pre-commit
  does not collapse manual column spacing.
- Standalone Isaac scripts must construct `SimulationApp` or Isaac Lab's
  `AppLauncher` before any Kit-dependent imports. The full import-order contract
  lives in `isaac/runtime/standalone/README.md`.

## Docker assumptions
- Containers often run with `--network=host`.
- GPU support is required (NVIDIA runtime).
- Headless must work by default.
- Design GPU-backed workloads with independent work units for parallel
  execution from the outset. Size and batch them to aim for 95% measured GPU
  utilization, and verify utilization during representative execution. When a
  dependency or another measured bottleneck prevents that target, preserve
  deterministic behavior and report the limiting factor instead of adding
  semantically meaningless work solely to inflate utilization.

## When uncertain
Don't do anything, ask for instructions.
