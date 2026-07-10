# Architecture and Process Improvement Review

## Scope

This review covers repository architecture and engineering processes: package
boundaries, dependency management, configuration, tests, quality gates,
containers, integration workflows, and documentation. It does not evaluate the
project's product or research goal.

The current architecture has a strong foundation:

- Runtime composition is centralized in `FleetOrchestrator` and documented by an
  explicit contract.
- Isaac Sim, ROS 2, PX4, and training dependencies are kept behind lazy package
  boundaries where practical.
- The commanding and bridge layers isolate pure decisions from side effects.
- Hydra is the configuration source of truth, and fleet identities and endpoints
  are resolved centrally.
- Integration runs isolate artifacts and provide detailed timing and flight
  diagnostics.
- ADRs and area-specific documents capture important architectural decisions.

The main opportunities are to enforce these decisions more consistently, make
builds and tests more reproducible, and reduce the cost of continued growth.

## Prioritized improvements

### 1. Fix host-test discovery

Priority: immediate

Status: completed

Pytest remains configured with `testpaths = ["tests"]`, while `norecursedirs`
now contains only build, cache, and virtual-environment paths. The default suite
therefore includes the host-side tests in `tests/isaac` and `tests/px4`, as
required by `docs/DEVELOPMENT_WORKFLOW.md`.

Observed during this review:

- The default suite reported 285 passed and 1 skipped.
- Running `tests/isaac` and `tests/px4` explicitly found another 47 tests, all of
  which passed.

Completed change:

- Removed `isaac`, `ros`, and `px4` from `norecursedirs`.
- Added a collection regression test that discovers the current top-level host
  test areas and verifies that default pytest collection includes each one. The
  guard does not depend on a fixed test count.

### 2. Establish one canonical host gate and run it in CI

Priority: immediate

Status: completed

`make check` is the canonical non-mutating host gate and runs:

- `uv lock --check`
- `uv run ruff check`
- `uv run ruff format --check`
- `uv run mypy src tests`
- `uv run --extra dev vulture --config pyproject.toml`
- `uv run pytest`

The CPU-only GitHub Actions pull-request workflow installs the locked development
environment and invokes the same `make check` target. Pre-commit remains the
formatting and commit-time convenience layer. GPU/container integration stays a
separate local or NVIDIA-runner gate and is not weakened by the host workflow.

This separates fast deterministic feedback from expensive simulator validation
without weakening either gate.

### 3. Separate dependency profiles and define the packaging boundary

Priority: high

The default project dependencies include Isaac Sim, pytest, and pyulog even
though the package architecture deliberately keeps many modules independent of
Isaac. Pytest already belongs to the development dependency set, and
`flight_analysis/README.md` says pyulog should be supplied through a throwaway
`uv --with` environment rather than become a permanent project dependency.

Recommended change:

- Keep only dependencies required by the reusable core in the default set.
- Move simulator/USD dependencies to an Isaac or simulation extra.
- Keep pytest and quality tools in the development set.
- Keep analysis-only dependencies in a dedicated analysis extra or continue to
  use the documented throwaway environment, but not both.
- Keep training dependencies isolated from runtime and host-test dependencies.

The repository should also explicitly choose between two packaging models:

1. Installable package: package Hydra configs and other required resources with
   the wheel and resolve them through package resources.
2. Checkout-only application: document and encode that assumption rather than
   presenting a package that appears relocatable.

At present, config loading derives `configs/` through
`Path(__file__).resolve().parents[3]`, while setuptools does not package that
directory. An installed wheel therefore does not have the same runtime contract
as a source checkout.

### 4. Make container builds hermetic and split development from execution images

Priority: high

The Isaac and ROS image chains inherit interactive editor and AI-tool layers.
Integration tests therefore depend on tools that are unrelated to runtime
execution. Several image inputs are also obtained through mutable branches,
unpinned installers, or `curl | bash`. The Neovim version is derived from the
host when Make is evaluated, further coupling image reproducibility to the
developer machine.

Integration containers also run an unpinned pip install at test time. This adds
network dependence, makes repeated runs slower, and permits two runs of the same
image to execute with different Python dependency versions.

Recommended change:

- Build minimal runtime/integration stages containing only execution
  dependencies.
- Layer optional interactive development stages on top for Neovim, shells, and
  assistant tools.
- Pin source checkouts to immutable commits or release tags and pin important
  base images by digest where operationally practical.
- Remove the dependency on the host's Neovim version from the default image
  build.
- Bake integration Python dependencies into the appropriate image, or install
  them from lock-derived constraints rather than unconstrained package names at
  test time.

### 5. Mechanically enforce package dependency direction

Priority: high

Status: completed

An AST-based architecture test discovers every top-level package under
`robotics_experiments`, requires each package to have an explicit dependency
policy, and rejects imports outside the allowed lower-level edges. The policy
keeps config at the base, pure/domain packages above it, runtime above the
domains, and host workflows at the outer edge. Same-package imports remain
allowed. The separate lightweight-import tests continue to enforce dependency
weight and lazy loading.

The config suite also discovers every top-level YAML sibling of `base.yaml` and
composes each experiment through the public config facade. New experiment files
therefore receive composition coverage automatically, including
`policy_hover_quad.yaml`.

### 6. Split the main growth hotspots by responsibility

Priority: medium

Several files remain cohesive today but are large enough that new features will
raise review cost and conflict frequency:

- `src/robotics_experiments/config/types.py` contains schemas for simulation,
  ROS, switches, PX4, fleet, RL, routines, and drivers.
- `src/robotics_experiments/commanding/routine.py` contains the routine model,
  decision state machine, predicates, envelope gates, geodetic helpers, and a
  long descriptor-to-step builder.
- `isaac/runtime/standalone/build_copter_usd_standalone.py` combines builder
  config parsing, mass-property calculations, USD authoring, and CLI wiring.
- `tests/support/fakes.py` contains fakes for several unrelated subsystems.

Recommended change:

- Split config schemas into private domain-oriented modules while preserving
  `robotics_experiments.config` as the only public facade.
- Separate routine state/decisions, predicates/gates, and descriptor builders
  behind the existing commanding facade.
- Move pure airframe parsing and mass-property calculations away from the
  standalone USD-authoring shell.
- Split test fakes by subsystem, retaining only genuinely cross-cutting helpers
  in the shared module.

These changes should preserve public exports and behavior. The purpose is
smaller ownership units, not a broader API.

### 7. Reduce configuration global-state coupling

Priority: medium

`get_config()` caches a mutable `EnvConfig`, clears process-global Hydra state
before composition, and returns shallowly immutable resolved vehicles whose
nested config objects remain mutable. This is convenient in a single-owner
process but creates avoidable coupling in an extension host where other
components may also use Hydra.

Recommended change:

- Introduce an explicit `load_config(path)` operation that performs composition
  and resolution without depending on an environment variable at its core
  boundary.
- Retain `get_config()` as the process-cached adapter used by production shells.
- Avoid clearing Hydra state owned by unrelated consumers where Hydra's context
  APIs permit isolated composition.
- Return an immutable resolved configuration graph, or clearly constrain
  mutation to a private construction phase before the object is cached.

This also simplifies tests, which currently reset global caches and environment
selection repeatedly.

### 8. Reconcile documentation policy with repository enforcement

Priority: medium

The repository instructions prohibit committing binary files, but the tree
already contains binary USD crates. The current pre-commit configuration checks
file size but does not enforce a binary-file policy.

Recommended change:

- Decide whether the existing USD crates are intentional grandfathered assets.
- If they are, document a narrow allowlist and add a check that rejects new
  binary files outside it.
- If binary assets are not allowed at all, replace or remove the existing
  exceptions so the repository matches its stated policy.

The root README should also include:

- A compact architecture map from config through domains, orchestrator, and
  host shells.
- A test matrix distinguishing host checks, Isaac-only integration, PX4
  integration, and RL gates.
- The canonical contributor command once `make check` becomes complete.

Finally, periodically groom `TODO.txt`. Some entries describe work already
represented in current tests or architecture, while other items are durable
backlog candidates that would be easier to track in `docs/backlog/` or issues.

## Suggested implementation sequence

1. [x] Correct pytest discovery and make the 47 previously omitted tests part
   of the default gate.
2. [x] Expand `make check` and add CPU-only CI.
3. [x] Add dynamic experiment-config coverage and the package dependency-direction
   test.
4. [ ] Reorganize Python dependency groups and decide the installable versus
   checkout-only packaging contract.
5. [ ] Make integration dependency installation hermetic and split minimal container
   stages from interactive development stages.
6. [ ] Refactor the large modules incrementally, one subsystem at a time, with no
   public API changes.
7. [ ] Isolate config composition and make resolved configuration immutable.
8. [ ] Reconcile binary-asset, README, and backlog documentation with the resulting
   workflow.

## Review validation

The following read-only checks were run while preparing this review:

- `uv run pytest`: 285 passed, 1 skipped.
- `uv run pytest tests/isaac tests/px4`: 47 passed.
- `uv run ruff check`: passed.
- `uv run ruff format --check`: passed.
- `uv run mypy src tests`: passed under the current configuration.
- `uv run mypy --check-untyped-defs src`: passed.
- `uv run --extra dev vulture --config pyproject.toml`: passed.
- `uv lock --check`: passed.

Container-backed integration tests were not run as part of this architecture and
process review.
