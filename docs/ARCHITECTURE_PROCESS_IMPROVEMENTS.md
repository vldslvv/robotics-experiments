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

Status: completed

The default dependency set now contains only reusable-core requirements:
NumPy, SciPy (required by the vendored Pegasus bridge), Hydra, and OmegaConf.
Pytest and quality tools live in `dev`; Isaac Sim and USD live in `simulation`;
and `rsl-rl-lib` remains isolated in `training`. Pyulog is not a project
dependency or extra: flight analysis continues to use the documented throwaway
`uv --no-project --with` environment.

The repository is explicitly a checkout-only application. It has no build
backend or setuptools package-discovery configuration, and
`[tool.uv] package = false` prevents uv from installing it as an editable
package. Make exports the checkout's `src/` on `PYTHONPATH`, while pytest keeps
its existing source-root configuration. Hydra configs and assets remain
checkout resources, matching the loader and container runtime contracts.

### 4. Make container builds hermetic and split development from execution images

Priority: high

Status: completed

The default Isaac and ROS targets now build minimal runtime/integration stages
from a small shared shell base. Neovim, Node.js, Codex, and Claude Code are
isolated in opt-in `isaac-sim-dev` and `ros-dev` stages with separate Make
targets; runtime images no longer inherit them. Development tool versions,
source checkouts, Isaac Lab/PX4 refs, and the Isaac/ROS/Ubuntu base manifests are
pinned. Image construction no longer reads the host's Neovim version and no
installer is piped directly into a shell. The ROS build excludes the unused
upstream humanoid example's mutable PyTorch/CUDA rosdep closure.

Hydra, OmegaConf, pymavlink, and their exact transitive versions are baked into
the Isaac image from `isaac/docker/runtime-requirements.txt`. Cache warm-up and
integration scenario scripts perform no package installation, eliminating
test-time network access and dependency drift.

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

Status: completed

Config schemas now live in private simulation, scene/ROS, PX4/fleet, RL,
commanding, and environment modules. `config/types.py` is a compatibility
aggregator and `robotics_experiments.config` remains the only public facade.

The commanding routine implementation is divided into private model, stepper,
predicate/gate, and descriptor-builder modules; `commanding/routine.py` keeps
the existing internal compatibility facade and the package-root public exports
are unchanged. Pure airframe dataclasses, Hydra parsing, and body mass-property
calculations live in `robotics_experiments.drone._airframe`, leaving the
standalone script responsible for USD authoring, batch generation, and CLI
wiring. Test support now has commanding and PX4-bridge fake modules, while the
shared fake module contains only quaternion data helpers used by both suites.

### 7. Reduce configuration global-state coupling

Priority: medium

Status: completed

`load_config(path)` is the explicit core operation: it resolves a supplied
experiment path, composes it with an isolated Hydra loader, enriches and
validates the fleet, and never reads the environment or clears `GlobalHydra`.
`get_config()` remains the process-cached production adapter and delegates its
environment-selected path to `load_config`.

Configuration dataclasses stay mutable only during Hydra construction and the
private enrichment phase. Before return, the loader freezes every config node,
converts nested lists to tuples, and converts authored step mappings to read-only
mapping proxies. The cached `EnvConfig`, global defaults, resolved vehicles,
per-vehicle overrides, and nested tunables therefore form one immutable graph.

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
4. [x] Reorganize Python dependency groups and decide the installable versus
   checkout-only packaging contract.
5. [x] Make integration dependency installation hermetic and split minimal container
   stages from interactive development stages.
6. [x] Refactor the large modules incrementally, one subsystem at a time, with no
   public API changes.
7. [x] Isolate config composition and make resolved configuration immutable.
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
