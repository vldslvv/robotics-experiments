# Development workflow

This is the reference for host-side Python, tests, quality gates, and Makefile
maintenance. `AGENTS.md` keeps only the hard rules and points here.

## Python environment

- This project uses `uv` for all host-side Python work. Do not call `pip`,
  `python -m venv`, or a system/global Python directly.
- The project virtualenv lives at `.venv` and is pinned to the Python in
  `.python-version` (3.11). Dependencies and the resolver index (including the
  NVIDIA index for Isaac Sim) are defined in `pyproject.toml`; exact versions
  are locked in `uv.lock`. Do not edit `uv.lock` by hand.
- This repository is a checkout-only application, not an installable wheel.
  `[tool.uv] package = false` encodes that boundary, and the Make entrypoints
  expose the checkout's `src/` through `PYTHONPATH`. Configuration and assets
  remain repository resources resolved from the checkout. Run Python commands
  from the repository root through the documented Make or `uv` entrypoints.
- The project uses a repo-local uv-managed Python install under `.uv-python/`
  so sandboxed tools do not need access to user-level Python installs under
  `/home/vv/.local` or `/home/vv/.pyenv`. `.uv-python/` is local machine state
  and must not be committed.
- Initialize or repair the core development environment with
  `uv sync --extra dev`. The default dependency set contains the ordinary
  dependencies reached by public package roots, including torch, pymavlink,
  Gymnasium, and rsl_rl. The `dev` extra adds pytest and quality tools, while
  the `simulation` extra adds Isaac Sim and USD for host-side asset/simulator
  work. Actual training runs in the Isaac Lab image, which also ships rsl_rl.
  Flight-analysis tools deliberately use the documented
  throwaway `uv --no-project --with pyulog --with numpy` environment instead of
  a project extra.
- If `.venv` is missing, points outside the repository, or cannot be read from
  the sandbox, recreate it with the repo-local interpreter:
  `UV_PYTHON_INSTALL_DIR=.uv-python uv python install 3.11`, then
  `UV_PYTHON_INSTALL_DIR=.uv-python uv sync --extra dev --managed-python`.
  After that, normal `uv run ...` and `uv sync ...` commands reuse `.venv`.
- Always work inside the project venv. Either activate it with
  `source .venv/bin/activate`, or prefix commands with `uv run` (for example
  `uv run pytest`, `uv run ruff check`). Prefer `uv run` for one-off commands so
  the correct interpreter is used without relying on shell state.
- Add or change dependencies with `uv add` / `uv remove` (or by editing
  `pyproject.toml` and running `uv lock`), never with ad hoc `pip install`. Keep
  `uv.lock` committed and in sync; `uv lock --check` must pass.
- Host-side Isaac Sim, ROS 2, and PX4 work that does not run inside their Docker
  containers still goes through this venv. Sync or invoke with
  `--extra simulation` when host-side code imports Isaac or USD. Throwaway
  `uv run --no-project ...` invocations are deliberate exceptions for one-off
  tooling that must not touch the project environment.

## Unit tests

- `make test` runs the host Python unit test suite (pytest, defined in
  `make/tests.mk`). It requires the project venv at `.venv`; create it first
  with `uv sync --extra dev` if missing. The target activates `.venv` and runs
  `python -m pytest`; the direct equivalent is `uv run pytest`. Unit tests live
  under `tests/` and do not need Docker, Isaac Sim, or PX4 running.
- Unit tests mirror the structure of the logic they cover. Package tests live
  under `tests/src/robotics_experiments/...`; tests for repo-root subsystems
  such as Isaac runtime helpers, PX4 scripts, and Make launch wiring live under
  matching `tests/isaac/...`, `tests/px4/...`, and `tests/make/...` paths.
- Shared unit-test fakes/helpers live in `tests/support/`. Define each fake
  once there rather than inline per test file, keep subsystem fakes in their
  named modules (`commanding_fakes`, `bridge_fakes`), and reserve `fakes` for
  genuinely cross-cutting data helpers. Use
  `tests.support.paths.REPO_ROOT` for repository-root path lookups instead of
  deriving parent depth from a test file location.
- Tests assert on public behavior, not internals: prefer public properties,
  return values, raised errors, and recorded contract seams (constructor kwargs a
  collaborator received, commands sent to a port, forces/dynamics applied to an
  articulation IO) over reading a unit's private state or asserting on a
  mock/interaction-log when a public observable exists.
- External private-member access is lint-enforced off via Ruff `SLF001` (in
  `[tool.ruff.lint] select`). The existing `per-file-ignores` in
  `pyproject.toml` are grandfathered exceptions for production seams against
  vendored Pegasus / Isaac Lab privates, fixture injection into the vendored
  backend, focused private-helper unit tests over import-gated modules, and
  deliberate corruption/fault injection.
- Do not add new lint/type ignores to `pyproject.toml`. If new external private
  access or another lint exception is genuinely unavoidable, use a narrow inline
  suppression at the offending line, such as `# noqa: SLF001`, with a brief
  reason.

## Quality gates

- `make check` is the canonical non-mutating host gate. It validates `uv.lock`,
  checks lint and formatting, runs mypy over every project-owned Python file
  except vendored Pegasus, runs vulture, and executes the complete host pytest
  suite. Every development tool is resolved with the `dev` extra so a fresh
  environment uses the project-pinned version instead of a similarly named
  transitive dependency. The CPU-only pull-request workflow runs this same
  target.
- Pre-commit remains the formatting and commit-time convenience layer. Install
  it once with `uv run pre-commit install` and run a whole-repository sweep with
  `uv run pre-commit run --all-files`. Mirror hooks may modify staged Python
  files, while `make check` never modifies the working tree.
- The shared mypy configuration enables `warn_unreachable` in pre-commit and
  `make check`. Vulture also checks unused code. Verify reachability warnings
  before deleting code: type narrowing and assignments inside closures can
  produce false positives. Keep necessary suppressions local and explained.
  An analyzer's nonzero exit status means its check did not pass; inspect its
  diagnostics separately from dependency or command-launch errors reported by `uv`.
- The hook list and the per-tool tunables (`.hadolint.yaml`, `.codespellrc`,
  `.yamllint.yaml`, `.taplo.toml`, `.binary-asset-allowlist`) live in
  `.pre-commit-config.yaml` and those dotfiles at the repo root. The binary
  asset hook rejects new binary content outside the documented grandfathered
  USD crates in `docs/BINARY_ASSETS.md`.
- For large-scope changes, especially public API, module-name, or runtime
  behavior changes, run `uv run pytest`, `make test-integration-all`, and
  `uv run pre-commit run --all-files` before completion.
- GPU/container integration is intentionally separate from the pull-request
  host workflow. Run it locally with `make test-integration-all` or on a
  scheduled, manually triggered, or self-hosted runner with an NVIDIA runtime.
- When a quality gate reports a failure, fix it along with the current task. If a
  flagged line is genuinely correct, silence it narrowly and on purpose with a
  scoped inline suppression such as `# shellcheck disable=...` or
  `# noqa: <code>` plus a brief reason.
- For intentionally aligned Python data tables, use Ruff/Black formatter pragmas
  (`# fmt: off` / `# fmt: on`) around the smallest useful block so pre-commit
  does not collapse manual column spacing.

## Makefile maintenance

- Keep Make recipes thin. If Makefile logic grows into a complex Bash script
  with substantial control flow, helpers, traps, or heavy `$$` escaping, extract
  it into a separate shell script and let Make prepare variables and call that
  script.
- Avoid `define`/`eval` target templates when they require nested Make/shell
  escaping such as `$$($(1)_VAR)`; prefer explicit targets with ordinary
  recursive Make variables so command-line overrides stay dynamic.

Asset preparation uses `make generate-assets HYDRA_CONFIG=<experiment-or-curriculum>`.
See [asset generation](ASSET_GENERATION.md) for switch profiles,
immutable outputs, locking and validation.
