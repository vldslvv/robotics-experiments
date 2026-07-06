You are performing one isolated Codex run.

Target module:

`{{MODULE_PATH}}`

The path is relative to the project root.

## Goal

Minimize the public API for the module `{{MODULE_PATH}}`.

The module should only export symbols that are actually used by other modules in the project. Everything else should be hidden, made private, removed from module exports, or kept internal as appropriate.

## Scope

Work primarily inside:

`{{MODULE_PATH}}`

You may inspect and update directly related tests and direct import sites elsewhere in the project when needed to preserve behavior.

Do not perform unrelated cleanup. Do not start broad architectural changes outside this module.

## Public API rules

1. Identify what the module currently exports.
2. Check which exported symbols are actually used by other modules in the project.
3. Keep exported only the symbols that are externally used or intentionally part of the module boundary.
4. Hide internal helpers.
5. Prefer removing unnecessary re-exports from `__init__.py`.
6. Prefer explicit `__all__` only when it improves clarity and is consistent with the surrounding codebase.
7. Do not break legitimate external imports.
8. Do not change runtime behavior except where required by the public API cleanup.
9. Keep the diff reviewable.

## Verification

After refactoring, run:

1. `pre-commit run --all-files`
2. The full test suite.
3. The integration test suite.

Use the repository's existing commands if they are documented in `README`, `AGENTS.md`, `Makefile`, `pyproject.toml`, CI config, or nearby project documentation.

Do not skip integration tests. If there is a canonical command for full tests that includes integration tests, use that. If full tests and integration tests are separate, run both.

If any command fails, fix the issue and rerun the relevant command. Do not commit while tests or pre-commit are failing.

## Commit

After successful refactoring and successful verification, create exactly one Git commit.

Use this exact commit message format:

`Refactored public API for the module {{MODULE_PATH}}`

Do not create the commit if verification fails.

## Final response

Summarize:

1. Which public exports were removed or hidden.
2. Which imports or tests were updated.
3. Which verification commands were run.
4. The commit hash.
