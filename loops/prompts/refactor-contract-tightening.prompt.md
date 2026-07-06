You are performing one isolated Codex run.

Target package:

`{{MODULE_PATH}}`

The path is relative to the project root.

## Goal

Rework the public contract of the package `{{MODULE_PATH}}`.

Treat this package as an architectural module with a single explicit public boundary. Code outside the package must depend only on symbols exported by the package root `__init__.py`.

The package boundary should be clear, enforceable, and coherent. Consumers should depend on the package contract, not on the package's internal module layout.

Public contract changes are allowed. You may remove, rename, consolidate, or redesign exported symbols if doing so produces a clearer and more maintainable package contract. Update all affected consumers and tests accordingly.

Broader changes are allowed when they support the package-boundary refactor. You may fix related or unrelated issues discovered during the work if they improve correctness, maintainability, consistency, or test reliability. Keep the final result coherent and verified.

## Core rule

Outside `{{MODULE_PATH}}`, imports from this package must go through the package root.

Allowed external import style:

    from package_name import PublicSymbol

Disallowed external import style:

    from package_name.some_submodule import SomeSymbol
    from package_name.internal_module import SomeSymbol
    from package_name._internal_module import SomeSymbol
    import package_name.some_submodule

Internal imports inside `{{MODULE_PATH}}` may continue to use internal submodules where appropriate.

## Public contract rules

1. Treat `{{MODULE_PATH}}/__init__.py` as the only supported public import surface for code outside the package.
2. Design the package root exports intentionally. Do not simply preserve the current public surface.
3. Public contract changes are acceptable when they improve modularity, clarity, or long-term maintainability.
4. Do not blindly re-export every symbol that is currently imported from outside the package.
5. Classify each external dependency on package internals:
   - if the dependency is legitimate, expose an intentional symbol through the package root;
   - if the dependency is an implementation leak, refactor the consuming code so it no longer depends on that internal symbol.
6. Keep the public API as small as practical, but not smaller than the real architectural needs of the project.
7. Export stable, meaningful concepts rather than low-level helpers or implementation details.
8. Keep helpers, implementation details, intermediate types, internal constants, and private utilities internal unless they are genuinely part of the package contract.
9. Add or update `__all__` in the package root when it clarifies and enforces the public contract.
10. Remove stale exports that are not used outside the package and are not part of the intended contract.
11. Rename or reorganize internal modules when doing so improves the boundary and does not create pointless churn.
12. Prefer a coherent final design over a minimal diff.

## Private naming rules

Use leading underscores to mark implementation details when it improves boundary clarity.

Appropriate uses:

1. Internal modules that should not be imported from outside the package, for example `_resolver.py`, `_adapters.py`, `_serialization.py`.
2. Internal helper functions, classes, and constants, for example `_normalize_config`, `_InternalState`, `_DEFAULT_TIMEOUT`.
3. Object attributes that are implementation details, for example `self._cache`, `self._client`, `self._state`.

Do not mechanically prefix every non-exported symbol with an underscore.

Avoid underscores when:

1. The symbol is internal to the package but is part of a clear internal subsystem and is widely used inside the package.
2. Renaming would create large churn without improving the public boundary.
3. The module name is already clearly internal by location and is not imported outside the package.
4. The symbol is part of a data model, schema, or object state that is intentionally visible to callers.

Use double-underscore name mangling only for rare class-internal collision avoidance. Do not use it as a general privacy mechanism.

## Scope

Work primarily on the target package and its consumers:

`{{MODULE_PATH}}`

You may also update:

1. Import sites elsewhere in the project that consume this package.
2. Tests that import from this package.
3. Package-level `__init__.py` files needed to define the public contract.
4. Internal module names and symbol names when renaming improves the boundary.
5. Documentation, comments, type hints, fixtures, and tests affected by the new contract.
6. Related or unrelated issues discovered during the refactor when fixing them improves correctness, maintainability, consistency, or test reliability.

Broader scope is acceptable. Keep the result coherent, but do not avoid necessary changes merely because they cross package boundaries.

## Process

1. Identify the package root and its `__init__.py`.
2. Identify all imports from this package outside `{{MODULE_PATH}}`.
3. Identify which external imports currently bypass the package root.
4. Classify each bypass as either:
   - legitimate public API need;
   - implementation leak;
   - obsolete or unnecessary dependency.
5. Design the intended package root public contract.
6. For legitimate public API needs:
   - expose an intentional symbol from `{{MODULE_PATH}}/__init__.py`;
   - include it in `__all__` if `__all__` is used.
7. For implementation leaks:
   - refactor the consuming code to stop depending on the internal symbol;
   - keep the symbol internal.
8. For obsolete dependencies:
   - remove the dependency;
   - simplify the affected code or tests.
9. Rewrite valid external imports to use the package root.
10. Rename internal modules or symbols when it clarifies the public/private boundary.
11. Fix additional issues discovered during the work when doing so improves the final result.
12. Preserve or intentionally update runtime behavior as appropriate.
13. Keep the final design coherent and maintainable.

## Verification

After the refactor, run:

1. `pre-commit run --all-files`
2. The full test suite.
3. The integration test suite.

Use the repository's existing commands if they are documented in `README`, `AGENTS.md`, `Makefile`, `pyproject.toml`, CI config, or nearby project documentation.

Do not skip integration tests. If there is a canonical command for full tests that includes integration tests, use that. If full tests and integration tests are separate, run both.

If any command fails, fix the issue and rerun the relevant command. Do not commit while tests or pre-commit are failing.

## Commit

After successful refactoring and successful verification, create exactly one Git commit.

Use this exact commit message format:

`Defined public contract for the package {{MODULE_PATH}}`

Do not create the commit if verification fails.

## Final response

Summarize:

1. The new package root public contract.
2. Which symbols were added, removed, renamed, or kept as exports.
3. Which external imports were rewritten to use the package root.
4. Which former external dependencies were treated as implementation leaks.
5. Which symbols or modules were renamed or marked internal.
6. Any additional issues fixed during the work.
7. Which verification commands were run.
8. The commit hash.
