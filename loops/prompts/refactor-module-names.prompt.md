You are performing one isolated Codex run.

Target package:

`{{MODULE_PATH}}`

The path is relative to the project root.

## Goal

Rework the public contract and internal module naming of the package `{{MODULE_PATH}}`.

Treat this package as an architectural module with a single explicit public boundary. Code outside the package must depend only on symbols exported by the package root `__init__.py`.

The package boundary should be clear, enforceable, and coherent. Consumers should depend on the package contract, not on the package's internal module layout.

## Core public API rule

Outside `{{MODULE_PATH}}`, imports from this package must go through the package root.

Allowed external import style:

    from package_name import PublicSymbol

Disallowed external import style:

    from package_name.some_submodule import SomeSymbol
    from package_name._some_submodule import SomeSymbol
    import package_name.some_submodule
    import package_name._some_submodule

Internal imports inside `{{MODULE_PATH}}` may continue to use internal submodules where appropriate.

## Module naming rule

Use module names to communicate whether the module contributes to the package's public contract.

1. If an internal module defines or hosts symbols that are exported as part of the package's public contract, the module itself should not have a leading underscore.

   Good:

       package/
         __init__.py
         config.py          # exports Config through package root
         controller.py      # exports Controller through package root

   Avoid:

       package/
         __init__.py
         _config.py         # but Config is public
         _controller.py     # but Controller is public

2. If a module contains only private implementation details, helpers, adapters, internal constants, private classes, or private functions, its filename may and usually should have a leading underscore.

   Good:

       package/
         _serialization.py
         _validation.py
         _adapters.py
         _helpers.py

3. Do not mechanically underscore every module that is not imported outside the package. A module can be internal to the package but still have a non-underscored name when it contains public concepts exported through the package root.

4. Use leading underscores for private-only modules and private-only symbols when doing so improves boundary clarity.

5. Avoid unnecessary churn. Rename modules only when the rename makes the public/private boundary clearer.

6. Use double-underscore name mangling only for rare class-internal collision avoidance. Do not use it as a general privacy mechanism.

## Public contract rules

1. Treat `{{MODULE_PATH}}/__init__.py` as the only supported public import surface for code outside the package.
2. Design the package root exports intentionally.
3. Public contract changes are allowed when they improve modularity, clarity, or maintainability.
4. Do not blindly re-export every symbol that is currently imported from outside the package.
5. Classify each external dependency on package internals:
   - if the dependency is legitimate, expose an intentional symbol through the package root;
   - if the dependency is an implementation leak, refactor the consuming code so it no longer depends on that internal symbol;
   - if the dependency is obsolete, remove it.
6. Keep the public API as small as practical, but not smaller than the real architectural needs of the project.
7. Export stable, meaningful concepts rather than low-level helpers or implementation details.
8. Keep helpers, implementation details, intermediate types, internal constants, and private utilities internal unless they are genuinely part of the package contract.
9. Add or update `__all__` in the package root when it clarifies and enforces the public contract.
10. Remove stale exports that are not used outside the package and are not part of the intended contract.
11. Prefer a coherent final design over a minimal diff.

## Scope

Work primarily on the target package and its consumers:

`{{MODULE_PATH}}`

You may update:

1. Import sites elsewhere in the project that consume this package.
2. Tests that import from this package.
3. Package-level `__init__.py` files needed to define the public contract.
4. Internal module names and symbol names when renaming improves boundary clarity.
5. Documentation, comments, type hints, fixtures, and tests affected by the new contract.
6. Related issues discovered during the refactor when fixing them improves correctness, maintainability, consistency, or test reliability.

Broader scope is acceptable when it supports the package-boundary refactor. Keep the result coherent.

## Process

1. Identify the package root and its `__init__.py`.
2. Identify all imports from this package outside `{{MODULE_PATH}}`.
3. Identify which external imports currently bypass the package root.
4. Classify each bypass as:
   - legitimate public API need;
   - implementation leak;
   - obsolete or unnecessary dependency.
5. Design the intended package root public contract.
6. For legitimate public API needs:
   - expose an intentional symbol from `{{MODULE_PATH}}/__init__.py`;
   - include it in `__all__` if `__all__` is used;
   - ensure the module defining that public symbol is not unnecessarily underscored.
7. For implementation leaks:
   - refactor the consuming code to stop depending on the internal symbol;
   - keep the implementation symbol internal.
8. For obsolete dependencies:
   - remove the dependency;
   - simplify the affected code or tests.
9. Rewrite valid external imports to use the package root.
10. Review module names:
    - modules that define public exported symbols should not be prefixed with `_`;
    - modules containing only private implementation details may be prefixed with `_`.
11. Rename internal modules or symbols when it clarifies the public/private boundary.
12. Fix additional issues discovered during the work when doing so improves the final result.
13. Preserve or intentionally update runtime behavior as appropriate.
14. Keep the final design coherent and maintainable.

## Verification

After the refactor, run:

1. `pre-commit run --all-files`
2. The unit test suite.

Do not run integration tests for this prompt.

Use the repository's existing commands if they are documented in `README`, `AGENTS.md`, `Makefile`, `pyproject.toml`, CI config, or nearby project documentation.

If there is no obvious unit test command, inspect the project configuration and choose the narrowest command that runs unit tests without integration tests.

If any command fails, fix the issue and rerun the relevant command. Do not commit while pre-commit or unit tests are failing.

## Commit

After successful refactoring and successful verification, create exactly one Git commit.

Use this exact commit message format:

`Defined public contract and module naming for the package {{MODULE_PATH}}`

Do not create the commit if verification fails.

## Final response

Summarize:

1. The new package root public contract.
2. Which symbols were added, removed, renamed, or kept as exports.
3. Which external imports were rewritten to use the package root.
4. Which former external dependencies were treated as implementation leaks.
5. Which modules were renamed because they define public exported symbols.
6. Which modules were kept or renamed with leading underscores because they contain only private implementation details.
7. Any additional issues fixed during the work.
8. Which verification commands were run.
9. The commit hash.
