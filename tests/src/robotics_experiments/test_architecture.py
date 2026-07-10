from __future__ import annotations

import ast
from pathlib import Path

from tests.support.paths import REPO_ROOT

_PACKAGE_ROOT = REPO_ROOT / "src" / "robotics_experiments"

# Dependencies point from a consumer to the lower-level packages it may import.
# Same-package imports are always allowed. Host-specific shells outside this
# package may import any public facade; this table protects the reusable package
# from acquiring a dependency back on those shells.
_ALLOWED_DEPENDENCIES = {
    "commanding": frozenset({"px4_bridge"}),
    "config": frozenset(),
    "drone": frozenset({"config"}),
    "environment": frozenset({"config"}),
    "px4_bridge": frozenset({"config"}),
    "rl": frozenset({"commanding", "px4_bridge"}),
    "runtime": frozenset(
        {"commanding", "config", "drone", "environment", "px4_bridge", "rl", "simulation"}
    ),
    "simulation": frozenset({"config"}),
    "workflows": frozenset(
        {
            "commanding",
            "config",
            "drone",
            "environment",
            "px4_bridge",
            "rl",
            "runtime",
            "simulation",
        }
    ),
}


def _current_package(path: Path) -> tuple[str, ...]:
    parts = path.relative_to(_PACKAGE_ROOT).with_suffix("").parts
    return parts[:-1]


def _project_import_targets(node: ast.Import | ast.ImportFrom, path: Path) -> set[str]:
    if isinstance(node, ast.Import):
        return {
            alias.name.split(".", maxsplit=2)[1]
            for alias in node.names
            if alias.name.startswith("robotics_experiments.")
        }

    if node.level:
        package = _current_package(path)
        keep = len(package) - node.level + 1
        base = package[:keep]
        if base:
            return {base[0]}
        if node.module:
            return {node.module.split(".", maxsplit=1)[0]}
        return {alias.name.split(".", maxsplit=1)[0] for alias in node.names}

    if node.module and node.module.startswith("robotics_experiments."):
        return {node.module.split(".", maxsplit=2)[1]}
    return set()


def test_internal_package_imports_follow_dependency_direction() -> None:
    package_names = {
        path.name for path in _PACKAGE_ROOT.iterdir() if (path / "__init__.py").is_file()
    }
    assert package_names == _ALLOWED_DEPENDENCIES.keys()

    violations: list[str] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        if path.parent == _PACKAGE_ROOT:
            continue
        source = path.relative_to(_PACKAGE_ROOT).parts[0]
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for target in _project_import_targets(node, path):
                if target == source:
                    continue
                if target not in _ALLOWED_DEPENDENCIES[source]:
                    relative_path = path.relative_to(REPO_ROOT)
                    violations.append(f"{relative_path}:{node.lineno}: {source} -> {target}")

    assert not violations, "Forbidden internal package imports:\n" + "\n".join(violations)
