from __future__ import annotations

import tomllib

from tests.support.paths import REPO_ROOT


def test_dependency_profiles_and_checkout_only_boundary() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert "build-system" not in project
    assert "setuptools" not in project["tool"]
    assert project["tool"]["uv"]["package"] is False
    assert project["tool"]["uv"]["prerelease"] == "explicit"
    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert project["tool"]["ruff"]["target-version"] == "py312"
    assert project["tool"]["mypy"]["python_version"] == "3.12"
    assert project["tool"]["uv"]["constraint-dependencies"] == ["tinyobjloader==2.0.0rc13"]
    assert (REPO_ROOT / ".python-version").read_text().strip() == "3.12"
    assert project["project"]["dependencies"] == [
        "numpy==2.3.1",
        "scipy>=1.11,<2",
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
    ]

    extras = project["project"]["optional-dependencies"]
    assert "pytest" in extras["dev"]
    assert extras["simulation"] == [
        "isaacsim==6.0.1.0",
        "isaacsim-core==6.0.1.0",
    ]
    assert extras["training"] == ["rsl-rl-lib==5.0.1"]
    assert project["dependency-groups"]["usd-build"] == ["usd-core>=25.11,<27"]
    assert all(
        "pyulog" not in requirement
        for requirements in extras.values()
        for requirement in requirements
    )
