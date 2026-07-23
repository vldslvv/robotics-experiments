from __future__ import annotations

import tomllib

from tests.support.paths import REPO_ROOT


def test_dependency_profiles_and_checkout_only_boundary() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())

    assert "build-system" not in project
    assert "setuptools" not in project["tool"]
    assert project["tool"]["uv"]["package"] is False
    assert project["project"]["dependencies"] == [
        "gymnasium==1.2.1",
        "numpy==1.26.0",
        "pymavlink==2.4.49",
        "rsl-rl-lib>=3.1.2,<4",
        "scipy>=1.11,<2",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "torch==2.7.0",
    ]

    extras = project["project"]["optional-dependencies"]
    assert "pytest" in extras["dev"]
    assert extras["simulation"] == [
        "isaacsim==5.1.0.0",
        "isaacsim-core==5.1.0.0",
        "usd-core>=25.11,<27",
    ]
    assert "training" not in extras
    assert all(
        "pyulog" not in requirement
        for requirements in extras.values()
        for requirement in requirements
    )
