from __future__ import annotations

import re

from tests.support.paths import REPO_ROOT


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def test_runtime_images_do_not_inherit_development_tools() -> None:
    dockerfile = _read("Dockerfile")

    assert "FROM base AS isaac-sim" in dockerfile
    assert "FROM isaac-sim AS isaac-sim-dev" in dockerfile
    assert "FROM base AS ros-base" in dockerfile
    assert "FROM ros AS ros-dev" in dockerfile
    assert "FROM ai-tools" not in dockerfile
    assert "| bash" not in dockerfile


def test_container_inputs_are_pinned() -> None:
    dockerfile = _read("Dockerfile")
    config = _read("make/config.mk")
    images = _read("make/images.mk")

    assert "@${DOCKER_DIGEST} AS base" in dockerfile
    assert re.search(r"FROM ubuntu:22\.04@sha256:[0-9a-f]{64} AS px4", dockerfile)
    assert "NVIM_VERSION" not in config + images
    assert "build-isaac-dev" in images
    assert "build-ros-dev" in images
    for variable in (
        "ISAACLAB_REF",
        "PX4_REF",
        "TOPIC_BASED_ROS2_CONTROL_REF",
        "ISAAC_ROS_WORKSPACES_REF",
    ):
        assert re.search(rf"^{variable}\s+\?=\s+[0-9a-f]{{40}}$", config, re.MULTILINE)


def test_integration_dependencies_are_baked_into_image() -> None:
    requirements = set(_read("isaac/docker/runtime-requirements.txt").splitlines())
    assert requirements == {
        "antlr4-python3-runtime==4.9.3",
        "fastcrc==0.3.6",
        "hydra-core==1.3.2",
        "lxml==4.9.3",
        "omegaconf==2.3.0",
        "packaging==23.0",
        "pymavlink==2.4.49",
        "PyYAML==6.0.2",
    }

    runners = "\n".join(
        _read(path)
        for path in (
            "make/scripts/run_isaac_test_in_container.sh",
            "make/scripts/run_isaac_integration_test.sh",
            "make/scripts/run_px4_integration_test.sh",
            "make/scripts/warm_isaac_cache.sh",
        )
    )
    assert "pip install" not in runners
    assert "ISAAC_TEST_PIP_PACKAGES" not in runners

    dockerfile = _read("Dockerfile")
    assert "--skip-keys python3-pytorch-pip" in dockerfile
    assert "PIP_BREAK_SYSTEM_PACKAGES" not in dockerfile
    assert "kconfiglib==14.1.0 empy==3.3.4 pyros-genmsg==0.5.8" in dockerfile


def test_isaac_lab_uses_pinned_minimal_dependencies() -> None:
    dockerfile = _read("Dockerfile")
    requirements = _read("isaac/docker/lab-runtime-requirements.txt").splitlines()

    assert requirements
    assert all("==" in requirement for requirement in requirements)
    assert "TERM=xterm ./isaaclab.sh --install" not in dockerfile
    assert "--no-build-isolation -e source/isaaclab -e source/isaaclab_rl" in dockerfile
