from __future__ import annotations

import re

from tests.support.paths import REPO_ROOT


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text()


def test_runtime_images_inherit_development_tools() -> None:
    dockerfile = _read("Dockerfile")
    images = _read("make/images.mk")

    assert "FROM development-tools AS isaac-sim" in dockerfile
    assert "FROM development-tools AS ros-base" in dockerfile
    assert "isaac-sim-dev" not in dockerfile + images
    assert "ros-dev" not in dockerfile + images
    assert "| bash" not in dockerfile


def test_isaac_and_ros_restore_the_interactive_terminal_environment() -> None:
    dockerfile = _read("Dockerfile")
    terminal_stage = dockerfile.split("FROM base AS terminal-tools", maxsplit=1)[1]
    terminal_stage = terminal_stage.split("FROM terminal-tools AS development-tools", maxsplit=1)[0]

    for package in (
        "fd-find",
        "file",
        "htop",
        "ranger",
        "ripgrep",
        "tree",
        "zsh",
        "zsh-syntax-highlighting",
    ):
        assert f"    {package} " in terminal_stage

    assert "ohmyzsh/ohmyzsh/master/tools/install.sh" in terminal_stage
    assert "git clone https://github.com/zsh-users/zsh-autosuggestions" in terminal_stage
    assert "git clone https://github.com/jeffreytse/zsh-vi-mode" in terminal_stage
    assert 'ZSH_THEME="robbyrussell"' in terminal_stage
    assert "  git\n  zsh-autosuggestions\n  zsh-vi-mode" in terminal_stage
    assert 'export ZSH_COMPDUMP="$ZSH_CACHE_DIR/zcompdump"' in terminal_stage
    assert "source /usr/share/zsh-syntax-highlighting/" in terminal_stage
    assert "FROM terminal-tools AS development-tools" in dockerfile


def test_container_inputs_are_pinned() -> None:
    dockerfile = _read("Dockerfile")
    config = _read("make/config.mk")
    images = _read("make/images.mk")

    assert "@${DOCKER_DIGEST} AS base" in dockerfile
    assert re.search(r"FROM ubuntu:22\.04@sha256:[0-9a-f]{64} AS px4", dockerfile)
    assert "NVIM_VERSION := $(shell command -v nvim" in config
    assert "ARG NVIM_TAG" in dockerfile
    assert 'git clone --depth 1 --branch "${NVIM_TAG}"' in dockerfile
    assert images.count('--build-arg NVIM_TAG="$(NVIM_VERSION)"') == 3
    assert "build-isaac: require-host-nvim" in images
    assert "build-isaac-lab: require-host-nvim" in images
    assert "build-ros: require-host-nvim" in images
    for variable in (
        "ISAACLAB_REF",
        "PX4_REF",
        "TOPIC_BASED_ROS2_CONTROL_REF",
        "ISAAC_ROS_WORKSPACES_REF",
    ):
        assert re.search(rf"^{variable}\s+\?=\s+[0-9a-f]{{40}}$", config, re.MULTILINE)
    assert "ISAAC_BASE_TAG   ?= 6.0.1" in config
    assert (
        "ISAAC_BASE_DIGEST ?= "
        "sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb" in config
    )
    assert "ISAACLAB_REF         ?= ffff603eafc6b74264a5261cc0183d6a65390d78" in config


def test_isaac_cache_mounts_are_version_isolated() -> None:
    config = _read("make/config.mk")
    docker_make = _read("make/docker.mk")

    assert "ISAAC_CACHE_ROOT       = $(ISAAC_HOME)/isaac-sim/cache/$(ISAAC_BASE_TAG)" in config
    for cache_name in ("kit", "ov", "pip", "glcache", "computecache", "zsh"):
        assert f"$(ISAAC_CACHE_ROOT)/{cache_name}" in config
    for container_path in (
        "/root/.cache/ov",
        "/root/.cache/pip",
        "/root/.cache/nvidia/GLCache",
        "/root/.nv/ComputeCache",
        "/root/.nvidia-omniverse/logs",
        "/root/.local/share/ov/data",
    ):
        assert docker_make.count(f":{container_path}") == 2
    assert ":/isaac-sim/.cache" not in docker_make
    assert ":/isaac-sim/.nvidia-omniverse" not in docker_make


def test_host_neovim_setup_is_mounted_with_additive_sync() -> None:
    dockerfile = _read("Dockerfile")
    docker_make = _read("make/docker.mk")

    assert dockerfile.count("alias vim='nvim'") == 2
    assert dockerfile.count("alias pyvim=") == 2
    assert "/isaac-sim/make_symlinks.sh || true" in dockerfile
    for host_path, container_path in (
        ("${HOME}/code/lazyvim-starter", "/root/.config/nvim"),
        ("${HOME}/.config/github-copilot", "/root/.config/github-copilot"),
        ("${HOME}/.local/share/nvim", "/root/.local/share/nvim"),
        ("${HOME}/.local/state/nvim", "/root/.local/state/nvim"),
        ("${HOME}/.cache/nvim", "/root/.cache/nvim"),
    ):
        assert host_path in docker_make
        assert container_path in docker_make
    assert "rsync -a --exclude='**/.git/**'" in docker_make
    assert "rsync -a --delete" not in docker_make

    editor_env = _read("isaac/runtime/standalone/set_python_env.sh")
    assert 'source "$SCRIPT_DIR/setup_python_env.sh"' in editor_env
    assert 'PROJECT_PYTHONPATH="${PYTHONPATH:-}"' in editor_env
    assert 'export PYTHONPATH="$PROJECT_PYTHONPATH:$PYTHONPATH:$SYMLINKS"' in editor_env
    assert "LD_LIBRARY_PATH" not in editor_env
    assert "env -u PYTHONPATH" not in editor_env
    assert "python3.11" not in editor_env
    assert "extscache/" not in editor_env


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
    assert "-e source/isaaclab -e source/isaaclab_physx -e source/isaaclab_rl" in dockerfile
    assert "isaacsim_lab_prebundles.pth" in dockerfile
    assert "omni.isaac.ml_archive/pip_prebundle" in dockerfile
    assert "glob('extscache/omni.warp.core-*')" in dockerfile
    assert "expected one Torch prebundle" in dockerfile
    assert "expected one Warp extension" in dockerfile
    assert "omni.warp.core-1.13.0+lx64" not in dockerfile
    assert "pip uninstall --yes wheel" in dockerfile
    assert "find_spec('wheel') is None" in dockerfile
    assert (
        "import flatdict, gymnasium, importlib.util, isaaclab, isaaclab_physx, "
        "isaaclab_rl, rsl_rl, torch, torchvision, warp" in dockerfile
    )
    assert "torch.__version__ == '2.11.0+cu128'" in dockerfile
    assert "torchvision.__version__ == '0.26.0+cu128'" in dockerfile
    assert "warp.__version__ == '1.13.0'" in dockerfile
    assert "flatdict==4.1.0" in requirements
    assert "lazy_loader==0.4" in requirements
    assert "rsl-rl-lib==5.0.1" in requirements
    assert "setuptools==81.0.0" in requirements
    assert "toml==0.10.2" in requirements
    assert "wheel==0.46.3" in requirements


def test_isaac_runtime_can_forward_quaternion_access_detector() -> None:
    docker_makefile = _read("make/docker.mk")

    assert "$(if $(WARN_ON_TORCH_QUATF_ACCESS),-e WARN_ON_TORCH_QUATF_ACCESS=" in docker_makefile
    assert "isaacsim.ros2.core/$(ROS_DISTRO)/lib" in docker_makefile
    assert "isaacsim.ros2.bridge/$(ROS_DISTRO)/lib" not in docker_makefile


def test_persistent_container_recreates_for_image_and_mount_changes() -> None:
    runner = _read("make/scripts/run_persistent_container.sh")

    assert "expected_image_id=" in runner
    assert "existing_image_id=" in runner
    assert 'recreate_reason="updated image"' in runner
    assert "expected_binds" in runner
    assert "existing_binds" in runner
    assert 'recreate_reason="updated volume mounts"' in runner
