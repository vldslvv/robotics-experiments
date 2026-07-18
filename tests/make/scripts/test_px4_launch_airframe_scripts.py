"""Static checks that the host-side PX4 launch wiring threads the airframe.

The per-vehicle PX4 SITL airframe (``PX4_SIM_MODEL`` / ``PX4_SYS_AUTOSTART``,
resolved onto each enabled ``VehicleConfig`` by ``config.load_config``) must
reach every PX4 container. These assert the persistent and integration launch
scripts propagate both env vars rather than relying on the old hard-coded
stale persistent container env, which silently flew hexacopters as quads.
"""

from tests.support.paths import REPO_ROOT

_ROOT = REPO_ROOT


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text()


def test_persistent_launch_derives_and_propagates_airframe():
    containers = _read("make/scripts/run_px4_persistent_containers.sh")
    helper = _read("make/scripts/px4_airframe_manifest_check.sh")
    # Derives the per-vehicle airframe from the resolved fleet, not a bare index.
    assert "load_config" in containers
    assert "vehicle.px4_sim_model" in containers
    # Threads both airframe env vars into the per-instance launch.
    assert "PX4_SIM_MODEL=" in containers
    assert "PX4_SYS_AUTOSTART=" in containers
    # Validates the PX4 image was rebuilt after airframe geometry/config changes.
    assert "px4_airframe_manifest_check.sh" in containers
    assert "check_px4_airframe_manifest_current" in containers
    assert "generate_px4_airframes.py" in helper
    assert "--print-manifest" in helper
    assert "/usr/local/share/robotics-experiments/px4-airframes.json" in helper
    assert "run make build-px4" in helper
    assert "generate_px4_airframes.py" not in containers

    instance = _read("px4/scripts/run_px4_persistent_instance.sh")
    assert '-e "PX4_SIM_MODEL=${PX4_SIM_MODEL}"' in instance
    assert '-e "PX4_SYS_AUTOSTART=${PX4_SYS_AUTOSTART}"' in instance
    assert '-e "PX4_TOTAL_INSTANCES=${PX4_TOTAL_INSTANCES}"' in instance
    assert "docker image inspect" in instance
    assert "container_env_value PX4_SIM_MODEL" in instance
    assert "container_env_value PX4_SYS_AUTOSTART" in instance
    assert "docker rm -f" in instance

    make_target = _read("make/containers.mk")
    assert "PX4_IMAGE='$(PX4_IMAGE)'" in make_target
    assert "PX4_SIM_MODEL='$(PX4_SIM_MODEL)'" in make_target
    assert "PX4_SYS_AUTOSTART='$(PX4_SYS_AUTOSTART)'" in make_target


def test_integration_launch_derives_and_propagates_airframe():
    integration = _read("make/scripts/run_px4_integration_test.sh")
    assert "load_config" in integration
    assert "vehicle.px4_sim_model" in integration
    assert '-e "PX4_SIM_MODEL=${sim_model}"' in integration
    assert '-e "PX4_SYS_AUTOSTART=${sys_autostart}"' in integration
    assert "px4_airframe_manifest_check.sh" in integration
    assert "check_px4_airframe_manifest_current" in integration
    assert "generate_px4_airframes.py" not in integration


def test_common_launcher_defaults_to_repo_generated_quad():
    common = _read("px4/docker/px4_sitl_common.sh")
    assert 'export PX4_SIM_MODEL="none_iris"' not in common
    # Default to the X500 V2 slot in PX4's custom model range.
    assert 'export PX4_SIM_MODEL="${PX4_SIM_MODEL:-none_x500_v2}"' in common
    assert 'export PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-22000}"' in common
    assert "none_quad_x" not in common
    assert ":-4001}" not in common


def test_px4_docker_build_generates_airframes_from_configs():
    dockerfile = _read("Dockerfile")

    assert "COPY configs/airframes" in dockerfile
    assert "generate_px4_airframes.py" in dockerfile
    assert "--cmake-list" in dockerfile
    assert "--manifest ${PX4_AIRFRAME_MANIFEST}" in dockerfile
    assert "COPY px4/docker/airframes/22001_none_hexa_x" not in dockerfile
    assert "22001_none_hexa_x/' CMakeLists.txt" not in dockerfile


def test_px4_build_defaults_pin_release_commit():
    config = _read("make/config.mk")
    dockerfile = _read("Dockerfile")

    assert "PX4_TAG            ?= v1.17.0" in config
    assert "PX4_REF            ?= d6f12ad1c4f70ad3230afd7d86e971421e02fef4" in config
    assert "ARG PX4_REF=d6f12ad1c4f70ad3230afd7d86e971421e02fef4" in dockerfile
