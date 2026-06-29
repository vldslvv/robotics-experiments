"""Static checks that the host-side PX4 launch wiring threads the airframe.

The per-vehicle PX4 SITL airframe (``PX4_SIM_MODEL`` / ``PX4_SYS_AUTOSTART``,
derived from the drone asset by ``config.get_enabled_px4_launch_configs``) must
reach every PX4 container. These assert the persistent and integration launch
scripts propagate both env vars rather than relying on the old hard-coded
``none_iris`` default, which silently flew hexacopters as quads.
"""

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text()


def test_persistent_launch_derives_and_propagates_airframe():
    containers = _read("make/scripts/run_px4_persistent_containers.sh")
    # Derives the per-vehicle airframe from the resolved fleet, not a bare index.
    assert "get_enabled_px4_launch_configs" in containers
    # Threads both airframe env vars into the per-instance launch.
    assert "PX4_SIM_MODEL=" in containers
    assert "PX4_SYS_AUTOSTART=" in containers

    instance = _read("px4/scripts/run_px4_persistent_instance.sh")
    assert '-e "PX4_SIM_MODEL=${PX4_SIM_MODEL}"' in instance
    assert '-e "PX4_SYS_AUTOSTART=${PX4_SYS_AUTOSTART}"' in instance

    make_target = _read("make/containers.mk")
    assert "PX4_SIM_MODEL='$(PX4_SIM_MODEL)'" in make_target
    assert "PX4_SYS_AUTOSTART='$(PX4_SYS_AUTOSTART)'" in make_target


def test_integration_launch_derives_and_propagates_airframe():
    integration = _read("make/scripts/run_px4_integration_test.sh")
    assert "get_enabled_px4_launch_configs" in integration
    assert '-e "PX4_SIM_MODEL=${sim_model}"' in integration
    assert '-e "PX4_SYS_AUTOSTART=${sys_autostart}"' in integration


def test_common_launcher_no_longer_hardcodes_none_iris():
    common = _read("px4/docker/px4_iris_common.sh")
    # The airframe is now caller-selected with a none_iris default, not a
    # hard-coded export that overrode the per-vehicle choice.
    assert 'export PX4_SIM_MODEL="none_iris"' not in common
    assert 'export PX4_SIM_MODEL="${PX4_SIM_MODEL:-none_iris}"' in common
    assert "PX4_SYS_AUTOSTART" in common
