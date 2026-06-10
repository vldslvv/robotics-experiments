import shutil
from pathlib import Path

import pytest

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiment"


def _group_defaults(group_name):
    from omegaconf import OmegaConf

    group_config = OmegaConf.load(_CONFIG_DIR / group_name / "default.yaml")
    group_config.pop("defaults", None)
    return group_config


def test_default_config_loads_from_root_experiment_config():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import EnvConfig, get_config, reset_config

    reset_config()

    config = get_config()
    expected_sim = _group_defaults("sim")
    expected_drone = _group_defaults("drone")
    expected_switches = _group_defaults("switches")
    expected_px4 = _group_defaults("px4")

    assert isinstance(config, EnvConfig)
    assert config.sim.physics_hz == expected_sim.physics_hz
    assert config.sim.rendering_hz == expected_sim.rendering_hz
    assert config.sim.solver_type == expected_sim.solver_type
    assert config.sim.gravity == expected_sim.gravity
    assert config.drone.prim_name == expected_drone.prim_name
    assert config.switches.num_switches == expected_switches.num_switches
    assert config.switches.state_publisher.topic == expected_switches.state_publisher.topic
    assert config.switches.state_publisher.qos.depth == expected_switches.state_publisher.qos.depth
    assert (
        config.switches.state_publisher.qos.durability
        == expected_switches.state_publisher.qos.durability
    )
    assert (
        config.switches.state_publisher.qos.reliability
        == expected_switches.state_publisher.qos.reliability
    )
    assert (
        config.switches.state_publisher.qos.history
        == expected_switches.state_publisher.qos.history
    )
    assert config.px4.connection_baseport == expected_px4.connection_baseport
    assert config.drone.usd_path.endswith("/quadcopter_500mm.usda")
    assert config.px4.num_rotors == 4
    assert config.px4.rotor_link_paths == [
        "body/propulsion/motor_1/prop",
        "body/propulsion/motor_3/prop",
        "body/propulsion/motor_2/prop",
        "body/propulsion/motor_4/prop",
    ]
    assert config.px4.connection_type == "tcpin"
    assert config.px4.px4_autolaunch is False
    assert config.px4.enabled is True
    assert config.px4.first_heartbeat_timeout_s == 5.0
    assert config.px4.heartbeat_rate_hz == 1.0
    assert config.px4.gps_enabled is True
    assert config.px4.gps_send_rate_hz == 250.0
    assert config.px4.input_scaling == [1000.0, 1000.0, 1000.0, 1000.0]
    assert config.px4.zero_position_armed == [100.0, 100.0, 100.0, 100.0]
    assert config.px4.rotor_constant == [8.54858e-6, 8.54858e-6, 8.54858e-6, 8.54858e-6]
    assert config.px4.rot_dir == [-1.0, -1.0, 1.0, 1.0]
    assert config.px4.linear_drag == [0.50, 0.30, 0.0]
    reset_config()


def test_experiment_config_dir_override_loads(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import EnvConfig, get_config, reset_config

    conf_dir = tmp_path / "experiment"
    shutil.copytree(_CONFIG_DIR, conf_dir)

    monkeypatch.setenv("EXPERIMENT_CONFIG_DIR", str(conf_dir))
    monkeypatch.setenv(
        "EXPERIMENT_CONFIG_OVERRIDES",
        "sim.physics_hz=123.0 drone.prim_name=OverrideDrone switches.num_switches=3",
    )
    reset_config()

    config = get_config()

    assert isinstance(config, EnvConfig)
    assert config.sim.physics_hz == 123.0
    assert config.drone.prim_name == "OverrideDrone"
    assert config.switches.num_switches == 3
    reset_config()
