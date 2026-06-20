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
    expected_isaac_lab = _group_defaults("isaac_lab")

    assert isinstance(config, EnvConfig)
    assert config.sim.physics_hz == expected_sim.physics_hz
    assert config.sim.rendering_hz == expected_sim.rendering_hz
    assert config.sim.solver_type == expected_sim.solver_type
    assert config.sim.gravity == expected_sim.gravity
    assert config.drone.prim_name == expected_drone.prim_name
    assert config.drone.spawn_stationary is True
    assert config.drone.stationary_ground_z == 0.0
    assert config.drone.stationary_ground_clearance_m == 0.05
    assert config.drone.stationary_spawn_z_fallback == 0.07
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
        config.switches.state_publisher.qos.history == expected_switches.state_publisher.qos.history
    )
    assert config.px4.connection_baseport == expected_px4.connection_baseport
    assert config.drone.usd_path.endswith("/quadcopter_500mm.usda")
    assert not hasattr(config.px4, "num_rotors")
    assert not hasattr(config.px4, "rotor_link_paths")
    assert config.px4.connection_type == "tcpin"
    assert config.px4.enabled is True
    assert config.px4.lockstep.first_heartbeat_timeout_s == 5.0
    assert config.px4.lockstep.heartbeat_rate_hz == 1.0
    assert config.px4.startup_settle_time_s == 1.0
    assert config.px4.gps_enabled is True
    assert config.px4.gps_send_rate_hz == 250.0
    assert config.px4.sensor_noise_enabled is True
    assert config.px4.input_scaling == [1000.0, 1000.0, 1000.0, 1000.0]
    assert config.px4.zero_position_armed == [100.0, 100.0, 100.0, 100.0]
    assert config.px4.rotor_constant == [8.54858e-6, 8.54858e-6, 8.54858e-6, 8.54858e-6]
    assert not hasattr(config.px4, "rot_dir")
    assert config.px4.linear_drag == [0.50, 0.30, 0.0]
    assert config.px4.debug_log_path == (
        "/isaac-sim/proj/px4/logs/{date}/{time}_v{vehicle}_bridge.log"
    )
    assert config.px4.debug_log_level == "DEBUG"
    assert config.px4.sensor_debug_log_interval == 50
    assert config.px4.max_tilt_angle_deg == 60.0
    assert (
        config.isaac_lab.drone_hover_episode_length_s
        == expected_isaac_lab.drone_hover_episode_length_s
    )
    assert config.isaac_lab.drone_hover_num_envs == expected_isaac_lab.drone_hover_num_envs
    assert config.isaac_lab.drone_hover_env_spacing == expected_isaac_lab.drone_hover_env_spacing
    assert config.isaac_lab.drone_hover_action_scale == expected_isaac_lab.drone_hover_action_scale
    expected_scenario = _group_defaults("scenario")
    assert config.scenario.scene == expected_scenario.scene
    assert config.scenario.takeoff_altitude_m == expected_scenario.takeoff_altitude_m
    assert config.scenario.min_takeoff_climb_m == expected_scenario.min_takeoff_climb_m
    assert config.scenario.gcs_baseport == expected_scenario.gcs_baseport
    assert config.scenario.max_downward_speed_m_s == expected_scenario.max_downward_speed_m_s
    assert (
        config.scenario.hover_max_horizontal_range_m
        == expected_scenario.hover_max_horizontal_range_m
    )
    reset_config()


def test_multi_drone_takeoff_config_composes_two_enabled_vehicles(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, reset_config

    monkeypatch.setenv("EXPERIMENT_CONFIG_NAME", "multi_drone_takeoff")
    reset_config()
    config = get_config()

    assert len(config.vehicles) == 2
    v0, v1 = config.vehicles
    assert (v0.id, v1.id) == (0, 1)
    assert v0.drone.prim_name == "Drone0"
    assert v1.drone.prim_name == "Drone1"
    assert v0.spawn_xy == (0.0, 0.0)
    assert v1.spawn_xy == (4.0, 0.0)
    assert v0.px4.enabled is True
    assert v1.px4.enabled is True
    # Distinct simulator endpoints: connection_baseport + vehicle_id.
    assert v0.px4.connection_baseport + v0.px4.vehicle_id == 4560
    assert v1.px4.connection_baseport + v1.px4.vehicle_id == 4561
    # Per-vehicle bridge logs carry the {vehicle} token so they never collide.
    assert "{vehicle}" in v0.px4.debug_log_path
    # Scenario tunables come from the scenario group.
    assert config.scenario.takeoff_altitude_m == 2.5
    reset_config()


def test_resolve_section_returns_provided_untouched():
    from robotics_experiments.config._resolve import resolve_section

    sentinel = object()
    # A provided value must be returned as-is without composing the config.
    assert resolve_section("drone", sentinel) is sentinel


def test_resolve_section_falls_back_to_named_section():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, reset_config
    from robotics_experiments.config._resolve import resolve_section

    reset_config()
    config = get_config()

    assert resolve_section("drone", None) is config.drone
    assert resolve_section("sim", None) is config.sim
    assert resolve_section("switches", None) is config.switches
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
