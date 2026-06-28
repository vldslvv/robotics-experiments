from pathlib import Path

import pytest

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiment"


def _group_defaults(group_name, file_stem="default"):
    from omegaconf import OmegaConf

    group_config = OmegaConf.load(_CONFIG_DIR / group_name / f"{file_stem}.yaml")
    group_config.pop("defaults", None)
    return group_config


def test_default_config_loads_from_root_experiment_config():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config._load import _reset_config as reset_config
    from robotics_experiments.config._types import EnvConfig

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
    # The default runtime experiment ships an empty (manual) routine and its
    # flight-envelope gates; the GCS link baseport lives in the px4 group.
    expected_routine = _group_defaults("routine")
    assert config.routine.steps == []
    assert config.routine.max_tilt_deg == expected_routine.max_tilt_deg
    assert config.px4.gcs_baseport == 14550
    assert config.scene.endswith("/Warehouse.usd")
    reset_config()


def test_takeoff_experiment_composes_routine_steps_gates_and_scene(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config._load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/integration_test_takeoff.yaml",
    )
    reset_config()

    config = get_config()
    # The takeoff routine variant overrides only steps; the gates come from the
    # default routine it inherits.
    expected_steps = _group_defaults("routine", "takeoff")
    expected_gates = _group_defaults("routine")

    assert config.scene.endswith("/MinimalIntegration.usda")
    assert [step["kind"] for step in config.routine.steps] == ["arm", "takeoff", "hover"]
    assert [step["kind"] for step in expected_steps.steps] == ["arm", "takeoff", "hover"]
    takeoff_step = config.routine.steps[1]
    assert takeoff_step["climb_m"] == 2.5
    assert takeoff_step["min_climb_m"] == 1.5
    assert config.routine.steps[2]["hold_s"] == 5.0
    assert config.routine.max_downward_speed_m_s == expected_gates.max_downward_speed_m_s
    assert (
        config.routine.hover_max_abs_horizontal_speed_m_s
        == expected_gates.hover_max_abs_horizontal_speed_m_s
    )
    # Single-drone fleet inherited from config.yaml.
    assert [vehicle.id for vehicle in config.vehicles] == [0]
    reset_config()


def test_two_drone_forward_land_config_composes(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config._load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/runtime_two_drone_forward_land.yaml",
    )
    reset_config()

    config = get_config()
    assert config.routine.steps == []
    assert [(vehicle.id, vehicle.spawn_xy, vehicle.prim_name) for vehicle in config.vehicles] == [
        (0, (0.0, -2.0), "Drone0"),
        (1, (0.0, 2.0), "Drone1"),
    ]
    assert [[step["kind"] for step in vehicle.routine.steps] for vehicle in config.vehicles] == [
        ["arm", "takeoff", "hover", "forward", "land"],
        ["arm", "takeoff", "hover", "forward", "land"],
    ]
    assert config.vehicles[0].routine.steps[3]["distance_m"] == 1.0
    assert config.vehicles[1].routine.steps[3]["distance_m"] == 2.0
    reset_config()


def test_resolve_fleet_merges_per_vehicle_routine_override():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from robotics_experiments.config import get_config
    from robotics_experiments.config._load import _reset_config as reset_config
    from robotics_experiments.config._load import _resolve_fleet as resolve_fleet

    reset_config()
    env = get_config()  # default config: global routine.steps is empty (manual)
    cfg = OmegaConf.create(
        {
            "drone": env.drone,
            "px4": env.px4,
            "routine": env.routine,
            "vehicles": [
                # Drone 0 inherits the (empty) global routine -> manual.
                {"id": 0, "spawn_xy": [0.0, 0.0], "drone": {"prim_name": "Drone0"}},
                # Drone 1 overrides only steps and one gate -> flies its own routine.
                {
                    "id": 1,
                    "spawn_xy": [4.0, 0.0],
                    "drone": {"prim_name": "Drone1"},
                    "routine": {
                        "steps": [
                            {"kind": "arm"},
                            {"kind": "takeoff", "climb_m": 2.0, "min_climb_m": 1.0},
                        ],
                        "max_tilt_deg": 20.0,
                    },
                },
            ],
        }
    )

    fleet = resolve_fleet(cfg)
    assert fleet[0].routine.steps == []  # inherited empty global -> manual
    assert [s["kind"] for s in fleet[1].routine.steps] == ["arm", "takeoff"]
    # The per-vehicle override changed only the named keys; the rest inherits.
    assert fleet[1].routine.max_tilt_deg == 20.0
    assert fleet[0].routine.max_tilt_deg == env.routine.max_tilt_deg
    reset_config()


def test_resolve_section_returns_provided_untouched():
    from robotics_experiments.config._resolve import _resolve_section as resolve_section

    sentinel = object()
    # A provided value must be returned as-is without composing the config.
    assert resolve_section("drone", sentinel) is sentinel


def test_resolve_section_falls_back_to_named_section():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config._load import _reset_config as reset_config
    from robotics_experiments.config._resolve import _resolve_section as resolve_section

    reset_config()
    config = get_config()

    assert resolve_section("drone", None) is config.drone
    assert resolve_section("sim", None) is config.sim
    assert resolve_section("switches", None) is config.switches
    reset_config()


def test_enabled_px4_helpers_match_active_config():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import (
        get_config,
        get_enabled_px4_instance_indices,
    )
    from robotics_experiments.config._load import _reset_config as reset_config

    reset_config()

    config = get_config()

    assert get_enabled_px4_instance_indices() == [
        vehicle.px4_instance_index for vehicle in config.vehicles if vehicle.px4.enabled
    ]
    reset_config()


def test_config_path_env_selects_top_level_hydra_config(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config._load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/integration_test_multi_drone_takeoff.yaml",
    )
    reset_config()

    config = get_config()

    assert [vehicle.id for vehicle in config.vehicles] == [0, 1]
    assert [vehicle.drone.prim_name for vehicle in config.vehicles] == ["Drone0", "Drone1"]
    assert [vehicle.mavlink_port for vehicle in config.vehicles] == [4560, 4561]
    # gcs_port is derived from px4.gcs_baseport + id.
    assert [vehicle.gcs_port for vehicle in config.vehicles] == [14550, 14551]
    assert [step["kind"] for step in config.routine.steps] == ["arm", "takeoff", "hover"]
    reset_config()
