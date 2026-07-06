from pathlib import Path

import pytest

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "experiment"
_AIRFRAME_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "airframes"


def _group_defaults(group_name, file_stem="default"):
    from omegaconf import OmegaConf

    group_config = OmegaConf.load(_CONFIG_DIR / group_name / f"{file_stem}.yaml")
    group_config.pop("defaults", None)
    return group_config


def _airframe_config(file_stem):
    from omegaconf import OmegaConf

    return OmegaConf.load(_AIRFRAME_CONFIG_DIR / f"{file_stem}.yaml")


def test_compute_rotor_constants_rejects_invalid_inputs():
    from robotics_experiments.config.load import _compute_rotor_constants

    with pytest.raises(ValueError, match="max_thrust_per_rotor_g must be positive"):
        _compute_rotor_constants(0.0, [100.0])

    with pytest.raises(ValueError, match="at least one rotor velocity"):
        _compute_rotor_constants(1000.0, [])

    with pytest.raises(ValueError, match="max_rotor_velocity values must be positive"):
        _compute_rotor_constants(1000.0, [100.0, 0.0])


def test_default_config_loads_from_root_experiment_config():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _compute_rotor_constants
    from robotics_experiments.config.load import _reset_config as reset_config
    from robotics_experiments.config.types import EnvConfig

    reset_config()

    config = get_config()
    expected_sim = _group_defaults("sim")
    expected_drone = _group_defaults("drone")
    expected_switches = _group_defaults("switches")
    expected_px4 = _group_defaults("px4")
    expected_rl = _group_defaults("rl")
    expected_press = _group_defaults("rl/press")

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
    assert config.drone.usd_path.endswith("/holybro_x500_v2.usda")
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
    assert config.px4.max_thrust_per_rotor_g == 1054.0
    # rotor_constant is not authored: it is derived from max_thrust_per_rotor_g +
    # max_rotor_velocity by config.load (here the global default fallback).
    assert config.px4.rotor_constant == _compute_rotor_constants(
        config.px4.max_thrust_per_rotor_g, config.px4.max_rotor_velocity
    )
    assert not hasattr(config.px4, "rot_dir")
    assert config.px4.linear_drag == [0.50, 0.30, 0.0]
    assert config.px4.debug_log_path == (
        "/isaac-sim/proj/px4/logs/{date}/{time}_v{vehicle}_bridge.log"
    )
    assert config.px4.debug_log_level == "DEBUG"
    assert config.px4.sensor_debug_log_interval == 50
    assert config.px4.max_tilt_angle_deg == 60.0
    assert config.rl.num_envs == expected_rl.num_envs
    assert config.rl.env_spacing == expected_rl.env_spacing
    assert config.rl.episode_length_s == expected_rl.episode_length_s
    assert config.rl.policy_rate_hz == expected_rl.policy_rate_hz
    assert config.rl.training.policy_name == expected_rl.training.policy_name == "hover"
    assert config.rl.action.kind == expected_rl.action.kind
    assert config.rl.action.action_dim == expected_rl.action.action_dim
    assert config.rl.action.controller_kp_vel == list(expected_rl.action.controller_kp_vel)
    # The hover task definition is config (task definition = one YAML block).
    assert config.rl.press.hover_task.reward_alive == expected_press.hover_task.reward_alive
    assert config.rl.press.hover_task.min_altitude_m == expected_press.hover_task.min_altitude_m
    assert config.rl.press.hover_task.reset_xy_half_m == expected_press.hover_task.reset_xy_half_m
    # The default runnable experiment selects the takeoff routine; the GCS link
    # baseport lives in the px4 group.
    expected_routine = _group_defaults("routine")
    assert [step["kind"] for step in config.routine.steps] == ["arm", "takeoff", "hover"]
    assert config.routine.max_tilt_deg == expected_routine.max_tilt_deg
    assert config.px4.gcs_baseport == 14550
    assert config.scene.endswith("/MinimalIntegration.usda")
    assert config.vehicles[0].ros_namespace == "/x500_01"
    assert config.vehicles[0].ros_frame_prefix == "x500_01"
    holybro = _airframe_config("holybro_x500_v2")
    assert (
        config.vehicles[0].px4.rotor_spin_visual_scale == holybro.px4.bridge.rotor_spin_visual_scale
    )
    reset_config()


def test_base_config_is_shared_defaults_only():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "base.yaml")

    assert "scene" not in cfg
    assert "vehicles" not in cfg


def test_press_training_policy_name_is_required_by_schema():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf
    from omegaconf.errors import MissingMandatoryValue

    from robotics_experiments.config.types import RlTrainingConfig

    defaults = _group_defaults("rl").training
    defaults.pop("policy_name")
    cfg = OmegaConf.merge(OmegaConf.structured(RlTrainingConfig), defaults)

    with pytest.raises(MissingMandatoryValue, match="policy_name"):
        OmegaConf.to_object(cfg)


def test_takeoff_experiment_composes_routine_steps_gates_and_scene(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/takeoff_quad.yaml",
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
    assert config.routine.steps[2]["hold_s"] == 3.0
    assert config.routine.max_downward_speed_m_s == expected_gates.max_downward_speed_m_s
    assert (
        config.routine.hover_max_abs_horizontal_speed_m_s
        == expected_gates.hover_max_abs_horizontal_speed_m_s
    )
    # The single-drone experiment declares its own fleet.
    assert [vehicle.id for vehicle in config.vehicles] == [0]
    reset_config()


def test_forward_quad_config_composes(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/forward_quad.yaml",
    )
    reset_config()

    config = get_config()
    assert config.routine.steps == []
    assert [(vehicle.id, vehicle.spawn_xy, vehicle.prim_name) for vehicle in config.vehicles] == [
        (0, (0.0, -2.0), "Drone0"),
    ]
    steps = config.vehicles[0].routine.steps
    assert [step["kind"] for step in steps] == ["arm", "takeoff", "hover", "forward", "land"]
    assert steps[3]["distance_m"] == 5.0
    reset_config()


def test_circle_land_quad_quad_config_composes(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/circle_land_quad_quad.yaml",
    )
    reset_config()

    config = get_config()
    assert config.routine.steps == []
    assert [(vehicle.id, vehicle.spawn_xy, vehicle.prim_name) for vehicle in config.vehicles] == [
        (0, (0.0, -2.0), "Drone0"),
        (1, (0.0, 2.0), "Drone1"),
    ]
    expected_step_kinds = ["arm", "takeoff", "hover", *(["goto"] * 12), "hover", "land"]
    assert [[step["kind"] for step in vehicle.routine.steps] for vehicle in config.vehicles] == [
        expected_step_kinds,
        expected_step_kinds,
    ]
    assert [
        (step["x"], step["y"], step["z"]) for step in config.vehicles[0].routine.steps[3:15]
    ] == [
        (1.0, -1.732, 2.6),
        (1.732, -1.0, 2.7),
        (2.0, 0.0, 2.8),
        (1.732, 1.0, 2.9),
        (1.0, 1.732, 3.0),
        (0.0, 2.0, 3.1),
        (-1.0, 1.732, 3.0),
        (-1.732, 1.0, 2.9),
        (-2.0, 0.0, 2.8),
        (-1.732, -1.0, 2.7),
        (-1.0, -1.732, 2.6),
        (0.0, -2.0, 2.5),
    ]
    assert [
        (step["x"], step["y"], step["z"]) for step in config.vehicles[1].routine.steps[3:15]
    ] == [
        (-1.0, 1.732, 3.6),
        (-1.732, 1.0, 3.7),
        (-2.0, 0.0, 3.8),
        (-1.732, -1.0, 3.9),
        (-1.0, -1.732, 4.0),
        (0.0, -2.0, 4.1),
        (1.0, -1.732, 4.0),
        (1.732, -1.0, 3.9),
        (2.0, 0.0, 3.8),
        (1.732, 1.0, 3.7),
        (1.0, 1.732, 3.6),
        (0.0, 2.0, 3.5),
    ]
    assert config.vehicles[1].routine.steps[1]["climb_m"] == 3.5
    reset_config()


def test_hexacopter_airframe_config_supplies_px4_bridge_and_sitl_defaults(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, get_enabled_px4_launch_configs
    from robotics_experiments.config.load import _compute_rotor_constants
    from robotics_experiments.config.load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/navigate_quad_hex.yaml",
    )
    reset_config()

    airframe = _airframe_config("hexacopter_s550")
    config = get_config()
    vehicle = config.vehicles[1]

    assert vehicle.drone.usd_path.endswith(f"/{airframe.output_filename}")
    assert vehicle.px4.input_offset == list(airframe.px4.bridge.input_offset)
    assert vehicle.px4.input_scaling == list(airframe.px4.bridge.input_scaling)
    assert vehicle.px4.zero_position_armed == list(airframe.px4.bridge.zero_position_armed)
    assert vehicle.px4.max_thrust_per_rotor_g == airframe.px4.bridge.max_thrust_per_rotor_g
    # rotor_constant is derived (not authored) from the airframe's cited thrust.
    assert vehicle.px4.rotor_constant == _compute_rotor_constants(
        airframe.px4.bridge.max_thrust_per_rotor_g, airframe.px4.bridge.max_rotor_velocity
    )
    assert vehicle.px4.rolling_moment_coefficient == list(
        airframe.px4.bridge.rolling_moment_coefficient
    )
    assert vehicle.px4.min_rotor_velocity == list(airframe.px4.bridge.min_rotor_velocity)
    assert vehicle.px4.max_rotor_velocity == list(airframe.px4.bridge.max_rotor_velocity)
    assert [
        (c.instance_index, c.sim_model, c.sys_autostart) for c in get_enabled_px4_launch_configs()
    ] == [
        (0, "none_x500_v2", 4019),
        (1, airframe.px4.sitl.sim_model, airframe.px4.sitl.sys_autostart),
    ]
    reset_config()


def test_resolve_fleet_merges_per_vehicle_routine_override():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config
    from robotics_experiments.config.load import _resolve_fleet as resolve_fleet
    from robotics_experiments.config.types import RoutineConfig

    reset_config()
    env = get_config()
    global_routine = OmegaConf.merge(
        OmegaConf.structured(RoutineConfig), _group_defaults("routine")
    )
    cfg = OmegaConf.create(
        {
            "drone": env.drone,
            "px4": env.px4,
            "routine": global_routine,
            "vehicles": [
                # Drone 0 inherits the (empty) global routine -> manual.
                {
                    "id": 0,
                    "ros": {"namespace": "/x500_01"},
                    "spawn_xy": [0.0, 0.0],
                    "drone": {"prim_name": "Drone0"},
                },
                # Drone 1 overrides only steps and one gate -> flies its own routine.
                {
                    "id": 1,
                    "ros": {"namespace": "/x500_02"},
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


def test_resolve_fleet_requires_ros_namespace():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config
    from robotics_experiments.config.load import _resolve_fleet as resolve_fleet

    reset_config()
    env = get_config()
    cfg = OmegaConf.create(
        {
            "drone": env.drone,
            "px4": env.px4,
            "routine": env.routine,
            "vehicles": [{"id": 0, "spawn_xy": [0.0, 0.0]}],
        }
    )

    with pytest.raises(ValueError, match=r"vehicle id=0 must define ros\.namespace"):
        resolve_fleet(cfg)
    reset_config()


@pytest.mark.parametrize(
    ("namespace", "match"),
    [
        ("x500_01", "must be absolute"),
        ("/", "must not be root"),
        ("/x500_01/", "must not end with"),
        ("/fleet//x500_01", "empty segments"),
        (" /x500_01", "whitespace"),
    ],
)
def test_resolve_fleet_rejects_invalid_ros_namespace(namespace, match):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config
    from robotics_experiments.config.load import _resolve_fleet as resolve_fleet

    reset_config()
    env = get_config()
    cfg = OmegaConf.create(
        {
            "drone": env.drone,
            "px4": env.px4,
            "routine": env.routine,
            "vehicles": [{"id": 0, "ros": {"namespace": namespace}, "spawn_xy": [0.0, 0.0]}],
        }
    )

    with pytest.raises(ValueError, match=match):
        resolve_fleet(cfg)
    reset_config()


@pytest.mark.parametrize(
    ("vehicles", "match"),
    [
        (
            [
                {"id": 0, "ros": {"namespace": "/x500_01"}, "spawn_xy": [0.0, 0.0]},
                {
                    "id": 1,
                    "ros": {"namespace": "/x500_01"},
                    "spawn_xy": [4.0, 0.0],
                    "drone": {"prim_name": "Drone1"},
                },
            ],
            "duplicate vehicle ros.namespace",
        ),
        (
            [
                {"id": 0, "ros": {"namespace": "/fleet/a_b"}, "spawn_xy": [0.0, 0.0]},
                {
                    "id": 1,
                    "ros": {"namespace": "/fleet_a/b"},
                    "spawn_xy": [4.0, 0.0],
                    "drone": {"prim_name": "Drone1"},
                },
            ],
            "duplicate vehicle ROS frame prefix",
        ),
        (
            [{"id": -1, "ros": {"namespace": "/x500_01"}, "spawn_xy": [0.0, 0.0]}],
            "vehicle id must be non-negative",
        ),
    ],
)
def test_resolve_fleet_validates_ros_identity(vehicles, match):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config
    from robotics_experiments.config.load import _resolve_fleet as resolve_fleet

    reset_config()
    env = get_config()
    cfg = OmegaConf.create(
        {
            "drone": env.drone,
            "px4": env.px4,
            "routine": env.routine,
            "vehicles": vehicles,
        }
    )

    with pytest.raises(ValueError, match=match):
        resolve_fleet(cfg)
    reset_config()


def test_resolve_drone_bridge_config_uses_airframe_constants():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, resolve_drone_bridge_config
    from robotics_experiments.config.load import _reset_config as reset_config

    reset_config()
    config = get_config()
    # The Isaac Lab env resolves the per-asset airframe constants (not the global
    # iris fallback) for the drone it spawns; they match the fleet's resolution.
    bridge = resolve_drone_bridge_config(config.drone.usd_path)
    assert bridge.max_thrust_per_rotor_g == config.vehicles[0].px4.max_thrust_per_rotor_g
    assert bridge.rotor_constant == config.vehicles[0].px4.rotor_constant
    assert bridge.max_rotor_velocity == config.vehicles[0].px4.max_rotor_velocity
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
    from robotics_experiments.config._resolve import _resolve_section as resolve_section
    from robotics_experiments.config.load import _reset_config as reset_config

    reset_config()
    config = get_config()

    assert resolve_section("drone", None) is config.drone
    assert resolve_section("sim", None) is config.sim
    assert resolve_section("switches", None) is config.switches
    reset_config()


def test_enabled_px4_launch_configs_match_active_config():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, get_enabled_px4_launch_configs
    from robotics_experiments.config.load import _reset_config as reset_config

    reset_config()

    config = get_config()

    assert [c.instance_index for c in get_enabled_px4_launch_configs()] == [
        vehicle.px4_instance_index for vehicle in config.vehicles if vehicle.px4.enabled
    ]
    reset_config()


def test_quad_default_resolves_to_generic_quad_x_airframe():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, get_enabled_px4_launch_configs
    from robotics_experiments.config.load import _reset_config as reset_config

    reset_config()
    config = get_config()
    assert config.vehicles[0].drone.usd_path.endswith("/holybro_x500_v2.usda")

    launch_configs = get_enabled_px4_launch_configs()
    assert [(c.instance_index, c.sim_model, c.sys_autostart) for c in launch_configs] == [
        (0, "none_x500_v2", 4019),
    ]
    reset_config()


def test_mixed_navigation_config_resolves_hexarotor_airframe(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_enabled_px4_launch_configs
    from robotics_experiments.config.load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/navigate_quad_hex.yaml",
    )
    reset_config()

    launch_configs = get_enabled_px4_launch_configs()
    assert [(c.instance_index, c.sim_model, c.sys_autostart) for c in launch_configs] == [
        (0, "none_x500_v2", 4019),
        (1, "none_hexa_x", 6001),
    ]
    reset_config()


def test_get_enabled_px4_launch_configs_omits_disabled_vehicles(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    import dataclasses

    from robotics_experiments.config import get_config, load
    from robotics_experiments.config.load import _reset_config as reset_config

    reset_config()
    enabled = get_config().vehicles[0]
    disabled = dataclasses.replace(
        enabled,
        id=1,
        px4_instance_index=1,
        prim_name="DisabledDrone",
        px4=dataclasses.replace(enabled.px4, enabled=False),
    )

    class _FakeConfig:
        vehicles = [enabled, disabled]

    monkeypatch.setattr(load, "get_config", lambda: _FakeConfig())
    launch_configs = load.get_enabled_px4_launch_configs()
    assert [c.instance_index for c in launch_configs] == [enabled.px4_instance_index]
    # monkeypatch restores load.get_config at teardown; reset the real cache too.
    monkeypatch.undo()
    reset_config()


def test_resolve_px4_sitl_launch_config_rejects_unknown_asset():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    import dataclasses

    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config
    from robotics_experiments.config.load import (
        _resolve_px4_sitl_launch_config as resolve_px4_sitl_launch_config,
    )

    reset_config()
    vehicle = get_config().vehicles[0]
    unknown = dataclasses.replace(
        vehicle,
        drone=dataclasses.replace(vehicle.drone, usd_path="/assets/mystery_drone.usda"),
    )

    with pytest.raises(ValueError, match=r"mystery_drone\.usda"):
        resolve_px4_sitl_launch_config(unknown)
    reset_config()


def test_config_path_env_selects_top_level_hydra_config(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config.load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/takeoff_quad_quad.yaml",
    )
    reset_config()

    config = get_config()

    assert [vehicle.id for vehicle in config.vehicles] == [0, 1]
    assert [vehicle.drone.prim_name for vehicle in config.vehicles] == ["Drone0", "Drone1"]
    assert [vehicle.mavlink_port for vehicle in config.vehicles] == [4560, 4561]
    # gcs_port is derived from px4.gcs_baseport + id.
    assert [vehicle.gcs_port for vehicle in config.vehicles] == [14550, 14551]
    assert [vehicle.ros_namespace for vehicle in config.vehicles] == ["/x500_01", "/x500_02"]
    assert [vehicle.ros_frame_prefix for vehicle in config.vehicles] == ["x500_01", "x500_02"]
    assert [step["kind"] for step in config.routine.steps] == ["arm", "takeoff", "hover"]
    reset_config()


def test_navigate_quad_hex_config_composes(monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, get_enabled_px4_launch_configs
    from robotics_experiments.config.load import _reset_config as reset_config

    monkeypatch.setenv(
        "ROBOTICS_EXPERIMENT_CONFIG_PATH",
        "configs/navigate_quad_hex.yaml",
    )
    reset_config()

    config = get_config()
    # A quad (id 0) and a hexacopter (id 1) fly the same navigation routine
    # concurrently in parallel lanes (hex waypoints offset +4 in x by its spawn).
    assert [vehicle.id for vehicle in config.vehicles] == [0, 1]
    assert [vehicle.prim_name for vehicle in config.vehicles] == ["Drone0", "Hexacopter0"]
    assert [vehicle.spawn_xy for vehicle in config.vehicles] == [(0.0, -2.0), (4.0, -2.0)]
    assert [vehicle.ros_namespace for vehicle in config.vehicles] == ["/x500_01", "/s550_01"]
    assert [vehicle.mavlink_port for vehicle in config.vehicles] == [4560, 4561]
    quad, hexa = config.vehicles
    assert quad.drone.usd_path.endswith("/holybro_x500_v2.usda")
    assert hexa.drone.usd_path.endswith("/hexacopter_s550.usda")
    for vehicle in config.vehicles:
        assert [step["kind"] for step in vehicle.routine.steps] == [
            "arm",
            "takeoff",
            "hover",
            "goto",
            "goto",
            "hover",
            "land",
        ]
    assert [(step["x"], step["y"], step["z"]) for step in quad.routine.steps[3:5]] == [
        (0.0, 2.0, 2.5),
        (3.0, 2.0, 2.5),
    ]
    assert [(step["x"], step["y"], step["z"]) for step in hexa.routine.steps[3:5]] == [
        (4.0, 2.0, 2.5),
        (7.0, 2.0, 2.5),
    ]
    # Each vehicle's PX4 container launches the per-asset airframe.
    assert [
        (c.instance_index, c.sim_model, c.sys_autostart) for c in get_enabled_px4_launch_configs()
    ] == [
        (0, "none_x500_v2", 4019),
        (1, "none_hexa_x", 6001),
    ]
    reset_config()
