import shutil

import pytest

from tests.support.paths import REPO_ROOT

_ROOT_CONFIG_DIR = REPO_ROOT / "configs"


def _write_experiment(
    tmp_path,
    name,
    *,
    vehicles,
    scene="@isaac/assets/TestScenes/MinimalIntegration.usda",
):
    from omegaconf import OmegaConf

    config_dir = tmp_path / "configs"
    shutil.copytree(_ROOT_CONFIG_DIR / "experiment", config_dir / "experiment")
    shutil.copytree(_ROOT_CONFIG_DIR / "airframes", config_dir / "airframes")
    shutil.copy2(_ROOT_CONFIG_DIR / "base.yaml", config_dir / "base.yaml")
    config_path = config_dir / f"{name}.yaml"
    OmegaConf.save(
        {
            "defaults": ["base", "_self_"],
            "scene": scene,
            "vehicles": vehicles,
        },
        config_path,
    )
    return config_path


def test_config_merges_partial_vehicles_over_globals_and_derives_fields(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config import reset_config_cache as reset_config

    config_path = _write_experiment(
        tmp_path,
        "two_vehicle_fleet",
        vehicles=[
            {"id": 0, "ros": {"namespace": "/x500_01"}, "spawn_xy": [0.0, 0.0]},
            {
                "id": 1,
                "ros": {"namespace": "/x500_02"},
                "spawn_xy": [2.0, 0.0],
                "drone": {"prim_name": "Drone1"},
                "px4": {"max_tilt_angle_deg": 45.0},
            },
        ],
    )
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", str(config_path))
    reset_config()

    env = get_config()
    vehicles = env.vehicles

    assert len(vehicles) == 2
    assert vehicles[0].drone == env.drone
    assert vehicles[0].px4.connection_baseport == env.px4.connection_baseport
    assert vehicles[0].px4.gcs_baseport == env.px4.gcs_baseport
    assert vehicles[0].px4.lockstep == env.px4.lockstep
    assert vehicles[0].px4.max_tilt_angle_deg == env.px4.max_tilt_angle_deg
    assert vehicles[1].drone.prim_name == "Drone1"
    assert vehicles[1].drone.usd_path == env.drone.usd_path
    assert vehicles[1].px4.max_tilt_angle_deg == 45.0
    assert vehicles[1].px4.connection_baseport == env.px4.connection_baseport
    assert [v.mavlink_port for v in vehicles] == [4560, 4561]
    assert [v.px4_instance_index for v in vehicles] == [0, 1]
    assert [v.gcs_port for v in vehicles] == [14550, 14551]
    assert [v.ros_namespace for v in vehicles] == ["/x500_01", "/x500_02"]
    assert [v.ros_frame_prefix for v in vehicles] == ["x500_01", "x500_02"]
    assert [v.prim_name for v in vehicles] == [env.drone.prim_name, "Drone1"]
    assert "{vehicle}" not in vehicles[0].debug_log_path
    assert vehicles[0].debug_log_path != vehicles[1].debug_log_path
    reset_config()


def test_config_accepts_empty_vehicle_list(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config import reset_config_cache as reset_config

    config_path = _write_experiment(tmp_path, "empty_fleet", vehicles=[])
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", str(config_path))
    reset_config()

    assert get_config().vehicles == []
    reset_config()


def test_config_supports_non_contiguous_vehicle_ids(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config import reset_config_cache as reset_config

    config_path = _write_experiment(
        tmp_path,
        "non_contiguous_fleet",
        vehicles=[
            {"id": 0, "ros": {"namespace": "/x500_01"}, "spawn_xy": [0.0, 0.0]},
            {
                "id": 5,
                "ros": {"namespace": "/x500_06"},
                "spawn_xy": [2.0, 0.0],
                "drone": {"prim_name": "Drone5"},
            },
        ],
    )
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", str(config_path))
    reset_config()

    vehicles = get_config().vehicles

    assert [v.id for v in vehicles] == [0, 5]
    assert [v.mavlink_port for v in vehicles] == [4560, 4565]
    assert [v.px4_instance_index for v in vehicles] == [0, 5]
    assert [v.gcs_port for v in vehicles] == [14550, 14555]
    assert [v.ros_namespace for v in vehicles] == ["/x500_01", "/x500_06"]
    assert [v.ros_frame_prefix for v in vehicles] == ["x500_01", "x500_06"]
    reset_config()


@pytest.mark.parametrize(
    ("vehicles", "match"),
    [
        (
            [
                {"id": 0, "ros": {"namespace": "/x500_01"}, "spawn_xy": [0.0, 0.0]},
                {
                    "id": 0,
                    "ros": {"namespace": "/x500_02"},
                    "spawn_xy": [2.0, 0.0],
                    "drone": {"prim_name": "Drone1"},
                },
            ],
            "duplicate vehicle id",
        ),
        (
            [
                {"id": 0, "ros": {"namespace": "/x500_01"}, "spawn_xy": [0.0, 0.0]},
                {
                    "id": 1,
                    "ros": {"namespace": "/x500_02"},
                    "spawn_xy": [2.0, 0.0],
                    "drone": {"prim_name": "Drone1"},
                    "px4": {"connection_baseport": 4559},
                },
            ],
            "duplicate vehicle MAVLink endpoint",
        ),
        (
            [
                {"id": 0, "ros": {"namespace": "/x500_01"}, "spawn_xy": [0.0, 0.0]},
                {"id": 1, "ros": {"namespace": "/x500_02"}, "spawn_xy": [2.0, 0.0]},
            ],
            "duplicate vehicle drone.prim_name",
        ),
        (
            [{"id": 0, "ros": {"namespace": "/x500_01"}, "spawn_xy": [1.0, 2.0, 3.0]}],
            "spawn_xy must have length 2",
        ),
        (
            [
                {
                    "id": 0,
                    "ros": {"namespace": "/x500_01"},
                    "spawn_xy": [0.0, 0.0],
                    "px4": {"debug_log_path": "/tmp/shared_bridge.log"},
                },
                {
                    "id": 1,
                    "ros": {"namespace": "/x500_02"},
                    "spawn_xy": [2.0, 0.0],
                    "drone": {"prim_name": "Drone1"},
                    "px4": {"debug_log_path": "/tmp/shared_bridge.log"},
                },
            ],
            "colliding vehicle debug_log_path",
        ),
    ],
)
def test_config_rejects_invalid_fleet(tmp_path, monkeypatch, vehicles, match):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config import reset_config_cache as reset_config

    config_path = _write_experiment(tmp_path, "invalid_fleet", vehicles=vehicles)
    monkeypatch.setenv("ROBOTICS_EXPERIMENT_CONFIG_PATH", str(config_path))
    reset_config()

    with pytest.raises(ValueError, match=match):
        get_config()
    reset_config()


def test_default_config_composes_to_one_vehicle_equal_to_globals():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config
    from robotics_experiments.config import reset_config_cache as reset_config

    reset_config()
    env = get_config()

    assert len(env.vehicles) == 1
    vehicle = env.vehicles[0]
    assert vehicle.id == 0
    assert vehicle.spawn_xy == (0.0, 0.0)
    assert vehicle.drone == env.drone
    assert vehicle.px4.connection_baseport == env.px4.connection_baseport
    assert vehicle.px4.lockstep == env.px4.lockstep
    assert vehicle.px4.max_tilt_angle_deg == env.px4.max_tilt_angle_deg
    assert vehicle.px4.max_thrust_per_rotor_g != env.px4.max_thrust_per_rotor_g
    assert vehicle.mavlink_port == 4560
    assert vehicle.px4_instance_index == 0
    assert vehicle.gcs_port == 14550
    assert vehicle.ros_namespace == "/x500_01"
    assert vehicle.ros_frame_prefix == "x500_01"
    assert vehicle.prim_name == env.drone.prim_name
    reset_config()
