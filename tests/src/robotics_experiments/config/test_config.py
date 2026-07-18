import shutil
from dataclasses import FrozenInstanceError
from typing import get_type_hints

import pytest

from tests.support.paths import REPO_ROOT

_ROOT = REPO_ROOT / "configs"
_EXPERIMENTS = sorted(path for path in _ROOT.glob("*.yaml") if path.name != "base.yaml")


def _write_experiment(tmp_path, name, vehicles):
    from omegaconf import OmegaConf

    root = tmp_path / "configs"
    shutil.copytree(_ROOT / "experiment", root / "experiment", dirs_exist_ok=True)
    shutil.copytree(_ROOT / "airframes", root / "airframes", dirs_exist_ok=True)
    shutil.copy2(_ROOT / "base.yaml", root / "base.yaml")
    defaults = ["base"]
    defaults.extend(
        {f"experiment/vehicle@vehicles.{label}": "holybro_x500_v2"} for label in vehicles
    )
    defaults.append("_self_")
    path = root / f"{name}.yaml"
    OmegaConf.save(
        {
            "defaults": defaults,
            "scene": "@isaac/assets/TestScenes/MinimalIntegration.usda",
            "vehicles": vehicles,
        },
        path,
    )
    return path


def _vehicle(vehicle_id, namespace, x, *, prim_name=None, **extra):
    spawn = {"position_xy": [x, 0.0]}
    if prim_name is not None:
        spawn["prim_name"] = prim_name
    return {
        "id": vehicle_id,
        "ros": {"namespace": namespace},
        "spawn": spawn,
        **extra,
    }


def test_load_config_is_isolated_and_immutable(monkeypatch):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    from robotics_experiments.config import load_config

    with initialize_config_dir(config_dir=str(_ROOT), version_base=None):
        foreign_hydra = GlobalHydra.instance().hydra
        config = load_config(_ROOT / "takeoff_quad.yaml")
        assert GlobalHydra.instance().hydra is foreign_hydra

    vehicle = config.vehicles["x500_01"]
    with pytest.raises(FrozenInstanceError):
        config.scene = "different.usda"
    with pytest.raises(FrozenInstanceError):
        vehicle.bridge.lockstep.lockstep_timeout_s = 99.0
    with pytest.raises(TypeError):
        vehicle.routine.steps[0]["kind"] = "land"
    with pytest.raises(TypeError):
        config.vehicles["other"] = vehicle


def test_load_config_caches_by_explicit_path():
    from robotics_experiments.config import load_config

    path = _ROOT / "takeoff_quad.yaml"
    assert load_config(path) is load_config(path)


def test_load_config_rejects_missing_path(tmp_path):
    from robotics_experiments.config import load_config

    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match=str(missing)):
        load_config(missing)


@pytest.mark.parametrize("path", _EXPERIMENTS, ids=lambda path: path.stem)
def test_every_shipped_experiment_composes(path):
    from robotics_experiments.config import ExperimentConfig, load_config

    config = load_config(path)
    assert isinstance(config, ExperimentConfig)
    assert config.scene
    assert config.vehicles


def test_default_experiment_has_only_global_root_config():
    from robotics_experiments.config import load_config

    config = load_config(_ROOT / "takeoff_quad.yaml")
    assert config.switches.spawn_seed == 0
    assert set(vars(config)) - {"_config_frozen"} == {
        "sim",
        "switches",
        "rl",
        "scene",
        "vehicles",
        "driver",
    }
    vehicle = config.vehicles["x500_01"]
    assert vehicle.spawn.position_xy == (0.0, 0.0)
    assert vehicle.spawn.usd_path.endswith("/holybro_x500_v2.usda")
    assert vehicle.bridge.rotor_constant
    assert [step["kind"] for step in vehicle.routine.steps] == ["arm", "takeoff", "hover"]
    assert config.rl.training.vehicle_id == 0


def test_vehicle_mapping_is_typed_and_routine_steps_are_not_any():
    from robotics_experiments.config import ExperimentConfig, VehicleConfig

    assert "VehicleConfig" in str(get_type_hints(ExperimentConfig)["vehicles"])
    routine_type = get_type_hints(VehicleConfig)["routine"]
    assert "RoutineStepConfig" in str(get_type_hints(routine_type)["steps"])


def test_empty_fleet_is_valid(tmp_path):
    from robotics_experiments.config import load_config

    config = load_config(_write_experiment(tmp_path, "empty", {}))
    assert config.vehicles == {}


def test_vehicle_order_and_derived_identity(tmp_path):
    from robotics_experiments.config import load_config

    vehicles = {
        "lead": _vehicle(5, "/fleet/lead", 2.0, prim_name="Lead"),
        "wing": _vehicle(1, "/fleet/wing", 4.0, prim_name="Wing"),
    }
    config = load_config(_write_experiment(tmp_path, "ordered", vehicles))
    assert list(config.vehicles) == ["lead", "wing"]
    lead, wing = config.vehicles.values()
    assert (lead.mavlink_port, lead.gcs_port) == (4565, 14555)
    assert (wing.mavlink_port, wing.gcs_port) == (4561, 14551)
    assert lead.debug_log_path != wing.debug_log_path


@pytest.mark.parametrize(
    ("vehicles", "match"),
    [
        (
            {
                "a": _vehicle(0, "/a", 0.0, prim_name="A"),
                "b": _vehicle(0, "/b", 2.0, prim_name="B"),
            },
            "duplicate vehicle id",
        ),
        (
            {
                "a": _vehicle(0, "/a", 0.0, prim_name="Same"),
                "b": _vehicle(1, "/b", 2.0, prim_name="Same"),
            },
            "duplicate vehicle spawn.prim_name",
        ),
        (
            {
                "a": _vehicle(0, "/fleet/a_b", 0.0, prim_name="A"),
                "b": _vehicle(1, "/fleet_a/b", 2.0, prim_name="B"),
            },
            "duplicate vehicle ROS frame prefix",
        ),
    ],
)
def test_fleet_validation(tmp_path, vehicles, match):
    from robotics_experiments.config import load_config

    with pytest.raises(ValueError, match=match):
        load_config(_write_experiment(tmp_path, "invalid", vehicles))


@pytest.mark.parametrize(
    ("namespace", "match"),
    [
        ("x500", "must be absolute"),
        ("/", "must not be root"),
        ("/x500/", "must not end with"),
        ("/fleet//x500", "empty segments"),
        (" /x500", "whitespace"),
    ],
)
def test_namespace_validation(tmp_path, namespace, match):
    from robotics_experiments.config import load_config

    vehicles = {"x500": _vehicle(0, namespace, 0.0)}
    with pytest.raises(ValueError, match=match):
        load_config(_write_experiment(tmp_path, "namespace", vehicles))


def test_per_vehicle_routines_are_independent(tmp_path):
    from robotics_experiments.config import load_config

    vehicles = {
        "manual": _vehicle(0, "/manual", 0.0, prim_name="Manual"),
        "flying": _vehicle(
            1,
            "/flying",
            2.0,
            prim_name="Flying",
            routine={"steps": [{"kind": "arm"}], "max_tilt_deg": 20.0},
        ),
    }
    config = load_config(_write_experiment(tmp_path, "routines", vehicles))
    assert config.vehicles["manual"].routine.steps == ()
    assert config.vehicles["flying"].routine.steps == ({"kind": "arm"},)
    assert config.vehicles["flying"].routine.max_tilt_deg == 20.0


def test_mixed_airframes_drive_bridge_and_px4_launch():
    from robotics_experiments.config import load_config

    config = load_config("configs/navigate_quad_hex.yaml")
    quad = config.vehicles["x500_01"]
    hexa = config.vehicles["s550_01"]
    assert hexa.spawn.usd_path.endswith("/hexacopter_s550.usda")
    assert len(quad.bridge.max_rotor_velocity) == 4
    assert len(hexa.bridge.max_rotor_velocity) == 6
    assert [
        (vehicle.id, vehicle.px4_sim_model, vehicle.px4_sys_autostart)
        for vehicle in config.vehicles.values()
    ] == [
        (0, "none_x500_v2", 22000),
        (1, "none_hexa_x", 22001),
    ]


def test_disabled_and_unknown_assets_affect_px4_launch(tmp_path):
    from robotics_experiments.config import load_config

    vehicles = {
        "disabled": _vehicle(
            0,
            "/disabled",
            0.0,
            bridge={"enabled": False},
            spawn={
                "position_xy": [0.0, 0.0],
                "usd_path": "/assets/unknown.usda",
            },
        )
    }
    config = load_config(_write_experiment(tmp_path, "disabled", vehicles))
    vehicle = config.vehicles["disabled"]
    assert (vehicle.px4_sim_model, vehicle.px4_sys_autostart) == ("", 0)

    vehicles["disabled"]["bridge"] = {"enabled": True}
    with pytest.raises(ValueError, match="unknown.usda"):
        load_config(_write_experiment(tmp_path, "unknown", vehicles))
