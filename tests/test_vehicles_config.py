import dataclasses

import pytest


def _composed_cfg(vehicles):
    """Build a composed-style DictConfig with structured global drone/px4.

    Mirrors what ``get_config`` hands ``_resolve_vehicles``: ``cfg.drone`` and
    ``cfg.px4`` are structured nodes (so per-vehicle partials merge over them and
    unknown keys are rejected), plus a loosely typed ``vehicles`` list.
    """
    from omegaconf import OmegaConf

    from robotics_experiments.config import get_config, reset_config

    reset_config()
    env = get_config()
    reset_config()
    return OmegaConf.create(
        {
            "drone": OmegaConf.structured(env.drone),
            "px4": OmegaConf.structured(env.px4),
            "vehicles": vehicles,
        }
    )


def _base_defaults():
    from robotics_experiments.config import get_config, reset_config

    reset_config()
    env = get_config()
    reset_config()
    return env.drone, env.px4


def test_resolve_vehicles_merges_partial_over_globals():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _resolve_vehicles

    cfg = _composed_cfg(
        [
            {"id": 0, "spawn_xy": [0.0, 0.0], "px4": {"vehicle_id": 0}},
            {
                "id": 1,
                "spawn_xy": [2.0, 0.0],
                "drone": {"prim_name": "Drone1"},
                "px4": {"vehicle_id": 1, "max_tilt_angle_deg": 45.0},
            },
        ]
    )
    base_drone, base_px4 = _base_defaults()

    vehicles = _resolve_vehicles(cfg)

    assert len(vehicles) == 2
    # Vehicle 0 inherits all global defaults.
    assert vehicles[0].drone == base_drone
    assert vehicles[0].px4 == base_px4
    # Vehicle 1: overridden fields win, untouched fields inherit.
    assert vehicles[1].drone.prim_name == "Drone1"
    assert vehicles[1].drone.usd_path == base_drone.usd_path
    assert vehicles[1].px4.max_tilt_angle_deg == 45.0
    assert vehicles[1].px4.connection_baseport == base_px4.connection_baseport
    # Resolved endpoints differ: baseport + vehicle_id.
    assert vehicles[0].px4.connection_baseport + vehicles[0].px4.vehicle_id == 4560
    assert vehicles[1].px4.connection_baseport + vehicles[1].px4.vehicle_id == 4561


def test_resolve_vehicles_rejects_empty_list():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _resolve_vehicles

    cfg = _composed_cfg([])
    with pytest.raises(ValueError, match="at least one entry"):
        _resolve_vehicles(cfg)


def _two_vehicles(**v1_overrides):
    from robotics_experiments.config import VehicleConfig

    base_drone, base_px4 = _base_defaults()
    v0 = VehicleConfig(
        id=0,
        spawn_xy=(0.0, 0.0),
        drone=dataclasses.replace(base_drone, prim_name="Drone0"),
        px4=dataclasses.replace(base_px4, vehicle_id=0),
    )
    v1 = VehicleConfig(
        id=v1_overrides.get("id", 1),
        spawn_xy=v1_overrides.get("spawn_xy", (2.0, 0.0)),
        drone=dataclasses.replace(
            base_drone, prim_name=v1_overrides.get("prim_name", "Drone1")
        ),
        px4=dataclasses.replace(
            base_px4,
            vehicle_id=v1_overrides.get("vehicle_id", 1),
            debug_log_path=v1_overrides.get("debug_log_path", base_px4.debug_log_path),
        ),
    )
    return [v0, v1]


def test_validate_vehicles_accepts_distinct_vehicles():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_vehicles

    _validate_vehicles(_two_vehicles())


def test_validate_vehicles_rejects_duplicate_id():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_vehicles

    with pytest.raises(ValueError, match="duplicate vehicle id"):
        _validate_vehicles(_two_vehicles(id=0))


def test_validate_vehicles_rejects_duplicate_endpoint():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_vehicles

    with pytest.raises(ValueError, match="duplicate vehicle MAVLink endpoint"):
        _validate_vehicles(_two_vehicles(vehicle_id=0))


def test_validate_vehicles_rejects_duplicate_prim_name():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_vehicles

    with pytest.raises(ValueError, match="duplicate vehicle drone.prim_name"):
        _validate_vehicles(_two_vehicles(prim_name="Drone0"))


def test_validate_vehicles_rejects_bad_spawn_xy():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_vehicles

    with pytest.raises(ValueError, match="spawn_xy must have length 2"):
        _validate_vehicles(_two_vehicles(spawn_xy=(1.0, 2.0, 3.0)))


def test_validate_vehicles_rejects_colliding_log_path():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_vehicles

    base_drone, base_px4 = _base_defaults()
    from robotics_experiments.config import VehicleConfig

    fixed = "/tmp/shared_bridge.log"  # no {vehicle} token -> collides
    vehicles = [
        VehicleConfig(
            id=0,
            spawn_xy=(0.0, 0.0),
            drone=dataclasses.replace(base_drone, prim_name="Drone0"),
            px4=dataclasses.replace(base_px4, vehicle_id=0, debug_log_path=fixed),
        ),
        VehicleConfig(
            id=1,
            spawn_xy=(2.0, 0.0),
            drone=dataclasses.replace(base_drone, prim_name="Drone1"),
            px4=dataclasses.replace(base_px4, vehicle_id=1, debug_log_path=fixed),
        ),
    ]
    with pytest.raises(ValueError, match="colliding vehicle debug_log_path"):
        _validate_vehicles(vehicles)


def test_default_config_composes_to_one_vehicle_equal_to_globals():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import get_config, reset_config

    reset_config()
    env = get_config()

    assert len(env.vehicles) == 1
    vehicle = env.vehicles[0]
    assert vehicle.id == 0
    assert vehicle.spawn_xy == (0.0, 0.0)
    # The single default vehicle reproduces today's global drone/px4 exactly.
    assert vehicle.drone == env.drone
    assert vehicle.px4 == env.px4
    reset_config()
