import dataclasses

import pytest


def _composed_cfg(vehicles):
    """Build a composed-style DictConfig with structured global drone/px4.

    Mirrors what ``get_config`` hands ``resolve_fleet``: ``cfg.drone`` and
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


def _resolved_vehicle(
    vehicle_id,
    prim_name,
    spawn_xy,
    *,
    mavlink_port=None,
    connection_baseport=None,
    debug_log_path=None,
    gcs_port=None,
):
    """Build a ``ResolvedVehicle`` with consistent derived fields for tests.

    Overrides let a single derived constraint be broken at a time (e.g. an
    explicit ``mavlink_port`` to force an endpoint collision).
    """
    from robotics_experiments.config import ResolvedVehicle

    base_drone, base_px4 = _base_defaults()
    px4 = base_px4
    if connection_baseport is not None:
        px4 = dataclasses.replace(px4, connection_baseport=connection_baseport)
    if debug_log_path is not None:
        px4 = dataclasses.replace(px4, debug_log_path=debug_log_path)
    drone = dataclasses.replace(base_drone, prim_name=prim_name)
    port = mavlink_port if mavlink_port is not None else px4.connection_baseport + vehicle_id
    return ResolvedVehicle(
        id=vehicle_id,
        spawn_xy=spawn_xy,
        drone=drone,
        px4=px4,
        mavlink_port=port,
        px4_instance_index=vehicle_id,
        gcs_port=gcs_port if gcs_port is not None else 14550 + vehicle_id,
        debug_log_path=px4.debug_log_path.replace("{vehicle}", str(vehicle_id)),
        prim_name=prim_name,
    )


def test_resolve_fleet_merges_partial_over_globals_and_derives_fields():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import resolve_fleet

    cfg = _composed_cfg(
        [
            {"id": 0, "spawn_xy": [0.0, 0.0]},
            {
                "id": 1,
                "spawn_xy": [2.0, 0.0],
                "drone": {"prim_name": "Drone1"},
                "px4": {"max_tilt_angle_deg": 45.0},
            },
        ]
    )
    base_drone, base_px4 = _base_defaults()

    vehicles = resolve_fleet(cfg)

    assert len(vehicles) == 2
    # Vehicle 0 inherits all global defaults.
    assert vehicles[0].drone == base_drone
    assert vehicles[0].px4 == base_px4
    # Vehicle 1: overridden fields win, untouched fields inherit.
    assert vehicles[1].drone.prim_name == "Drone1"
    assert vehicles[1].drone.usd_path == base_drone.usd_path
    assert vehicles[1].px4.max_tilt_angle_deg == 45.0
    assert vehicles[1].px4.connection_baseport == base_px4.connection_baseport
    # Derived runtime fields come from id, not a hand-set vehicle_id.
    assert vehicles[0].mavlink_port == 4560
    assert vehicles[1].mavlink_port == 4561
    assert vehicles[0].px4_instance_index == 0
    assert vehicles[1].px4_instance_index == 1
    assert vehicles[0].gcs_port == 14550
    assert vehicles[1].gcs_port == 14551
    assert vehicles[0].prim_name == base_drone.prim_name
    assert vehicles[1].prim_name == "Drone1"
    # The {vehicle} token is rendered to id; date/time tokens stay for runtime.
    assert "{vehicle}" not in vehicles[0].debug_log_path
    assert vehicles[0].debug_log_path != vehicles[1].debug_log_path


def test_resolve_fleet_accepts_empty_list():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import resolve_fleet

    assert resolve_fleet(_composed_cfg([])) == []


def test_resolve_fleet_supports_non_contiguous_ids():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import resolve_fleet

    cfg = _composed_cfg(
        [
            {"id": 0, "spawn_xy": [0.0, 0.0]},
            {"id": 5, "spawn_xy": [2.0, 0.0], "drone": {"prim_name": "Drone5"}},
        ]
    )

    vehicles = resolve_fleet(cfg)

    assert [v.id for v in vehicles] == [0, 5]
    # Ports and instance indices follow id, not the list position.
    assert [v.mavlink_port for v in vehicles] == [4560, 4565]
    assert [v.px4_instance_index for v in vehicles] == [0, 5]
    assert [v.gcs_port for v in vehicles] == [14550, 14555]


def _two_vehicles(**v1_overrides):
    v0 = _resolved_vehicle(0, "Drone0", (0.0, 0.0))
    v1 = _resolved_vehicle(
        v1_overrides.get("id", 1),
        v1_overrides.get("prim_name", "Drone1"),
        v1_overrides.get("spawn_xy", (2.0, 0.0)),
        mavlink_port=v1_overrides.get("mavlink_port"),
        debug_log_path=v1_overrides.get("debug_log_path"),
    )
    return [v0, v1]


def test_validate_fleet_accepts_distinct_vehicles():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_fleet

    _validate_fleet(_two_vehicles())


def test_validate_fleet_rejects_duplicate_id():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_fleet

    with pytest.raises(ValueError, match="duplicate vehicle id"):
        _validate_fleet(_two_vehicles(id=0))


def test_validate_fleet_rejects_duplicate_endpoint():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_fleet

    # Same derived MAVLink port as vehicle 0 (e.g. a per-vehicle baseport
    # override) on a distinct id.
    with pytest.raises(ValueError, match="duplicate vehicle MAVLink endpoint"):
        _validate_fleet(_two_vehicles(mavlink_port=4560))


def test_validate_fleet_rejects_duplicate_prim_name():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_fleet

    with pytest.raises(ValueError, match="duplicate vehicle drone.prim_name"):
        _validate_fleet(_two_vehicles(prim_name="Drone0"))


def test_validate_fleet_rejects_bad_spawn_xy():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_fleet

    with pytest.raises(ValueError, match="spawn_xy must have length 2"):
        _validate_fleet(_two_vehicles(spawn_xy=(1.0, 2.0, 3.0)))


def test_validate_fleet_rejects_colliding_log_path():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config.load import _validate_fleet

    fixed = "/tmp/shared_bridge.log"  # no {vehicle} token -> collides
    vehicles = [
        _resolved_vehicle(0, "Drone0", (0.0, 0.0), debug_log_path=fixed),
        _resolved_vehicle(1, "Drone1", (2.0, 0.0), debug_log_path=fixed),
    ]
    with pytest.raises(ValueError, match="colliding vehicle debug_log_path"):
        _validate_fleet(vehicles)


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
    # Derived fields reproduce the single-drone baseline.
    assert vehicle.mavlink_port == 4560
    assert vehicle.px4_instance_index == 0
    assert vehicle.gcs_port == 14550
    assert vehicle.prim_name == env.drone.prim_name
    reset_config()
