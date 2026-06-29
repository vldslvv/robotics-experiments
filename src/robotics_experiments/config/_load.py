import functools
import os
from pathlib import Path
from typing import cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from ._schemas import _register_schemas
from ._types import (
    DroneConfig,
    EnvConfig,
    Px4BridgeConfig,
    Px4SitlLaunchConfig,
    ResolvedVehicle,
    RoutineConfig,
)

__all__ = [
    "get_config",
    "get_enabled_px4_instance_indices",
    "get_enabled_px4_launch_configs",
    "resolve_px4_sitl_launch_config",
]

_DEFAULT_CONF_DIR = Path(__file__).resolve().parents[3] / "configs"
_DEFAULT_CONFIG_NAME = "config"
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"


def _resolve_config_path() -> Path:
    raw_path = os.environ.get(_CONFIG_PATH_ENV)
    if raw_path is None:
        return _DEFAULT_CONF_DIR / f"{_DEFAULT_CONFIG_NAME}.yaml"

    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path.resolve()


def _validate_fleet(vehicles: list[ResolvedVehicle]) -> None:
    """Raise ``ValueError`` on any structurally invalid resolved fleet.

    Operates on the derived runtime fields (``mavlink_port`` / ``debug_log_path``)
    so it never recomputes ``connection_baseport + id``. Invalid configuration
    must fail loudly at resolution time rather than surface later as a port
    collision or an overwritten log.
    """
    ids: set[int] = set()
    endpoints: set[tuple[str, str, int]] = set()
    prim_names: set[str] = set()
    log_paths: set[str] = set()
    for v in vehicles:
        if len(v.spawn_xy) != 2:
            raise ValueError(f"vehicle id={v.id} spawn_xy must have length 2, got {v.spawn_xy!r}")
        if v.id in ids:
            raise ValueError(f"duplicate vehicle id={v.id}")
        ids.add(v.id)

        endpoint = (v.px4.connection_type, v.px4.connection_ip, v.mavlink_port)
        if endpoint in endpoints:
            raise ValueError(f"duplicate vehicle MAVLink endpoint {endpoint} (vehicle id={v.id})")
        endpoints.add(endpoint)

        if v.prim_name in prim_names:
            raise ValueError(
                f"duplicate vehicle drone.prim_name {v.prim_name!r} (vehicle id={v.id})"
            )
        prim_names.add(v.prim_name)

        # debug_log_path already has the {vehicle} token rendered to id; the
        # date/time tokens stay literal here. Two vehicles whose template lacks
        # {vehicle} render identically and would overwrite each other's logs.
        # Empty path disables logging (no collision).
        if v.debug_log_path:
            if v.debug_log_path in log_paths:
                raise ValueError(
                    f"colliding vehicle debug_log_path {v.px4.debug_log_path!r} "
                    f"(vehicle id={v.id}); add a {{vehicle}} token to disambiguate"
                )
            log_paths.add(v.debug_log_path)


def _resolve_fleet(cfg) -> list[ResolvedVehicle]:
    """Enrich the composed config's ``vehicles`` into a resolved fleet.

    The single enrichment step from authored Hydra data to a self-describing
    runtime fleet. Each ``vehicles`` entry is a *partial* override; its
    ``drone`` / ``px4`` partials are merged over the structured global
    ``cfg.drone`` / ``cfg.px4`` (``OmegaConf.merge`` both fills defaults and
    rejects unknown keys) into a pure resolved config, from which the derived
    runtime fields (``mavlink_port``, ``px4_instance_index``, the
    ``px4.gcs_baseport + id`` GCS port, the rendered ``debug_log_path``,
    ``prim_name``) are computed once. An empty list is a valid empty fleet
    (0..N is handled uniformly).
    """
    raw = list(cfg.get("vehicles") or [])
    resolved: list[ResolvedVehicle] = []
    for entry in raw:
        drone = OmegaConf.merge(cfg.drone, entry.get("drone", {}))
        px4 = OmegaConf.merge(cfg.px4, entry.get("px4", {}))
        routine = OmegaConf.merge(cfg.routine, entry.get("routine", {}))
        drone_cfg = cast(DroneConfig, OmegaConf.to_object(drone))
        px4_cfg = cast(Px4BridgeConfig, OmegaConf.to_object(px4))
        routine_cfg = cast(RoutineConfig, OmegaConf.to_object(routine))
        vehicle_id = int(entry["id"])
        resolved.append(
            ResolvedVehicle(
                id=vehicle_id,
                spawn_xy=tuple(entry["spawn_xy"]),
                drone=drone_cfg,
                px4=px4_cfg,
                routine=routine_cfg,
                mavlink_port=px4_cfg.connection_baseport + vehicle_id,
                px4_instance_index=vehicle_id,
                gcs_port=px4_cfg.gcs_baseport + vehicle_id,
                debug_log_path=px4_cfg.debug_log_path.replace("{vehicle}", str(vehicle_id)),
                prim_name=drone_cfg.prim_name,
            )
        )
    _validate_fleet(resolved)
    return resolved


@functools.cache
def get_config() -> EnvConfig:
    """Return the process-wide composed :class:`EnvConfig`, loading once.

    The first call composes Hydra against the selected top-level experiment
    config and caches the resulting dataclass.
    Subsequent calls return the cached instance - safe to call from any extension's
    ``on_startup`` and from per-step hot paths.

    Returns:
        The composed ``EnvConfig`` instance.
    """
    _register_schemas()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    config_path = _resolve_config_path()
    with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
        cfg = compose(config_name=config_path.stem)

    # to_object turns the loosely typed cfg.vehicles into raw dicts; replace
    # them with the enriched, validated resolved fleet before caching/returning.
    env = cast(EnvConfig, OmegaConf.to_object(cfg))
    env.vehicles = _resolve_fleet(cfg)
    return env


# Maps the spawned drone USD asset (by file name) to its PX4 SITL airframe as
# (PX4_SIM_MODEL, PX4_SYS_AUTOSTART). Keyed by the asset basename so a
# per-vehicle drone.usd_path override selects the matching PX4 airframe. Both
# entries use a "none*" sim model that still starts simulator_mavlink (not
# Gazebo/jMAVSim). Extend this when adding a new PX4-flyable asset.
_PX4_AIRFRAME_BY_ASSET: dict[str, tuple[str, int]] = {
    "quadcopter_500mm.usda": ("none_iris", 10016),
    "hexacopter_s550.usda": ("none_hexa_x", 6001),
}


def resolve_px4_sitl_launch_config(vehicle: ResolvedVehicle) -> Px4SitlLaunchConfig:
    """Derive the PX4 SITL launch airframe for one resolved vehicle.

    Pure: reads only resolved config values (no filesystem, Docker, Hydra
    composition, subprocess, or environment access). The drone USD asset name
    selects the PX4 ``PX4_SIM_MODEL`` / ``PX4_SYS_AUTOSTART`` pair so a
    hexacopter asset launches PX4's hex geometry instead of the quad
    ``none_iris`` default. An unknown PX4-flyable asset raises ``ValueError``
    (naming the asset and prim) rather than silently flying the wrong airframe.
    """
    asset = Path(vehicle.drone.usd_path).name
    airframe = _PX4_AIRFRAME_BY_ASSET.get(asset)
    if airframe is None:
        raise ValueError(
            f"no PX4 SITL airframe mapping for drone asset {vehicle.drone.usd_path!r} "
            f"(prim {vehicle.prim_name!r}); add it to config._load._PX4_AIRFRAME_BY_ASSET"
        )
    sim_model, sys_autostart = airframe
    return Px4SitlLaunchConfig(
        instance_index=vehicle.px4_instance_index,
        sim_model=sim_model,
        sys_autostart=sys_autostart,
    )


def get_enabled_px4_launch_configs() -> list[Px4SitlLaunchConfig]:
    """Return PX4 SITL launch specs for the PX4-enabled vehicles, in fleet order.

    The host-side launch scripts iterate these to start one PX4 container per
    vehicle, threading each spec's ``instance_index`` / ``sim_model`` /
    ``sys_autostart`` in as ``PX4_INSTANCE`` / ``PX4_SIM_MODEL`` /
    ``PX4_SYS_AUTOSTART`` so each container launches the airframe matching its
    drone asset.
    """
    return [resolve_px4_sitl_launch_config(v) for v in get_config().vehicles if v.px4.enabled]


def get_enabled_px4_instance_indices() -> list[int]:
    """Return the PX4 ``-i`` instance indices of the PX4-enabled vehicles.

    The host-side launch scripts iterate these (the resolved
    ``px4_instance_index``, in fleet order) instead of a blind ``0..count``
    loop, so non-contiguous vehicle ids keep their derived MAVLink/PX4 port
    offsets.
    """
    return [c.instance_index for c in get_enabled_px4_launch_configs()]


def _reset_config() -> None:
    """Drop the cached config so the next :func:`get_config` recomposes.

    Intended for tests and for interactive reload workflows; production
    extensions should not need to call this.
    """
    get_config.cache_clear()
