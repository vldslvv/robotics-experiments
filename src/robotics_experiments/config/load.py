import os
import shlex
from pathlib import Path
from typing import Optional, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .schemas import EnvConfig, register_schemas
from .types import DroneConfig, Px4BridgeConfig, VehicleConfig

__all__ = ["get_config", "reset_config"]

_CACHED: Optional[EnvConfig] = None

_DEFAULT_CONF_DIR = Path(__file__).resolve().parents[3] / "configs" / "experiment"
_DEFAULT_CONFIG_NAME = "config"


def _resolve_conf_dir() -> Path:
    """Return the YAML root directory, honoring ``EXPERIMENT_CONFIG_DIR``.

    Returns:
        Absolute ``Path`` to the directory that contains ``config.yaml`` and
        the ``sim/``, ``drone/``, ``switches/`` groups.
    """
    override = os.environ.get("EXPERIMENT_CONFIG_DIR", "")
    return Path(override).resolve() if override else _DEFAULT_CONF_DIR.resolve()


def _resolve_overrides() -> list[str]:
    """Parse space-separated Hydra overrides from ``EXPERIMENT_CONFIG_OVERRIDES``.

    Returns:
        List of dot-list override strings (e.g. ``["sim.physics_hz=200"]``),
        or an empty list when the env var is unset.
    """
    raw = os.environ.get("EXPERIMENT_CONFIG_OVERRIDES", "").strip()
    return shlex.split(raw) if raw else []


def _validate_vehicles(vehicles: list[VehicleConfig]) -> None:
    """Raise ``ValueError`` on any structurally invalid multi-vehicle config.

    Invalid configuration must fail loudly at resolution time rather than
    surface later as a port collision or an overwritten log.
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

        endpoint = (
            v.px4.connection_type,
            v.px4.connection_ip,
            v.px4.connection_baseport + v.px4.vehicle_id,
        )
        if endpoint in endpoints:
            raise ValueError(f"duplicate vehicle MAVLink endpoint {endpoint} (vehicle id={v.id})")
        endpoints.add(endpoint)

        if v.drone.prim_name in prim_names:
            raise ValueError(
                f"duplicate vehicle drone.prim_name {v.drone.prim_name!r} (vehicle id={v.id})"
            )
        prim_names.add(v.drone.prim_name)

        # Render only the {vehicle} token; date/time tokens stay literal. Two
        # vehicles whose template lacks {vehicle} render identically and would
        # overwrite each other's logs. Empty path disables logging (no collision).
        if v.px4.debug_log_path:
            rendered = v.px4.debug_log_path.replace("{vehicle}", str(v.px4.vehicle_id))
            if rendered in log_paths:
                raise ValueError(
                    f"colliding vehicle debug_log_path {v.px4.debug_log_path!r} "
                    f"(vehicle id={v.id}); add a {{vehicle}} token to disambiguate"
                )
            log_paths.add(rendered)


def _resolve_vehicles(cfg) -> list[VehicleConfig]:
    """Build the typed, validated vehicle list from the composed config.

    Each ``vehicles`` entry is a *partial* override; its ``drone`` / ``px4``
    partials are merged over the structured global ``cfg.drone`` / ``cfg.px4``
    (``OmegaConf.merge`` both fills defaults and rejects unknown keys).
    """
    raw = list(cfg.get("vehicles") or [])
    if not raw:
        raise ValueError("config 'vehicles' must contain at least one entry")
    resolved: list[VehicleConfig] = []
    for entry in raw:
        drone = OmegaConf.merge(cfg.drone, entry.get("drone", {}))
        px4 = OmegaConf.merge(cfg.px4, entry.get("px4", {}))
        resolved.append(
            VehicleConfig(
                id=int(entry["id"]),
                spawn_xy=tuple(entry["spawn_xy"]),
                drone=cast(DroneConfig, OmegaConf.to_object(drone)),
                px4=cast(Px4BridgeConfig, OmegaConf.to_object(px4)),
            )
        )
    _validate_vehicles(resolved)
    return resolved


def get_config() -> EnvConfig:
    """Return the process-wide composed :class:`EnvConfig`, loading once.

    The first call composes Hydra against the YAML tree resolved from
    ``EXPERIMENT_CONFIG_DIR`` (default: the repository's
    ``configs/experiment`` directory) and caches the resulting dataclass.
    Subsequent calls return the cached instance - safe to call from any
    extension's ``on_startup`` and from per-step hot paths.

    Returns:
        The composed ``EnvConfig`` instance.
    """
    global _CACHED
    if _CACHED is not None:
        return _CACHED

    register_schemas()

    conf_dir = str(_resolve_conf_dir())
    config_name = os.environ.get("EXPERIMENT_CONFIG_NAME", "") or _DEFAULT_CONFIG_NAME
    overrides = _resolve_overrides()

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=conf_dir, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)

    # to_object turns the loosely typed cfg.vehicles into raw dicts; replace
    # them with the typed, validated list before caching/returning.
    env = cast(EnvConfig, OmegaConf.to_object(cfg))
    env.vehicles = _resolve_vehicles(cfg)
    _CACHED = env
    return _CACHED


def reset_config() -> None:
    """Drop the cached config so the next :func:`get_config` recomposes.

    Intended for tests and for interactive reload workflows; production
    extensions should not need to call this.
    """
    global _CACHED
    _CACHED = None
