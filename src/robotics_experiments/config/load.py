import os
import shlex
from pathlib import Path
from typing import Optional, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from .schemas import EnvConfig, register_schemas

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

    _CACHED = cast(EnvConfig, OmegaConf.to_object(cfg))
    return _CACHED


def reset_config() -> None:
    """Drop the cached config so the next :func:`get_config` recomposes.

    Intended for tests and for interactive reload workflows; production
    extensions should not need to call this.
    """
    global _CACHED
    _CACHED = None
