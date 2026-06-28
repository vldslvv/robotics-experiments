from typing import TypeVar

__all__: list[str] = []

_T = TypeVar("_T")


def _resolve_section(name: str, provided: _T | None) -> _T:
    """Return ``provided`` if set, else the named section of the composed config.

    Collapses the repeated ``if x is None: x = get_config().<section>`` fallback
    into one place. The :func:`get_config` import stays inside the function so a
    lightweight ``import robotics_experiments`` does not pull in Hydra until a
    fallback actually fires.

    Args:
        name: Attribute on the composed :class:`EnvConfig` to read when
            ``provided`` is ``None`` (for example ``"drone"`` or ``"sim"``).
        provided: A caller-supplied config section, or ``None`` to resolve.

    Returns:
        ``provided`` unchanged when it is not ``None``; otherwise
        ``getattr(get_config(), name)``.
    """
    if provided is not None:
        return provided
    from robotics_experiments.config import get_config

    return getattr(get_config(), name)
