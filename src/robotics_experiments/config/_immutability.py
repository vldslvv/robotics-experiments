from dataclasses import FrozenInstanceError, fields, is_dataclass
from types import MappingProxyType
from typing import Any, TypeVar, cast

__all__: list[str] = []


class _FreezableConfig:
    """Mutable during Hydra construction, immutable after loader enrichment."""

    _config_frozen = False

    def __setattr__(self, name: str, value: Any) -> None:
        if self._config_frozen:
            raise FrozenInstanceError(f"cannot assign to field {name!r}")
        object.__setattr__(self, name, value)


_T = TypeVar("_T")


def _freeze_config_graph(value: _T) -> _T:
    """Freeze a resolved config graph after all derivation and validation."""

    def freeze(node: Any) -> Any:
        if isinstance(node, list):
            return tuple(freeze(item) for item in node)
        if isinstance(node, dict):
            return MappingProxyType({key: freeze(item) for key, item in node.items()})
        if isinstance(node, tuple):
            return tuple(freeze(item) for item in node)
        if is_dataclass(node) and not isinstance(node, type):
            for field in fields(node):
                object.__setattr__(node, field.name, freeze(getattr(node, field.name)))
            if isinstance(node, _FreezableConfig):
                object.__setattr__(node, "_config_frozen", True)
        return node

    return cast(_T, freeze(value))
