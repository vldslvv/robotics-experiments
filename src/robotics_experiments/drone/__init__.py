"""Drone spawning and randomization logic."""

__all__ = ["spawn_drone", "despawn_drone"]


def __getattr__(name: str):
    if name in __all__:
        from . import spawn

        return getattr(spawn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
