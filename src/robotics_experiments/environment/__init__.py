"""Environment spawning, randomization, and interaction logic."""

from ._switches import SwitchState

__all__ = [
    "SwitchState",
    "despawn_switches",
    "init_articulations",
    "spawn_switches",
    "step_switches",
    "validate_switches",
]


def __getattr__(name: str):
    if name in {"despawn_switches", "spawn_switches", "validate_switches"}:
        from . import _stage

        return getattr(_stage, name)
    if name in {"init_articulations", "step_switches"}:
        from . import _step

        return getattr(_step, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
