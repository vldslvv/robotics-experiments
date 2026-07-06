"""Public contract for environment spawning and interaction logic.

The Isaac-dependent stage and stepping implementations are imported lazily so
the package root stays importable on hosts without Isaac Sim installed.
"""

from robotics_experiments.environment.switches import SwitchState, apply_button_sample

__all__ = [
    "SwitchState",
    "apply_button_sample",
    "despawn_switches",
    "init_articulations",
    "spawn_switches",
    "step_switches",
    "validate_switches",
]


def __getattr__(name: str):
    if name in {"despawn_switches", "spawn_switches", "validate_switches"}:
        from robotics_experiments.environment import stage

        return getattr(stage, name)
    if name in {"init_articulations", "step_switches"}:
        from robotics_experiments.environment import step

        return getattr(step, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
