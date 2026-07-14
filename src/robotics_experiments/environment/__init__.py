"""Public contract for environment spawning and interaction logic.

The Isaac-dependent stage and stepping implementations are imported lazily so
the package root stays importable on hosts without Isaac Sim installed.
"""

from robotics_experiments.environment.switches import SwitchState, apply_button_sample

__all__ = [
    "SwitchState",
    "SwitchesRuntimePhase",
    "SwitchesRuntimeState",
    "adopt_switches_stage",
    "apply_button_sample",
    "create_switches_runtime",
    "despawn_switches",
    "init_articulations",
    "pause_switches_runtime",
    "play_switches_runtime",
    "spawn_switches",
    "step_switches",
    "step_switches_runtime",
    "teardown_switches_runtime",
    "validate_switches",
]


def __getattr__(name: str):
    if name in {
        "SwitchesRuntimePhase",
        "SwitchesRuntimeState",
        "adopt_switches_stage",
        "create_switches_runtime",
        "pause_switches_runtime",
        "play_switches_runtime",
        "step_switches_runtime",
        "teardown_switches_runtime",
    }:
        from robotics_experiments.environment import _runtime

        return getattr(_runtime, name)
    if name in {"despawn_switches", "spawn_switches", "validate_switches"}:
        from robotics_experiments.environment import stage

        return getattr(stage, name)
    if name in {"init_articulations", "step_switches"}:
        from robotics_experiments.environment import step

        return getattr(step, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
