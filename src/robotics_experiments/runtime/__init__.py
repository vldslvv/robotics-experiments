"""Host-agnostic fleet runtime orchestrator.

This is the topmost domain layer: it composes ``config`` -> ``drone`` /
``simulation`` / ``px4_bridge`` / ``commanding`` into the one runtime pipeline
(open scene -> apply settings -> spawn fleet -> start the lockstep bridge group ->
build per-drone routine drivers -> teardown) that the thin per-host shells drive
(the live ``experiment_runtime`` Kit extension and the standalone scenario
runner).

Isaac-coupled leaf package: ``_bridge_host`` and ``orchestrator`` import
``carb``/``omni`` at module scope (the documented domain-leaf exception), so they
are loaded lazily here and ``import robotics_experiments.runtime`` stays free of
Isaac. Kept off the core import path; ``tests/test_core_imports.py`` guards it.
The pure ``_scene`` helper is safe to import eagerly elsewhere.
"""

__all__ = [
    "FleetDriverEnv",
    "FleetOrchestrator",
]

_LAZY = {
    "FleetDriverEnv": "orchestrator",
    "FleetOrchestrator": "orchestrator",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f"{__name__}.{module}"), name)
