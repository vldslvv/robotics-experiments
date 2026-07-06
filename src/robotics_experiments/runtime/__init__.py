"""Public contract for the host-agnostic fleet runtime.

This is the topmost domain layer: it composes ``config`` -> ``drone`` /
``simulation`` / ``px4_bridge`` / ``commanding`` into the one runtime pipeline
(open scene -> apply settings -> spawn fleet -> start the lockstep bridge group ->
build per-drone routine drivers -> teardown) that the thin per-host shells drive
(the live ``experiment_runtime`` Kit extension and the standalone scenario
runner).

The package root is the only supported import surface for code outside
``robotics_experiments.runtime``. Internal modules may move; consumers should
depend on these exported runtime concepts instead.

Isaac-coupled leaf package: ``bridge_host`` and ``orchestrator`` import
``carb``/``omni`` at module scope (the documented domain-leaf exception), so
exports are loaded lazily here and ``import robotics_experiments.runtime`` stays
free of Isaac. Kept off the core import path; ``tests/test_core_imports.py``
guards it.
"""

__all__ = [
    "BridgeHost",
    "BridgeHostGroup",
    "BridgeHostParams",
    "FleetDriverEnv",
    "FleetOrchestrator",
    "SpawnedDrone",
]

_LAZY = {
    "BridgeHost": "bridge_host",
    "BridgeHostGroup": "bridge_host",
    "BridgeHostParams": "bridge_host",
    "FleetDriverEnv": "orchestrator",
    "FleetOrchestrator": "orchestrator",
    "SpawnedDrone": "orchestrator",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if module == "bridge_host":
        import robotics_experiments.runtime.bridge_host as bridge_host

        return getattr(bridge_host, name)
    if module == "orchestrator":
        import robotics_experiments.runtime.orchestrator as orchestrator

        return getattr(orchestrator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
