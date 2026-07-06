"""Public drone stage-authoring contract.

The Isaac-specific implementation is imported lazily so host-side imports of
``robotics_experiments.drone`` do not require Isaac Sim, USD, or carb.
"""

__all__ = ["spawn_drone", "despawn_drone"]


def __getattr__(name: str):
    if name in __all__:
        import robotics_experiments.drone.spawn as spawn

        return getattr(spawn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
