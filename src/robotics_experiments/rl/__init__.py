"""Shared reinforcement-learning adapters and runtime helpers.

Lazy package: importing :mod:`robotics_experiments.rl` must not pull training
dependencies such as rsl_rl, torch, or Isaac Lab onto the core import path.
"""

from __future__ import annotations

__all__ = ["press", "training"]


def __getattr__(name: str):
    if name == "press":
        import robotics_experiments.rl.press as press

        return press
    if name == "training":
        import robotics_experiments.rl.training as training

        return training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
