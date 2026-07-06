"""Shared reinforcement-learning adapters and runtime helpers.

Lazy package: importing :mod:`robotics_experiments.rl` must not pull training
dependencies such as rsl_rl, torch, or Isaac Lab onto the core import path.
"""

from __future__ import annotations

__all__ = ["press", "training"]


def __getattr__(name: str):
    if name == "press":
        from . import press

        return press
    if name == "training":
        from . import training

        return training
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
