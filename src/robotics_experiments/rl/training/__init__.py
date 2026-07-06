"""Internal training adapters for RL tasks (training-only dependencies).

Lazy package: importing it must not pull rsl_rl/torch/isaaclab. The concrete
adapter module is imported on attribute access only, so the core import path and
the pure task/domain modules never depend on the training stack. Consumers
outside ``robotics_experiments.rl.training`` must import training entrypoints
from this package root, or use the higher-level ``robotics_experiments.rl``
facade.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_experiments.rl.training.rsl_rl import (
        TrainingResult,
        build_runner,
        export_policy,
        train,
    )

__all__ = ["TrainingResult", "build_runner", "export_policy", "train"]

_RSL_RL_ATTRS = set(__all__)


def __getattr__(name: str):
    if name in _RSL_RL_ATTRS:
        import robotics_experiments.rl.training.rsl_rl as adapter_module

        return getattr(adapter_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
