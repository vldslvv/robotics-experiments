"""Training adapters for RL tasks (training-only dependencies).

Lazy package: importing it must not pull rsl_rl/torch/isaaclab -- the concrete
adapter modules are imported on attribute access only, so the core import path
and the pure task/domain modules never depend on the training stack. The rsl_rl
adapter is the shipped implementation; ``_skrl.py`` / ``_rl_games.py`` are
drop-in later behind the same functional contract (``build_runner`` /
``train`` / ``export_policy``).
"""

from __future__ import annotations

__all__ = ["PolicyRunner", "TrainResult", "build_runner", "export_policy", "train"]

_RSL_RL_ATTRS = {"PolicyRunner", "TrainResult", "build_runner", "export_policy", "train"}


def __getattr__(name: str):
    if name in _RSL_RL_ATTRS:
        from . import _rsl_rl

        return getattr(_rsl_rl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
