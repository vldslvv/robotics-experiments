"""rsl_rl training adapter: the ONLY module importing ``rsl_rl``.

Runs inside the Isaac Lab image (rsl_rl and isaaclab_rl ship with
``isaaclab.sh --install``); it is never imported on the core path (the lazy
``robotics_experiments.rl`` package root defers it). The package-root contract --
``build_runner`` / ``train`` / ``export_policy`` plus ``TrainingResult`` over a
private runner Protocol -- is what a future ``skrl.py`` / ``rl_games.py``
re-implements.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = ["TrainingResult", "build_runner", "export_policy", "train"]


class _RunnerLogger(Protocol):
    """Logger state exposed by RSL-RL 5 runners."""

    log_dir: str


class _PolicyRunner(Protocol):
    """The runner surface the adapter contract relies on."""

    export_policy_to_jit: Callable[[str, str], None]
    logger: _RunnerLogger

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = ...) -> None:
        """Run the training loop for the given number of iterations."""
        ...


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of a training run."""

    iterations: int
    log_dir: str


def _train_cfg(training_cfg: Any) -> dict[str, Any]:
    """Map ``RlTrainingConfig`` onto the rsl_rl OnPolicyRunner cfg dict."""
    return {
        # Single observation group: the env emits {"policy": tensor} and the
        # critic shares it (no privileged observations yet).
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": int(training_cfg.num_steps_per_env),
        "save_interval": int(training_cfg.save_interval),
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [int(d) for d in training_cfg.actor_hidden_dims],
            "activation": str(training_cfg.activation),
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": float(training_cfg.init_noise_std),
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [int(d) for d in training_cfg.critic_hidden_dims],
            "activation": str(training_cfg.activation),
            "obs_normalization": False,
            "distribution_cfg": None,
        },
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": float(training_cfg.learning_rate),
            "schedule": "adaptive",
            "gamma": float(training_cfg.gamma),
            "lam": float(training_cfg.lam),
            "entropy_coef": float(training_cfg.entropy_coef),
            "clip_param": float(training_cfg.clip_param),
            "value_loss_coef": float(training_cfg.value_loss_coef),
            "use_clipped_value_loss": True,
            "num_learning_epochs": int(training_cfg.num_learning_epochs),
            "num_mini_batches": int(training_cfg.num_mini_batches),
            "desired_kl": float(training_cfg.desired_kl),
            "max_grad_norm": float(training_cfg.max_grad_norm),
        },
    }


def build_runner(env: Any, training_cfg: Any, *, log_dir: str, device: str) -> _PolicyRunner:
    """Wrap a DirectRLEnv for rsl_rl and build the OnPolicyRunner."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    return OnPolicyRunner(wrapped, _train_cfg(training_cfg), log_dir=log_dir, device=device)


def train(runner: _PolicyRunner, iterations: int) -> TrainingResult:
    """Run the training loop and return the run summary."""
    runner.learn(num_learning_iterations=int(iterations), init_at_random_ep_len=True)
    return TrainingResult(iterations=int(iterations), log_dir=str(runner.logger.log_dir))


def export_policy(runner: _PolicyRunner, out_dir: Path, *, metadata: dict[str, Any]) -> None:
    """Write the policy artifact: TorchScript actor + metadata sidecar.

    The exported module maps a ``[N, obs_dim]`` float32 batch to the
    ``[N, action_dim]`` action mean (deterministic inference); the metadata is
    the runtime policy-step loader validates at load time.
    """
    from robotics_experiments.rl.press import METADATA_FILENAME, POLICY_FILENAME

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(str(out_dir), POLICY_FILENAME)
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
