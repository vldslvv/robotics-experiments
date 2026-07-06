"""rsl_rl training adapter: the ONLY module importing ``rsl_rl``.

Runs inside the Isaac Lab image (rsl_rl and isaaclab_rl ship with
``isaaclab.sh --install``); it is never imported on the core path (the lazy
``rl.training`` package defers it). The functional contract --
``build_runner`` / ``train`` / ``export_policy`` over the :class:`PolicyRunner`
Protocol -- is what a future ``_skrl.py`` / ``_rl_games.py`` re-implements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = ["PolicyRunner", "TrainResult", "build_runner", "export_policy", "train"]


class PolicyRunner(Protocol):
    """The runner surface the adapter contract relies on."""

    alg: Any

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = ...) -> None:
        """Run the training loop for the given number of iterations."""
        ...


@dataclass(frozen=True)
class TrainResult:
    """Outcome of a training run."""

    iterations: int
    log_dir: str


def _train_cfg(training_cfg: Any) -> dict[str, Any]:
    """Map ``RlTrainingConfig`` onto the rsl_rl OnPolicyRunner cfg dict."""
    return {
        # Single observation group: the env emits {"policy": tensor} and the
        # critic shares it (no privileged observations yet).
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": int(training_cfg.num_steps_per_env),
        "save_interval": int(training_cfg.save_interval),
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": float(training_cfg.init_noise_std),
            "noise_std_type": "scalar",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [int(d) for d in training_cfg.actor_hidden_dims],
            "critic_hidden_dims": [int(d) for d in training_cfg.critic_hidden_dims],
            "activation": str(training_cfg.activation),
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


def build_runner(env: Any, training_cfg: Any, *, log_dir: str, device: str) -> PolicyRunner:
    """Wrap a DirectRLEnv for rsl_rl and build the OnPolicyRunner."""
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    return OnPolicyRunner(wrapped, _train_cfg(training_cfg), log_dir=log_dir, device=device)


def train(runner: PolicyRunner, iterations: int) -> TrainResult:
    """Run the training loop and return the run summary."""
    runner.learn(num_learning_iterations=int(iterations), init_at_random_ep_len=True)
    return TrainResult(iterations=int(iterations), log_dir=str(getattr(runner, "log_dir", "")))


def _actor_module(runner: PolicyRunner) -> Any:
    # rsl_rl 3.x names the PPO policy `alg.policy`; 2.x used `alg.actor_critic`.
    policy = getattr(runner.alg, "policy", None)
    if policy is None:
        policy = runner.alg.actor_critic
    return policy.actor


def export_policy(runner: PolicyRunner, out_dir: Path, *, metadata: dict[str, Any]) -> None:
    """Write the policy artifact: TorchScript actor + metadata sidecar.

    The exported module maps a ``[N, obs_dim]`` float32 batch to the
    ``[N, action_dim]`` action mean (deterministic inference); the metadata is
    the contract :func:`robotics_experiments.rl.press.policy.load_policy`
    validates at load time.
    """
    import copy

    import torch

    from robotics_experiments.rl.press.policy import METADATA_FILENAME, POLICY_FILENAME

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actor = copy.deepcopy(_actor_module(runner)).to("cpu").eval()
    example = torch.zeros(1, int(metadata["obs_dim"]), dtype=torch.float32)
    scripted = torch.jit.trace(actor, example)
    scripted.save(str(out_dir / POLICY_FILENAME))
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2) + "\n")
