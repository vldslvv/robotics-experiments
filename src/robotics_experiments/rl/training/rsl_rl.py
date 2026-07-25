"""rsl_rl training adapter: the ONLY module importing ``rsl_rl``.

Runs inside the Isaac Lab image (rsl_rl and isaaclab_rl ship with
``isaaclab.sh --install``); it is never imported on the core path (the lazy
``robotics_experiments.rl`` package root defers it). The package-root contract --
``build_runner`` / ``train`` / ``load_checkpoint`` / ``deterministic_actor`` /
``initialize_actor_from_checkpoint`` / ``export_policy`` plus ``TrainingResult``
over a private runner Protocol -- is what a future ``skrl.py`` / ``rl_games.py``
re-implements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "TrainingResult",
    "build_runner",
    "deterministic_actor",
    "export_policy",
    "initialize_actor_from_checkpoint",
    "load_checkpoint",
    "train",
]


class _PolicyRunner(Protocol):
    """The runner surface the adapter contract relies on."""

    alg: Any

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = ...) -> None:
        """Run the training loop for the given number of iterations."""
        ...

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """Restore a runner checkpoint."""
        del path, load_optimizer, map_location
        return {}


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
            "schedule": str(training_cfg.schedule),
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
    return TrainingResult(iterations=int(iterations), log_dir=str(getattr(runner, "log_dir", "")))


def load_checkpoint(*, runner: _PolicyRunner, checkpoint_path: Path, device: str) -> None:
    """Restore policy weights for export without resuming optimizer state."""
    runner.load(
        str(checkpoint_path),
        load_optimizer=False,
        map_location=device,
    )


def _policy_module(runner: _PolicyRunner) -> Any:
    # rsl_rl 3.x names the PPO policy `alg.policy`; 2.x used `alg.actor_critic`.
    policy = getattr(runner.alg, "policy", None)
    return policy if policy is not None else runner.alg.actor_critic


def initialize_actor_from_checkpoint(
    *,
    runner: _PolicyRunner,
    checkpoint_path: Path,
    device: str,
) -> None:
    """Load only compatible actor parameters into a freshly built runner.

    The critic, action standard deviation, optimizer, and runner iteration are
    deliberately left exactly as initialized by ``build_runner``.
    """
    import torch

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint {checkpoint_path} has no model_state_dict")
    source_actor = {
        name.removeprefix("actor."): value
        for name, value in model_state.items()
        if name.startswith("actor.")
    }
    target_actor = _policy_module(runner).actor
    target_state = target_actor.state_dict()
    if source_actor.keys() != target_state.keys():
        raise ValueError("initialization checkpoint actor architecture is incompatible")
    incompatible_shapes = {
        name: (tuple(source_actor[name].shape), tuple(target.shape))
        for name, target in target_state.items()
        if source_actor[name].shape != target.shape
    }
    if incompatible_shapes:
        raise ValueError(
            "initialization checkpoint observation/action dimensions or actor "
            f"architecture are incompatible: {incompatible_shapes}"
        )
    target_actor.load_state_dict(source_actor, strict=True)


def _actor_module(runner: _PolicyRunner) -> Any:
    return _policy_module(runner).actor


def deterministic_actor(*, runner: _PolicyRunner) -> Any:
    """Return the checkpoint-loaded actor mean in evaluation mode."""
    actor = _actor_module(runner)
    actor.eval()
    return actor


def export_policy(runner: _PolicyRunner, out_dir: Path, *, metadata: dict[str, Any]) -> None:
    """Write the policy artifact: TorchScript actor + metadata sidecar.

    The exported module maps a ``[N, obs_dim]`` float32 batch to the
    ``[N, action_dim]`` action mean (deterministic inference); the metadata is
    the runtime policy-step loader validates at load time.
    """
    import copy

    import torch

    from robotics_experiments.rl import METADATA_FILENAME, POLICY_FILENAME

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actor = copy.deepcopy(_actor_module(runner)).to("cpu").eval()
    example = torch.zeros(1, int(metadata["obs_dim"]), dtype=torch.float32)
    scripted = torch.jit.trace(actor, example)
    scripted.save(str(out_dir / POLICY_FILENAME))
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
