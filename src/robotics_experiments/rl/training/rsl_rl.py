"""rsl_rl training adapter: the ONLY module importing ``rsl_rl``.

Runs inside the Isaac Lab image (rsl_rl and isaaclab_rl ship with
``isaaclab.sh --install``); it is never imported on the core path. The ordinary
``rsl_rl`` and torch dependencies are eager, while the Isaac Lab adapter stays
deferred until the application is running. The package-root contract --
``build_runner`` / ``configure_phase`` / ``train_chunk`` / ``save_checkpoint`` /
``restore_checkpoint`` / ``deterministic_actor`` / ``initialize_actor_output`` /
``export_policy`` plus ``TrainingResult``
over a private runner Protocol -- is what a future ``skrl.py`` / ``rl_games.py``
re-implements.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from rsl_rl.runners import OnPolicyRunner

from robotics_experiments.config import RlPolicyConfig, RlTrainingConfig, RlTrainingPhaseConfig
from robotics_experiments.rl import METADATA_FILENAME, POLICY_FILENAME

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

__all__ = [
    "TrainingResult",
    "build_runner",
    "configure_phase",
    "deterministic_actor",
    "export_policy",
    "initialize_actor_output",
    "restore_checkpoint",
    "save_checkpoint",
    "train_chunk",
]


class _ActorCritic(Protocol):
    actor: torch.nn.Module

    def state_dict(self) -> dict[str, Any]:
        """Return the complete actor-critic parameter state."""
        ...


class _Algorithm(Protocol):
    policy: _ActorCritic
    actor_critic: _ActorCritic
    optimizer: torch.optim.Optimizer
    learning_rate: float
    entropy_coef: float
    schedule: str


class _PolicyRunner(Protocol):
    """The runner surface the adapter contract relies on."""

    alg: _Algorithm
    log_dir: str
    save_interval: int
    current_learning_iteration: int

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = ...) -> None:
        """Run the training loop for the given number of iterations."""
        ...

    def load(
        self,
        path: str,
        load_optimizer: bool = True,
        map_location: str | None = None,
    ) -> dict[str, Any]:
        """Restore a runner checkpoint."""
        del path, load_optimizer, map_location
        return {}


@dataclass(frozen=True)
class TrainingResult:
    """Outcome of a training run."""

    iterations: int
    log_dir: str


def _train_cfg(
    *,
    training_config: RlTrainingConfig,
    phase: RlTrainingPhaseConfig,
    policy_config: RlPolicyConfig,
) -> dict[str, Any]:
    """Map PPO and task-owned policy config onto the rsl_rl runner dict."""
    return {
        # Single observation group: the env emits {"policy": tensor} and the
        # critic shares it (no privileged observations yet).
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
        "num_steps_per_env": int(training_config.num_steps_per_env),
        # The orchestration loop owns checkpoint cadence. RSL-RL still writes
        # the terminal model at the end of every learn() chunk.
        "save_interval": 2**31 - 1,
        "empirical_normalization": False,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": float(policy_config.actor.initial_noise_std),
            "noise_std_type": "scalar",
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [int(d) for d in policy_config.actor.hidden_dims],
            "critic_hidden_dims": [int(d) for d in policy_config.critic.hidden_dims],
            "activation": str(policy_config.activation),
        },
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": float(phase.learning_rate),
            "schedule": str(phase.schedule),
            "gamma": float(training_config.gamma),
            "lam": float(training_config.lam),
            "entropy_coef": float(phase.entropy_coef),
            "clip_param": float(training_config.clip_param),
            "value_loss_coef": float(training_config.value_loss_coef),
            "use_clipped_value_loss": True,
            "num_learning_epochs": int(training_config.num_learning_epochs),
            "num_mini_batches": int(training_config.num_mini_batches),
            "desired_kl": float(training_config.desired_kl),
            "max_grad_norm": float(training_config.max_grad_norm),
        },
    }


def build_runner(
    *,
    env: DirectRLEnv,
    training_config: RlTrainingConfig,
    phase: RlTrainingPhaseConfig,
    policy_config: RlPolicyConfig,
    log_dir: str,
    device: str,
) -> _PolicyRunner:
    """Wrap a DirectRLEnv for rsl_rl and build the OnPolicyRunner."""
    # Isaac Lab must be imported only after AppLauncher starts.
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    return OnPolicyRunner(
        wrapped,
        _train_cfg(
            training_config=training_config,
            phase=phase,
            policy_config=policy_config,
        ),
        log_dir=log_dir,
        device=device,
    )


def configure_phase(
    *,
    runner: _PolicyRunner,
    phase: RlTrainingPhaseConfig,
) -> None:
    """Apply declarative optimizer values without replacing optimizer state."""
    runner.alg.learning_rate = float(phase.learning_rate)
    runner.alg.entropy_coef = float(phase.entropy_coef)
    runner.alg.schedule = str(phase.schedule)
    for parameter_group in runner.alg.optimizer.param_groups:
        parameter_group["lr"] = float(phase.learning_rate)


def train_chunk(
    *,
    runner: _PolicyRunner,
    completed_iterations: int,
    iterations: int,
    randomize_initial_episode_lengths: bool,
) -> TrainingResult:
    """Train exact new updates and number the terminal model by completed updates."""
    if completed_iterations < 0 or iterations <= 0:
        raise ValueError("completed_iterations must be non-negative and iterations positive")
    runner.current_learning_iteration = completed_iterations + 1
    runner.learn(
        num_learning_iterations=iterations,
        init_at_random_ep_len=randomize_initial_episode_lengths,
    )
    return TrainingResult(
        iterations=iterations,
        log_dir=str(getattr(runner, "log_dir", "")),
    )


def save_checkpoint(
    *,
    runner: _PolicyRunner,
    checkpoint_path: Path,
    iteration: int,
) -> None:
    """Persist one evaluation checkpoint without external-logger side effects."""
    runner.current_learning_iteration = iteration
    torch.save(
        {
            "model_state_dict": _policy_module(runner).state_dict(),
            "optimizer_state_dict": runner.alg.optimizer.state_dict(),
            "iter": runner.current_learning_iteration,
            "infos": None,
        },
        checkpoint_path,
    )


def restore_checkpoint(*, runner: _PolicyRunner, checkpoint_path: Path, device: str) -> int:
    """Restore actor, critic, optimizer, and learner iteration for refinement."""
    runner.load(
        str(checkpoint_path),
        load_optimizer=True,
        map_location=device,
    )
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    iteration = int(payload["iter"])
    runner.current_learning_iteration = iteration
    return iteration


def _policy_module(runner: _PolicyRunner) -> _ActorCritic:
    # rsl_rl 3.x names the PPO policy `alg.policy`; 2.x used `alg.actor_critic`.
    policy = getattr(runner.alg, "policy", None)
    return policy if policy is not None else runner.alg.actor_critic


def initialize_actor_output(*, runner: _PolicyRunner, method: str) -> None:
    """Apply a task-neutral near-zero initialization to the actor output."""
    if method != "scratch-neutral-output":
        raise ValueError(f"unsupported actor output initialization {method!r}")
    actor = _policy_module(runner).actor
    output_layers = tuple(
        module for module in actor.modules() if isinstance(module, torch.nn.Linear)
    )
    if not output_layers:
        raise ValueError("actor has no linear output layer")
    output = output_layers[-1]
    torch.nn.init.orthogonal_(output.weight, gain=0.01)
    if output.bias is not None:
        torch.nn.init.zeros_(output.bias)


def _actor_module(runner: _PolicyRunner) -> torch.nn.Module:
    return _policy_module(runner).actor


def deterministic_actor(*, runner: _PolicyRunner) -> torch.nn.Module:
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
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actor = copy.deepcopy(_actor_module(runner)).to("cpu").eval()
    example = torch.zeros(1, int(metadata["obs_dim"]), dtype=torch.float32)
    scripted = torch.jit.trace(actor, example)
    scripted.save(str(out_dir / POLICY_FILENAME))
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
