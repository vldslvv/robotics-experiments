"""Complete learner storage. RNG is evidence; new attempts deliberately reseed.

There is no persistent return normalizer or independent scheduler in this PPO
configuration. Environments, rollout storage and episode accumulators are excluded.
"""

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch

__all__ = ["LearnerProgress", "save_checkpoint", "load_checkpoint", "restore_checkpoint"]

_SCHEMA_VERSION = 1


class _ActorCritic(Protocol):
    actor: torch.nn.Module
    actor_obs_normalizer: torch.nn.Module
    critic: torch.nn.Module
    critic_obs_normalizer: torch.nn.Module

    def load_state_dict(self, *, state_dict: dict[str, Any], strict: bool) -> Any:
        del state_dict, strict

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
    log_dir: str | None
    save_interval: int
    current_learning_iteration: int
    learner_updates: int
    completed_updates: int
    logger: Any
    writer: Any
    logger_type: str
    save: Any

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = ...) -> None:
        """Run the training loop for the given number of iterations."""
        ...


def _policy_module(runner: _PolicyRunner) -> _ActorCritic:
    # rsl_rl 3.x names the PPO policy `alg.policy`; 2.x used `alg.actor_critic`.
    policy = getattr(runner.alg, "policy", None)
    return policy if policy is not None else runner.alg.actor_critic


@dataclass(frozen=True)
class LearnerProgress:
    iteration: int
    learner_updates: int
    curriculum_updates: int


def _rng_state() -> dict[str, Any]:
    # NumPy's type stub does not narrow the legacy=True return to its tuple form.
    numpy_state = cast(tuple, np.random.get_state(legacy=True))
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _validate(payload: dict[str, Any]) -> None:
    if (
        set(payload)
        != {"schema_version", "model_state_dict", "optimizer_state_dict", "progress", "rng_state"}
        or payload["schema_version"] != _SCHEMA_VERSION
    ):
        raise ValueError("unsupported or incomplete learner checkpoint")
    progress = payload["progress"]
    if set(progress) != {"iteration", "learner_updates", "curriculum_updates"} or any(
        type(v) is not int or v < 0 for v in progress.values()
    ):
        raise ValueError("invalid learner progress")
    state = payload["model_state_dict"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("invalid learner model state")
    if (
        not any(k.startswith("actor.") for k in state)
        or not any(k.startswith("critic.") for k in state)
        or not {"std", "log_std"}.intersection(state)
    ):
        raise ValueError("learner must include actor, critic and exploration parameters")
    optimizer = payload["optimizer_state_dict"]
    if set(optimizer) != {"state", "param_groups"} or not optimizer["param_groups"]:
        raise ValueError("invalid learner optimizer state")
    parameters = [index for group in optimizer["param_groups"] for index in group["params"]]
    if progress["learner_updates"] and set(optimizer["state"]) != set(parameters):
        raise ValueError("incomplete Adam optimizer state")
    for state in optimizer["state"].values():
        if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError("incomplete Adam optimizer moments or step")
    rng = payload["rng_state"]
    if set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("incomplete RNG snapshot")
    random.Random().setstate(rng["python"])
    numpy = rng["numpy"]
    np.random.RandomState().set_state(
        (
            numpy["algorithm"],
            numpy["keys"].numpy().astype(np.uint32),
            numpy["position"],
            numpy["has_gauss"],
            numpy["cached_gaussian"],
        )
    )
    torch.Generator().set_state(rng["torch_cpu"].cpu())
    if not isinstance(rng["torch_cuda"], list) or any(
        not isinstance(state, torch.Tensor) or state.dtype != torch.uint8
        for state in rng["torch_cuda"]
    ):
        raise ValueError("invalid CUDA RNG snapshot")


def save_checkpoint(
    *, runner: _PolicyRunner, checkpoint_path: Path, iteration: int, curriculum_updates: int
) -> None:
    """Save all learned state and random-stream evidence at a completed update."""
    if getattr(runner.alg, "rnd", None) is not None:
        raise ValueError("RND learner state is not supported by this checkpoint schema")
    runner.current_learning_iteration = iteration
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model_state_dict": _policy_module(runner).state_dict(),
        "optimizer_state_dict": runner.alg.optimizer.state_dict(),
        "progress": asdict(
            LearnerProgress(
                iteration=iteration,
                learner_updates=runner.learner_updates,
                curriculum_updates=curriculum_updates,
            )
        ),
        "rng_state": _rng_state(),
    }
    _validate(payload)
    torch.save(payload, checkpoint_path)


def load_checkpoint(*, checkpoint_path: Path) -> dict[str, Any]:
    """Read and validate the strict tensor/primitive format on CPU."""
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _validate(payload)
    return payload


def restore_checkpoint(*, runner: _PolicyRunner, checkpoint_path: Path) -> LearnerProgress:
    """Restore model, optimizer and branch progress; leave RNG and total work alone.

    The caller applies its phase schedule afterwards and starts fresh rollouts.
    Both normalizers are part of model_state_dict, including their sample counts.
    """
    payload = load_checkpoint(checkpoint_path=checkpoint_path)
    if getattr(runner.alg, "rnd", None) is not None:
        raise ValueError("RND learner state is not supported by this checkpoint schema")
    if not isinstance(runner.alg.optimizer, torch.optim.Adam):
        raise ValueError("checkpoint schema supports the configured Adam optimizer only")
    groups = payload["optimizer_state_dict"]["param_groups"]
    states = payload["optimizer_state_dict"]["state"]
    for saved_group, live_group in zip(groups, runner.alg.optimizer.param_groups, strict=True):
        for index, parameter in zip(saved_group["params"], live_group["params"], strict=True):
            if index in states and any(
                states[index][name].shape != parameter.shape for name in ("exp_avg", "exp_avg_sq")
            ):
                raise ValueError("incompatible optimizer moment shape")
    policy = _policy_module(runner)
    for normalizer in (policy.actor_obs_normalizer, policy.critic_obs_normalizer):
        for name, buffer in normalizer.named_buffers(recurse=False):
            if buffer.is_inference():
                normalizer.register_buffer(name, buffer.clone())
    policy.load_state_dict(state_dict=payload["model_state_dict"], strict=True)
    runner.alg.optimizer.load_state_dict(payload["optimizer_state_dict"])
    runner.alg.learning_rate = float(runner.alg.optimizer.param_groups[0]["lr"])
    progress = LearnerProgress(**payload["progress"])
    runner.current_learning_iteration = progress.iteration
    runner.learner_updates = progress.learner_updates
    return progress
