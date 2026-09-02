"""rsl_rl training adapter: the ONLY module importing ``rsl_rl``.

Runs inside the Isaac Lab image (rsl_rl and isaaclab_rl ship with
``isaaclab.sh --install``); it is never imported on the core path. The ordinary
``rsl_rl`` and torch dependencies are eager, while the Isaac Lab adapter stays
deferred until the application is running. The package-root contract --
``build_runner`` / ``configure_phase`` / ``train_chunk`` / ``save_checkpoint`` /
``restore_checkpoint`` / ``configure_monitoring`` / ``monitoring_step`` /
``deterministic_actor`` / ``initialize_actor_output`` / ``export_policy`` plus ``TrainingResult``
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
    "configure_monitoring",
    "deterministic_actor",
    "export_policy",
    "initialize_actor_output",
    "monitoring_step",
    "restore_checkpoint",
    "save_checkpoint",
    "train_chunk",
]


class _ActorCritic(Protocol):
    actor: torch.nn.Module
    critic: torch.nn.Module

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
    logger: Any
    writer: Any
    logger_type: str
    save: Any

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


class _PresentationWriter:
    """Map RSL-RL logging onto the strict project TensorBoard contract."""

    def __init__(
        self,
        *,
        writer,
        phase: RlTrainingPhaseConfig,
        phase_number: int,
        policy: _ActorCritic,
    ) -> None:
        self._writer = writer
        self._phase = phase
        self._phase_number = phase_number
        self._policy = policy
        self._actor_parameters = self._snapshot(policy.actor)
        self._critic_parameters = self._snapshot(policy.critic)
        self._last_learner_step: int | None = None
        self._new_update_pending = False
        self._presentation_step = 0
        self._collection_time_s = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    @property
    def presentation_step(self) -> int:
        return self._presentation_step

    def configure_phase(self, *, phase: RlTrainingPhaseConfig, phase_number: int) -> None:
        self._phase = phase
        self._phase_number = phase_number

    def begin_update(self) -> None:
        self._new_update_pending = True

    def _write(self, *, tag: str, value, step: int) -> None:
        self._writer.add_scalar(
            tag,
            value,
            step,
            new_style=True,
            double_precision=True,
        )

    @staticmethod
    def _snapshot(module: torch.nn.Module) -> tuple[torch.Tensor, ...]:
        return tuple(parameter.detach().cpu().clone() for parameter in module.parameters())

    @classmethod
    def _relative_parameter_change(
        cls,
        *,
        module: torch.nn.Module,
        previous: tuple[torch.Tensor, ...],
    ) -> tuple[float, tuple[torch.Tensor, ...]]:
        current = cls._snapshot(module)
        change_squared = sum(
            float(torch.sum((now - before).square()))
            for now, before in zip(current, previous, strict=True)
        )
        previous_squared = sum(float(torch.sum(value.square())) for value in previous)
        denominator = max(previous_squared**0.5, torch.finfo(torch.float32).eps)
        return change_squared**0.5 / denominator, current

    def _write_parameter_changes(self) -> None:
        actor_change, self._actor_parameters = self._relative_parameter_change(
            module=self._policy.actor,
            previous=self._actor_parameters,
        )
        critic_change, self._critic_parameters = self._relative_parameter_change(
            module=self._policy.critic,
            previous=self._critic_parameters,
        )
        self._write(
            tag="train/04_optimization/actor_relative_parameter_change",
            value=actor_change,
            step=self._presentation_step,
        )
        self._write(
            tag="train/04_optimization/critic_relative_parameter_change",
            value=critic_change,
            step=self._presentation_step,
        )

    def refresh_parameters(self) -> None:
        self._actor_parameters = self._snapshot(self._policy.actor)
        self._critic_parameters = self._snapshot(self._policy.critic)

    def _start_step(self, *, learner_step: int) -> None:
        if not self._new_update_pending and learner_step == self._last_learner_step:
            return
        self._new_update_pending = False
        self._last_learner_step = learner_step
        self._presentation_step += 1
        step = self._presentation_step
        self._write(tag="train/05_schedule/phase", value=self._phase_number, step=step)
        self._write(
            tag="train/04_optimization/learning_rate",
            value=self._phase.learning_rate,
            step=step,
        )
        if self._phase.range_undershoot_min_m is not None:
            self._write(
                tag="train/05_schedule/recovery_boundary_m",
                value=self._phase.range_undershoot_min_m,
                step=step,
            )
        self._write(tag="train/05_schedule/checkpoint_update", value=learner_step, step=step)

    def add_scalar(self, tag: str, scalar_value, global_step: int, /, *args, **kwargs) -> None:
        del args, kwargs
        if tag in {
            "Loss/learning_rate",
            "Train/mean_episode_length/time",
            "Train/mean_reward/time",
        }:
            return
        self._start_step(learner_step=int(global_step))
        step = self._presentation_step
        if tag.startswith("train/"):
            mapped = tag
        elif tag in {"Loss/value", "Loss/value_function"}:
            mapped = "train/04_optimization/critic_return_mse"
        elif tag == "Loss/surrogate":
            mapped = "train/04_optimization/ppo_clipped_surrogate_loss"
        elif tag == "Loss/entropy":
            mapped = "train/04_optimization/action_distribution_entropy"
        elif tag == "Policy/mean_noise_std":
            mapped = "train/04_optimization/policy_noise_standard_deviation"
        elif tag == "Train/mean_reward":
            mapped = "train/02_progress/recent_100_episode_return_mean"
        elif tag == "Train/mean_episode_length":
            mapped = "train/02_progress/recent_100_episode_length_mean_steps"
        elif tag == "Perf/total_fps":
            mapped = "train/06_performance/fps"
        elif tag in {"Perf/collection_time", "Perf/collection time"}:
            self._collection_time_s = float(scalar_value)
            mapped = "train/06_performance/rollout_time_s"
        elif tag == "Perf/learning_time":
            learning_time_s = float(scalar_value)
            self._write(
                tag="train/06_performance/total_update_time_s",
                value=self._collection_time_s + learning_time_s,
                step=step,
            )
            if torch.cuda.is_available():
                self._write(
                    tag="system/train/01_gpu/torch_gpu_memory_allocated_gib_peak",
                    value=torch.cuda.max_memory_allocated() / 1024.0**3,
                    step=step,
                )
                self._write(
                    tag="system/train/01_gpu/torch_gpu_memory_reserved_gib_peak",
                    value=torch.cuda.max_memory_reserved() / 1024.0**3,
                    step=step,
                )
                torch.cuda.reset_peak_memory_stats()
            mapped = "train/06_performance/optimizer_update_time_s"
        else:
            raise ValueError(f"unregistered RSL-RL metric tag {tag!r}")
        self._write(tag=mapped, value=scalar_value, step=step)
        if tag == "Perf/learning_time":
            self._write_parameter_changes()


class _MetricLogger:
    """Retain RSL-RL aggregation/console behavior without its event writer."""

    log_dir = None
    disable_logs = False

    def __init__(self, *, base, writer: _PresentationWriter) -> None:
        self._base = base
        self.writer = writer
        self._base.writer = writer
        self._base.logger_type = "tensorboard"
        self._base.disable_logs = False

    def _call_enabled(self, method, /, *args, **kwargs):
        self._base.log_dir = "metrics-enabled"
        try:
            return method(*args, **kwargs)
        finally:
            self._base.log_dir = None

    def process_env_step(self, *args, **kwargs) -> None:
        self._call_enabled(self._base.process_env_step, *args, **kwargs)

    def log(self, *args, **kwargs) -> None:
        self.writer.begin_update()
        self._call_enabled(self._base.log, *args, **kwargs)

    def save_model(self, *args, **kwargs) -> None:
        del args, kwargs


def _ignore_internal_checkpoint(*args, **kwargs) -> None:
    del args, kwargs


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
        # The orchestration loop owns checkpoint cadence. The runner receives
        # no log directory, so its internal periodic/final save path is inert.
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
    log_dir: str | None,
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


def configure_monitoring(
    *,
    runner: _PolicyRunner,
    writer,
    phase: RlTrainingPhaseConfig,
    phase_number: int,
) -> None:
    """Install the sole namespaced TensorBoard writer for one attempt."""
    presentation = _PresentationWriter(
        writer=writer,
        phase=phase,
        phase_number=phase_number,
        policy=_policy_module(runner),
    )
    if hasattr(runner, "logger"):
        runner.logger = _MetricLogger(base=runner.logger, writer=presentation)
        return
    runner.writer = presentation
    runner.logger_type = "tensorboard"
    runner.log_dir = str(writer.log_dir)
    runner.save = _ignore_internal_checkpoint


def monitoring_step(*, runner: _PolicyRunner) -> int:
    """Return total attempted updates, independent of learner restoration."""
    logger = getattr(runner, "logger", None)
    if isinstance(logger, _MetricLogger):
        return logger.writer.presentation_step
    if isinstance(getattr(runner, "writer", None), _PresentationWriter):
        return runner.writer.presentation_step
    raise RuntimeError("training monitoring has not been configured")


def configure_phase(
    *,
    runner: _PolicyRunner,
    phase: RlTrainingPhaseConfig,
    phase_number: int = 1,
) -> None:
    """Apply declarative optimizer values without replacing optimizer state."""
    runner.alg.learning_rate = float(phase.learning_rate)
    runner.alg.entropy_coef = float(phase.entropy_coef)
    runner.alg.schedule = str(phase.schedule)
    for parameter_group in runner.alg.optimizer.param_groups:
        parameter_group["lr"] = float(phase.learning_rate)
    if isinstance(getattr(runner, "logger", None), _MetricLogger):
        runner.logger.writer.configure_phase(phase=phase, phase_number=phase_number)
    elif isinstance(getattr(runner, "writer", None), _PresentationWriter):
        runner.writer.configure_phase(phase=phase, phase_number=phase_number)


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
    logger = getattr(runner, "logger", None)
    if isinstance(logger, _MetricLogger):
        logger.writer.begin_update()
    elif isinstance(getattr(runner, "writer", None), _PresentationWriter):
        runner.writer.begin_update()
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
    _refresh_parameter_monitoring(runner=runner)
    return iteration


def _policy_module(runner: _PolicyRunner) -> _ActorCritic:
    # rsl_rl 3.x names the PPO policy `alg.policy`; 2.x used `alg.actor_critic`.
    policy = getattr(runner.alg, "policy", None)
    return policy if policy is not None else runner.alg.actor_critic


def _refresh_parameter_monitoring(*, runner: _PolicyRunner) -> None:
    logger = getattr(runner, "logger", None)
    writer = logger.writer if isinstance(logger, _MetricLogger) else getattr(runner, "writer", None)
    if isinstance(writer, _PresentationWriter):
        writer.refresh_parameters()


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
    _refresh_parameter_monitoring(runner=runner)


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
