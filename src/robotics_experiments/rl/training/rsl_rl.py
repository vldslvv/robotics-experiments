"""rsl_rl training adapter: the ONLY module importing ``rsl_rl``.

Runs inside the Isaac Lab image (rsl_rl and isaaclab_rl ship with
``isaaclab.sh --install``); it is never imported on the core path. The ordinary
``rsl_rl`` and torch dependencies are eager, while the Isaac Lab adapter stays
deferred until the application is running. The package-root contract --
``build_runner`` / ``configure_phase`` / ``train_chunk`` /
``configure_monitoring`` /
``deterministic_actor`` / ``initialize_actor_output`` / ``export_policy``
over a private runner Protocol -- is what a future ``skrl.py`` / ``rl_games.py``
re-implements. Complete learner storage belongs to ``checkpoint.py``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.runners import OnPolicyRunner

from robotics_experiments.config import RlPolicyConfig, RlTrainingConfig, RlTrainingPhaseConfig
from robotics_experiments.rl import METADATA_FILENAME, POLICY_FILENAME
from robotics_experiments.rl.training.checkpoint import _policy_module, _PolicyRunner

if TYPE_CHECKING:
    from isaaclab.envs import DirectRLEnv

__all__ = [
    "build_runner",
    "configure_phase",
    "configure_monitoring",
    "deterministic_actor",
    "export_policy",
    "initialize_actor_output",
    "train_chunk",
]


class _PresentationWriter:
    """Map RSL-RL logging onto the strict project monitoring contract."""

    def __init__(
        self,
        *,
        writer,
        phase: RlTrainingPhaseConfig,
        phase_number: int,
        runner: _PolicyRunner,
    ) -> None:
        self._writer = writer
        self._phase = phase
        self._phase_number = phase_number
        self._runner = runner
        self._policy = _policy_module(runner)
        self._actor_parameters = self._snapshot(self._policy.actor)
        self._critic_parameters = self._snapshot(self._policy.critic)
        self._last_learner_step: int | None = None
        self._new_update_pending = False
        self._collection_time_s = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
            tag="training/Relative change in policy weights",
            value=actor_change,
            step=self._runner.completed_updates,
        )
        self._write(
            tag="training/Relative change in value model weights",
            value=critic_change,
            step=self._runner.completed_updates,
        )

    def refresh_parameters(self) -> None:
        self._actor_parameters = self._snapshot(self._policy.actor)
        self._critic_parameters = self._snapshot(self._policy.critic)

    def _start_step(self, *, learner_step: int) -> None:
        if not self._new_update_pending and learner_step == self._last_learner_step:
            return
        self._new_update_pending = False
        self._last_learner_step = learner_step
        self._runner.completed_updates += 1
        self._runner.learner_updates += 1
        step = self._runner.completed_updates
        self._write(tag="training/Training phase (number)", value=self._phase_number, step=step)
        self._write(
            tag="training/Optimizer learning rate",
            value=self._phase.learning_rate,
            step=step,
        )
        self._write(
            tag="training/Updates in current learner checkpoint",
            value=self._runner.learner_updates,
            step=step,
        )

    def add_scalar(self, tag: str, scalar_value, global_step: int, /, *args, **kwargs) -> None:
        del args, kwargs
        if tag in {
            "Loss/learning_rate",
            "Train/mean_episode_length/time",
            "Train/mean_reward/time",
        }:
            return
        self._start_step(learner_step=int(global_step))
        step = self._runner.completed_updates
        if tag.startswith("training/"):
            mapped = tag
        elif tag in {"Loss/value", "Loss/value_function"}:
            mapped = "training/Critic return prediction loss"
        elif tag == "Loss/surrogate":
            mapped = "training/PPO policy update loss"
        elif tag == "Loss/entropy":
            mapped = "training/Action distribution entropy"
        elif tag == "Policy/mean_noise_std":
            mapped = "training/Exploration noise (normalized action standard deviation)"
        elif tag == "Train/mean_reward":
            mapped = "training/Mean total reward over last 100 episodes"
        elif tag == "Train/mean_episode_length":
            mapped = "training/Mean length of last 100 episodes (policy steps)"
        elif tag == "Perf/total_fps":
            mapped = "training/Training transitions per second"
        elif tag in {"Perf/collection_time", "Perf/collection time"}:
            self._collection_time_s = float(scalar_value)
            mapped = "training/Rollout collection time per update (seconds)"
        elif tag == "Perf/learning_time":
            learning_time_s = float(scalar_value)
            self._write(
                tag="training/Total time per update (seconds)",
                value=self._collection_time_s + learning_time_s,
                step=step,
            )
            if torch.cuda.is_available():
                self._write(
                    tag="system/training/Peak GPU memory allocated by PyTorch (GiB)",
                    value=torch.cuda.max_memory_allocated() / 1024.0**3,
                    step=step,
                )
                self._write(
                    tag="system/training/Peak GPU memory reserved by PyTorch (GiB)",
                    value=torch.cuda.max_memory_reserved() / 1024.0**3,
                    step=step,
                )
                torch.cuda.reset_peak_memory_stats()
            mapped = "training/PPO optimization time per update (seconds)"
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
        self._base.logger_type = "trackio"
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
            "actor_obs_normalization": bool(policy_config.normalize_observations),
            "critic_obs_normalization": bool(policy_config.normalize_observations),
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
    """Install the sole namespaced monitoring writer for one attempt."""
    runner.learner_updates = 0
    runner.completed_updates = 0
    presentation = _PresentationWriter(
        writer=writer,
        phase=phase,
        phase_number=phase_number,
        runner=runner,
    )
    if hasattr(runner, "logger"):
        runner.logger = _MetricLogger(base=runner.logger, writer=presentation)
        return
    runner.writer = presentation
    runner.logger_type = "trackio"
    runner.log_dir = str(writer.log_dir)
    runner.save = _ignore_internal_checkpoint


def configure_phase(
    *,
    runner: _PolicyRunner,
    phase: RlTrainingPhaseConfig,
    phase_number: int = 1,
) -> None:
    """Apply optimizer and normalization settings without replacing learner state."""
    _refresh_parameter_monitoring(runner=runner)
    runner.alg.learning_rate = float(phase.learning_rate)
    runner.alg.entropy_coef = float(phase.entropy_coef)
    runner.alg.schedule = str(phase.schedule)
    for parameter_group in runner.alg.optimizer.param_groups:
        parameter_group["lr"] = float(phase.learning_rate)
    policy = _policy_module(runner)
    for normalizer in (policy.actor_obs_normalizer, policy.critic_obs_normalizer):
        if isinstance(normalizer, EmpiricalNormalization):
            # RSL-RL enters train mode for every chunk; its sample limit survives that.
            normalizer.until = None if phase.update_observation_normalization else 0
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
) -> int:
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
    return iterations


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


def _actor_module(*, runner: _PolicyRunner) -> torch.nn.Module:
    policy = _policy_module(runner)
    return torch.nn.Sequential(policy.actor_obs_normalizer, policy.actor)


def deterministic_actor(*, runner: _PolicyRunner) -> torch.nn.Module:
    """Return the checkpoint-loaded actor mean in evaluation mode."""
    actor = _actor_module(runner=runner)
    actor.eval()
    return actor


def export_policy(runner: _PolicyRunner, out_dir: Path, *, metadata: dict[str, Any]) -> None:
    """Write the policy artifact: TorchScript actor + metadata sidecar.

    The exported module maps a ``[N, obs_dim]`` float32 batch to the
    ``[N, action_dim]`` action mean using the frozen input normalizer and actor.
    The runtime policy-step loader validates the metadata at load time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actor = copy.deepcopy(_actor_module(runner=runner)).to("cpu").eval()
    example = torch.zeros(1, int(metadata["obs_dim"]), dtype=torch.float32)
    scripted = torch.jit.trace(actor, example)
    scripted.save(str(out_dir / POLICY_FILENAME))
    (out_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
