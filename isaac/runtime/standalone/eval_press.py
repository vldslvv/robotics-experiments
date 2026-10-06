"""Independently evaluate an already selected F3 press policy artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from robotics_experiments.config import ExperimentConfig, load_config, load_curriculum_config
from robotics_experiments.rl import (
    POLICY_FILENAME,
    PolicyInputs,
    prepare_policy_inputs,
    validate_policy_artifact,
)
from robotics_experiments.rl.evaluation import PolicyEvaluation
from robotics_experiments.rl.press import (
    create_press_acceptance_criteria,
    evaluate_press_acceptance,
    press_policy_contract,
)
from robotics_experiments.rl.training.publication import _REPORT_SCHEMA_VERSION
from robotics_experiments.workflows.isaac_lab._contact_evaluation import _evaluate_contact_policy
from robotics_experiments.workflows.isaac_lab.training import _config_values

if TYPE_CHECKING:
    from robotics_experiments.workflows.isaac_lab import PressApproachEvaluationResult

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_RUN_ID_ENV = "RL_RUN_ID"


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class _HoverFeedback:
    """Position/yaw feedback using only the production schema-3 observation."""

    def __init__(self, *, config: ExperimentConfig) -> None:
        self._action = config.rl.action
        self._evaluation = config.rl.press.evaluation

    def __call__(self, observation: torch.Tensor, /) -> torch.Tensor:
        body_velocity = self._evaluation.feasibility_position_gain_per_s * observation[:, :3]
        body_velocity -= self._evaluation.feasibility_velocity_damping * observation[:, 3:6]
        # Transpose the goal-to-body rotation to recover drone yaw in the goal frame.
        yaw_error = torch.atan2(observation[:, 15], observation[:, 12])
        yaw_rate = -self._evaluation.feasibility_yaw_gain_per_s * yaw_error
        return torch.cat(
            (
                body_velocity / self._action.max_body_velocity_mps,
                (yaw_rate / self._action.max_yaw_rate_radps).unsqueeze(-1),
            ),
            dim=-1,
        ).clamp(-1.0, 1.0)


class _ContactFeedback(_HoverFeedback):
    """Diagnostic press/retreat control through the same actor observation."""

    def __init__(self, *, config: ExperimentConfig, tool_offset_m: torch.Tensor) -> None:
        super().__init__(config=config)
        self._tool_offset_m = tool_offset_m
        self._contact = config.rl.press.contact
        self._contact_only = config.rl.press.task == "contact"
        self._goal = config.rl.press.goal
        self.reset()

    def reset(self) -> None:
        """Start a fresh feedback controller for each evaluation seed."""
        self._contact_steps: torch.Tensor | None = None
        self._compression_velocity: torch.Tensor | None = None

    def __call__(self, observation: torch.Tensor, /) -> torch.Tensor:
        action = super().__call__(observation)
        contact = (observation[:, 24] > 0.5) | (
            observation[:, 21] > self._contact.feasibility_compression_m / 2
        )
        if self._contact_steps is None:
            self._contact_steps = torch.zeros_like(observation[:, 24])
        if self._compression_velocity is None:
            self._compression_velocity = torch.full_like(
                observation[:, 24], self._contact.feasibility_approach_mps
            )
        speed = torch.full_like(observation[:, 24], self._contact.feasibility_approach_mps)
        normal = observation[:, 12:15]
        alignment_error = observation[:, :3]
        if not self._contact_only:
            tangent = observation[:, 15:18]
            vertical = torch.linalg.cross(normal, tangent)
            tool = self._tool_offset_m.to(observation)
            rotated_tool = normal * tool[0] + tangent * tool[1] + vertical * tool[2]
            # Align the pad through approach and pressing, including vehicle
            # tilt before first contact. The offset comes from the actual asset.
            action[:, :3] += (
                self._evaluation.feasibility_position_gain_per_s
                * (rotated_tool - tool)
                * (observation[:, 25] < 0.5)[:, None]
                / self._action.max_body_velocity_mps
            )
            alignment_error = alignment_error + rotated_tool - tool
        normal_error = (alignment_error * normal).sum(dim=-1)
        tangent_error = alignment_error - normal_error[:, None] * normal
        aligned = (
            torch.linalg.vector_norm(tangent_error, dim=-1) <= self._goal.alignment_error_max_m
        ) & (
            torch.linalg.vector_norm(observation[:, 6:9], dim=-1)
            <= self._goal.angular_speed_max_radps
        )
        speed = torch.where(aligned, speed, 0.0)
        if self._contact_only:
            self._compression_velocity = torch.where(
                contact,
                self._compression_velocity
                + (
                    self._contact.feasibility_compression_gain_per_s
                    * (self._contact.feasibility_compression_m - observation[:, 21])
                ),
                self._contact.feasibility_approach_mps,
            ).clamp(-self._action.max_body_velocity_mps, self._action.max_body_velocity_mps)
            speed = torch.where(contact, self._compression_velocity, speed)
        else:
            self._contact_steps = torch.where(contact & aligned, self._contact_steps + 1, 0)
            speed = torch.where(
                aligned,
                speed + self._contact_steps * self._contact.feasibility_press_increment_mps,
                0.0,
            )
        # Goal offset remains the retreat target throughout the episode.
        retreat = observation[:, 25] > 0.5
        approach_action = speed / self._action.max_body_velocity_mps
        normal_action = (action[:, :3] * normal).sum(dim=-1)
        retreat_action = normal_action.clamp(
            -self._contact.feasibility_retreat_mps / self._action.max_body_velocity_mps,
            self._contact.feasibility_retreat_mps / self._action.max_body_velocity_mps,
        )
        if not self._contact_only:
            # A close, misaligned handoff needs clearance before the pad can
            # move across the housing face. Holding normal velocity is not enough.
            approach_action = torch.where(aligned, approach_action, retreat_action.clamp(max=0))
        # Replace wall-normal motion in the contact frame, preserving lateral
        # feedback when the vehicle pitches under button load.
        commanded_normal = torch.where(retreat, retreat_action, approach_action)
        action[:, :3] += (commanded_normal - normal_action)[:, None] * normal
        return action.clamp(-1, 1)


def _run_feasibility(*, config: ExperimentConfig, inputs: PolicyInputs, output_dir: Path) -> None:
    from robotics_experiments.workflows.isaac_lab import (
        evaluate_press_approach_policy,
        make_isaac_lab_env,
        press_policy_evaluation,
    )

    if config.rl.action.kind != "velocity":
        raise ValueError("task feasibility requires the velocity action interface")
    contract = press_policy_contract(config=config, controller=inputs.controller)
    seeds = tuple(config.rl.press.evaluation.validation_seeds)
    output_dir = _repo_root() / output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "schema_version": 1,
        "mode": "task-feasibility",
        "task": config.rl.press.task,
        "resolved_configuration": _config_values(config),
        "set_name": "validation",
        "seeds": seeds,
        "episodes_per_seed": config.rl.press.evaluation.episodes_per_seed,
        "episode_length_s": config.rl.episode_length_s,
        "controller_path": str(inputs.controller_path),
        "controller_digest": inputs.controller_digest,
        "switch_asset": inputs.switch_asset.record(),
        "controller_run_id": inputs.controller.run_id,
        "flight_parameters_path": str(inputs.flight_parameters_path),
        "flight_parameters_digest": inputs.flight_parameters_digest,
        "flight_parameters_run_id": inputs.flight_parameters.run_id,
        "feedback": {
            name: getattr(config.rl.press.evaluation, name)
            for name in (
                "feasibility_position_gain_per_s",
                "feasibility_velocity_damping",
                "feasibility_yaw_gain_per_s",
            )
        },
    }
    report_path = output_dir / "feasibility.json"
    report_path.write_text(json.dumps(record, indent=2) + "\n")
    try:
        env, _env_cfg = make_isaac_lab_env(
            task_id=config.rl.training.task_id,
            training=False,
            experiment_config=config,
            controller_settings=inputs.controller.selected_settings,
            seed=seeds[0],
            num_envs=config.rl.press.evaluation.episodes_per_seed,
            policy_contract=contract,
            flight_parameters=inputs.flight_parameters,
            hover_reset_samples=inputs.hover_reset_samples,
            switch_asset=inputs.switch_asset,
        )
        try:
            if config.rl.press.task in {"contact", "press"}:
                from isaaclab.utils import math as math_utils

                env.reset(seed=seeds[0])
                pad_id = int(env.drone.find_bodies("pad")[0][0])
                tool_offset = math_utils.quat_apply_inverse(
                    env.drone.data.root_quat_w[:1],
                    env.drone.data.body_pos_w[:1, pad_id] - env.drone.data.root_pos_w[:1],
                )[0]
                record["feedback"]["tool_offset_m"] = tool_offset.cpu().tolist()
                report_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                feedback = _ContactFeedback(config=config, tool_offset_m=tool_offset)
                evaluated = _evaluate_contact_policy(
                    env=env,
                    policy=feedback,
                    seeds=seeds,
                    episodes_per_seed=config.rl.press.evaluation.episodes_per_seed,
                    iteration=0,
                    checkpoint_path=report_path,
                    policy_digest="",
                    source="task-feasibility",
                    report=_report,
                    reset_policy=feedback.reset,
                )
                result_record = {
                    "metrics": {metric.name: metric.value for metric in evaluated.metrics},
                    "reward_means": {reward.name: reward.mean for reward in evaluated.reward_means},
                }
            else:
                result = evaluate_press_approach_policy(
                    env=env,
                    policy=_HoverFeedback(config=config),
                    seeds=seeds,
                    episodes_per_seed=config.rl.press.evaluation.episodes_per_seed,
                    hold_steps=config.rl.press.goal.stable_steps,
                    flight_parameters=inputs.flight_parameters,
                    report=_report,
                    trace_worst_episode=True,
                )
                evaluated = press_policy_evaluation(
                    iteration=0,
                    checkpoint_path=report_path,
                    policy_digest="",
                    result=result,
                    source="hover-feasibility",
                )
                result_record = asdict(result)
        finally:
            env.close()
        acceptance = evaluate_press_acceptance(
            criteria=create_press_acceptance_criteria(
                config=config.rl.press,
                flight_parameters=inputs.flight_parameters,
            ),
            evaluation=evaluated,
        )
        record.update(result=result_record, acceptance=asdict(acceptance), passed=acceptance.passed)
        report_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        if not acceptance.passed:
            raise AssertionError("task feasibility failed: " + ", ".join(acceptance.reasons))
        _report(f"task feasibility PASSED; report={report_path}")
    except (Exception, KeyboardInterrupt) as exc:
        record.update(error=str(exc) or type(exc).__name__, passed=False)
        report_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        raise


def _run(
    *, config: ExperimentConfig, artifact_dir: Path, inputs: PolicyInputs
) -> PressApproachEvaluationResult | PolicyEvaluation:
    # AppLauncher must start before the Isaac Lab workflow import.
    from robotics_experiments.workflows.isaac_lab import (
        evaluate_press_approach_policy,
        make_isaac_lab_env,
        press_policy_evaluation,
    )

    evaluation = config.rl.press.evaluation
    seeds = tuple(int(seed) for seed in evaluation.test_seeds)
    metadata = validate_policy_artifact(artifact_dir=artifact_dir)
    contract = metadata.contract
    if contract.curriculum_stage != config.rl.press.task:
        raise ValueError(
            f"press artifact stage {contract.curriculum_stage!r} does not match "
            f"configured stage {config.rl.press.task!r}"
        )
    env, _env_cfg = make_isaac_lab_env(
        task_id=config.rl.training.task_id,
        training=False,
        experiment_config=config,
        controller_settings=metadata.controller_settings,
        seed=seeds[0],
        num_envs=evaluation.episodes_per_seed,
        policy_contract=contract,
        flight_parameters=metadata.flight_parameters,
        hover_reset_samples=inputs.hover_reset_samples,
        switch_asset=inputs.switch_asset,
    )
    try:
        policy = torch.jit.load(str(artifact_dir / POLICY_FILENAME), map_location=env.device)
        policy.eval()
        if config.rl.press.task in {"contact", "press"}:
            result = _evaluate_contact_policy(
                env=env,
                policy=policy,
                seeds=seeds,
                episodes_per_seed=evaluation.episodes_per_seed,
                iteration=int(metadata.values["selected_checkpoint_iteration"]),
                checkpoint_path=artifact_dir / POLICY_FILENAME,
                policy_digest=hashlib.sha256(
                    (artifact_dir / POLICY_FILENAME).read_bytes()
                ).hexdigest(),
                source="independent",
                report=_report,
            )
        else:
            result = evaluate_press_approach_policy(
                env=env,
                policy=policy,
                seeds=seeds,
                episodes_per_seed=evaluation.episodes_per_seed,
                hold_steps=config.rl.press.goal.stable_steps,
                flight_parameters=metadata.flight_parameters,
                report=_report,
                trace_worst_episode=True,
            )
    finally:
        env.close()

    if not isinstance(result, PolicyEvaluation):
        summary = result.summary
        _report(
            "press approach evaluation: "
            f"episodes={summary.episodes} "
            f"success={summary.success_rate:.3f} "
            f"collision={summary.collision_rate:.3f} "
            f"other_flight_failure={summary.other_flight_failure_rate:.3f} "
            f"timeout={summary.timeout_rate:.3f} "
            f"min_closest_range_m={summary.min_closest_range_m:.5f} "
            f"longest_hover_mean_s={summary.episode_longest_hover_duration_mean_s:.3f} "
            f"hover_occupancy_rate={summary.hover_envelope_occupancy_rate:.3f} "
            f"hover_utilization_mean={summary.hover_envelope_utilization_mean:.3f}"
        )
    for reward in result.reward_means:
        _report(f"reward_mean {reward.name}={reward.mean:.6f}")
    digest = hashlib.sha256((artifact_dir / POLICY_FILENAME).read_bytes()).hexdigest()
    evaluated = (
        result
        if isinstance(result, PolicyEvaluation)
        else press_policy_evaluation(
            iteration=int(metadata.values["selected_checkpoint_iteration"]),
            checkpoint_path=artifact_dir / POLICY_FILENAME,
            policy_digest=digest,
            result=result,
            source="independent",
        )
    )
    acceptance = evaluate_press_acceptance(
        criteria=create_press_acceptance_criteria(
            config=config.rl.press,
            flight_parameters=metadata.flight_parameters,
        ),
        evaluation=evaluated,
    )
    if not acceptance.passed:
        raise AssertionError("press acceptance failed: " + ", ".join(acceptance.reasons))
    _report("press approach evaluation PASSED")
    return result


def _resolve_artifact_dir(*, explicit_artifact_dir: Path | None, config_path: Path) -> Path:
    if explicit_artifact_dir is not None:
        return explicit_artifact_dir.resolve(strict=True)
    curriculum = load_curriculum_config(config_path)
    policy_root = _repo_root() / ".build/training" / curriculum.policy_name
    run_id = os.environ.get(_RUN_ID_ENV)
    if not run_id:
        return (policy_root / "latest").resolve(strict=True)
    report = json.loads((policy_root / run_id / "curriculum_report.json").read_text())
    published = [stage for stage in report["stages"] if stage["published"]]
    if not published:
        raise ValueError("curriculum has no published stage")
    return Path(published[-1]["report_path"]).resolve(strict=True).parent


def _producer_config(artifact_dir: Path) -> tuple[ExperimentConfig, PolicyInputs]:
    report = json.loads((artifact_dir / "training_report.json").read_text())
    if report["schema_version"] != _REPORT_SCHEMA_VERSION or report["report_kind"] != "training":
        raise ValueError("unsupported producer training report")
    provenance = report["provenance"]
    config = load_config(provenance["config_path"])
    evidence = provenance["inputs"]
    inputs = prepare_policy_inputs(
        config=config,
        repo_root=Path(__file__).resolve().parents[3],
        controller_artifact=evidence["controller"]["path"],
        flight_parameters_artifact=evidence["flight_parameters"]["path"],
    )
    return config, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hover-feasibility", action="store_true")
    parser.add_argument("--task-feasibility", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--scene", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()

    config_path = Path(os.environ[_CONFIG_PATH_ENV])
    feasibility = args.hover_feasibility or args.task_feasibility
    if feasibility:
        config = load_config(config_path)
        inputs = prepare_policy_inputs(config=config, repo_root=_repo_root())
    else:
        artifact_dir = _resolve_artifact_dir(
            explicit_artifact_dir=args.artifact_dir, config_path=config_path
        )
        config, inputs = _producer_config(artifact_dir)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    # Kit installs its own handler; retain Python interruption and report cleanup.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if feasibility:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            _run_feasibility(
                config=config,
                inputs=inputs,
                output_dir=args.output_dir
                or _repo_root() / ".build" / "hover-feasibility" / run_id,
            )
        else:
            _run(config=config, inputs=inputs, artifact_dir=artifact_dir)
    except (Exception, KeyboardInterrupt) as exc:  # noqa: BLE001 - Make failure sentinel
        _report(f"ERROR: {str(exc) or type(exc).__name__}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
