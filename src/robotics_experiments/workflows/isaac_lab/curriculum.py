"""Sequential curriculum coordination; no training environment lives here."""

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotics_experiments.config import CurriculumConfig, load_config
from robotics_experiments.rl import prepare_policy_inputs
from robotics_experiments.workflows.isaac_lab.training import (
    _config_values,
    _write_json,
    pin_learner,
    publish_training,
    run_training,
    validate_learner,
)

__all__ = ["CurriculumResult", "run_curriculum"]


@dataclass(frozen=True)
class CurriculumResult:
    complete: bool
    report_path: Path


def run_curriculum(
    *,
    config: CurriculumConfig,
    config_path: Path,
    repo_root: Path,
    run_id: str | None,
    procedural_smoke: bool,
) -> CurriculumResult:
    """Pin all inputs, then train, accept and publish each stage in order."""
    stages = []
    for stage in config.stages:
        path = (repo_root / stage.config_path).resolve(strict=True)
        experiment = load_config(path)
        inputs = prepare_policy_inputs(
            config=experiment, repo_root=repo_root, procedural_smoke=procedural_smoke
        )
        stages.append((stage.name, path, experiment, inputs))
    learner = (
        pin_learner(report_path=repo_root / config.initial_learner)
        if config.initial_learner
        else None
    )
    if learner:
        validate_learner(learner=learner, config=stages[0][2], inputs=stages[0][3])
    resolved_id = run_id or f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.{os.getpid()}"
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", resolved_id) is None:
        raise ValueError(
            "RL_RUN_ID must be one path component containing letters, digits, '.', '_' or '-'"
        )
    run_dir = repo_root / ".build/training" / config.policy_name / resolved_id
    run_dir.mkdir(parents=True, exist_ok=False)
    report_path = run_dir / "curriculum_report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "run_id": resolved_id,
        "configuration": {
            "path": str(config_path.resolve(strict=True)),
            "resolved": _config_values(config),
        },
        "initial_learner": asdict(learner) if learner else None,
        "status": "running",
        "completed_updates": 0,
        "stages": [],
    }
    passed_stages: list[dict[str, str]] = []
    _write_json(path=report_path, payload=report)
    try:
        for index, (name, path, experiment, inputs) in enumerate(stages):
            stage_dir = run_dir.parent / f"{run_dir.name}.stage_{index + 1:02d}"
            entry = {
                "name": name,
                "report_path": str(stage_dir / "training_report.json"),
                "accepted": False,
                "published": False,
                "completed_updates": 0,
            }
            report["stages"].append(entry)
            _write_json(path=report_path, payload=report)
            result = run_training(
                config=experiment,
                config_path=path,
                inputs=inputs,
                initial_learner=learner,
                run_dir=stage_dir,
                monitoring_dir=run_dir / "monitoring",
                update_offset=report["completed_updates"],
                procedural_smoke=procedural_smoke,
            )
            entry.update(
                report_path=str(result.report_path),
                accepted=result.accepted,
                completed_updates=result.completed_updates,
            )
            report["completed_updates"] += result.completed_updates
            _write_json(path=report_path, payload=report)
            if not result.accepted:
                report["status"] = "stopped"
                break
            if result.selected_checkpoint is None:
                raise RuntimeError("accepted stage returned no selected learner")
            learner = result.selected_checkpoint
            passed_stages.append(
                {
                    "name": name,
                    "report_path": str(result.report_path),
                    "report_digest": learner.report_digest,
                    "policy_digest": learner.policy_digest,
                }
            )
            publish_training(
                result=result, curriculum_run_id=resolved_id, passed_stages=passed_stages.copy()
            )
            entry["published"] = True
            _write_json(path=report_path, payload=report)
        else:
            report["status"] = "complete"
    except (Exception, KeyboardInterrupt) as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = str(exc)
        # Failed stages retain partial work, without rewriting their terminal reports.
        if report["stages"]:
            entry = report["stages"][-1]
            stage_report = Path(entry["report_path"])
            if stage_report.is_file():
                try:
                    partial = json.loads(stage_report.read_text())
                    completed = partial["progress"]["completed_updates"]
                    report["completed_updates"] += completed - entry["completed_updates"]
                    entry["completed_updates"] = completed
                except (OSError, ValueError, KeyError) as evidence_error:
                    report["evidence_error"] = str(evidence_error)
        raise
    finally:
        _write_json(path=report_path, payload=report)
    return CurriculumResult(complete=report["status"] == "complete", report_path=report_path)
