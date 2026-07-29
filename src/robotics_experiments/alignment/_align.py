"""Generic candidate expansion, evaluation, scoring, and selection."""

from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._metrics import _finite_metric_result, metric
from robotics_experiments.alignment._models import (
    AlignmentArtifact,
    CandidateScore,
    ControllerDefinition,
    ControllerSettings,
)
from robotics_experiments.config import load_config

__all__: list[str] = []


def _fingerprint(value: object, /) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plain(value: Any, /) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _settings_candidates(
    definition: ControllerDefinition,
    /,
) -> tuple[ControllerSettings, ...]:
    if not definition.parameters:
        return (definition.base_settings,)
    candidates: list[ControllerSettings] = []
    value_sets = [parameter.candidates for parameter in definition.parameters]
    for values in itertools.product(*value_sets):
        replacements = {
            parameter.field: value
            for parameter, value in zip(definition.parameters, values, strict=True)
        }
        candidates.append(replace(definition.base_settings, **replacements))
    return tuple(candidates)


def _validate_pair(
    target: ControllerDefinition,
    reference: ControllerDefinition,
    /,
) -> None:
    mismatches = []
    for name in (
        "alignment_id",
        "experiment_config",
        "vehicle",
        "airframe",
        "command_rate_hz",
        "command_interface",
        "probes",
        "selection_metric",
    ):
        if getattr(target, name) != getattr(reference, name):
            mismatches.append(name)
    if target.backend != "isaac-lab" or reference.backend != "px4-sitl":
        raise ValueError("alignment v1 requires an isaac-lab target and px4-sitl reference")
    if mismatches:
        raise ValueError(f"incompatible controller definitions: {', '.join(mismatches)}")


def _git_source(*, repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot resolve alignment source provenance: {exc}") from exc
    return commit, bool(status)


def _resolved_experiment_fingerprint(
    *,
    target: ControllerDefinition,
    repo_root: Path,
) -> str:
    experiment_path = Path(target.experiment_config)
    if not experiment_path.is_absolute():
        experiment_path = repo_root / experiment_path
    experiment = load_config(experiment_path)
    try:
        vehicle = experiment.vehicles[target.vehicle]
    except KeyError as exc:
        raise ValueError(
            f"alignment vehicle {target.vehicle!r} is absent from " f"{target.experiment_config}"
        ) from exc
    mismatches = []
    if Path(vehicle.spawn.usd_path).name != target.airframe:
        mismatches.append(f"airframe {Path(vehicle.spawn.usd_path).name!r} != {target.airframe!r}")
    if experiment.rl.policy_rate_hz != target.command_rate_hz:
        mismatches.append(
            f"command cadence {experiment.rl.policy_rate_hz} != " f"{target.command_rate_hz}"
        )
    action = experiment.rl.action
    if action.kind != "velocity" or action.action_dim != target.command_interface.dimension:
        mismatches.append(f"action contract {(action.kind, action.action_dim)!r} is incompatible")
    if (
        action.max_body_velocity_mps != target.command_interface.linear_scale_mps
        or action.max_yaw_rate_radps != target.command_interface.yaw_rate_scale_radps
    ):
        mismatches.append("normalized command scales differ")
    if mismatches:
        raise ValueError(
            "controller definition is incompatible with its resolved experiment: "
            + "; ".join(mismatches)
        )
    return _fingerprint(_plain(experiment))


def align(
    target: ControllerDefinition,
    reference: ControllerDefinition,
    /,
) -> AlignmentArtifact:
    """Evaluate every declarative candidate and select it by the named metric."""
    _validate_pair(target, reference)
    reference_traces = reference.evaluator.evaluate(
        definition=reference,
        settings=reference.base_settings,
        reference_traces=None,
    )
    score_metric = metric(name=target.selection_metric)
    candidate_scores: list[CandidateScore] = []
    metric_results = {}
    for index, settings in enumerate(_settings_candidates(target)):
        try:
            target_traces = target.evaluator.evaluate(
                definition=target,
                settings=settings,
                reference_traces=reference_traces,
            )
            result = score_metric(reference_traces, target_traces)
            if not _finite_metric_result(result=result):
                raise ValueError("registered metric returned a non-finite value")
            available_metrics = (
                result.metrics.keys() | result.fit_metrics.keys() | result.validation_metrics.keys()
            )
            missing_metrics = set(target.publication_metrics) - available_metrics
            if missing_metrics:
                raise ValueError(
                    "registered metric omitted declared publication metrics "
                    f"{sorted(missing_metrics)!r}"
                )
            metric_results[index] = result
            candidate_scores.append(
                CandidateScore(
                    index=index,
                    settings=settings,
                    usable=True,
                    metrics=result.metrics,
                    ranking_values=result.ranking_values,
                    failure_reason=None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - backend failures make only this candidate unusable
            candidate_scores.append(
                CandidateScore(
                    index=index,
                    settings=settings,
                    usable=False,
                    metrics={},
                    ranking_values=(),
                    failure_reason=str(exc),
                )
            )
    usable = [candidate for candidate in candidate_scores if candidate.usable]
    if not usable:
        raise RuntimeError("every alignment candidate was unusable; no artifact was published")
    selected = min(usable, key=lambda candidate: (candidate.ranking_values, candidate.index))
    selected_result = metric_results[selected.index]
    quality_reasons = []
    if selected_result.validation_metrics["worst_axis_velocity_nrmse"] > 0.15:
        quality_reasons.append("worst-axis validation velocity NRMSE exceeds 15%")
    repo_root = Path(__file__).resolve().parents[3]
    source_commit, source_dirty = _git_source(repo_root=repo_root)
    experiment_fingerprint = _resolved_experiment_fingerprint(
        target=target,
        repo_root=repo_root,
    )
    probe_suite_fingerprint = _fingerprint([asdict(profile) for profile in target.probes])
    comparison_fingerprint = _fingerprint(
        {
            "target_definition": target.definition_fingerprint,
            "reference_definition": reference.definition_fingerprint,
            "probe_suite": probe_suite_fingerprint,
            "selection_metric": target.selection_metric,
            "experiment": experiment_fingerprint,
            "interface": asdict(target.command_interface),
            "target_image": target.image,
            "reference_image": reference.image,
        }
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return AlignmentArtifact(
        schema_version=1,
        alignment_id=target.alignment_id,
        run_id=run_id,
        target_controller_id=target.controller_id,
        reference_controller_id=reference.controller_id,
        target_backend=target.backend,
        reference_backend=reference.backend,
        target_definition_fingerprint=target.definition_fingerprint,
        reference_definition_fingerprint=reference.definition_fingerprint,
        comparison_fingerprint=comparison_fingerprint,
        experiment_config=target.experiment_config,
        experiment_fingerprint=experiment_fingerprint,
        vehicle=target.vehicle,
        airframe=target.airframe,
        command_rate_hz=target.command_rate_hz,
        command_interface=target.command_interface,
        probe_suite_fingerprint=probe_suite_fingerprint,
        selection_metric=target.selection_metric,
        selected_settings=selected.settings,
        declared_parameters=target.parameters,
        candidates=tuple(candidate_scores),
        selected_candidate_index=selected.index,
        selected_ranking_values=selected.ranking_values,
        fit_metrics=selected_result.fit_metrics,
        validation_metrics=selected_result.validation_metrics,
        advisory_quality_passed=not quality_reasons,
        advisory_quality_reasons=tuple(quality_reasons),
        source_commit=source_commit,
        source_dirty=source_dirty,
        target_image=target.image,
        reference_image=reference.image,
    )
