"""Generic candidate expansion, evaluation, scoring, and selection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._metrics import _finite_metric_result, metric
from robotics_experiments.alignment._models import (
    AlignmentArtifact,
    CandidateScore,
    ControllerDefinition,
    ControllerSettings,
    MetricResult,
    ProbeTrace,
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


@dataclass(frozen=True)
class _SearchOutcome:
    candidate_scores: tuple[CandidateScore, ...]
    metric_results: Mapping[int, MetricResult]
    target_traces: Mapping[ControllerSettings, tuple[ProbeTrace, ...]]
    selected: CandidateScore


def _deduplicate_settings(
    settings: list[ControllerSettings],
    /,
) -> tuple[ControllerSettings, ...]:
    return tuple(dict.fromkeys(settings))


def _staged_beam_search(
    *,
    target: ControllerDefinition,
    reference_traces: tuple[ProbeTrace, ...],
) -> _SearchOutcome:
    """Evaluate deterministic parameter stages without materializing a Cartesian sweep."""
    score_metric = metric(name=target.selection_metric)
    scores_by_settings: dict[ControllerSettings, CandidateScore] = {}
    results_by_index: dict[int, MetricResult] = {}
    traces_by_settings: dict[ControllerSettings, tuple[ProbeTrace, ...]] = {}

    def record(
        *,
        settings: ControllerSettings,
        index: int,
        target_traces: tuple[ProbeTrace, ...] | None,
        failure_reason: str | None,
    ) -> CandidateScore:
        try:
            if failure_reason is not None:
                raise RuntimeError(failure_reason)
            if target_traces is None:
                raise RuntimeError("backend returned no target traces")
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
            score = CandidateScore(
                index=index,
                settings=settings,
                usable=True,
                metrics=result.metrics,
                ranking_values=result.ranking_values,
                failure_reason=None,
            )
            results_by_index[index] = result
            if getattr(target.evaluator, "_retain_traces", True):
                traces_by_settings[settings] = target_traces
        except Exception as exc:  # noqa: BLE001 - a backend failure invalidates one candidate
            score = CandidateScore(
                index=index,
                settings=settings,
                usable=False,
                metrics={},
                ranking_values=(),
                failure_reason=str(exc),
            )
        scores_by_settings[settings] = score
        return score

    def evaluate_many(
        settings: tuple[ControllerSettings, ...],
        /,
    ) -> list[CandidateScore]:
        pending = tuple(value for value in settings if value not in scores_by_settings)
        if pending:
            start_index = len(scores_by_settings)
            batch_evaluate = getattr(target.evaluator, "evaluate_batch", None)
            if batch_evaluate is None:
                evaluations: tuple[tuple[ProbeTrace, ...] | str, ...] = tuple()
                serial_results: list[tuple[ProbeTrace, ...] | str] = []
                for value in pending:
                    try:
                        serial_results.append(
                            target.evaluator.evaluate(
                                definition=target,
                                settings=value,
                                reference_traces=reference_traces,
                            )
                        )
                    except Exception as exc:  # noqa: BLE001 - candidate-local failure
                        serial_results.append(str(exc))
                evaluations = tuple(serial_results)
            else:
                try:
                    evaluations = batch_evaluate(
                        definition=target,
                        settings=pending,
                        reference_traces=reference_traces,
                    )
                except Exception as exc:  # noqa: BLE001 - a batch failure invalidates its candidates
                    evaluations = tuple(str(exc) for _value in pending)
                if len(evaluations) != len(pending):
                    failure = (
                        "backend batch returned "
                        f"{len(evaluations)} results for {len(pending)} candidates"
                    )
                    evaluations = tuple(failure for _value in pending)
            for offset, (value, evaluation) in enumerate(zip(pending, evaluations, strict=True)):
                record(
                    settings=value,
                    index=start_index + offset,
                    target_traces=None if isinstance(evaluation, str) else evaluation,
                    failure_reason=evaluation if isinstance(evaluation, str) else None,
                )
        return [scores_by_settings[value] for value in settings]

    def evaluate(settings: ControllerSettings, /) -> CandidateScore:
        return evaluate_many((settings,))[0]

    def retain(
        scores: list[CandidateScore],
        /,
    ) -> tuple[ControllerSettings, ...]:
        ordered = sorted(scores, key=lambda score: (score.ranking_values, score.index))
        if target.search.strategy == "staged-beam-v1":
            return tuple(score.settings for score in ordered[: target.search.beam_width])
        retained_scores: list[CandidateScore] = []
        beam_width = target.search.beam_width
        objective_count = max(
            (len(results_by_index[score.index].beam_values) for score in ordered),
            default=0,
        )
        for objective_index in range(objective_count):
            candidates = [
                score
                for score in ordered
                if len(results_by_index[score.index].beam_values) > objective_index
            ]
            champion = min(
                candidates,
                key=lambda score: (
                    results_by_index[score.index].beam_values[objective_index],
                    score.ranking_values,
                    score.index,
                ),
            )
            if champion not in retained_scores:
                retained_scores.append(champion)
            if len(retained_scores) == beam_width:
                break
        for score in ordered:
            if score not in retained_scores:
                retained_scores.append(score)
            if len(retained_scores) == beam_width:
                break
        return tuple(score.settings for score in retained_scores)

    retained = (target.base_settings,)
    evaluate(target.base_settings)
    stages = tuple(sorted({parameter.stage for parameter in target.parameters}))
    for stage in stages:
        stage_parameters = tuple(
            parameter for parameter in target.parameters if parameter.stage == stage
        )
        for _pass in range(target.search.passes_per_stage):
            for parameter in stage_parameters:
                expanded = _deduplicate_settings(
                    [
                        replace(settings, **{parameter.field: candidate})
                        for settings in retained
                        for candidate in parameter.candidates
                    ]
                )
                usable = evaluate_many(expanded)
                usable = [score for score in usable if score.usable]
                if not usable:
                    raise RuntimeError(
                        f"every alignment candidate was unusable while fitting "
                        f"stage {stage} field {parameter.field!r}"
                    )
                retained = retain(usable)
    final_scores = evaluate_many(retained)
    final_usable = [score for score in final_scores if score.usable]
    if not final_usable:
        raise RuntimeError("every alignment candidate was unusable; no artifact was published")
    selected = min(
        final_usable,
        key=lambda candidate: (candidate.ranking_values, candidate.index),
    )
    candidates = tuple(sorted(scores_by_settings.values(), key=lambda candidate: candidate.index))
    return _SearchOutcome(
        candidate_scores=candidates,
        metric_results=results_by_index,
        target_traces=traces_by_settings,
        selected=selected,
    )


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
        raise ValueError("alignment requires an isaac-lab target and px4-sitl reference")
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
    resolved = _plain(experiment)
    del resolved["alignment"]["execution"]
    return _fingerprint(resolved)


def align(
    target: ControllerDefinition,
    reference: ControllerDefinition,
    /,
) -> AlignmentArtifact:
    """Run deterministic staged beam search and publish its held-out selection."""
    _validate_pair(target, reference)
    reference_traces = reference.evaluator.evaluate(
        definition=reference,
        settings=reference.base_settings,
        reference_traces=None,
    )
    outcome = _staged_beam_search(
        target=target,
        reference_traces=reference_traces,
    )
    selected = outcome.selected
    selected_result = outcome.metric_results[selected.index]
    quality_reasons = []
    quality_thresholds: tuple[tuple[str, float, str], ...] = (
        (
            "braking_distance_error_m",
            0.005,
            "braking-distance error exceeds 5 mm",
        ),
        (
            "residual_speed_error_mps",
            0.01,
            "residual-speed error exceeds 0.01 m/s",
        ),
        ("stop_time_error_s", 0.10, "stop-time error exceeds 0.10 s"),
        (
            "worst_axis_velocity_nrmse",
            0.10,
            "worst-axis velocity NRMSE exceeds 10%",
        ),
        ("yaw_rate_nrmse", 0.10, "yaw-rate NRMSE exceeds 10%"),
        ("heading_error_deg", 3.0, "heading error exceeds 3 degrees"),
        (
            "combined_endpoint_error_m",
            0.01,
            "combined-path endpoint error exceeds 10 mm",
        ),
    )
    if target.selection_metric not in (
        "policy-execution-controller-quality-v2",
        "policy-execution-controller-quality-v3",
    ):
        quality_thresholds = (
            (
                "worst_axis_velocity_nrmse",
                0.15,
                "worst-axis validation velocity NRMSE exceeds 15%",
            ),
        )
    for metric_name, threshold, reason in quality_thresholds:
        if selected_result.validation_metrics[metric_name] > threshold:
            quality_reasons.append(reason)
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
        schema_version=3,
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
        candidates=outcome.candidate_scores,
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
