"""Generic candidate expansion, evaluation, scoring, and selection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from robotics_experiments.alignment._metrics import _finite_metric_result, metric
from robotics_experiments.alignment._models import (
    AlignmentResult,
    CandidateScore,
    ControllerArtifact,
    ControllerDefinition,
    ControllerSettings,
    MetricResult,
    ParameterSpec,
    ProbeTrace,
    SettingValue,
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
    trace_sources: Mapping[ControllerSettings, ControllerSettings]
    selected: CandidateScore
    statistics: Mapping[str, int]
    generated_refinements: tuple[Mapping[str, Any], ...]


def _effective_controller_key(
    settings: ControllerSettings,
    /,
) -> tuple[tuple[str, float | tuple[float, ...]], ...]:
    """Return the exact float32 values active in the controller-v1 contract."""
    key = []
    for item in fields(settings):
        value = getattr(settings, item.name)
        if isinstance(value, tuple):
            canonical: float | tuple[float, ...] = tuple(float(np.float32(part)) for part in value)
        else:
            canonical = float(np.float32(value))
        key.append((item.name, canonical))
    return tuple(key)


def _midpoint(left: SettingValue, right: SettingValue, /) -> SettingValue | None:
    if isinstance(left, tuple) and isinstance(right, tuple):
        differing = [
            index
            for index, values in enumerate(zip(left, right, strict=True))
            if values[0] != values[1]
        ]
        if len(differing) != 1:
            return None
        index = differing[0]
        result = list(left)
        result[index] = (left[index] + right[index]) / 2.0
        return result[0], result[1], result[2]
    if isinstance(left, tuple) or isinstance(right, tuple):
        return None
    return (left + right) / 2.0


def _deduplicate_settings(
    settings: list[ControllerSettings],
    /,
) -> tuple[ControllerSettings, ...]:
    return tuple(dict.fromkeys(settings))


def _refinement_candidates(
    *,
    parameter_field: str,
    grid: tuple[SettingValue, ...],
    retained: tuple[ControllerSettings, ...],
    champions: tuple[ControllerSettings, ...],
    min_survivors: int,
) -> tuple[tuple[SettingValue, ...], tuple[Mapping[str, Any], ...]]:
    retained_values = tuple(getattr(settings, parameter_field) for settings in retained)
    counts = Counter(retained_values)
    champion_values = {getattr(settings, parameter_field) for settings in champions}
    supported = {
        value
        for value, count in counts.items()
        if count >= min_survivors or value in champion_values
    }
    candidates: list[SettingValue] = []
    records: list[Mapping[str, Any]] = []

    def add(*, left: SettingValue, right: SettingValue, reason: str) -> None:
        candidate = _midpoint(left, right)
        if candidate is None or candidate in grid or candidate in candidates:
            return
        candidates.append(candidate)
        records.append(
            {
                "field": parameter_field,
                "left": left,
                "right": right,
                "candidate": candidate,
                "reason": reason,
            }
        )

    if grid and not isinstance(grid[0], tuple):
        ordered = tuple(sorted(float(value) for value in grid if not isinstance(value, tuple)))
        for scalar_value in sorted(
            float(item) for item in supported if not isinstance(item, tuple)
        ):
            if scalar_value not in ordered:
                continue
            index = ordered.index(scalar_value)
            reason = "gate champion" if scalar_value in champion_values else "retained basin"
            if index:
                add(left=ordered[index - 1], right=scalar_value, reason=reason)
            if index + 1 < len(ordered):
                add(left=scalar_value, right=ordered[index + 1], reason=reason)
    else:
        ordered_vectors = tuple(value for value in grid if isinstance(value, tuple))
        for vector_value in (item for item in supported if isinstance(item, tuple)):
            reason = "gate champion" if vector_value in champion_values else "retained basin"
            for axis in range(3):
                axis_neighbors = sorted(
                    (
                        item
                        for item in ordered_vectors
                        if all(
                            item[other] == vector_value[other]
                            for other in range(3)
                            if other != axis
                        )
                    ),
                    key=lambda item: item[axis],
                )
                if vector_value not in axis_neighbors:
                    continue
                index = axis_neighbors.index(vector_value)
                if index:
                    add(left=axis_neighbors[index - 1], right=vector_value, reason=reason)
                if index + 1 < len(axis_neighbors):
                    add(left=vector_value, right=axis_neighbors[index + 1], reason=reason)
    return tuple(candidates), tuple(records)


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
    trace_sources: dict[ControllerSettings, ControllerSettings] = {}
    settings_by_effective_key: dict[
        tuple[tuple[str, float | tuple[float, ...]], ...], ControllerSettings
    ] = {}
    statistics = {
        "attempted_settings": 0,
        "exact_cache_hits": 0,
        "effective_key_aliases": 0,
        "submitted_unique_candidates": 0,
        "skipped_fixed_point_passes": 0,
    }
    generated_refinements: list[Mapping[str, Any]] = []
    next_index = 0

    def record(
        *,
        settings: ControllerSettings,
        index: int,
        target_traces: tuple[ProbeTrace, ...] | None,
        metric_result: MetricResult | None,
        failure_reason: str | None,
    ) -> CandidateScore:
        try:
            if failure_reason is not None:
                raise RuntimeError(failure_reason)
            if target_traces is None and metric_result is None:
                raise RuntimeError("backend returned no target traces")
            result = (
                metric_result
                if metric_result is not None
                else score_metric(reference_traces, target_traces)
            )
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
                metrics={
                    **result.fit_metrics,
                    **result.validation_metrics,
                    **result.metrics,
                },
                ranking_values=result.ranking_values,
                failure_reason=None,
            )
            results_by_index[index] = result
            if getattr(target.evaluator, "_retain_traces", True):
                assert target_traces is not None
                traces_by_settings[settings] = target_traces
            trace_sources[settings] = settings
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

    def record_alias(
        *,
        settings: ControllerSettings,
        index: int,
        source_settings: ControllerSettings,
    ) -> CandidateScore:
        source = scores_by_settings[source_settings]
        score = CandidateScore(
            index=index,
            settings=settings,
            usable=source.usable,
            metrics=source.metrics,
            ranking_values=source.ranking_values,
            failure_reason=source.failure_reason,
        )
        scores_by_settings[settings] = score
        if source.usable:
            results_by_index[index] = results_by_index[source.index]
            trace_sources[settings] = trace_sources[source_settings]
            if source_settings in traces_by_settings:
                traces_by_settings[settings] = traces_by_settings[source_settings]
        return score

    def evaluate_many(
        settings: tuple[ControllerSettings, ...],
        /,
    ) -> list[CandidateScore]:
        nonlocal next_index
        statistics["attempted_settings"] += len(settings)
        statistics["exact_cache_hits"] += sum(value in scores_by_settings for value in settings)
        pending: list[tuple[ControllerSettings, int]] = []
        aliases: list[tuple[ControllerSettings, int, ControllerSettings]] = []
        reserved_by_key: dict[
            tuple[tuple[str, float | tuple[float, ...]], ...], ControllerSettings
        ] = {}
        for value in settings:
            if value in scores_by_settings:
                continue
            index = next_index
            next_index += 1
            key = _effective_controller_key(value)
            source = settings_by_effective_key.get(key) or reserved_by_key.get(key)
            if source is None:
                pending.append((value, index))
                reserved_by_key[key] = value
            else:
                aliases.append((value, index, source))
                statistics["effective_key_aliases"] += 1
        if pending:
            pending_settings = tuple(value for value, _index in pending)
            statistics["submitted_unique_candidates"] += len(pending_settings)
            score_evaluate = getattr(target.evaluator, "evaluate_scored_batch", None)
            batch_evaluate = getattr(target.evaluator, "evaluate_batch", None)
            if score_evaluate is not None:
                try:
                    evaluations = score_evaluate(
                        definition=target,
                        settings=pending_settings,
                        reference_traces=reference_traces,
                    )
                except Exception as exc:  # noqa: BLE001 - a batch failure invalidates candidates
                    evaluations = tuple(str(exc) for _value in pending_settings)
            elif batch_evaluate is None:
                evaluations = tuple()
                serial_results: list[tuple[ProbeTrace, ...] | str] = []
                for value in pending_settings:
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
                        settings=pending_settings,
                        reference_traces=reference_traces,
                    )
                except Exception as exc:  # noqa: BLE001 - a batch failure invalidates its candidates
                    evaluations = tuple(str(exc) for _value in pending_settings)
                if len(evaluations) != len(pending_settings):
                    failure = (
                        "backend batch returned "
                        f"{len(evaluations)} results for {len(pending_settings)} candidates"
                    )
                    evaluations = tuple(failure for _value in pending_settings)
            for (value, index), evaluation in zip(pending, evaluations, strict=True):
                record(
                    settings=value,
                    index=index,
                    target_traces=(evaluation if isinstance(evaluation, tuple) else None),
                    metric_result=(evaluation if isinstance(evaluation, MetricResult) else None),
                    failure_reason=evaluation if isinstance(evaluation, str) else None,
                )
                key = _effective_controller_key(value)
                settings_by_effective_key[key] = value
        for value, index, source in aliases:
            record_alias(settings=value, index=index, source_settings=source)
        return [scores_by_settings[value] for value in settings]

    def evaluate(settings: ControllerSettings, /) -> CandidateScore:
        return evaluate_many((settings,))[0]

    initial_settings = _deduplicate_settings(
        [target.base_settings, *target.search.initial_settings]
    )
    retained = initial_settings

    def retain(
        scores: list[CandidateScore],
        /,
    ) -> tuple[ControllerSettings, ...]:
        ordered = sorted(scores, key=lambda score: (score.ranking_values, score.index))
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

    def objective_champions(
        scores: list[CandidateScore],
        /,
    ) -> tuple[ControllerSettings, ...]:
        ordered = sorted(scores, key=lambda score: (score.ranking_values, score.index))
        objective_count = max(
            (len(results_by_index[score.index].beam_values) for score in ordered),
            default=0,
        )
        champions = []
        for objective_index in range(objective_count):
            eligible = [
                score
                for score in ordered
                if len(results_by_index[score.index].beam_values) > objective_index
            ]
            champion = min(
                eligible,
                key=lambda score: (
                    results_by_index[score.index].beam_values[objective_index],
                    score.ranking_values,
                    score.index,
                ),
            )
            if champion.settings not in champions:
                champions.append(champion.settings)
        return tuple(champions)

    def effective_beam(
        settings: tuple[ControllerSettings, ...],
        /,
    ) -> tuple[tuple[tuple[str, float | tuple[float, ...]], ...], ...]:
        return tuple(_effective_controller_key(value) for value in settings)

    def sweep_parameter(*, parameter: ParameterSpec, candidates: tuple[SettingValue, ...]) -> None:
        nonlocal retained
        expanded = _deduplicate_settings(
            [
                *retained,
                *(
                    replace(settings, **{parameter.field: candidate})
                    for settings in retained
                    for candidate in candidates
                ),
            ]
        )
        evaluated = evaluate_many(expanded)
        usable = [score for score in evaluated if score.usable]
        if not usable:
            raise RuntimeError(
                f"every alignment candidate was unusable while fitting "
                f"stage {parameter.stage} field {parameter.field!r}"
            )
        usable = [score for score in scores_by_settings.values() if score.usable]
        retained = retain(usable)

    initial_usable = [score for score in evaluate_many(initial_settings) if score.usable]
    if not initial_usable:
        raise RuntimeError("every initial alignment setting was unusable")
    retained = tuple(score.settings for score in initial_usable)
    grids = {
        parameter.field: tuple(
            dict.fromkeys(
                (
                    *parameter.candidates,
                    *(getattr(settings, parameter.field) for settings in initial_settings),
                )
            )
        )
        for parameter in target.parameters
    }
    stages = tuple(sorted({parameter.stage for parameter in target.parameters}))
    for stage in stages:
        stage_parameters = tuple(
            parameter for parameter in target.parameters if parameter.stage == stage
        )
        for pass_index in range(target.search.passes_per_stage):
            before = effective_beam(retained)
            for parameter in stage_parameters:
                sweep_parameter(parameter=parameter, candidates=parameter.candidates)
            if effective_beam(retained) == before:
                statistics["skipped_fixed_point_passes"] += (
                    target.search.passes_per_stage - pass_index - 1
                )
                break
        for level in range(1, target.search.refinement_levels + 1):
            before = effective_beam(retained)
            all_usable = [score for score in scores_by_settings.values() if score.usable]
            champions = objective_champions(all_usable)
            generated_any = False
            for parameter in stage_parameters:
                candidates, records = _refinement_candidates(
                    parameter_field=parameter.field,
                    grid=grids[parameter.field],
                    retained=retained,
                    champions=champions,
                    min_survivors=target.search.min_refinement_survivors,
                )
                if not candidates:
                    continue
                generated_any = True
                generated_refinements.extend(
                    {**record, "stage": stage, "level": level} for record in records
                )
                grids[parameter.field] = tuple(
                    dict.fromkeys((*grids[parameter.field], *candidates))
                )
                sweep_parameter(parameter=parameter, candidates=candidates)
            if not generated_any or effective_beam(retained) == before:
                statistics["skipped_fixed_point_passes"] += target.search.refinement_levels - level
                break
    final_scores = list(scores_by_settings.values())
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
        trace_sources=trace_sources,
        selected=selected,
        statistics=statistics,
        generated_refinements=tuple(generated_refinements),
    )


def _validate_pair(
    target: ControllerDefinition,
    reference: ControllerDefinition,
    /,
) -> None:
    mismatches = []
    for name in (
        "alignment_id",
        "controller_schema_version",
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
) -> AlignmentResult:
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
        (
            "vertical_policy_envelope_nrmse",
            0.10,
            "vertical policy-envelope velocity NRMSE exceeds 10%",
        ),
        ("yaw_rate_nrmse", 0.10, "yaw-rate NRMSE exceeds 10%"),
        ("heading_error_deg", 3.0, "heading error exceeds 3 degrees"),
        (
            "combined_endpoint_error_m",
            0.01,
            "combined-path endpoint error exceeds 10 mm",
        ),
    )
    if target.selection_metric != "controller-quality":
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
            "controller_artifact_schema": 1,
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
    controller = ControllerArtifact(
        artifact_kind="robotics-experiments-controller",
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
        selected_settings=selected.settings,
        settings_fingerprint=_fingerprint(asdict(selected.settings)),
        advisory_quality_passed=not quality_reasons,
        advisory_quality_reasons=tuple(quality_reasons),
        source_commit=source_commit,
        source_dirty=source_dirty,
        target_image=target.image,
        reference_image=reference.image,
    )
    return AlignmentResult(
        controller=controller,
        selection_metric=target.selection_metric,
        base_settings=target.base_settings,
        declared_parameters=target.parameters,
        candidates=outcome.candidate_scores,
        selected_candidate_index=selected.index,
    )
