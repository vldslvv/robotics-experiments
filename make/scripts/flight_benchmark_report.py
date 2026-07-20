#!/usr/bin/env python3
"""Validate, summarize, and compare full-GUI flight benchmark runs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import re
import statistics
from collections.abc import Mapping
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = ["main"]

_PHASES = ("connected_disarmed", "armed_takeoff", "airborne")
_REPETITIONS = 3


def _read_json(path: Path, /) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _canonical_config(value: Any, /) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_config(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.name != "_config_frozen"
        }
    if isinstance(value, Mapping):
        return [[str(key), _canonical_config(child)] for key, child in value.items()]
    if isinstance(value, (list, tuple)):
        return [_canonical_config(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _canonical_config(value.value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported resolved-config value {type(value).__name__}")


def _fingerprint(args: argparse.Namespace, /) -> int:
    from robotics_experiments.config import load_config

    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    resolved = _canonical_config(config)
    encoded = json.dumps(resolved, ensure_ascii=True, separators=(",", ":"))
    enabled = [
        (label, vehicle) for label, vehicle in config.vehicles.items() if vehicle.bridge.enabled
    ]
    if not enabled:
        raise ValueError("flight benchmark requires at least one bridge-enabled vehicle")
    try:
        portable_config_path = str(config_path.relative_to(Path.cwd()))
    except ValueError:
        portable_config_path = str(config_path)
    payload = {
        "config_path": portable_config_path,
        "config_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "vehicle_labels": [label for label, _vehicle in enabled],
        "isaac_image": args.isaac_image,
        "isaac_image_id": args.isaac_image_id,
        "px4_image": args.px4_image,
        "px4_image_id": args.px4_image_id,
        "px4_revision": args.px4_revision,
        "source_commit": args.source_commit,
        "px4_readiness_timeout_s": max(
            float(vehicle.routine.bridge_heartbeat_timeout_s) for _label, vehicle in enabled
        ),
    }
    _write_json(Path(args.output).resolve(), payload)
    print(math.ceil(payload["px4_readiness_timeout_s"]), payload["config_sha256"])
    return 0


def _finite(values: Any, label: str, /) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} has no samples")
    samples = [float(value) for value in values]
    if not all(math.isfinite(value) for value in samples):
        raise ValueError(f"{label} contains a non-finite sample")
    return samples


def _percentile(values: list[float], fraction: float, /) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _summary(values: list[float], /) -> dict[str, float]:
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _pearson(xs: list[float], ys: list[float], /) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator if denominator else 0.0


def _nvidia_samples(root: Path, /) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {phase: {} for phase in _PHASES}
    for path in sorted((root / "nvidia").glob("metrics_*.json")):
        for phase in _read_json(path):
            phase_name = str(phase.get("phase_name", ""))
            if phase_name not in result:
                continue
            for measurement in phase.get("measurements", []):
                name = str(measurement.get("name", ""))
                values = measurement.get("value")
                if not isinstance(values, list) or not values:
                    continue
                if name.endswith("App_Update Frametime Samples"):
                    result[phase_name]["app_frametime_ms"] = _finite(values, name)
                elif name.endswith("GPU Frametime Samples"):
                    result[phase_name]["gpu_frametime_ms"] = _finite(values, name)
                elif name.endswith("Physics Frametime Samples"):
                    result[phase_name]["physx_frametime_ms"] = _finite(values, name)
    return result


def _memory_mb(value: str, /) -> float:
    match = re.fullmatch(r"([0-9.]+)\s*([A-Za-z]+)", value.strip())
    if match is None:
        raise ValueError(f"invalid Docker memory value {value!r}")
    number, unit = match.groups()
    multipliers = {
        "B": 1.0 / (1024.0 * 1024.0),
        "KiB": 1.0 / 1024.0,
        "MiB": 1.0,
        "GiB": 1024.0,
    }
    return float(number) * multipliers[unit]


def _container_resources(root: Path, /) -> dict[str, Any]:
    groups: dict[str, dict[str, list[float]]] = {
        "isaac": {"cpu_percent": [], "memory_mb": []},
        "px4_fleet": {"cpu_percent": [], "memory_mb": []},
    }
    path = root / "container-stats.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        name = str(sample["Name"])
        group = "isaac" if name.endswith("-isaac") else "px4_fleet"
        groups[group]["cpu_percent"].append(float(str(sample["CPUPerc"]).rstrip("%")))
        memory_used = str(sample["MemUsage"]).partition("/")[0].strip()
        groups[group]["memory_mb"].append(_memory_mb(memory_used))
    for group, metrics in groups.items():
        if not metrics["cpu_percent"] or not metrics["memory_mb"]:
            raise ValueError(f"{path} has no {group} resource samples")
    return {
        group: {metric: _summary(values) for metric, values in metrics.items()}
        for group, metrics in groups.items()
    }


def _slow_frame_correlations(
    *, frames: list[dict[str, Any]], components: list[dict[str, Any]]
) -> dict[str, Any]:
    valid_frames = [frame for frame in frames if float(frame["interval_ms"]) > 0.0]
    intervals = {int(frame["frame"]): float(frame["interval_ms"]) for frame in valid_frames}
    component_by_frame: dict[int, dict[str, float]] = {}
    for sample in components:
        frame = int(sample["frame"])
        metric = str(sample["metric"])
        per_metric = component_by_frame.setdefault(frame, {})
        per_metric[metric] = per_metric.get(metric, 0.0) + float(sample["duration_ms"])
    threshold = _percentile(list(intervals.values()), 0.95)
    slow_frames = {frame for frame, interval in intervals.items() if interval >= threshold}
    metrics = sorted({metric for values in component_by_frame.values() for metric in values})
    correlations = {}
    for metric in metrics:
        common = sorted(set(intervals) & set(component_by_frame))
        xs = [intervals[frame] for frame in common]
        ys = [component_by_frame[frame].get(metric, 0.0) for frame in common]
        slow_values = [
            component_by_frame[frame].get(metric, 0.0)
            for frame in slow_frames
            if frame in component_by_frame
        ]
        correlations[metric] = {
            "pearson": _pearson(xs, ys),
            "all_frame_mean_ms": statistics.fmean(ys) if ys else 0.0,
            "slow_frame_mean_ms": statistics.fmean(slow_values) if slow_values else 0.0,
        }
    return {
        "slow_frame_threshold_ms": threshold,
        "slow_frame_count": len(slow_frames),
        "components": correlations,
    }


def _summarize_phase(
    *,
    raw: dict[str, Any],
    phase_name: str,
    nvidia: dict[str, list[float]],
    measurement_level: str,
) -> dict[str, Any]:
    phase = raw["phases"][phase_name]
    sim_elapsed_s = float(phase["end_sim_time_s"]) - float(phase["start_sim_time_s"])
    wall_elapsed_s = float(phase["wall_elapsed_s"])
    if wall_elapsed_s <= 0.0 or sim_elapsed_s < 0.0:
        raise ValueError(f"{phase_name} has invalid elapsed timing")
    result: dict[str, Any] = {
        "sim_elapsed_s": sim_elapsed_s,
        "wall_elapsed_s": wall_elapsed_s,
        "real_time_factor": sim_elapsed_s / wall_elapsed_s,
    }
    if measurement_level in ("nvidia", "full"):
        frames = phase["app_frames"]
        intervals = _finite(
            [frame["interval_ms"] for frame in frames if float(frame["interval_ms"]) > 0.0],
            f"{phase_name} app frames",
        )
        physics_steps = _finite(
            [frame["physics_steps"] for frame in frames],
            f"{phase_name} physics steps per frame",
        )
        budget_ms = 1000.0 / float(raw["metadata"]["rendering_hz"])
        violations = sum(value > budget_ms for value in intervals)
        result.update(
            {
                "app_frame_ms": _summary(intervals),
                "physics_steps_per_frame": _summary(physics_steps),
                "frame_budget_ms": budget_ms,
                "frame_budget_violation_count": violations,
                "frame_budget_violation_percent": 100.0 * violations / len(intervals),
            }
        )
        render_values = phase.get("render_intervals_ms", [])
        result["render_interval_ms"] = _summary(
            _finite(render_values, f"{phase_name} render intervals")
        )
        result["nvidia_ms"] = {
            metric: _summary(values) for metric, values in sorted(nvidia.items())
        }
        expected_nvidia = {"app_frametime_ms", "gpu_frametime_ms", "physx_frametime_ms"}
        missing_nvidia = sorted(expected_nvidia - set(nvidia))
        if missing_nvidia:
            raise ValueError(f"{phase_name} is missing NVIDIA metrics {missing_nvidia}")
    if measurement_level == "full":
        resources = [sample for sample in raw["resource_samples"] if sample["phase"] == phase_name]
        components = [
            sample for sample in raw["component_samples"] if sample["phase"] == phase_name
        ]
        by_metric: dict[str, list[dict[str, Any]]] = {}
        for sample in components:
            by_metric.setdefault(str(sample["metric"]), []).append(sample)
        component_metrics = {}
        for metric, samples in sorted(by_metric.items()):
            entities: dict[str, list[float]] = {}
            for sample in samples:
                entities.setdefault(str(sample["entity"]), []).append(float(sample["duration_ms"]))
            component_metrics[metric] = {
                "all": _summary([float(sample["duration_ms"]) for sample in samples]),
                "entities": {
                    entity: _summary(values) for entity, values in sorted(entities.items())
                },
            }
        result["process_cpu_percent"] = _summary(
            _finite([sample["cpu_percent"] for sample in resources], f"{phase_name} CPU")
        )
        result["process_rss_mb"] = _summary(
            _finite([sample["rss_mb"] for sample in resources], f"{phase_name} RSS")
        )
        result["components_ms"] = component_metrics
        result["slow_frame_correlations"] = _slow_frame_correlations(
            frames=frames,
            components=components,
        )
    return result


def _summarize_repeat(path: Path, /) -> dict[str, Any]:
    raw = _read_json(path)
    if raw.get("schema_version") != 1 or raw.get("status") != "complete":
        raise ValueError(f"{path} is not a complete schema-1 flight sample")
    if set(raw.get("phases", {})) != set(_PHASES):
        raise ValueError(f"{path} phases are not {_PHASES}")
    if float(raw["phases"]["airborne"]["end_sim_time_s"]) - float(
        raw["phases"]["airborne"]["start_sim_time_s"]
    ) < float(raw["metadata"]["airborne_duration_s"]):
        raise ValueError(f"{path} captured less than its configured airborne duration")
    nvidia = _nvidia_samples(path.parent)
    measurement_level = str(raw["metadata"].get("measurement_level", "full"))
    if measurement_level not in ("lifecycle", "nvidia", "full"):
        raise ValueError(f"unsupported measurement level {measurement_level!r}")
    return {
        "source": str(path.parent),
        "metadata": raw["metadata"],
        "environment": raw["environment"],
        "phases": {
            phase: _summarize_phase(
                raw=raw,
                phase_name=phase,
                nvidia=nvidia[phase],
                measurement_level=measurement_level,
            )
            for phase in _PHASES
        },
        "container_resources": _container_resources(path.parent),
    }


def _contract(repeat: dict[str, Any], fingerprint: dict[str, Any], /) -> dict[str, Any]:
    metadata = repeat["metadata"]
    contract = {
        "methodology_version": metadata["methodology_version"],
        "scene": metadata["scene"],
        "config_sha256": metadata["config_sha256"],
        "physics_hz": metadata["physics_hz"],
        "rendering_hz": metadata["rendering_hz"],
        "solver_type": metadata["solver_type"],
        "renderer": metadata["renderer"],
        "render_resolution": metadata["render_resolution"],
        "vehicle_count": metadata["vehicle_count"],
        "vehicle_labels": metadata["vehicle_labels"],
        "vehicle_ids": metadata["vehicle_ids"],
        "airborne_duration_s": metadata["airborne_duration_s"],
        "px4_revision": fingerprint["px4_revision"],
        "measurement_level": metadata["measurement_level"],
    }
    return contract


def _hardware(environment: dict[str, Any], /) -> dict[str, Any]:
    return {
        "cpu_model": environment["cpu_model"],
        "cpu_governor": environment["cpu_governor"],
        "logical_cpus": environment["logical_cpus"],
        "gpu_models": environment["gpu_models"],
    }


def _median_tree(values: list[Any], /) -> Any:
    first = values[0]
    if isinstance(first, dict):
        expected = set(first)
        if any(not isinstance(value, dict) or set(value) != expected for value in values[1:]):
            raise ValueError("measured repetitions expose different metric trees")
        return {key: _median_tree([value[key] for value in values]) for key in sorted(expected)}
    if isinstance(first, (int, float)) and all(isinstance(value, (int, float)) for value in values):
        return statistics.median(float(value) for value in values)
    if any(value != first for value in values[1:]):
        raise ValueError("measured repetitions expose different non-numeric metric values")
    return first


def _trace(profile_root: Path, run_root: Path, /) -> str:
    traces = sorted(profile_root.rglob("*.tracy"))
    if len(traces) != 1 or traces[0].stat().st_size == 0:
        raise ValueError(f"profile run must contain one non-empty .tracy file, found {len(traces)}")
    return str(traces[0].relative_to(run_root))


def _markdown_report(report: dict[str, Any], /) -> str:
    lines = [
        "# Full-GUI flight benchmark",
        "",
        f"Status: **{report['status']}** (diagnostic; no acceptance gate)",
        "",
        f"Isaac image: `{report['environment']['isaac_image']}`",
        "",
        f"Measurement level: `{report['contract'].get('measurement_level', 'full')}`",
        "",
        f"Tracy trace: `{report['profile_trace']}`",
        "",
        "| Phase | App p50 ms | App p95 ms | App max ms | RTF | " "Budget violations |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for phase in _PHASES:
        metric = report["metrics"]["phases"][phase]
        app = metric.get("app_frame_ms")
        if app is None:
            lines.append(f"| {phase} | n/a | n/a | n/a | {metric['real_time_factor']:.3f} | n/a |")
            continue
        lines.append(
            f"| {phase} | {app['p50']:.3f} | {app['p95']:.3f} | {app['max']:.3f} | "
            f"{metric['real_time_factor']:.3f} | {metric['frame_budget_violation_percent']:.1f}% |"
        )
    components = report["metrics"]["phases"]["airborne"].get("components_ms", {})
    if components:
        lines.extend(["", "## Airborne component p95", ""])
        for name, metric in components.items():
            lines.append(f"- {name}: {metric['all']['p95']:.3f} ms")
    return "\n".join(lines) + "\n"


def _build(args: argparse.Namespace, /) -> int:
    raw_root = Path(args.raw_root).resolve()
    run_root = Path(args.run_root).resolve()
    fingerprint = _read_json(Path(args.fingerprint).resolve())
    report: dict[str, Any]
    try:
        sample_paths = sorted(raw_root.glob("repeat-*/samples.json"))
        if len(sample_paths) != _REPETITIONS:
            raise ValueError(
                f"found {len(sample_paths)} measured repetitions; expected {_REPETITIONS}"
            )
        repeats = [_summarize_repeat(path) for path in sample_paths]
        metadata = repeats[0]["metadata"]
        environment = repeats[0]["environment"]
        for repeat in repeats[1:]:
            repeat_contract = {
                key: value for key, value in repeat["metadata"].items() if key != "repetition"
            }
            metadata_contract = {
                key: value for key, value in metadata.items() if key != "repetition"
            }
            if repeat_contract != metadata_contract:
                raise ValueError("measured repetitions have different methodology metadata")
            if repeat["environment"] != environment:
                raise ValueError("measured repetitions have different environments")
        if metadata["config_sha256"] != fingerprint["config_sha256"]:
            raise ValueError("measured repetitions do not match the resolved-config fingerprint")
        if metadata["vehicle_labels"] != fingerprint["vehicle_labels"]:
            raise ValueError("measured vehicle order does not match the fingerprint")
        fingerprint_environment = {
            key: fingerprint[key]
            for key in (
                "isaac_image",
                "isaac_image_id",
                "px4_image",
                "px4_image_id",
                "source_commit",
            )
        }
        if {key: environment[key] for key in fingerprint_environment} != fingerprint_environment:
            raise ValueError("measured environment does not match the run fingerprint")
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "diagnostic_only": True,
            "repetitions": _REPETITIONS,
            "environment": environment,
            "hardware": _hardware(environment),
            "contract": _contract(repeats[0], fingerprint),
            "fingerprint": fingerprint,
            "profile_trace": _trace(Path(args.profile_root).resolve(), run_root),
            "metrics": _median_tree(
                [
                    {
                        "phases": repeat["phases"],
                        "container_resources": repeat["container_resources"],
                    }
                    for repeat in repeats
                ]
            ),
            "repeats": repeats,
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "invalid",
            "diagnostic_only": True,
            "reason": str(exc),
        }
    _write_json(run_root / "report.json", report)
    if report["status"] == "complete":
        (run_root / "report.md").write_text(_markdown_report(report), encoding="utf-8")
        print(f"Flight benchmark report: {run_root / 'report.md'}")
        return 0
    (run_root / "report.md").write_text(
        f"# Full-GUI flight benchmark\n\nStatus: **invalid**\n\nReason: {report['reason']}\n",
        encoding="utf-8",
    )
    return 2


def _numeric_leaves(value: Any, prefix: str = "", /) -> dict[str, float]:
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_numeric_leaves(child, child_prefix))
        return result
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return {prefix: float(value)}
    return {}


def _markdown_comparison(comparison: dict[str, Any], /) -> str:
    by_metric = {item["metric"]: item for item in comparison["metrics"]}
    lines = [
        "# Isaac flight benchmark comparison",
        "",
        f"Status: **{comparison['status']}**",
        "",
        "## Airborne summary",
        "",
        "| Metric | Reference | Candidate | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    summary_metrics = (
        "phases.airborne.app_frame_ms.p50",
        "phases.airborne.app_frame_ms.p95",
        "phases.airborne.nvidia_ms.gpu_frametime_ms.p95",
        "phases.airborne.nvidia_ms.physx_frametime_ms.p95",
        "phases.airborne.real_time_factor",
    )
    for metric in summary_metrics:
        item = by_metric[metric]
        change = (
            f"{item['percent_change']:+.1f}%"
            if item["percent_change"] is not None
            else f"{item['delta']:+.4f} (absolute)"
        )
        lines.append(
            f"| {item['metric']} | {item['reference']:.4f} | {item['candidate']:.4f} | "
            f"{change} |"
        )
    diagnostics = comparison["airborne_deltas_ms"]
    lines.extend(
        [
            "",
            "Airborne p95 deltas: "
            f"app {diagnostics['app']:+.3f} ms, GPU {diagnostics['gpu']:+.3f} ms, "
            f"PhysX {diagnostics['physx']:+.3f} ms.",
            "",
            "## Airborne project components",
            "",
            "| Component p95 | Reference ms | Candidate ms | Change |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    components = sorted(
        (
            item
            for item in comparison["metrics"]
            if item["metric"].startswith("phases.airborne.components_ms.")
            and item["metric"].endswith(".all.p95")
        ),
        key=lambda item: item["candidate"] - item["reference"],
        reverse=True,
    )
    for item in components:
        component = item["metric"].split(".components_ms.", 1)[1].split(".", 1)[0]
        change = (
            f"{item['percent_change']:+.1f}%"
            if item["percent_change"] is not None
            else f"{item['delta']:+.4f} (absolute)"
        )
        lines.append(
            f"| {component} | {item['reference']:.4f} | {item['candidate']:.4f} | " f"{change} |"
        )
    return "\n".join(lines) + "\n"


def _compare(args: argparse.Namespace, /) -> int:
    reference_path = Path(args.reference).resolve()
    candidate_path = Path(args.candidate).resolve()
    if reference_path.is_dir():
        reference_path /= "report.json"
    if candidate_path.is_dir():
        candidate_path /= "report.json"
    reference = _read_json(reference_path)
    candidate = _read_json(candidate_path)
    if reference.get("status") != "complete" or candidate.get("status") != "complete":
        raise ValueError("comparison requires two complete flight benchmark reports")
    if any(
        report["contract"].get("measurement_level", "full") != "full"
        for report in (reference, candidate)
    ):
        raise ValueError("cross-version comparison requires full measurement reports")
    if reference["contract"] != candidate["contract"]:
        raise ValueError("flight benchmark contracts differ")
    if reference["hardware"] != candidate["hardware"]:
        raise ValueError("flight benchmark hardware fingerprints differ")
    ref_values = _numeric_leaves(reference["metrics"])
    candidate_values = _numeric_leaves(candidate["metrics"])
    if set(ref_values) != set(candidate_values):
        raise ValueError("flight benchmark reports expose different numeric metric trees")
    comparisons = []
    for metric in sorted(ref_values):
        baseline = ref_values[metric]
        value = candidate_values[metric]
        delta = value - baseline
        percent = 100.0 * delta / baseline if baseline else None
        comparisons.append(
            {
                "metric": metric,
                "reference": baseline,
                "candidate": value,
                "delta": delta,
                "percent_change": percent,
                "ratio": value / baseline if baseline else None,
            }
        )
    regressions = sorted(
        [item for item in comparisons if item["delta"] > 0.0],
        key=lambda item: item["delta"],
        reverse=True,
    )
    component_deltas = sorted(
        [
            item
            for item in comparisons
            if item["metric"].startswith("phases.airborne.components_ms.")
            and item["metric"].endswith(".all.p95")
        ],
        key=lambda item: item["delta"],
        reverse=True,
    )
    airborne_deltas = {
        "app": (
            candidate_values["phases.airborne.app_frame_ms.p95"]
            - ref_values["phases.airborne.app_frame_ms.p95"]
        ),
        "gpu": (
            candidate_values["phases.airborne.nvidia_ms.gpu_frametime_ms.p95"]
            - ref_values["phases.airborne.nvidia_ms.gpu_frametime_ms.p95"]
        ),
        "physx": (
            candidate_values["phases.airborne.nvidia_ms.physx_frametime_ms.p95"]
            - ref_values["phases.airborne.nvidia_ms.physx_frametime_ms.p95"]
        ),
    }
    comparison = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "diagnostic_only": True,
        "reference": str(reference_path),
        "candidate": str(candidate_path),
        "airborne_deltas_ms": airborne_deltas,
        "airborne_component_p95": component_deltas,
        "metrics": comparisons,
        "largest_regressions": regressions,
    }
    output_root = Path(args.output_root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_root / f"flight-comparison.{run_id}.json"
    markdown_path = output_root / f"flight-comparison.{run_id}.md"
    _write_json(json_path, comparison)
    markdown_path.write_text(_markdown_comparison(comparison), encoding="utf-8")
    print(f"Flight comparison: {markdown_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--config", required=True)
    fingerprint.add_argument("--output", required=True)
    fingerprint.add_argument("--isaac-image", required=True)
    fingerprint.add_argument("--isaac-image-id", required=True)
    fingerprint.add_argument("--px4-image", required=True)
    fingerprint.add_argument("--px4-image-id", required=True)
    fingerprint.add_argument("--px4-revision", required=True)
    fingerprint.add_argument("--source-commit", required=True)
    fingerprint.set_defaults(action=_fingerprint)
    build = subparsers.add_parser("build")
    build.add_argument("--raw-root", required=True)
    build.add_argument("--profile-root", required=True)
    build.add_argument("--run-root", required=True)
    build.add_argument("--fingerprint", required=True)
    build.set_defaults(action=_build)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output-root", required=True)
    compare.set_defaults(action=_compare)
    return parser


def main() -> int:
    """Run the flight benchmark reporting CLI."""
    args = _parser().parse_args()
    try:
        return int(args.action(args))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"flight benchmark: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
