#!/usr/bin/env python3
"""Normalize, compare, and accept Isaac-core benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["main"]

_EXPECTED_REPETITIONS = 3
_WORKLOADS = (
    "kit_only_unlimited",
    "render_only_unlimited",
    "scene_physics_unlimited",
    "drone_physics_unlimited",
    "drone_articulation_unlimited",
    "drone_articulation_realtime",
)
_COMPONENT_SAMPLE_GROUPS = (
    "articulation_state_read",
    "rotor_visual_update",
    "dynamics_apply",
)
_TOLERANCES = {
    "drone_articulation_unlimited": {
        "app_update_p50_ms": {"direction": "max", "relative_percent": 10.0},
        "app_update_p95_ms": {"direction": "max", "relative_percent": 15.0},
        "physics_p50_ms": {"direction": "max", "relative_percent": 10.0},
        "physics_p95_ms": {"direction": "max", "relative_percent": 15.0},
        "gpu_p50_ms": {"direction": "max", "relative_percent": 10.0},
        "gpu_p95_ms": {"direction": "max", "relative_percent": 15.0},
    },
    "drone_articulation_realtime": {
        "app_update_p50_ms": {"direction": "max", "relative_percent": 10.0},
        "app_update_p95_ms": {"direction": "max", "relative_percent": 15.0},
        "real_time_factor": {
            "direction": "min",
            "relative_percent": 5.0,
            "absolute_min": 0.95,
        },
    },
}


def _read_json(path: Path, /) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _finite_samples(values: Any, label: str, /) -> list[float]:
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


def _sample_summary(values: list[float], prefix: str, /) -> dict[str, float]:
    return {
        f"{prefix}_p50_ms": _percentile(values, 0.50),
        f"{prefix}_p95_ms": _percentile(values, 0.95),
        f"{prefix}_p99_ms": _percentile(values, 0.99),
    }


def _measurement_samples(*, path: Path, suffix: str, required: bool = True) -> list[float]:
    phases = _read_json(path)
    for phase in phases:
        for measurement in phase.get("measurements", []):
            if str(measurement.get("name", "")).endswith(suffix):
                if not measurement.get("value") and not required:
                    return []
                return _finite_samples(measurement.get("value"), f"{path}: {suffix}")
    if required:
        raise ValueError(f"{path} does not contain {suffix}")
    return []


def _contract(metadata: dict[str, Any], /) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key != "workload"}


def _hardware(environment: dict[str, Any], /) -> dict[str, Any]:
    gpu_models = environment["gpu_models"]
    return {
        "cpu_model": environment["cpu_model"],
        "logical_cpus": environment["logical_cpus"],
        "cpu_governor": environment["cpu_governor"],
        "gpu_count": len(gpu_models),
        "gpu_models": gpu_models,
    }


def _summarize_repeat(samples_path: Path, /) -> dict[str, Any]:
    raw = _read_json(samples_path)
    workload = str(raw["metadata"]["workload"])
    target_app_updates = raw["metadata"]["target_app_updates"]
    target_physics_ticks = raw["metadata"]["target_physics_ticks"]
    if (target_app_updates is None) == (target_physics_ticks is None):
        raise ValueError(f"{samples_path} must declare exactly one sample target")
    app_update_count = int(raw["app_update_count"])
    physics_step_count = int(raw["physics_step_count"])
    if target_app_updates is not None and app_update_count < int(target_app_updates):
        raise ValueError(
            f"{samples_path} captured {app_update_count} app updates; "
            f"expected at least {target_app_updates}"
        )
    if target_physics_ticks is not None and physics_step_count < int(target_physics_ticks):
        raise ValueError(
            f"{samples_path} captured {physics_step_count} physics ticks; "
            f"expected at least {target_physics_ticks}"
        )
    wall_elapsed_s = float(raw["wall_elapsed_s"])
    if not math.isfinite(wall_elapsed_s) or wall_elapsed_s <= 0.0:
        raise ValueError(f"{samples_path} has invalid wall elapsed time")

    metrics_files = sorted(samples_path.parent.glob("metrics_*.json"))
    if len(metrics_files) != 1:
        raise ValueError(f"{samples_path.parent} must contain exactly one metrics_*.json file")

    app_samples = _measurement_samples(path=metrics_files[0], suffix="App_Update Frametime Samples")
    gpu_samples = _measurement_samples(
        path=metrics_files[0],
        suffix="GPU Frametime Samples",
        required=workload != "kit_only_unlimited",
    )
    physics_expected = target_physics_ticks is not None
    physics_samples = _measurement_samples(
        path=metrics_files[0],
        suffix="Physics Frametime Samples",
        required=physics_expected,
    )
    sample_groups = raw["samples_ms"]
    metrics = {
        **_sample_summary(app_samples, "app_update"),
        "wall_elapsed_s": wall_elapsed_s,
    }
    sample_counts = {"app_update": len(app_samples)}
    if gpu_samples:
        metrics.update(_sample_summary(gpu_samples, "gpu"))
        sample_counts["gpu"] = len(gpu_samples)
    render_samples = sample_groups.get("render_update_interval")
    if render_samples:
        finite_render_samples = _finite_samples(
            render_samples,
            f"{samples_path}: render update intervals",
        )
        metrics.update(_sample_summary(finite_render_samples, "render_interval"))
        sample_counts["render_interval"] = len(finite_render_samples)
    if physics_expected:
        sim_elapsed_s = float(raw["sim_elapsed_s"])
        real_time_factor = float(raw["real_time_factor"])
        if (
            not math.isfinite(sim_elapsed_s)
            or sim_elapsed_s <= 0.0
            or not math.isfinite(real_time_factor)
            or real_time_factor <= 0.0
        ):
            raise ValueError(f"{samples_path} has invalid simulated timing")
        step_samples = _finite_samples(
            sample_groups.get("physics_step_interval"),
            f"{samples_path}: physics step intervals",
        )
        metrics.update(_sample_summary(physics_samples, "physics"))
        metrics.update(_sample_summary(step_samples, "step_interval"))
        metrics.update(
            {
                "real_time_factor": real_time_factor,
                "sim_elapsed_s": sim_elapsed_s,
            }
        )
        sample_counts.update(
            {
                "physics": len(physics_samples),
                "step_interval": len(step_samples),
            }
        )

    component_groups = (
        _COMPONENT_SAMPLE_GROUPS if workload.startswith("drone_articulation_") else ()
    )
    for group in component_groups:
        component_samples = _finite_samples(
            sample_groups.get(group),
            f"{samples_path}: {group}",
        )
        metrics.update(_sample_summary(component_samples, group))
        sample_counts[group] = len(component_samples)

    return {
        "workload": workload,
        "source": str(samples_path.parent),
        "contract": _contract(raw["metadata"]),
        "hardware": _hardware(raw["environment"]),
        "environment": raw["environment"],
        "sample_counts": sample_counts,
        "app_update_count": app_update_count,
        "physics_step_count": physics_step_count,
        "metrics": metrics,
    }


def _require_consistent(repeats: list[dict[str, Any]], field: str, workload: str, /) -> Any:
    expected = repeats[0][field]
    for repeat in repeats[1:]:
        if repeat[field] != expected:
            raise ValueError(f"{workload} repetitions have different {field} values")
    return expected


def _aggregate_metrics(repeats: list[dict[str, Any]], /) -> dict[str, float]:
    names = set(repeats[0]["metrics"])
    for repeat in repeats[1:]:
        names.intersection_update(repeat["metrics"])
    return {
        name: statistics.median(float(repeat["metrics"][name]) for repeat in repeats)
        for name in sorted(names)
    }


def _copy_raw(repeats: dict[str, list[dict[str, Any]]], run_root: Path, /) -> None:
    for workload, workload_repeats in repeats.items():
        for index, repeat in enumerate(workload_repeats, start=1):
            destination = run_root / "raw" / workload / f"repeat-{index}"
            source = Path(repeat["source"])
            try:
                relative_source = source.relative_to(run_root)
            except ValueError:
                shutil.copytree(source, destination)
                relative_source = destination.relative_to(run_root)
            repeat["source"] = str(relative_source)


def _profile_trace(profile_root: Path, run_root: Path, /) -> str:
    traces = sorted(profile_root.rglob("*.tracy"))
    if len(traces) != 1:
        raise ValueError(f"profile run must contain exactly one .tracy file, found {len(traces)}")
    trace = traces[0]
    if trace.stat().st_size == 0:
        raise ValueError(f"profile trace is empty: {trace}")
    try:
        relative = trace.relative_to(run_root)
    except ValueError:
        destination = run_root / "profile" / "isaac_core.tracy"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(trace, destination)
        relative = destination.relative_to(run_root)
    return str(relative)


def _attribution_deltas(metrics: dict[str, dict[str, float]], /) -> dict[str, dict[str, float]]:
    transitions = {
        "renderer": ("kit_only_unlimited", "render_only_unlimited"),
        "scene_physics": ("render_only_unlimited", "scene_physics_unlimited"),
        "drone_physics": ("scene_physics_unlimited", "drone_physics_unlimited"),
        "articulation_io": ("drone_physics_unlimited", "drone_articulation_unlimited"),
    }
    deltas = {}
    for label, (before_name, after_name) in transitions.items():
        before = metrics[before_name]
        after = metrics[after_name]
        shared = sorted(set(before) & set(after))
        deltas[label] = {
            metric: after[metric] - before[metric]
            for metric in shared
            if metric.endswith(("_p50_ms", "_p95_ms", "_p99_ms"))
        }
    return deltas


def _comparison(
    candidate_metrics: dict[str, dict[str, float]],
    baseline_metrics: dict[str, dict[str, float]],
    tolerances: dict[str, dict[str, dict[str, Any]]],
    /,
) -> list[dict[str, Any]]:
    comparisons = []
    for workload, workload_tolerances in tolerances.items():
        for metric, tolerance in workload_tolerances.items():
            candidate = candidate_metrics[workload][metric]
            baseline = baseline_metrics[workload][metric]
            relative = float(tolerance["relative_percent"]) / 100.0
            if tolerance["direction"] == "max":
                limit = baseline * (1.0 + relative)
                passed = candidate <= limit
            else:
                limit = baseline * (1.0 - relative)
                limit = max(limit, float(tolerance.get("absolute_min", -math.inf)))
                passed = candidate >= limit
            comparisons.append(
                {
                    "workload": workload,
                    "metric": metric,
                    "baseline": baseline,
                    "candidate": candidate,
                    "limit": limit,
                    "direction": tolerance["direction"],
                    "passed": passed,
                }
            )
    return comparisons


def _markdown(report: dict[str, Any], /) -> str:
    lines = [
        "# Isaac core performance benchmark",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Isaac image: `{report['environment']['isaac_image']}`",
        "",
        f"Repetitions: {report['repetitions']}",
        "",
    ]
    reason = report.get("reason")
    if reason:
        lines.extend([f"Reason: {reason}", ""])
    if report.get("profile_trace"):
        lines.extend([f"Profiler trace: `{report['profile_trace']}`", ""])
    if report.get("comparisons"):
        lines.extend(
            [
                "| Workload | Metric | Baseline | Candidate | Limit | Result |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for item in report["comparisons"]:
            operator = "<=" if item["direction"] == "max" else ">="
            result = "PASS" if item["passed"] else "FAIL"
            lines.append(
                f"| {item['workload']} | {item['metric']} | "
                f"{item['baseline']:.4f} | {item['candidate']:.4f} | "
                f"{operator} {item['limit']:.4f} | {result} |"
            )
        lines.append("")
    lines.extend(["## Aggregate metrics", ""])
    for workload, workload_metrics in report["metrics"].items():
        lines.extend([f"### {workload}", ""])
        for name, value in workload_metrics.items():
            lines.append(f"- {name}: {value:.4f}")
        lines.append("")
    lines.extend(["## Attribution deltas", ""])
    lines.append(
        "Signed phase deltas are diagnostic only; parallel CPU/GPU work means "
        "they are not additive."
    )
    lines.append("")
    for contributor, contributor_metrics in report.get("attribution_deltas", {}).items():
        lines.extend([f"### {contributor}", ""])
        for name, value in contributor_metrics.items():
            lines.append(f"- {name}: {value:+.4f}")
        lines.append("")
    return "\n".join(lines)


def _build(args: argparse.Namespace, /) -> int:
    run_root = Path(args.run_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    profile_root = Path(args.profile_root).resolve()
    baseline_path = Path(args.baseline).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "report.json").exists():
        raise ValueError(f"benchmark report already exists: {run_root / 'report.json'}")
    report: dict[str, Any]
    try:
        discovered = [_summarize_repeat(path) for path in sorted(raw_root.rglob("samples.json"))]
        unknown_workloads = sorted({item["workload"] for item in discovered} - set(_WORKLOADS))
        if unknown_workloads:
            raise ValueError(f"unexpected benchmark workloads: {unknown_workloads}")
        repeats = {
            workload: [item for item in discovered if item["workload"] == workload]
            for workload in _WORKLOADS
        }
        for workload, workload_repeats in repeats.items():
            if len(workload_repeats) != _EXPECTED_REPETITIONS:
                raise ValueError(
                    f"{workload} has {len(workload_repeats)} repetitions; "
                    f"expected {_EXPECTED_REPETITIONS}"
                )
            _require_consistent(workload_repeats, "contract", workload)
            _require_consistent(workload_repeats, "hardware", workload)

        hardware = repeats[_WORKLOADS[0]][0]["hardware"]
        for workload in _WORKLOADS[1:]:
            if repeats[workload][0]["hardware"] != hardware:
                raise ValueError("workloads ran on different hardware")

        contracts = {
            workload: workload_repeats[0]["contract"]
            for workload, workload_repeats in repeats.items()
        }
        metrics = {
            workload: _aggregate_metrics(workload_repeats)
            for workload, workload_repeats in repeats.items()
        }
        _copy_raw(repeats, run_root)
        report = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "no_baseline",
            "repetitions": _EXPECTED_REPETITIONS,
            "environment": {
                "isaac_image": args.isaac_image,
                "isaac_image_id": args.isaac_image_id,
                "isaac_versions": sorted(
                    {
                        repeat["environment"]["isaac_version"]
                        for workload_repeats in repeats.values()
                        for repeat in workload_repeats
                    }
                ),
            },
            "hardware": hardware,
            "contracts": contracts,
            "metrics": metrics,
            "attribution_deltas": _attribution_deltas(metrics),
            "profile_trace": _profile_trace(profile_root, run_root),
            "repeats": repeats,
            "comparisons": [],
        }
        if baseline_path.exists():
            baseline = _read_json(baseline_path)
            if baseline["hardware"] != hardware:
                report["status"] = "incomparable"
                report["reason"] = "hardware fingerprint differs from the accepted baseline"
            elif baseline["contracts"] != contracts:
                report["status"] = "incomparable"
                report["reason"] = "workload contract differs from the accepted baseline"
            else:
                report["comparisons"] = _comparison(
                    metrics,
                    baseline["metrics"],
                    baseline["tolerances"],
                )
                report["status"] = (
                    "pass" if all(item["passed"] for item in report["comparisons"]) else "fail"
                )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "invalid",
            "reason": str(exc),
            "repetitions": 0,
            "environment": {
                "isaac_image": args.isaac_image,
                "isaac_image_id": args.isaac_image_id,
            },
            "metrics": {},
            "attribution_deltas": {},
            "comparisons": [],
        }

    _write_json(run_root / "report.json", report)
    (run_root / "report.md").write_text(_markdown(report), encoding="utf-8")
    history = run_root.parent / "history.jsonl"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "created_at": report["created_at"],
                    "run_root": run_root.name,
                    "status": report["status"],
                    "isaac_image": args.isaac_image,
                    "metrics": report["metrics"],
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(f"Isaac core benchmark report: {run_root / 'report.md'}")
    return 0 if report["status"] == "pass" else 1 if report["status"] == "fail" else 2


def _accept(args: argparse.Namespace, /) -> int:
    result_path = Path(args.result).resolve()
    report = _read_json(result_path)
    if report.get("status") in {"invalid", "incomparable"}:
        raise ValueError(f"cannot accept a {report.get('status')} benchmark result")
    if report.get("repetitions") != _EXPECTED_REPETITIONS:
        raise ValueError(f"benchmark result must contain {_EXPECTED_REPETITIONS} repetitions")
    profile_trace = report.get("profile_trace")
    if not isinstance(profile_trace, str) or not profile_trace:
        raise ValueError("benchmark result does not reference a profiler trace")
    trace_path = result_path.parent / profile_trace
    if not trace_path.is_file() or trace_path.stat().st_size == 0:
        raise ValueError(f"benchmark profiler trace is missing or empty: {trace_path}")

    baseline_path = Path(args.baseline).resolve()
    existing = _read_json(baseline_path) if baseline_path.exists() else None
    if existing is not None and existing["hardware"] != report["hardware"]:
        raise ValueError("result hardware differs from the existing baseline")
    baseline = {
        "schema_version": 2,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "accepted_from": result_path.parent.name,
        "repetitions": report["repetitions"],
        "environment": report["environment"],
        "hardware": report["hardware"],
        "contracts": report["contracts"],
        "metrics": report["metrics"],
        "tolerances": existing["tolerances"] if existing is not None else _TOLERANCES,
    }
    _write_json(baseline_path, baseline)
    print(f"Accepted Isaac core benchmark baseline: {baseline_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="normalize and compare one benchmark run")
    build.add_argument("--raw-root", required=True)
    build.add_argument("--profile-root", required=True)
    build.add_argument("--run-root", required=True)
    build.add_argument("--baseline", required=True)
    build.add_argument("--isaac-image", required=True)
    build.add_argument("--isaac-image-id", required=True)
    build.set_defaults(action=_build)

    accept = subparsers.add_parser("accept", help="replace the committed baseline")
    accept.add_argument("--result", required=True)
    accept.add_argument("--baseline", required=True)
    accept.set_defaults(action=_accept)
    return parser


def main() -> int:
    """Run the report command-line interface."""
    args = _parser().parse_args()
    try:
        return int(args.action(args))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"isaac core benchmark: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
