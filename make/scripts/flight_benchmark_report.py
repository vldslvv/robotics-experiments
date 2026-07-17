#!/usr/bin/env python3
"""Normalize, compare, and accept Isaac/PX4 flight benchmark results."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = ["main"]

_EXPECTED_REPETITIONS = 3
_WORKLOADS = ("physics_only", "px4_flight")
_STARTUP_PATTERN = re.compile(r"stage=simulation_app_start elapsed_s=([0-9.]+)")
_TOLERANCES = {
    "physics_only": {
        "physics_p50_ms": {"direction": "max", "relative_percent": 10.0},
        "physics_p95_ms": {"direction": "max", "relative_percent": 15.0},
        "step_interval_p50_ms": {"direction": "max", "relative_percent": 10.0},
        "step_interval_p95_ms": {"direction": "max", "relative_percent": 15.0},
    },
    "px4_flight": {
        "step_interval_p50_ms": {"direction": "max", "relative_percent": 10.0},
        "step_interval_p95_ms": {"direction": "max", "relative_percent": 15.0},
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


def _measurement_samples(path: Path, suffix: str, /) -> list[float]:
    phases = _read_json(path)
    for phase in phases:
        for measurement in phase.get("measurements", []):
            if str(measurement.get("name", "")).endswith(suffix):
                return _finite_samples(measurement.get("value"), f"{path}: {suffix}")
    raise ValueError(f"{path} does not contain {suffix}")


def _startup_seconds(integration_run_root: Path, /) -> float | None:
    for path in sorted(integration_run_root.glob("isaac/logs/*.log")):
        match = _STARTUP_PATTERN.search(path.read_text(encoding="utf-8", errors="replace"))
        if match:
            return float(match.group(1))
    return None


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


def _summarize_repeat(step_path: Path, /) -> dict[str, Any]:
    step = _read_json(step_path)
    workload = str(step["metadata"]["workload"])
    metrics_files = sorted(step_path.parent.glob("metrics_*.json"))
    if len(metrics_files) != 1:
        raise ValueError(f"{step_path.parent} must contain exactly one metrics_*.json file")

    app_samples = _measurement_samples(metrics_files[0], "App_Update Frametime Samples")
    physics_samples = _measurement_samples(metrics_files[0], "Physics Frametime Samples")
    step_samples = _finite_samples(
        step.get("physics_step_interval_samples_ms"),
        f"{step_path}: physics step intervals",
    )
    metrics = {
        **_sample_summary(app_samples, "app_update"),
        **_sample_summary(physics_samples, "physics"),
        **_sample_summary(step_samples, "step_interval"),
        "real_time_factor": float(step["real_time_factor"]),
        "sim_elapsed_s": float(step["sim_elapsed_s"]),
        "wall_elapsed_s": float(step["wall_elapsed_s"]),
    }
    startup_s = _startup_seconds(step_path.parent.parent)
    if startup_s is not None:
        metrics["simulation_app_start_s"] = startup_s
    return {
        "workload": workload,
        "source": str(step_path.parent),
        "contract": _contract(step["metadata"]),
        "hardware": _hardware(step["environment"]),
        "environment": step["environment"],
        "sample_counts": {
            "app_update": len(app_samples),
            "physics": len(physics_samples),
            "step_interval": len(step_samples),
        },
        "physics_step_count": int(step["physics_step_count"]),
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
        "# Flight performance benchmark",
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
    for workload, metrics in report["metrics"].items():
        lines.extend([f"### {workload}", ""])
        for name, value in metrics.items():
            lines.append(f"- {name}: {value:.4f}")
        lines.append("")
    return "\n".join(lines)


def _build(args: argparse.Namespace, /) -> int:
    run_root = Path(args.run_root).resolve()
    raw_root = Path(args.raw_root).resolve()
    baseline_path = Path(args.baseline).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if (run_root / "report.json").exists():
        raise ValueError(f"benchmark report already exists: {run_root / 'report.json'}")
    report: dict[str, Any]
    try:
        discovered = [
            _summarize_repeat(path) for path in sorted(raw_root.rglob("step_intervals.json"))
        ]
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
            "schema_version": 1,
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
                "px4_ref": args.px4_ref,
            },
            "hardware": hardware,
            "contracts": contracts,
            "metrics": metrics,
            "repeats": repeats,
            "comparisons": [],
        }
        if baseline_path.exists():
            baseline = _read_json(baseline_path)
            if baseline["hardware"] != hardware:
                report["status"] = "incomparable"
                report["reason"] = "hardware fingerprint differs from the accepted baseline"
            elif baseline["environment"]["px4_ref"] != args.px4_ref:
                report["status"] = "incomparable"
                report["reason"] = "PX4 source reference differs from the accepted baseline"
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
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "invalid",
            "reason": str(exc),
            "repetitions": 0,
            "environment": {
                "isaac_image": args.isaac_image,
                "isaac_image_id": args.isaac_image_id,
                "px4_ref": args.px4_ref,
            },
            "metrics": {},
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
    print(f"Flight benchmark report: {run_root / 'report.md'}")
    return 0 if report["status"] == "pass" else 1 if report["status"] == "fail" else 2


def _accept(args: argparse.Namespace, /) -> int:
    result_path = Path(args.result).resolve()
    report = _read_json(result_path)
    if report.get("status") in {"invalid", "incomparable"}:
        raise ValueError(f"cannot accept a {report.get('status')} benchmark result")
    if report.get("repetitions") != _EXPECTED_REPETITIONS:
        raise ValueError(f"benchmark result must contain {_EXPECTED_REPETITIONS} repetitions")

    baseline_path = Path(args.baseline).resolve()
    existing = _read_json(baseline_path) if baseline_path.exists() else None
    if existing is not None and existing["hardware"] != report["hardware"]:
        raise ValueError("result hardware differs from the existing baseline")
    baseline = {
        "schema_version": 1,
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
    print(f"Accepted flight benchmark baseline: {baseline_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="normalize and compare one benchmark run")
    build.add_argument("--raw-root", required=True)
    build.add_argument("--run-root", required=True)
    build.add_argument("--baseline", required=True)
    build.add_argument("--isaac-image", required=True)
    build.add_argument("--isaac-image-id", required=True)
    build.add_argument("--px4-ref", required=True)
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
        raise SystemExit(f"flight benchmark: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
