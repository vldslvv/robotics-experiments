"""Atomic publication labels for completed policy-training runs."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

__all__ = ["publish_policy_labels"]

_TRAINING_REPORT = "training_report.json"
_REPORT_SCHEMA_VERSION = 29


def _publish_label(*, policy_root: Path, run_dir: Path, label: str) -> None:
    link = policy_root / label
    temporary = policy_root / f".{label}.{os.getpid()}.tmp"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(run_dir.name)
    temporary.replace(link)


def _comparable_best_value(
    *,
    policy_root: Path,
    comparison_fingerprint: str,
    metric_name: str,
    direction: str,
) -> float | None:
    best = policy_root / "best"
    if not best.is_symlink():
        return None
    target = Path(os.readlink(best))
    if target.is_absolute() or len(target.parts) != 1:
        return None
    report_path = policy_root / target / _TRAINING_REPORT
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if (
        report.get("schema_version") != _REPORT_SCHEMA_VERSION
        or report.get("comparison_fingerprint") != comparison_fingerprint
    ):
        return None
    selected = (report.get("selected") or {}).get("metric")
    if not isinstance(selected, dict):
        return None
    if selected.get("name") != metric_name or selected.get("direction") != direction:
        return None
    value = selected.get("value")
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def publish_policy_labels(
    *,
    policy_root: Path,
    run_dir: Path,
    comparison_fingerprint: str,
    metric_name: str,
    metric_value: float,
    direction: str,
) -> bool:
    """Advance ``latest`` and conditionally advance comparable ``best``."""
    if not comparison_fingerprint or not metric_name:
        raise ValueError("policy comparison fingerprint and metric name must be non-empty")
    if not math.isfinite(metric_value):
        raise ValueError("policy comparison metric must be finite")
    if direction not in {"minimize", "maximize"}:
        raise ValueError("policy comparison direction must be 'minimize' or 'maximize'")
    best_value = _comparable_best_value(
        policy_root=policy_root,
        comparison_fingerprint=comparison_fingerprint,
        metric_name=metric_name,
        direction=direction,
    )
    advance_best = (
        best_value is None
        or (direction == "minimize" and metric_value < best_value)
        or (direction == "maximize" and metric_value > best_value)
    )
    _publish_label(policy_root=policy_root, run_dir=run_dir, label="latest")
    if advance_best:
        _publish_label(policy_root=policy_root, run_dir=run_dir, label="best")
    return advance_best
