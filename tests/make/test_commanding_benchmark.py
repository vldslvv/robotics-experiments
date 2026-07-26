from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from make.scripts.commanding_benchmark import (
    bootstrap_effect_interval,
    build_report,
    build_workload_fixtures,
    exact_permutation_p_value,
    holm_adjust,
)
from tests.support.paths import REPO_ROOT

SCRIPT = REPO_ROOT / "make/scripts/commanding_benchmark.py"
WORKLOADS = (
    "goto_cached_transmit",
    "policy_cached_transmit",
    "policy_evaluate_transmit",
    "driver_goto_drive",
)
POLICY = {
    "test": "exact two-sided permutation test of median difference",
    "multiple_testing": "Holm",
    "family_alpha": 0.05,
    "practical_regression_percent": 5.0,
    "bootstrap_confidence": 0.95,
    "bootstrap_resamples": 10_000,
}


def _candidate(
    *,
    samples: dict[str, list[float]],
    fingerprint: dict[str, object] | None = None,
) -> dict[str, Any]:
    return {
        "source_commit": "b42ebaffbd636a6935dfb5334ef4d74e363c3a85",
        "repetitions": 9,
        "fingerprint": {"host": "test"} if fingerprint is None else fingerprint,
        "environment": {"cpu_model": "Test CPU"},
        "raw_workers": [f"raw/worker-{index:02d}.json" for index in range(1, 10)],
        "workloads": {
            name: {
                "description": name,
                "worker_medians_ns": samples[name],
            }
            for name in WORKLOADS
        },
    }


def _baseline(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "accepted_at": "2026-01-01T00:00:00+00:00",
        "accepted_from": "test",
        "source_commit": candidate["source_commit"],
        "repetitions": 9,
        "policy": POLICY,
        "fingerprint": candidate["fingerprint"],
        "environment": candidate["environment"],
        "workloads": candidate["workloads"],
    }


def _constant_samples(value: float) -> dict[str, list[float]]:
    return {name: [value] * 9 for name in WORKLOADS}


def _spread_samples(multiplier: float) -> dict[str, list[float]]:
    values = [multiplier * value for value in range(96, 105)]
    return {name: values for name in WORKLOADS}


def _write_json(path: Path, payload: Any, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _accept(
    *,
    result: Path,
    baseline: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "accept",
            "--result",
            str(result),
            "--baseline",
            str(baseline),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_workload_fixtures_validate_public_behavior() -> None:
    fixtures = build_workload_fixtures()

    assert tuple(fixture.name for fixture in fixtures) == WORKLOADS
    for fixture in fixtures:
        fixture.validate(fixture.operation())


def test_workload_fixtures_do_not_import_isaac() -> None:
    code = """
import sys
from make.scripts.commanding_benchmark import build_workload_fixtures
for fixture in build_workload_fixtures():
    fixture.validate(fixture.operation())
forbidden = ("carb", "isaacsim", "omni", "pxr")
loaded = [name for name in sys.modules if name.split(".", 1)[0] in forbidden]
raise SystemExit(1 if loaded else 0)
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = f"{REPO_ROOT / 'src'}:{REPO_ROOT}"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_exact_permutation_test_detects_separated_medians() -> None:
    p_value = exact_permutation_p_value(
        baseline=list(range(1, 10)),
        candidate=list(range(10, 19)),
    )

    assert p_value == pytest.approx(140 / 48_620)


def test_holm_correction_is_step_down_and_monotonic() -> None:
    adjusted = holm_adjust(p_values={"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.20})

    assert adjusted == pytest.approx(
        {
            "a": 0.04,
            "c": 0.09,
            "b": 0.09,
            "d": 0.20,
        }
    )


def test_bootstrap_effect_interval_is_deterministic() -> None:
    first = bootstrap_effect_interval(
        baseline=[98.0, 99.0, 100.0, 101.0, 102.0],
        candidate=[103.0, 104.0, 105.0, 106.0, 107.0],
        seed=17,
        resamples=500,
    )
    second = bootstrap_effect_interval(
        baseline=[98.0, 99.0, 100.0, 101.0, 102.0],
        candidate=[103.0, 104.0, 105.0, 106.0, 107.0],
        seed=17,
        resamples=500,
    )

    assert first == second
    assert first[0] > 0.0


def test_report_requires_significance_and_five_percent_regression() -> None:
    accepted_candidate = _candidate(samples=_spread_samples(1.0))
    baseline = _baseline(accepted_candidate)

    below_threshold = build_report(
        candidate=_candidate(samples=_spread_samples(1.0499)),
        baseline=baseline,
    )
    at_threshold = build_report(
        candidate=_candidate(samples=_spread_samples(1.05)),
        baseline=baseline,
    )

    assert below_threshold["status"] == "pass"
    assert {item["classification"] for item in below_threshold["comparisons"].values()} == {
        "significant_below_threshold"
    }
    assert at_threshold["status"] == "fail"
    assert all(item["failed"] for item in at_threshold["comparisons"].values())


def test_report_calls_significant_improvements_diagnostic() -> None:
    baseline_candidate = _candidate(samples=_spread_samples(1.0))

    report = build_report(
        candidate=_candidate(samples=_spread_samples(0.9)),
        baseline=_baseline(baseline_candidate),
    )

    assert report["status"] == "pass"
    assert {item["classification"] for item in report["comparisons"].values()} == {"improvement"}


def test_report_does_not_fail_a_large_noisy_delta() -> None:
    baseline_values = [90.0, 90.0, 90.0, 90.0, 100.0, 110.0, 110.0, 110.0, 110.0]
    candidate_values = [90.0, 90.0, 90.0, 90.0, 106.0, 110.0, 110.0, 110.0, 110.0]
    baseline_candidate = _candidate(samples={name: baseline_values for name in WORKLOADS})

    report = build_report(
        candidate=_candidate(samples={name: candidate_values for name in WORKLOADS}),
        baseline=_baseline(baseline_candidate),
    )

    assert report["status"] == "pass"
    assert {item["classification"] for item in report["comparisons"].values()} == {"noisy_increase"}


def test_report_rejects_malformed_samples() -> None:
    malformed = _constant_samples(100.0)
    malformed["goto_cached_transmit"] = [100.0] * 8

    with pytest.raises(ValueError, match="expected 9"):
        build_report(candidate=_candidate(samples=malformed), baseline=None)

    non_finite = _constant_samples(100.0)
    non_finite["goto_cached_transmit"][0] = float("nan")
    with pytest.raises(ValueError, match="non-positive or non-finite"):
        build_report(candidate=_candidate(samples=non_finite), baseline=None)


def test_report_marks_fingerprint_change_incomparable() -> None:
    baseline_candidate = _candidate(
        samples=_constant_samples(100.0),
        fingerprint={"host": "baseline"},
    )

    report = build_report(
        candidate=_candidate(
            samples=_constant_samples(100.0),
            fingerprint={"host": "candidate"},
        ),
        baseline=_baseline(baseline_candidate),
    )

    assert report["status"] == "incomparable"
    assert "fingerprint" in report["reason"]


def test_baseline_accepts_complete_no_baseline_result(tmp_path: Path) -> None:
    candidate = _candidate(samples=_constant_samples(100.0))
    report = build_report(candidate=candidate, baseline=None)
    run_root = tmp_path / "commanding.test"
    result = run_root / "report.json"
    _write_json(result, report)
    for raw_worker in report["raw_workers"]:
        _write_json(run_root / raw_worker, {"complete": True})
    baseline_path = tmp_path / "baseline.json"

    completed = _accept(result=result, baseline=baseline_path)

    assert completed.returncode == 0, completed.stderr
    baseline = json.loads(baseline_path.read_text())
    assert baseline["source_commit"] == candidate["source_commit"]
    assert baseline["policy"] == POLICY
    assert len(baseline["workloads"]["driver_goto_drive"]["worker_medians_ns"]) == 9


@pytest.mark.parametrize("status", ["invalid", "incomparable"])
def test_baseline_refuses_invalid_or_incomparable_result(
    tmp_path: Path,
    status: str,
) -> None:
    result = tmp_path / "report.json"
    _write_json(result, {"status": status})

    completed = _accept(result=result, baseline=tmp_path / "baseline.json")

    assert completed.returncode != 0
    assert "cannot accept" in completed.stderr


def test_baseline_refuses_missing_raw_worker(tmp_path: Path) -> None:
    report = build_report(
        candidate=_candidate(samples=_constant_samples(100.0)),
        baseline=None,
    )
    result = tmp_path / "commanding.test" / "report.json"
    _write_json(result, report)

    completed = _accept(result=result, baseline=tmp_path / "baseline.json")

    assert completed.returncode != 0
    assert "raw worker" in completed.stderr


def test_baseline_refuses_no_baseline_result_over_existing_baseline(tmp_path: Path) -> None:
    candidate = _candidate(samples=_constant_samples(100.0))
    report = build_report(candidate=candidate, baseline=None)
    run_root = tmp_path / "commanding.test"
    result = run_root / "report.json"
    _write_json(result, report)
    for raw_worker in report["raw_workers"]:
        _write_json(run_root / raw_worker, {"complete": True})
    baseline_path = tmp_path / "baseline.json"
    _write_json(baseline_path, _baseline(candidate))

    completed = _accept(result=result, baseline=baseline_path)

    assert completed.returncode != 0
    assert "no_baseline result over an existing baseline" in completed.stderr


def test_make_help_readme_and_host_only_wiring() -> None:
    benchmarks_make = (REPO_ROOT / "make/benchmarks.mk").read_text(encoding="utf-8")
    root_make = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "benchmark-commanding:" in benchmarks_make
    assert "benchmark-commanding-accept:" in benchmarks_make
    commanding_recipe = benchmarks_make.split("benchmark-commanding:", 1)[1].split(
        ".PHONY: benchmark-commanding-accept",
        1,
    )[0]
    assert "docker" not in commanding_recipe.lower()
    assert "benchmark-commanding" in root_make
    assert "benchmark-commanding" in readme

    completed = subprocess.run(
        ["make", "help"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "make benchmark-commanding " in completed.stdout
    assert "make benchmark-commanding-accept RESULT=..." in completed.stdout
