"""Run configured independent hover trials and publish a flight envelope."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from robotics_experiments.flight_parameters import (
    FlightConfig,
    derive_flight_envelope,
    finalize_flight_envelope_bundle,
    load_flight_config,
    load_flight_trace,
    publish_flight_envelope,
    write_flight_envelope,
)
from robotics_experiments.flight_parameters.px4_ulog import verify_px4_hover_trial

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _run_id(value: str, /) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("flight-parameter run id contains unsupported characters")
    return value


def _source_provenance(repo_root: Path, /) -> tuple[str, bool]:
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
        raise RuntimeError(f"cannot resolve flight campaign source provenance: {exc}") from exc
    return commit, bool(status)


def _run_trial(
    *,
    repo_root: Path,
    config_path: Path,
    config: FlightConfig,
    run_id: str,
    trial_index: int,
    trial_dir: Path,
) -> None:
    command = (
        "make",
        "test-integration-flight-parameters-sample",
        f"FLIGHT_CONFIG={config_path}",
        f"HYDRA_CONFIG={config.experiment_config}",
        f"FLIGHT_PARAMETERS_RUN_ID={run_id}",
        f"FLIGHT_PARAMETERS_TRIAL_INDEX={trial_index}",
        f"FLIGHT_PARAMETERS_TRIAL_DIR={trial_dir}",
        f"PX4_TEST_CONTAINER_NAME=flight-parameters-{run_id}-{trial_index:03d}",
    )
    subprocess.run(command, cwd=repo_root, check=True)
    verify_px4_hover_trial(
        config=config,
        raw_trace_path=trial_dir / "raw_trace.json",
        ulog_path=trial_dir / "flight.ulg",
        trial_dir=trial_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".build/flight-envelopes"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    config = load_flight_config(config_path)
    run_id = _run_id(args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = (args.output_root / config.flight_id / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir()

    trials = tuple(range(1, config.scenario.repetitions + 1))
    if config.execution.worker_count == 1:
        for index in trials:
            _run_trial(
                repo_root=repo_root,
                config_path=config_path,
                config=config,
                run_id=run_id,
                trial_index=index,
                trial_dir=evidence_dir / f"trial-{index:03d}",
            )
    else:
        with ThreadPoolExecutor(max_workers=config.execution.worker_count) as executor:
            for offset in range(0, len(trials), config.execution.worker_count):
                futures = [
                    executor.submit(
                        _run_trial,
                        repo_root=repo_root,
                        config_path=config_path,
                        config=config,
                        run_id=run_id,
                        trial_index=index,
                        trial_dir=evidence_dir / f"trial-{index:03d}",
                    )
                    for index in trials[offset : offset + config.execution.worker_count]
                ]
                for future in futures:
                    future.result()

    traces = tuple(
        load_flight_trace(evidence_dir / f"trial-{index:03d}" / "trace.json") for index in trials
    )
    commit, dirty = _source_provenance(repo_root)
    envelope = derive_flight_envelope(
        config=config,
        run_id=run_id,
        traces=traces,
        source_commit=commit,
        source_dirty=dirty,
    )
    write_flight_envelope(envelope=envelope, run_dir=run_dir)
    (run_dir / "campaign.json").write_text(
        json.dumps(
            {
                "artifact_kind": "robotics-experiments-flight-campaign",
                "schema_version": 1,
                "flight_config": config_path.relative_to(repo_root).as_posix(),
                "config_fingerprint": config.config_fingerprint,
                "worker_count": config.execution.worker_count,
                "trial_count": len(traces),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    finalize_flight_envelope_bundle(run_dir)
    latest = publish_flight_envelope(run_dir)
    print(f"Published flight envelope: {run_dir}")
    print(f"Latest: {latest} -> {latest.readlink()}")


if __name__ == "__main__":
    main()
