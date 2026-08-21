"""Run configured independent hover trials and publish flight parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from robotics_experiments.flight_parameters import (
    FlightParametersConfig,
    ResponseTraceEvidence,
    derive_flight_parameters,
    finalize_flight_parameters_bundle,
    load_flight_parameters_config,
    load_flight_trace,
    publish_flight_parameters,
    write_flight_parameters,
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


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_trial(
    *,
    repo_root: Path,
    config_path: Path,
    config: FlightParametersConfig,
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


def _trial_is_complete(trial_dir: Path, /) -> bool:
    return all(
        (trial_dir / name).is_file()
        for name in (
            "raw_trace.json",
            "flight.ulg",
            "bridge.log",
            "hover_verification.json",
            "trace.json",
        )
    )


def _run_response_repetition(
    *,
    repo_root: Path,
    config: FlightParametersConfig,
    run_id: str,
    repetition: int,
    response_dir: Path,
) -> ResponseTraceEvidence:
    subprocess.run(
        (
            "make",
            "test-integration-flight-parameters-response-sample",
            f"HYDRA_CONFIG={config.response.experiment_config}",
            f"FLIGHT_PARAMETERS_RESPONSE_DEFINITION={config.response.definition}",
            f"FLIGHT_PARAMETERS_RESPONSE_DIR={response_dir}",
            f"PX4_TEST_CONTAINER_NAME=flight-parameters-{run_id}-response-{repetition:03d}",
        ),
        cwd=repo_root,
        check=True,
    )
    reference_path = response_dir / "flight_parameters_response.json"
    ulog_path = response_dir / "flight_parameters.ulg"
    bridge_log_path = response_dir / "flight_parameters_bridge.log"
    subprocess.run(
        (
            "uv",
            "run",
            "--with",
            "pyulog",
            "python",
            "flight_analysis/tools/controller_state_replay.py",
            "--config",
            config.response.experiment_config,
            "--ulog",
            str(ulog_path),
            "--augment-reference",
            str(reference_path),
        ),
        cwd=repo_root,
        check=True,
    )
    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    traces = payload.get("traces") if isinstance(payload, dict) else None
    if not isinstance(traces, list) or not traces:
        raise ValueError("flight-parameters response capture produced no traces")
    return ResponseTraceEvidence(
        repetition=repetition,
        profile_count=len(traces),
        reference_path=reference_path.relative_to(response_dir.parents[1]).as_posix(),
        reference_sha256=_sha256(reference_path),
        ulog_path=ulog_path.relative_to(response_dir.parents[1]).as_posix(),
        ulog_sha256=_sha256(ulog_path),
        bridge_log_path=bridge_log_path.relative_to(response_dir.parents[1]).as_posix(),
        bridge_log_sha256=_sha256(bridge_log_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path(".build/flight-parameters"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    config = load_flight_parameters_config(config_path)
    created_at = datetime.now(timezone.utc)
    run_id = _run_id(args.run_id or created_at.strftime("%Y%m%dT%H%M%SZ"))
    run_dir = (args.output_root / config.flight_parameters_id / run_id).resolve()
    resume = args.run_id is not None and run_dir.exists()
    if resume and (run_dir / "manifest.json").exists():
        raise ValueError(f"cannot resume completed flight-parameters run {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=resume)
    evidence_dir = run_dir / "evidence"
    evidence_dir.mkdir(exist_ok=resume)

    trials = tuple(range(1, config.hover.repetitions + 1))
    pending_trials = tuple(
        index for index in trials if not _trial_is_complete(evidence_dir / f"trial-{index:03d}")
    )
    if config.execution.worker_count == 1:
        for index in pending_trials:
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
            for offset in range(0, len(pending_trials), config.execution.worker_count):
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
                    for index in pending_trials[offset : offset + config.execution.worker_count]
                ]
                for future in futures:
                    future.result()

    traces = tuple(
        load_flight_trace(evidence_dir / f"trial-{index:03d}" / "trace.json") for index in trials
    )
    response_traces = tuple(
        _run_response_repetition(
            repo_root=repo_root,
            config=config,
            run_id=run_id,
            repetition=repetition,
            response_dir=run_dir / "response" / f"repetition-{repetition:03d}",
        )
        for repetition in range(1, config.response.repetitions + 1)
    )
    commit, dirty = _source_provenance(repo_root)
    flight_parameters = derive_flight_parameters(
        config=config,
        run_id=run_id,
        traces=traces,
        response_traces=response_traces,
        source_commit=commit,
        source_dirty=dirty,
        created_at_utc=created_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
    )
    write_flight_parameters(flight_parameters=flight_parameters, run_dir=run_dir)
    (run_dir / "campaign.json").write_text(
        json.dumps(
            {
                "artifact_kind": "robotics-experiments-flight-parameters-campaign",
                "schema_version": 1,
                "flight_parameters_config": config_path.relative_to(repo_root).as_posix(),
                "config_fingerprint": config.config_fingerprint,
                "worker_count": config.execution.worker_count,
                "hover_trial_count": len(traces),
                "response_repetition_count": len(response_traces),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    finalize_flight_parameters_bundle(run_dir)
    latest = publish_flight_parameters(run_dir)
    print(f"Published flight parameters: {run_dir}")
    print(f"Latest: {latest} -> {latest.readlink()}")


if __name__ == "__main__":
    main()
