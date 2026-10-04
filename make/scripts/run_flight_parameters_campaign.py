"""Functional host orchestration for capture, verification, compression and publication."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robotics_experiments.flight_parameters import (
    FlightParametersConfig,
    StabilityReport,
    analyze_flight_parameters_stability,
    compress_flight_evidence,
    flight_evidence_sha256,
    flight_seeds,
    load_flight_dataset,
    load_flight_parameters_config,
)
from robotics_experiments.flight_parameters._dataset import (
    finalize_flight_dataset,
    write_flight_dataset,
)
from robotics_experiments.flight_parameters.px4_ulog import verify_flight_recording

__all__ = ["main"]

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class _Flight:
    index: int
    seed: int
    directory: str


@dataclass(frozen=True)
class _CompletedFlight:
    flight: _Flight
    retained_bytes: int


def _flight_result(*, flight: _Flight, checkpoint: dict[str, Any]) -> _CompletedFlight:
    return _CompletedFlight(
        flight=flight,
        retained_bytes=sum(item["size_bytes"] for item in checkpoint["files"]),
    )


def _run_id(value: str, /) -> str:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("flight-parameter run id contains unsupported characters")
    return value


def _sha256(path: Path, /) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(*, path: Path, value: Any) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise ValueError(f"existing campaign record differs: {path}")
        return
    pending = path.with_name(path.name + ".pending")
    pending.write_text(encoded)
    pending.replace(path)


def _flights(config: FlightParametersConfig, /) -> tuple[_Flight, ...]:
    return tuple(
        _Flight(index, seed, f"flights/flight-{index:03d}")
        for index, seed in enumerate(flight_seeds(config=config))
    )


def _definition(config: FlightParametersConfig, /) -> dict[str, Any]:
    definition = asdict(config)
    for field in ("execution", "config_fingerprint"):
        definition.pop(field)
    return definition


def _capture_root(
    *, repo_root: Path, config: FlightParametersConfig, run_id: str, flight: _Flight
) -> Path:
    return (
        repo_root
        / ".build/docker/integration/flight-parameters"
        / config.flight_parameters_id
        / run_id
        / flight.directory
    )


def _file_record(*, path: Path, root: Path) -> dict[str, Any]:
    record = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".gz":
        record["content_sha256"] = flight_evidence_sha256(path)
    return record


def _checked_path(*, root: Path, record: dict[str, Any]) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("evidence record path must be relative")
    path = root / relative
    if (
        not path.resolve().is_relative_to(root.resolve())
        or not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or _sha256(path) != record["sha256"]
    ):
        raise ValueError(f"flight checkpoint evidence differs: {path}")
    if "content_sha256" in record and flight_evidence_sha256(path) != record["content_sha256"]:
        raise ValueError(f"flight checkpoint contents differ: {path}")
    return path


def _capture_flight(
    *,
    repo_root: Path,
    config_path: Path,
    config: FlightParametersConfig,
    run_dir: Path,
    flight: _Flight,
) -> None:
    directory = run_dir / flight.directory
    receipt = directory / "capture.json"
    if receipt.exists():
        for item in json.loads(receipt.read_text())["files"]:
            _checked_path(root=directory, record=item)
        return
    if directory.exists():
        failed_root = run_dir / "failed" / flight.directory
        failed_root.mkdir(parents=True, exist_ok=True)
        destination = failed_root / f"attempt-{len(tuple(failed_root.iterdir())) + 1:03d}"
        directory.rename(destination)
        print(f"Retained incomplete capture: {destination}", flush=True)
    directory.mkdir(parents=True)
    integration_root = _capture_root(
        repo_root=repo_root, config=config, run_id=run_dir.name, flight=flight
    )
    common = (
        "make",
        "-o",
        "generate-assets",
        f"FLIGHT_CONFIG={config_path}",
        f"INTEGRATION_ROOT={integration_root}",
        f"PX4_TEST_CONTAINER_NAME=flight-parameters-{run_dir.name}-{flight.index:03d}",
    )
    command = (
        *common,
        "test-integration-flight-parameters-sample",
        f"HYDRA_CONFIG={config.experiment_config}",
        f"FLIGHT_COUNT={config.flight_count}",
        f"FLIGHT_PARAMETERS_RUN_ID={run_dir.name}",
        f"FLIGHT_PARAMETERS_FLIGHT_INDEX={flight.index}",
        f"FLIGHT_PARAMETERS_TRIAL_DIR={directory}",
    )
    names = ("raw_flight.json", "motion.jsonl", "flight.ulg", "bridge.log")
    previous_runs = set(integration_root.iterdir()) if integration_root.exists() else set()
    subprocess.run(command, cwd=repo_root, check=True)
    new_runs = set(integration_root.iterdir()) - previous_runs
    sources = tuple(
        path
        for root in new_runs
        for path in root.rglob("*")
        if path.is_file() and (path.suffix == ".ulg" or path.name.endswith("_bridge.log"))
    )
    bridge_sources = tuple(path for path in sources if path.name.endswith("_bridge.log"))
    if len(bridge_sources) != 1:
        raise ValueError("capture requires one completed harness bridge log")
    bridge = directory / names[-1]
    # The standalone copy precedes teardown; retain the complete closed log.
    prefix_digest = hashlib.sha256()
    remaining = bridge.stat().st_size
    with bridge_sources[0].open("rb") as stream:
        while remaining:
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ValueError("completed harness bridge log is shorter than captured prefix")
            prefix_digest.update(chunk)
            remaining -= len(chunk)
    if prefix_digest.hexdigest() != _sha256(bridge):
        raise ValueError("completed harness bridge log differs from captured prefix")
    shutil.copyfile(bridge_sources[0], bridge)
    files = [_file_record(path=directory / name, root=directory) for name in names]
    # Only matching logs in this flight's private harness root may be removed later.
    log_hashes = {item["sha256"] for item in files[2:]}
    duplicates = [
        _file_record(path=path, root=repo_root)
        for path in sorted(sources)
        if path.is_file()
        and (path.suffix == ".ulg" or path.name.endswith("_bridge.log"))
        and _sha256(path) in log_hashes
    ]
    _write_json(
        path=receipt,
        value={
            "flight": asdict(flight),
            "files": files,
            "duplicates": duplicates,
        },
    )


def _process_flight(
    *, repo_root: Path, config: FlightParametersConfig, run_dir: Path, flight: _Flight
) -> None:
    directory = run_dir / flight.directory
    reference_path = verify_flight_recording(config=config, directory=directory)
    subprocess.run(
        (
            "uv",
            "run",
            "--with",
            "pyulog",
            "python",
            "flight_analysis/tools/controller_state_replay.py",
            "--config",
            config.experiment_config,
            "--ulog",
            str(directory / "flight.ulg"),
            "--augment-reference",
            str(reference_path),
        ),
        cwd=repo_root,
        check=True,
    )
    payload = json.loads(reference_path.read_text())
    if (payload["flight_index"], payload["seed"], payload["run_id"]) != (
        flight.index,
        flight.seed,
        run_dir.name,
    ):
        raise ValueError("flight metadata differs from its definition")
    if payload.get("controller_state_method") != "px4-ulog-controller-boundary-v2" or any(
        not stage.get("initial_controller_state_valid") for stage in payload["stages"][1:]
    ):
        raise ValueError("response controller boundary reconstruction is incomplete")


def _cleanup_copies(
    *,
    repo_root: Path,
    config: FlightParametersConfig,
    run_dir: Path,
    flight: _Flight,
    completed: dict[str, Any],
) -> None:
    directory = run_dir / flight.directory
    for record in completed["files"]:
        path = directory / record["path"]
        if path.suffix == ".gz":
            original = path.with_suffix("")
            if original.exists():
                if _sha256(original) != record["content_sha256"]:
                    raise ValueError(f"uncompressed evidence differs: {original}")
                original.unlink()
    capture = json.loads((directory / "capture.json").read_text())
    allowed_root = _capture_root(
        repo_root=repo_root, config=config, run_id=run_dir.name, flight=flight
    ).resolve()
    for record in capture["duplicates"]:
        path = repo_root / record["path"]
        if not path.resolve().is_relative_to(allowed_root):
            raise ValueError("duplicate log is outside this flight's harness root")
        if path.exists():
            _checked_path(root=repo_root, record=record).unlink()


def _completed_flight(
    *, repo_root: Path, config: FlightParametersConfig, run_dir: Path, flight: _Flight
) -> _CompletedFlight | None:
    directory = run_dir / flight.directory
    checkpoint = directory / "complete.json"
    if not checkpoint.exists():
        return None
    completed = json.loads(checkpoint.read_text())
    if completed["flight"] != asdict(flight) or not completed["completed"]:
        raise ValueError("flight checkpoint input identity differs")
    for item in completed["files"]:
        _checked_path(root=directory, record=item)
    _cleanup_copies(
        repo_root=repo_root, config=config, run_dir=run_dir, flight=flight, completed=completed
    )
    return _flight_result(flight=flight, checkpoint=completed)


def _run_flight(
    *,
    repo_root: Path,
    config_path: Path,
    config: FlightParametersConfig,
    run_dir: Path,
    flight: _Flight,
) -> _CompletedFlight:
    print(f"Flight {flight.directory}: seed={flight.seed}", flush=True)
    _capture_flight(
        repo_root=repo_root, config_path=config_path, config=config, run_dir=run_dir, flight=flight
    )
    started = time.monotonic()
    _process_flight(repo_root=repo_root, config=config, run_dir=run_dir, flight=flight)
    directory = run_dir / flight.directory
    capture = json.loads((directory / "capture.json").read_text())
    logs = tuple(directory / item["path"] for item in capture["files"][2:])
    compressed = tuple(
        compress_flight_evidence(path=path, level=config.execution.gzip_level) for path in logs
    )
    retained = tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path not in logs and not path.name.endswith(".pending")
    )
    files = [_file_record(path=path, root=directory) for path in retained]
    completed = {
        "flight": asdict(flight),
        "completed": True,
        "files": files,
    }
    _write_json(path=directory / "complete.json", value=completed)
    _cleanup_copies(
        repo_root=repo_root, config=config, run_dir=run_dir, flight=flight, completed=completed
    )
    print(
        f"Completed {flight.directory}: "
        f"compressed={sum(path.stat().st_size for path in compressed)} "
        f"processing_s={time.monotonic() - started:.1f}",
        flush=True,
    )
    return _flight_result(flight=flight, checkpoint=completed)


def _check_space(
    *,
    run_dir: Path,
    remaining: int,
    config: FlightParametersConfig,
    completed: tuple[_CompletedFlight, ...],
) -> None:
    estimate = max(
        (
            config.execution.estimated_archive_bytes_per_flight,
            *(flight.retained_bytes for flight in completed),
        )
    )
    needed = (
        config.execution.minimum_free_bytes
        + min(config.execution.worker_count, remaining) * config.execution.raw_flight_budget_bytes
        + remaining * estimate
    )
    free = shutil.disk_usage(run_dir).free
    print(
        f"Storage: remaining_flights={remaining} estimated_archive_bytes={remaining * estimate} "
        f"required_free_bytes={needed} available_bytes={free}",
        flush=True,
    )
    if free < needed:
        raise OSError(
            "insufficient space for remaining archives, concurrent raw flights and reserve; "
            "completed evidence is resumable"
        )


def _collect_flights(
    *,
    repo_root: Path,
    config_path: Path,
    config: FlightParametersConfig,
    run_dir: Path,
    flights: tuple[_Flight, ...],
) -> tuple[_CompletedFlight, ...]:
    completed = {}
    for flight in flights:
        record = _completed_flight(
            repo_root=repo_root, config=config, run_dir=run_dir, flight=flight
        )
        if record is not None:
            completed[flight.directory] = record
    pending = tuple(flight for flight in flights if flight.directory not in completed)
    with ThreadPoolExecutor(max_workers=config.execution.worker_count) as executor:
        for offset in range(0, len(pending), config.execution.worker_count):
            current = load_flight_parameters_config(
                path=config_path, flight_count=config.flight_count
            )
            if _definition(current) != _definition(config):
                raise ValueError("flight definition changed during the campaign")
            _check_space(
                run_dir=run_dir,
                remaining=len(flights) - len(completed),
                config=config,
                completed=tuple(completed.values()),
            )
            batch = pending[offset : offset + config.execution.worker_count]
            futures = tuple(
                executor.submit(
                    _run_flight,
                    repo_root=repo_root,
                    config_path=config_path,
                    config=config,
                    run_dir=run_dir,
                    flight=flight,
                )
                for flight in batch
            )
            for flight, future in zip(batch, futures, strict=True):
                completed[flight.directory] = future.result()
    return tuple(completed[flight.directory] for flight in flights)


def _write_stability_report(*, path: Path, report: StabilityReport) -> None:
    encoded = json.dumps(asdict(report), indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"existing stability report differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".json.pending")
    pending.write_text(encoded, encoding="utf-8")
    pending.replace(path)


def _publish_campaign(
    *, run_dir: Path, comparison_run_dirs: tuple[Path, ...], settings_path: Path
) -> None:
    report = analyze_flight_parameters_stability(
        run_dir=run_dir,
        comparison_run_dirs=comparison_run_dirs,
        settings_path=settings_path,
    )
    _write_stability_report(
        path=run_dir.parent / "stability" / f"{run_dir.name}.json", report=report
    )
    latest = finalize_flight_dataset(run_dir)
    print(f"Published flight parameters: {run_dir}")
    print(f"Latest: {latest} -> {latest.readlink()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--flight-count", type=int)
    parser.add_argument(
        "--stability-config", type=Path, default=Path("configs/flight/stability/default.yaml")
    )
    parser.add_argument("--compare-run", type=Path, action="append", default=[])
    parser.add_argument("--analyze-run", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path(".build/flight-parameters"))
    parser.add_argument("--run-id")
    args = parser.parse_args()

    if args.analyze_run is not None:
        if args.analysis_output is None:
            parser.error("--analyze-run requires --analysis-output outside the producer bundle")
        producer_paths = (args.analyze_run, *args.compare_run)
        output = args.analysis_output.resolve()
        if any(output.is_relative_to(path.resolve()) for path in producer_paths):
            parser.error("analysis output must be outside every producer bundle")
        report = analyze_flight_parameters_stability(
            run_dir=args.analyze_run,
            comparison_run_dirs=tuple(args.compare_run),
            settings_path=args.stability_config,
        )
        _write_stability_report(path=output, report=report)
        print(f"Stability: {report.status}; report: {output}")
        return
    if args.analysis_output is not None:
        parser.error("--analysis-output requires --analyze-run")
    if args.config is None:
        parser.error("--config is required for capture or resuming publication")
    repo_root = Path.cwd().resolve()
    config_path = args.config.resolve()
    config = load_flight_parameters_config(path=config_path, flight_count=args.flight_count)
    created_at = datetime.now(timezone.utc)
    run_id = _run_id(args.run_id or created_at.strftime("%Y%m%dT%H%M%SZ"))
    run_dir = (args.output_root / config.flight_parameters_id / run_id).resolve()
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    with (run_dir.parent / f".{run_id}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"flight campaign is already running: {run_id}") from exc
        resume = args.run_id is not None and run_dir.exists()
        if resume and (run_dir / "manifest.json").exists():
            raise ValueError(f"cannot resume completed flight-parameters run {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=resume)
        _write_json(path=run_dir / "definition.json", value=_definition(config))
        if (run_dir / "flight_parameters.json").exists():
            load_flight_dataset(run_dir)
            _publish_campaign(
                run_dir=run_dir,
                comparison_run_dirs=tuple(args.compare_run),
                settings_path=args.stability_config,
            )
            return

        flights = _flights(config)
        completed = _collect_flights(
            repo_root=repo_root,
            config_path=config_path,
            config=config,
            run_dir=run_dir,
            flights=flights,
        )
        write_flight_dataset(
            run_dir=run_dir,
            payload={
                "flight_parameters_id": config.flight_parameters_id,
                "run_id": run_id,
                "completed": True,
                "definition": _file_record(path=run_dir / "definition.json", root=run_dir),
                "flights": [
                    _file_record(
                        path=run_dir / record.flight.directory / "complete.json", root=run_dir
                    )
                    for record in completed
                ],
            },
        )
        _publish_campaign(
            run_dir=run_dir,
            comparison_run_dirs=tuple(args.compare_run),
            settings_path=args.stability_config,
        )


if __name__ == "__main__":
    main()
