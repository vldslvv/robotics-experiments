"""Functional host orchestration for capture, verification, compression and publication."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import platform
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import groupby
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

from robotics_experiments.flight_parameters import (
    FlightParametersConfig,
    ResponseTraceEvidence,
    StabilityReport,
    analyze_flight_parameters_stability,
    compress_flight_evidence,
    derive_flight_parameters,
    finalize_flight_parameters_bundle,
    flight_evidence_sha256,
    load_flight_parameters,
    load_flight_parameters_config,
    load_flight_trace,
    publish_flight_parameters,
    write_flight_parameters,
)
from robotics_experiments.flight_parameters.px4_ulog import verify_px4_hover_trial

__all__ = ["main"]

_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class _Flight:
    kind: str
    index: int
    set_name: str
    seed: int
    directory: str


@dataclass(frozen=True)
class _Processing:
    gzip_level: int
    minimum_free_bytes: int
    raw_flight_budget_bytes: int
    estimated_archive_bytes_per_flight: int


@dataclass(frozen=True)
class _CompletedFlight:
    flight: _Flight
    response: ResponseTraceEvidence | None
    raw_log_bytes: int
    compressed_log_bytes: int
    retained_bytes: int
    capture_wall_s: float
    processing_wall_s: float


def _flight_result(*, flight: _Flight, checkpoint: dict[str, Any]) -> _CompletedFlight:
    response = checkpoint["response"]
    return _CompletedFlight(
        flight=flight,
        response=ResponseTraceEvidence(**response) if response is not None else None,
        raw_log_bytes=checkpoint["raw_log_bytes"],
        compressed_log_bytes=checkpoint["compressed_log_bytes"],
        retained_bytes=sum(item["size_bytes"] for item in checkpoint["files"]),
        capture_wall_s=checkpoint["capture_wall_s"],
        processing_wall_s=checkpoint["processing_wall_s"],
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


def _processing(path: Path, /) -> _Processing:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict) or set(value) != set(_Processing.__annotations__):
        raise ValueError("flight processing settings fields mismatch")
    if any(type(item) is not int or item < 0 for item in value.values()):
        raise ValueError("flight processing settings must be nonnegative integers")
    settings = _Processing(**cast(dict[str, int], value))
    if (
        settings.gzip_level > 9
        or min(settings.raw_flight_budget_bytes, settings.estimated_archive_bytes_per_flight) == 0
    ):
        raise ValueError("invalid compression level or flight storage budget")
    return settings


def _flights(config: FlightParametersConfig, /) -> tuple[_Flight, ...]:
    result: list[_Flight] = []
    for kind, prefix in (("hover", "evidence/trial"), ("response", "response/repetition")):
        entries = tuple(
            (group.name, seed) for group in config.sets for seed in getattr(group, f"{kind}_seeds")
        )
        result.extend(
            _Flight(kind, index, name, seed, f"{prefix}-{index:03d}")
            for index, (name, seed) in enumerate(entries, start=1)
        )
    return tuple(result)


def _source_provenance(repo_root: Path, /) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args), cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout

    names = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    source = {}
    for name in sorted(set(names)):
        path = Path(name)
        if name == "src/robotics_experiments/flight_parameters/_stability.py" or name.startswith(
            "configs/flight/stability/"
        ):
            continue
        if (
            name
            and (path.parts[0] in {"src", "configs", "make", "flight_analysis", "isaac", "px4"})
            and (
                path.suffix in {".py", ".yaml", ".sh", ".mk", ".toml", ".usda"}
                or path.name.startswith("Dockerfile")
            )
        ):
            source[name] = _sha256(repo_root / path) if (repo_root / path).is_file() else "deleted"
    for name in ("Makefile", "pyproject.toml", "uv.lock", ".python-version"):
        source[name] = _sha256(repo_root / name)
    image_names = subprocess.run(
        (
            "make",
            "--no-print-directory",
            "-s",
            "--eval=flight-parameters-images:;@printf '%s\\n' '$(ISAAC_LAB_IMAGE)' '$(PX4_IMAGE)'",
            "flight-parameters-images",
        ),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    images = subprocess.run(
        ("docker", "image", "inspect", *image_names),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "source_commit": git("rev-parse", "HEAD").strip(),
        "source_dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
        "source_sha256": source,
        "host_python": platform.python_version(),
        "host_packages": {name: version(name) for name in ("numpy", "pyulog", "omegaconf")},
        "images": dict(
            zip(image_names, (item["Id"] for item in json.loads(images.stdout)), strict=True)
        ),
    }


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
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "content_sha256": flight_evidence_sha256(path),
    }


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
    if flight_evidence_sha256(path) != record["content_sha256"]:
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
        f"PX4_TEST_CONTAINER_NAME=flight-parameters-{run_dir.name}-{flight.kind}-{flight.index:03d}",
    )
    if flight.kind == "hover":
        command = (
            *common,
            "test-integration-flight-parameters-sample",
            f"HYDRA_CONFIG={config.experiment_config}",
            f"FLIGHT_PARAMETERS_RUN_ID={run_dir.name}",
            f"FLIGHT_PARAMETERS_TRIAL_INDEX={flight.index}",
            f"FLIGHT_PARAMETERS_TRIAL_DIR={directory}",
        )
        names = ("raw_trace.json", "flight.ulg", "bridge.log")
    else:
        command = (
            *common,
            "test-integration-flight-parameters-response-sample",
            f"HYDRA_CONFIG={config.response.experiment_config}",
            f"FLIGHT_PARAMETERS_RESPONSE_REPETITION={flight.index}",
            f"FLIGHT_PARAMETERS_RESPONSE_DEFINITION={config.response.definition}",
            f"FLIGHT_PARAMETERS_RESPONSE_DIR={directory}",
        )
        names = ("raw_response.json", "flight_parameters.ulg", "flight_parameters_bridge.log")
    previous_runs = set(integration_root.iterdir()) if integration_root.exists() else set()
    started = time.monotonic()
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
    if flight.kind == "response":
        (directory / "flight_parameters_response.json").rename(directory / "raw_response.json")
    files = [_file_record(path=directory / name, root=directory) for name in names]
    # Only matching logs in this flight's private harness root may be removed later.
    log_hashes = {item["sha256"] for item in files[1:]}
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
            "capture_wall_s": time.monotonic() - started,
        },
    )


def _process_flight(
    *, repo_root: Path, config: FlightParametersConfig, run_dir: Path, flight: _Flight
) -> ResponseTraceEvidence | None:
    directory = run_dir / flight.directory
    if flight.kind == "hover":
        verify_px4_hover_trial(
            config=config,
            raw_trace_path=directory / "raw_trace.json",
            ulog_path=directory / "flight.ulg",
            trial_dir=directory,
        )
        trace = load_flight_trace(directory / "trace.json")
        if (
            trace.campaign_run_id,
            trace.trial_index,
            trace.set_name,
            trace.seed,
            trace.config_fingerprint,
        ) != (run_dir.name, flight.index, flight.set_name, flight.seed, config.config_fingerprint):
            raise ValueError("verified hover identity differs from the flight definition")
        return None
    reference_path = directory / "flight_parameters_response.json"
    reference_path.write_bytes((directory / "raw_response.json").read_bytes())
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
            str(directory / "flight_parameters.ulg"),
            "--augment-reference",
            str(reference_path),
        ),
        cwd=repo_root,
        check=True,
    )
    payload = json.loads(reference_path.read_text())
    traces = payload.get("traces", [])
    if (
        (payload.get("set_name"), payload.get("seed")) != (flight.set_name, flight.seed)
        or payload.get("controller_state_method") != "px4-ulog-controller-boundary-v2"
        or not traces
        or any(
            not trace.get("completed")
            or trace.get("failure_reason")
            or not trace.get("initial_controller_state_valid")
            for trace in traces
        )
    ):
        raise ValueError("response capture is incomplete or differs from the flight definition")
    return ResponseTraceEvidence(
        repetition=flight.index,
        set_name=flight.set_name,
        seed=flight.seed,
        profile_count=len(traces),
        reference_path=reference_path.relative_to(run_dir).as_posix(),
        reference_sha256=_sha256(reference_path),
        ulog_path=f"{flight.directory}/flight_parameters.ulg.gz",
        ulog_sha256=flight_evidence_sha256(directory / "flight_parameters.ulg"),
        bridge_log_path=f"{flight.directory}/flight_parameters_bridge.log.gz",
        bridge_log_sha256=flight_evidence_sha256(directory / "flight_parameters_bridge.log"),
    )


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
    if (
        completed["flight"] != asdict(flight)
        or completed["config_fingerprint"] != config.config_fingerprint
        or completed["capture_inputs_sha256"] != _sha256(run_dir / "capture_inputs.json")
    ):
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
    settings: _Processing,
) -> _CompletedFlight:
    print(f"Flight {flight.directory}: set={flight.set_name} seed={flight.seed}", flush=True)
    _capture_flight(
        repo_root=repo_root, config_path=config_path, config=config, run_dir=run_dir, flight=flight
    )
    started = time.monotonic()
    response = _process_flight(repo_root=repo_root, config=config, run_dir=run_dir, flight=flight)
    directory = run_dir / flight.directory
    capture = json.loads((directory / "capture.json").read_text())
    logs = tuple(directory / item["path"] for item in capture["files"][1:])
    compressed = tuple(
        compress_flight_evidence(path=path, level=settings.gzip_level) for path in logs
    )
    retained = tuple(
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path not in logs and not path.name.endswith(".pending")
    )
    files = [_file_record(path=path, root=directory) for path in retained]
    completed = {
        "flight": asdict(flight),
        "config_fingerprint": config.config_fingerprint,
        "capture_inputs_sha256": _sha256(run_dir / "capture_inputs.json"),
        "files": files,
        "response": asdict(response) if response is not None else None,
        "capture_wall_s": capture["capture_wall_s"],
        "processing_wall_s": time.monotonic() - started,
        "raw_log_bytes": sum(path.stat().st_size for path in logs),
        "compressed_log_bytes": sum(path.stat().st_size for path in compressed),
    }
    _write_json(path=directory / "complete.json", value=completed)
    _cleanup_copies(
        repo_root=repo_root, config=config, run_dir=run_dir, flight=flight, completed=completed
    )
    print(
        f"Completed {flight.directory}: raw={completed['raw_log_bytes']} "
        f"compressed={completed['compressed_log_bytes']} "
        f"processing_s={completed['processing_wall_s']:.1f}",
        flush=True,
    )
    return _flight_result(flight=flight, checkpoint=completed)


def _check_space(
    *,
    run_dir: Path,
    settings: _Processing,
    remaining: int,
    workers: int,
    completed: tuple[_CompletedFlight, ...],
) -> None:
    estimate = max(
        (
            settings.estimated_archive_bytes_per_flight,
            *(flight.retained_bytes for flight in completed),
        )
    )
    needed = (
        settings.minimum_free_bytes
        + min(workers, remaining) * settings.raw_flight_budget_bytes
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
    settings: _Processing,
    provenance: dict[str, Any],
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
        for _, entries in groupby(pending, key=lambda flight: flight.kind):
            group = tuple(entries)
            for offset in range(0, len(group), config.execution.worker_count):
                if _source_provenance(repo_root) != provenance:
                    raise ValueError("capture source or runtime changed during the campaign")
                _check_space(
                    run_dir=run_dir,
                    settings=settings,
                    remaining=len(flights) - len(completed),
                    workers=config.execution.worker_count,
                    completed=tuple(completed.values()),
                )
                batch = group[offset : offset + config.execution.worker_count]
                futures = tuple(
                    executor.submit(
                        _run_flight,
                        repo_root=repo_root,
                        config_path=config_path,
                        config=config,
                        run_dir=run_dir,
                        flight=flight,
                        settings=settings,
                    )
                    for flight in batch
                )
                for flight, future in zip(batch, futures, strict=True):
                    completed[flight.directory] = future.result()
    if _source_provenance(repo_root) != provenance:
        raise ValueError("capture source or runtime changed during the campaign")
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
    _write_stability_report(path=run_dir / "stability.json", report=report)
    finalize_flight_parameters_bundle(run_dir)
    latest = publish_flight_parameters(run_dir)
    print(f"Published flight parameters: {run_dir}")
    print(f"Latest: {latest} -> {latest.readlink()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--processing-config", type=Path, default=Path("configs/flight/processing/default.yaml")
    )
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
    config = load_flight_parameters_config(config_path)
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
        if (run_dir / "flight_parameters.json").exists():
            completed = load_flight_parameters(run_dir)
            if (
                completed.config_fingerprint != config.config_fingerprint
                or completed.run_id != run_id
            ):
                raise ValueError("cannot resume analysis with a different capture configuration")
            _publish_campaign(
                run_dir=run_dir,
                comparison_run_dirs=tuple(args.compare_run),
                settings_path=args.stability_config,
            )
            return

        settings = _processing(args.processing_config)
        flights = _flights(config)
        provenance = _source_provenance(repo_root)
        inputs_path = run_dir / "capture_inputs.json"
        inputs = {
            "schema_version": 1,
            "run_id": run_id,
            "config_fingerprint": config.config_fingerprint,
            "processing": asdict(settings),
            "provenance": provenance,
            "flights": [asdict(flight) for flight in flights],
        }
        _write_json(path=inputs_path, value=inputs)
        records = _collect_flights(
            repo_root=repo_root,
            config_path=config_path,
            config=config,
            run_dir=run_dir,
            flights=flights,
            settings=settings,
            provenance=provenance,
        )
        traces = tuple(
            load_flight_trace(run_dir / flight.directory / "trace.json")
            for flight in flights
            if flight.kind == "hover"
        )
        response_traces = tuple(
            record.response for record in records if record.response is not None
        )
        parameters = derive_flight_parameters(
            config=config,
            run_id=run_id,
            traces=traces,
            response_traces=response_traces,
            source_commit=provenance["source_commit"],
            source_dirty=provenance["source_dirty"],
            created_at_utc=created_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        )
        campaign_path = run_dir / "campaign.json"
        if not campaign_path.exists():
            _write_json(
                path=campaign_path,
                value={
                    "artifact_kind": "robotics-experiments-flight-parameters-campaign",
                    "schema_version": 1,
                    "flight_parameters_config": config_path.relative_to(repo_root).as_posix(),
                    "config_fingerprint": config.config_fingerprint,
                    "worker_count": config.execution.worker_count,
                    "hover_trial_count": len(traces),
                    "response_repetition_count": len(response_traces),
                    "failed_capture_attempts": len(
                        tuple((run_dir / "failed").glob("*/*/attempt-*"))
                    ),
                    "raw_log_bytes": sum(record.raw_log_bytes for record in records),
                    "compressed_log_bytes": sum(record.compressed_log_bytes for record in records),
                    "capture_wall_s": sum(record.capture_wall_s for record in records),
                    "processing_wall_s": sum(record.processing_wall_s for record in records),
                },
            )
        write_flight_parameters(flight_parameters=parameters, run_dir=run_dir)
        _publish_campaign(
            run_dir=run_dir,
            comparison_run_dirs=tuple(args.compare_run),
            settings_path=args.stability_config,
        )


if __name__ == "__main__":
    main()
