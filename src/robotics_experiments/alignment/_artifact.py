"""Compact controller contracts, compressed selection records, and publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._evaluation import evaluate_final
from robotics_experiments.alignment._flight_parameters import load_alignment_dataset
from robotics_experiments.alignment._io import (
    _canonical_json,
    _finite,
    _object,
    _parameters,
    _plain,
    _settings,
    _string,
)
from robotics_experiments.alignment._metrics import metric
from robotics_experiments.alignment._models import (
    AlignmentResult,
    CandidateScore,
    ControllerArtifact,
    ControllerSettings,
    QualityGateResult,
)
from robotics_experiments.controller import load_controller_artifact

__all__: list[str] = []

_PENDING_QUALIFICATION = "controller audit and integration qualification pending"

_CONTROLLER_SCHEMA_VERSION = 1
_SELECTION_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1
_SETTINGS_FIELDS = {item.name for item in fields(ControllerSettings)}


def _replace_text(*, path: Path, payload: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


def _write_new_text(*, path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"alignment result already exists: {path}")
    _replace_text(path=path, payload=payload)


def _zstd(*, arguments: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    executable = shutil.which("zstd")
    if executable is None:
        raise RuntimeError("zstd executable is required to finalize alignment artifacts")
    try:
        return subprocess.run(
            [executable, *arguments],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"zstd failed: {detail}") from exc


def _compress_file(
    *,
    source: Path,
    destination: Path,
    compression_arguments: Sequence[str] = ("-19", "-T1"),
) -> None:
    if destination.exists():
        raise FileExistsError(f"alignment result already exists: {destination}")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        _zstd(
            arguments=(
                "-q",
                "-f",
                *compression_arguments,
                str(source),
                "-o",
                str(temporary),
            )
        )
        if _zstd(arguments=("-q", "-t", str(temporary))).returncode:
            raise RuntimeError(
                f"compressed alignment payload verification failed for {destination}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _compress_json(*, payload: str, destination: Path) -> None:
    source = destination.with_name(f".{destination.name}.{os.getpid()}.json")
    source.write_text(payload)
    try:
        _compress_file(source=source, destination=destination)
        if _zstd(arguments=("-q", "-d", "-c", str(destination))).stdout != payload.encode():
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"compressed alignment payload verification failed for {destination}"
            )
    finally:
        source.unlink(missing_ok=True)


def _read_zstd_json(*, path: Path, name: str) -> dict[str, Any]:
    try:
        loaded = json.loads(_zstd(arguments=("-q", "-d", "-c", str(path))).stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"cannot load {name} {path}: {exc}") from exc
    return _object(value=loaded, name=str(path))


def _require_passed_diagnostic(*, path: Path, name: str) -> None:
    try:
        payload = _object(value=json.loads(path.read_text()), name=name)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load {name} {path}: {exc}") from exc
    if payload.get("schema_version") != 1 or type(payload.get("passed")) is not bool:
        raise ValueError(f"{name} must be a schema-1 pass/fail report")
    if not payload["passed"]:
        failures = payload.get("failures")
        detail = "; ".join(failures) if isinstance(failures, list) else "reported failure"
        raise ValueError(f"{name} failed: {detail}")


def _zstd_uncompressed_size(path: Path, /) -> int:
    listing = _zstd(arguments=("-l", "-v", str(path))).stdout.decode()
    for line in listing.splitlines():
        if "Decompressed Size:" in line:
            detail = line.split("Decompressed Size:", maxsplit=1)[1]
            value = detail.rsplit("(", maxsplit=1)[1].split(" B)", maxsplit=1)[0]
            return int(value.replace(",", ""))
    raise RuntimeError(f"zstd did not report the uncompressed size for {path}")


def _selection_payload(result: AlignmentResult, /) -> dict[str, Any]:
    base = asdict(result.base_settings)
    return {
        "schema_version": _SELECTION_SCHEMA_VERSION,
        "alignment_id": result.controller.alignment_id,
        "run_id": result.controller.run_id,
        "selection_metric": result.selection_metric,
        "base_settings": base,
        "declared_parameters": _plain(result.declared_parameters),
        "selected_candidate_index": result.selected_candidate_index,
        "candidates": [
            {
                "index": candidate.index,
                "set_name": candidate.set_name,
                "settings_overrides": {
                    name: value
                    for name, value in asdict(candidate.settings).items()
                    if value != base[name]
                },
                "metrics": dict(candidate.metrics),
                "ranking_values": list(candidate.ranking_values),
                "usable": candidate.usable,
                "failure_reason": candidate.failure_reason,
            }
            for candidate in result.candidates
        ],
    }


def load_selection(*, run_dir: Path, controller: ControllerArtifact) -> AlignmentResult:
    """Reconstruct and validate every candidate from compressed overrides."""

    selection = _read_zstd_json(path=run_dir / "selection.json.zst", name="selection artifact")
    expected = {
        "schema_version",
        "alignment_id",
        "run_id",
        "selection_metric",
        "base_settings",
        "declared_parameters",
        "selected_candidate_index",
        "candidates",
    }
    if set(selection) != expected:
        raise ValueError("selection artifact fields mismatch")
    if selection["schema_version"] != _SELECTION_SCHEMA_VERSION:
        raise ValueError("selection artifact has unsupported schema_version")
    if (
        selection["alignment_id"] != controller.alignment_id
        or selection["run_id"] != controller.run_id
    ):
        raise ValueError("controller and selection identities differ")
    base = _settings(value=selection["base_settings"], name="base_settings")
    declared = _parameters(value=selection["declared_parameters"], base_settings=base)
    rows = selection["candidates"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("selection candidates must be a non-empty list")
    candidates = []
    base_payload = asdict(base)
    for row_index, raw_row in enumerate(rows):
        name = f"candidates[{row_index}]"
        row = _object(value=raw_row, name=name)
        expected_row = {
            "index",
            "set_name",
            "settings_overrides",
            "metrics",
            "ranking_values",
            "usable",
            "failure_reason",
        }
        if set(row) != expected_row or row["index"] != row_index:
            raise ValueError(f"{name} fields or index mismatch")
        if row["set_name"] not in {"train", "validation"}:
            raise ValueError(f"{name} has an invalid selection recording set")
        overrides = _object(value=row["settings_overrides"], name=f"{name}.settings_overrides")
        extra = overrides.keys() - _SETTINGS_FIELDS
        if extra:
            raise ValueError(f"{name} has unknown settings overrides {sorted(extra)!r}")
        settings_payload = dict(base_payload)
        settings_payload.update(overrides)
        metrics = _object(value=row["metrics"], name=f"{name}.metrics")
        ranking = row["ranking_values"]
        usable = row["usable"]
        failure = row["failure_reason"]
        if not isinstance(ranking, list) or type(usable) is not bool or (usable and not ranking):
            raise ValueError(f"{name}.usable/ranking_values are inconsistent")
        if failure is not None and not isinstance(failure, str):
            raise ValueError(f"{name}.failure_reason must be a string or null")
        candidates.append(
            CandidateScore(
                index=row_index,
                set_name=row["set_name"],
                settings=_settings(value=settings_payload, name=f"{name}.settings"),
                usable=usable,
                metrics={
                    key: _finite(value=value, name=f"{name}.metrics.{key}")
                    for key, value in metrics.items()
                },
                ranking_values=tuple(
                    _finite(value=value, name=f"{name}.ranking_values") for value in ranking
                ),
                failure_reason=failure,
            )
        )
    selected_index = selection["selected_candidate_index"]
    if type(selected_index) is not int or not 0 <= selected_index < len(candidates):
        raise ValueError("selected_candidate_index is out of range")
    selected = candidates[selected_index]
    if not selected.usable or selected.settings != controller.selected_settings:
        raise ValueError("selected candidate does not match controller settings")
    selection_metric = _string(value=selection["selection_metric"], name="selection_metric")
    metric(name=selection_metric)
    return AlignmentResult(
        controller=controller,
        selection_metric=selection_metric,
        base_settings=base,
        declared_parameters=declared,
        candidates=tuple(candidates),
        selected_candidate_index=selected_index,
    )


def write_alignment_result(*, result: AlignmentResult, run_dir: Path) -> tuple[Path, Path]:
    """Write deterministic controller and compressed selection contracts."""

    run_dir.mkdir(parents=True, exist_ok=True)
    controller_path = run_dir / "controller.json"
    selection_path = run_dir / "selection.json.zst"
    controller_payload = _canonical_json(_plain(result.controller)) + "\n"
    if len(controller_payload.encode()) >= 16 * 1024:
        raise ValueError("controller.json must remain below 16 KiB")
    _write_new_text(path=controller_path, payload=controller_payload)
    try:
        _compress_json(
            payload=_canonical_json(_selection_payload(result)) + "\n",
            destination=selection_path,
        )
        load_selection(run_dir=run_dir, controller=load_controller_artifact(controller_path))
    except Exception:
        controller_path.unlink(missing_ok=True)
        selection_path.unlink(missing_ok=True)
        raise
    return controller_path, selection_path


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(*, role: str, path: Path, run_dir: Path) -> dict[str, Any]:
    size = path.stat().st_size
    compressed = path.suffix == ".zst"
    return {
        "role": role,
        "path": path.relative_to(run_dir).as_posix(),
        "size_bytes": size,
        "sha256": _sha256(path),
        "compression": "zstd" if compressed else "none",
        "uncompressed_size_bytes": (_zstd_uncompressed_size(path) if compressed else size),
    }


def finalize_alignment_bundle(*, run_dir: Path) -> Path:
    """Evaluate completed evidence, compress outputs in place, and inventory it."""
    controller = load_controller_artifact(run_dir)
    load_selection(run_dir=run_dir, controller=controller)
    dataset, _membership = load_alignment_dataset(run_dir / "dataset.json")
    if (
        dataset.artifact_digest != controller.flight_parameters_fingerprint
        or dataset.run_id != controller.flight_parameters_run_id
    ):
        raise ValueError("finalization source does not match the selected controller")
    for role in ("train", "validation", "test"):
        directory = run_dir if role == "train" else run_dir / role
        _require_passed_diagnostic(
            path=directory / "actuator_audit.json", name=f"{role} input audit"
        )
        replay = json.loads((directory / "controller_boundary_replay.json").read_text())
        if (
            not replay.get("completed")
            or replay.get("settings_fingerprint") != controller.settings_fingerprint
            or replay.get("source_digest") != dataset.artifact_digest
            or replay.get("set_name") != role
        ):
            raise ValueError(f"{role} controller replay did not complete for the selected inputs")
    qualification = json.loads((run_dir / "qualification.json").read_text())
    integration = json.loads((run_dir / "integration.json").read_text())
    for report in (qualification, integration):
        if (
            not report.get("completed")
            or report.get("settings_fingerprint") != controller.settings_fingerprint
            or report.get("flight_parameters_fingerprint")
            != controller.flight_parameters_fingerprint
        ):
            raise ValueError("evaluation did not complete for the selected controller inputs")
    if qualification.get("set_name") != "test":
        raise ValueError("final evaluation requires test flights")
    evaluation = evaluate_final(
        response_checks=tuple(QualityGateResult(**check) for check in qualification["checks"]),
        live_checks=tuple(QualityGateResult(**check) for check in integration["checks"]),
        replay=json.loads((run_dir / "test/controller_boundary_replay.json").read_text()),
    )
    qualification.update(
        score=evaluation.score,
        passed=evaluation.accepted,
        failures=evaluation.failures,
        checks=[_plain(check) for check in evaluation.checks],
    )
    _replace_text(
        path=run_dir / "qualification.json", payload=_canonical_json(qualification) + "\n"
    )
    for role in ("train", "validation", "test"):
        directory = run_dir if role == "train" else run_dir / role
        source = directory / "target_candidates.json"
        destination = source.with_suffix(".json.zst")
        _compress_file(
            source=source,
            destination=destination,
            compression_arguments=("-8", "--long=27", "-T0"),
        )
        source.unlink()
    controller = replace(
        controller,
        advisory_quality_passed=evaluation.accepted,
        advisory_quality_reasons=evaluation.failures,
    )
    _replace_text(
        path=run_dir / "controller.json", payload=_canonical_json(_plain(controller)) + "\n"
    )
    _replace_text(
        path=run_dir / "progress.json",
        payload=_canonical_json(
            {
                "phase": "complete",
                "accepted": evaluation.accepted,
                "test/final_score": evaluation.score,
                "artifact": str(run_dir / "controller.json"),
            }
        )
        + "\n",
    )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "alignment_id": controller.alignment_id,
        "run_id": controller.run_id,
        "files": [
            _manifest_entry(role=path.stem, path=path, run_dir=run_dir)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    manifest_path = run_dir / "manifest.json"
    _write_new_text(path=manifest_path, payload=_canonical_json(manifest) + "\n")
    validate_alignment_bundle(run_dir)
    return manifest_path


def validate_alignment_bundle(run_dir: Path, /) -> None:
    """Validate controller/selection consistency and every manifest payload."""

    controller = load_controller_artifact(run_dir)
    load_selection(run_dir=run_dir, controller=controller)
    manifest_path = run_dir / "manifest.json"
    try:
        manifest = _object(value=json.loads(manifest_path.read_text()), name=str(manifest_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load alignment manifest {manifest_path}: {exc}") from exc
    expected_manifest = {"schema_version", "alignment_id", "run_id", "files"}
    if set(manifest) != expected_manifest:
        raise ValueError("alignment manifest fields mismatch")
    if manifest["schema_version"] != _MANIFEST_SCHEMA_VERSION:
        raise ValueError("alignment manifest has unsupported schema_version")
    if (
        manifest["alignment_id"] != controller.alignment_id
        or manifest["run_id"] != controller.run_id
    ):
        raise ValueError("controller and manifest identities differ")
    entries = manifest["files"]
    if not isinstance(entries, list) or not entries:
        raise ValueError("alignment manifest files must be a non-empty list")
    inventoried_paths: set[Path] = set()
    for index, raw_entry in enumerate(entries):
        entry = _object(value=raw_entry, name=f"manifest.files[{index}]")
        expected = {
            "role",
            "path",
            "size_bytes",
            "sha256",
            "compression",
            "uncompressed_size_bytes",
        }
        if set(entry) != expected:
            raise ValueError(f"manifest.files[{index}] fields mismatch")
        relative = Path(_string(value=entry["path"], name=f"manifest.files[{index}].path"))
        if relative.is_absolute() or ".." in relative.parts or relative == Path("manifest.json"):
            raise ValueError("manifest paths must be relative payload paths")
        if relative in inventoried_paths:
            raise ValueError(f"manifest contains duplicate path {relative}")
        inventoried_paths.add(relative)
        path = run_dir / relative
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"manifest size mismatch for {relative}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"manifest hash mismatch for {relative}")
        uncompressed_size = (
            _zstd_uncompressed_size(path) if entry["compression"] == "zstd" else path.stat().st_size
        )
        if uncompressed_size != entry["uncompressed_size_bytes"]:
            raise ValueError(f"manifest uncompressed size mismatch for {relative}")
    payload_paths = {
        path.relative_to(run_dir)
        for path in run_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if inventoried_paths != payload_paths:
        raise ValueError("alignment manifest does not inventory every payload exactly once")


def _replace_relative_symlink(*, link: Path, target: Path) -> None:
    relative_target = os.path.relpath(target, start=link.parent)
    temporary = link.with_name(f".{link.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(relative_target, target_is_directory=True)
    os.replace(temporary, link)


def publish_controller(*, run_dir: Path) -> None:
    """Advance ``latest`` and comparable ``best`` links after bundle validation."""

    validate_alignment_bundle(run_dir)
    controller = load_controller_artifact(run_dir)
    root = run_dir.parent
    best_link = root / "best"
    selected = run_dir
    score = _finite(
        value=json.loads((run_dir / "qualification.json").read_text())["score"],
        name="final score",
        nonnegative=True,
    )
    if best_link.exists():
        best = load_controller_artifact(best_link)
        if best.comparison_fingerprint == controller.comparison_fingerprint:
            validate_alignment_bundle(best_link.resolve())
            best_score = json.loads((best_link / "qualification.json").read_text())["score"]
            if best_score <= score:
                selected = best_link.resolve()
    _replace_relative_symlink(link=best_link, target=selected)
    _replace_relative_symlink(link=root / "latest", target=run_dir)
