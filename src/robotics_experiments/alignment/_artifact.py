"""Compact controller contracts, compressed selection records, and publication."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from robotics_experiments.alignment._io import (
    _canonical_json,
    _fingerprint,
    _finite,
    _interface,
    _load_v6_alignment_for_migration,
    _object,
    _parameters,
    _settings,
    _string,
)
from robotics_experiments.alignment._metrics import metric
from robotics_experiments.alignment._models import (
    AlignmentResult,
    CandidateScore,
    ControllerArtifact,
    ControllerSettings,
)

__all__: list[str] = []

_CONTROLLER_SCHEMA_VERSION = 1
_SELECTION_SCHEMA_VERSION = 1
_MANIFEST_SCHEMA_VERSION = 1
_SETTINGS_FIELDS = {item.name for item in fields(ControllerSettings)}


def _plain(value: Any, /) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _write_new_text(*, path: Path, payload: str) -> None:
    if path.exists():
        raise FileExistsError(f"alignment result already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload)
    os.replace(temporary, path)


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


def _zstd_uncompressed_size(path: Path, /) -> int:
    listing = _zstd(arguments=("-l", "-v", str(path))).stdout.decode()
    for line in listing.splitlines():
        if "Decompressed Size:" in line:
            detail = line.split("Decompressed Size:", maxsplit=1)[1]
            value = detail.rsplit("(", maxsplit=1)[1].split(" B)", maxsplit=1)[0]
            return int(value.replace(",", ""))
    raise RuntimeError(f"zstd did not report the uncompressed size for {path}")


def resolve_controller_path(path: str | Path, /) -> Path:
    """Resolve a controller file, run directory, or publication link."""

    artifact_path = Path(path)
    if artifact_path.name == "alignment.json":
        raise ValueError("legacy alignment.json artifacts are unsupported")
    if artifact_path.is_dir():
        artifact_path = artifact_path / "controller.json"
    return artifact_path


def _controller_from_payload(*, artifact: dict[str, Any], name: str) -> ControllerArtifact:
    expected = {item.name for item in fields(ControllerArtifact)}
    missing = expected - artifact.keys()
    extra = artifact.keys() - expected
    if missing or extra:
        raise ValueError(
            f"{name} fields mismatch: missing={sorted(missing)!r}, extra={sorted(extra)!r}"
        )
    if artifact["schema_version"] != _CONTROLLER_SCHEMA_VERSION:
        raise ValueError(f"{name} has unsupported schema_version={artifact['schema_version']!r}")
    target_backend = _string(value=artifact["target_backend"], name="target_backend")
    reference_backend = _string(value=artifact["reference_backend"], name="reference_backend")
    if target_backend != "isaac-lab" or reference_backend != "px4-sitl":
        raise ValueError(
            "controller artifact has unsupported target/reference backend contract "
            f"{target_backend!r}/{reference_backend!r}"
        )
    settings = _settings(value=artifact["selected_settings"], name="selected_settings")
    settings_fingerprint = _string(
        value=artifact["settings_fingerprint"], name="settings_fingerprint"
    )
    if settings_fingerprint != _fingerprint(asdict(settings)):
        raise ValueError("controller artifact settings_fingerprint does not match settings")
    quality = artifact["advisory_quality_passed"]
    dirty = artifact["source_dirty"]
    if type(quality) is not bool or type(dirty) is not bool:
        raise ValueError("controller artifact quality and dirty provenance must be booleans")
    reasons = artifact["advisory_quality_reasons"]
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("advisory_quality_reasons must be a string list")
    return ControllerArtifact(
        schema_version=1,
        alignment_id=_string(value=artifact["alignment_id"], name="alignment_id"),
        run_id=_string(value=artifact["run_id"], name="run_id"),
        target_controller_id=_string(
            value=artifact["target_controller_id"], name="target_controller_id"
        ),
        reference_controller_id=_string(
            value=artifact["reference_controller_id"], name="reference_controller_id"
        ),
        target_backend=target_backend,
        reference_backend=reference_backend,
        target_definition_fingerprint=_string(
            value=artifact["target_definition_fingerprint"],
            name="target_definition_fingerprint",
        ),
        reference_definition_fingerprint=_string(
            value=artifact["reference_definition_fingerprint"],
            name="reference_definition_fingerprint",
        ),
        comparison_fingerprint=_string(
            value=artifact["comparison_fingerprint"], name="comparison_fingerprint"
        ),
        experiment_config=_string(value=artifact["experiment_config"], name="experiment_config"),
        experiment_fingerprint=_string(
            value=artifact["experiment_fingerprint"], name="experiment_fingerprint"
        ),
        vehicle=_string(value=artifact["vehicle"], name="vehicle"),
        airframe=_string(value=artifact["airframe"], name="airframe"),
        command_rate_hz=_finite(
            value=artifact["command_rate_hz"], name="command_rate_hz", positive=True
        ),
        command_interface=_interface(value=artifact["command_interface"], name="command_interface"),
        probe_suite_fingerprint=_string(
            value=artifact["probe_suite_fingerprint"], name="probe_suite_fingerprint"
        ),
        selected_settings=settings,
        settings_fingerprint=settings_fingerprint,
        advisory_quality_passed=quality,
        advisory_quality_reasons=tuple(reasons),
        source_commit=_string(value=artifact["source_commit"], name="source_commit"),
        source_dirty=dirty,
        target_image=_string(value=artifact["target_image"], name="target_image"),
        reference_image=_string(value=artifact["reference_image"], name="reference_image"),
    )


def load_controller_artifact(path: str | Path, /) -> ControllerArtifact:
    """Load and strictly validate the compact training controller contract."""

    artifact_path = resolve_controller_path(path)
    try:
        loaded = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load controller artifact {artifact_path}: {exc}") from exc
    return _controller_from_payload(
        artifact=_object(value=loaded, name=str(artifact_path)),
        name="controller artifact",
    )


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
            "settings_overrides",
            "metrics",
            "ranking_values",
            "usable",
            "failure_reason",
        }
        if set(row) != expected_row or row["index"] != row_index:
            raise ValueError(f"{name} fields or index mismatch")
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


def finalize_alignment_bundle(
    *,
    run_dir: Path,
    compressed_target_source: bool = False,
    migration_provenance: Mapping[str, Any] | None = None,
) -> Path:
    """Reorganize, compress, inventory, and validate one completed run."""

    controller = load_controller_artifact(run_dir)
    load_selection(run_dir=run_dir, controller=controller)
    evidence = run_dir / "evidence"
    diagnostics = run_dir / "diagnostics"
    stages = diagnostics / "stages"
    acceptance = diagnostics / "acceptance"
    evidence.mkdir(exist_ok=True)
    diagnostics.mkdir(exist_ok=True)
    stages.mkdir(exist_ok=True)
    acceptance.mkdir(exist_ok=True)
    moves = {
        "reference.json": evidence / "reference_traces.json",
        "reference.ulg": evidence / "reference.ulg",
        "bridge.log": evidence / "bridge.log",
        "actuator_audit.json": diagnostics / "actuator_audit.json",
        "controller_boundary_replay.json": diagnostics / "controller_boundary_replay.json",
        "gpu_utilization.json": diagnostics / "gpu_utilization.json",
        "preflight-alignment.json": stages / "preflight-alignment.json",
        "search_realization.json": stages / "search_realization.json",
        "manifest.json": stages / "capture_manifest.json",
    }
    optional_moves = {"preflight-alignment.json", "search_realization.json", "manifest.json"}
    for source_name, destination in moves.items():
        source = run_dir / source_name
        if not source.is_file():
            if source_name in optional_moves:
                continue
            raise FileNotFoundError(f"required alignment output is missing: {source}")
        source.rename(destination)
    target_source = run_dir / (
        "target_candidates.json.zst" if compressed_target_source else "target_candidates.json"
    )
    target_destination = evidence / "target_traces.json.zst"
    if not target_source.is_file():
        raise FileNotFoundError(f"required alignment output is missing: {target_source}")
    if compressed_target_source:
        shutil.copy2(target_source, target_destination)
        _zstd(arguments=("-q", "-t", str(target_destination)))
    else:
        _compress_file(
            source=target_source,
            destination=target_destination,
            compression_arguments=("-8", "--long=27", "-T0"),
        )
    target_source.unlink()
    role_by_path = {
        "controller.json": "controller",
        "selection.json.zst": "selection",
        "evidence/reference_traces.json": "reference-traces",
        "evidence/reference.ulg": "reference-ulog",
        "evidence/bridge.log": "bridge-log",
        "evidence/target_traces.json.zst": "target-traces",
        "diagnostics/actuator_audit.json": "actuator-audit",
        "diagnostics/controller_boundary_replay.json": "controller-boundary-replay",
        "diagnostics/gpu_utilization.json": "gpu-utilization",
        "diagnostics/stages/preflight-alignment.json": "preflight",
        "diagnostics/stages/search_realization.json": "search-realization",
        "diagnostics/stages/capture_manifest.json": "capture-manifest",
    }
    role_by_path = {
        relative: role for relative, role in role_by_path.items() if (run_dir / relative).is_file()
    }
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(run_dir).as_posix()
        role_by_path.setdefault(
            relative,
            (
                "acceptance-log"
                if relative.startswith("diagnostics/acceptance/")
                else "stage-diagnostic"
            ),
        )
    manifest = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "alignment_id": controller.alignment_id,
        "run_id": controller.run_id,
        "files": [
            _manifest_entry(role=role, path=run_dir / relative, run_dir=run_dir)
            for relative, role in role_by_path.items()
        ],
    }
    if migration_provenance is not None:
        manifest["migration"] = dict(migration_provenance)
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
    if "migration" in manifest:
        expected_manifest.add("migration")
        if not isinstance(manifest["migration"], dict):
            raise ValueError("alignment manifest migration provenance must be a mapping")
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
    result = load_selection(run_dir=run_dir, controller=controller)
    ranking = result.candidates[result.selected_candidate_index].ranking_values
    root = run_dir.parent
    best_link = root / "best"
    replace_best = True
    if best_link.exists() and (best_link / "controller.json").is_file():
        best = load_controller_artifact(best_link)
        if best.comparison_fingerprint == controller.comparison_fingerprint:
            best_result = load_selection(run_dir=best_link.resolve(), controller=best)
            best_ranking = best_result.candidates[
                best_result.selected_candidate_index
            ].ranking_values
            replace_best = (ranking, controller.run_id) < (best_ranking, best.run_id)
    if replace_best:
        _replace_relative_symlink(link=best_link, target=run_dir)
    _replace_relative_symlink(link=root / "latest", target=run_dir)


def _summary_actuator_audit(source: Path, destination: Path, /) -> None:
    payload = _object(value=json.loads(source.read_text()), name=str(source))
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise ValueError("legacy actuator audit profiles must be a list")
    summary_profiles = []
    for raw_profile in profiles:
        profile = _object(value=raw_profile, name="legacy actuator audit profile")
        topics = {}
        for topic_name in (
            "vehicle_thrust_setpoint",
            "vehicle_torque_setpoint",
            "actuator_motors",
            "hover_thrust_estimate",
        ):
            topic = _object(value=profile[topic_name], name=topic_name)
            timestamps = topic.get("timestamp_us")
            if not isinstance(timestamps, list):
                raise ValueError(f"legacy actuator audit {topic_name} timestamps must be a list")
            topics[topic_name] = {
                "sample_count": len(timestamps),
                "first_timestamp_us": timestamps[0] if timestamps else None,
                "last_timestamp_us": timestamps[-1] if timestamps else None,
            }
        bridge = _object(value=profile["bridge"], name="legacy actuator audit bridge")
        actuator_timestamps = bridge.get("actuator_timestamp_us")
        if not isinstance(actuator_timestamps, list):
            raise ValueError("legacy actuator audit bridge timestamps must be a list")
        summary_profiles.append(
            {
                "profile": profile["profile"],
                "split": profile["split"],
                "axis": profile["axis"],
                "start_sim_time_us": profile["start_sim_time_us"],
                "end_sim_time_us": profile["end_sim_time_us"],
                "topic_coverage": topics,
                "bridge": {"sample_count": len(actuator_timestamps)},
                "metrics": profile["metrics"],
            }
        )
    summary = {
        key: payload[key]
        for key in (
            "schema_version",
            "method",
            "config",
            "reference",
            "bridge_log",
            "airframe",
            "composed_mass_kg",
            "limits",
            "passed",
            "failures",
        )
    }
    summary["profiles"] = summary_profiles
    destination.write_text(_canonical_json(summary) + "\n")


def migrate_v6_alignment(*, source_dir: Path, run_dir: Path, run_id: str) -> None:
    """Build and publish a self-contained controller-v1 sibling from schema v6."""

    if run_dir.exists():
        raise FileExistsError(f"migration destination already exists: {run_dir}")
    checksum_path = source_dir / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"migration source is missing {checksum_path.name}")
    for line in checksum_path.read_text().splitlines():
        digest, relative = line.split(maxsplit=1)
        source = source_dir / relative.lstrip("* ")
        if not source.is_file() or _sha256(source) != digest:
            raise ValueError(f"legacy checksum mismatch for {relative}")
    legacy_path = source_dir / "alignment.json"
    legacy = _load_v6_alignment_for_migration(legacy_path)
    comparison_fingerprint = _fingerprint(
        {
            "controller_artifact_schema": 1,
            "legacy_comparison_fingerprint": legacy.comparison_fingerprint,
        }
    )
    controller = ControllerArtifact(
        schema_version=1,
        alignment_id=legacy.alignment_id,
        run_id=run_id,
        target_controller_id=legacy.target_controller_id,
        reference_controller_id=legacy.reference_controller_id,
        target_backend=legacy.target_backend,
        reference_backend=legacy.reference_backend,
        target_definition_fingerprint=legacy.target_definition_fingerprint,
        reference_definition_fingerprint=legacy.reference_definition_fingerprint,
        comparison_fingerprint=comparison_fingerprint,
        experiment_config=legacy.experiment_config,
        experiment_fingerprint=legacy.experiment_fingerprint,
        vehicle=legacy.vehicle,
        airframe=legacy.airframe,
        command_rate_hz=legacy.command_rate_hz,
        command_interface=legacy.command_interface,
        probe_suite_fingerprint=legacy.probe_suite_fingerprint,
        selected_settings=legacy.selected_settings,
        settings_fingerprint=_fingerprint(asdict(legacy.selected_settings)),
        advisory_quality_passed=legacy.advisory_quality_passed,
        advisory_quality_reasons=legacy.advisory_quality_reasons,
        source_commit=legacy.source_commit,
        source_dirty=legacy.source_dirty,
        target_image=legacy.target_image,
        reference_image=legacy.reference_image,
    )
    result = AlignmentResult(
        controller=controller,
        selection_metric=legacy.selection_metric,
        base_settings=legacy.candidates[0].settings,
        declared_parameters=legacy.declared_parameters,
        candidates=legacy.candidates,
        selected_candidate_index=legacy.selected_candidate_index,
    )
    run_dir.mkdir(parents=True)
    try:
        write_alignment_result(result=result, run_dir=run_dir)
        for name in (
            "reference.json",
            "reference.ulg",
            "bridge.log",
            "controller_boundary_replay.json",
            "gpu_utilization.json",
            "target_candidates.json.zst",
        ):
            shutil.copy2(source_dir / name, run_dir / name)
        _summary_actuator_audit(
            source_dir / "actuator_audit.json",
            run_dir / "actuator_audit.json",
        )
        acceptance = source_dir / "acceptance"
        if acceptance.is_dir():
            shutil.copytree(acceptance, run_dir / "diagnostics" / "acceptance")
        provenance = {
            "source_alignment_id": legacy.alignment_id,
            "source_run_id": legacy.run_id,
            "source_schema_version": legacy.schema_version,
            "source_artifact_sha256": _sha256(legacy_path),
            "source_checksum_file_sha256": _sha256(checksum_path),
            "legacy_comparison_fingerprint": legacy.comparison_fingerprint,
        }
        finalize_alignment_bundle(
            run_dir=run_dir,
            compressed_target_source=True,
            migration_provenance=provenance,
        )
        publish_controller(run_dir=run_dir)
    except Exception:
        shutil.rmtree(run_dir)
        raise
