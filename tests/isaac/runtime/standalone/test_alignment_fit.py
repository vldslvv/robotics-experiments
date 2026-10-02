"""Host-safe tests for alignment fitting artifact headers."""

import importlib.util
import json
import sys
from dataclasses import replace
from types import ModuleType

import pytest

from tests.support.paths import REPO_ROOT


def _load_script() -> ModuleType:
    path = REPO_ROOT / "isaac/runtime/standalone/alignment_fit.py"
    spec = importlib.util.spec_from_file_location("alignment_fit", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("data_key", ["candidates", "traces"])
def test_diagnostic_image_reads_compact_streamed_header(tmp_path, data_key):
    module = _load_script()
    path = tmp_path / "diagnostics.json"
    image = 'registry/image:tag-"escaped"'
    header = '{"schema_version":3,"image":' + json.dumps(image) + f',"{data_key}":{{"large":"'
    path.write_text(header + "x" * (128 * 1024), encoding="utf-8")

    assert module._diagnostic_image(path) == image  # noqa: SLF001 - focused reader test


def test_diagnostic_image_rejects_image_after_trace_data(tmp_path):
    module = _load_script()
    path = tmp_path / "diagnostics.json"
    path.write_text('{"traces":[],"image":"late"}', encoding="utf-8")

    with pytest.raises(ValueError, match="no image header"):
        module._diagnostic_image(path)  # noqa: SLF001 - focused reader test


def test_fit_uses_prepared_bundle_after_authored_label_moves(tmp_path, monkeypatch):
    module = _load_script()
    target_path = REPO_ROOT / "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"
    reference_path = REPO_ROOT / "configs/alignment/controllers/px4_sitl_x500_velocity.yaml"
    producer = tmp_path / "producer"
    producer.mkdir()
    newer_producer = tmp_path / "newer-producer"
    newer_producer.mkdir()
    latest = tmp_path / "latest"
    latest.symlink_to(newer_producer, target_is_directory=True)
    run_dir = tmp_path / "alignment"
    run_dir.mkdir()
    (run_dir / "flight_parameters").symlink_to(producer, target_is_directory=True)
    for name in ("target_candidates.json", "reference.json"):
        (run_dir / name).write_text(json.dumps({"image": "captured-image", "traces": []}))
    authored = {
        target_path: replace(
            module.load_controller_definition(target_path),
            flight_parameters_artifact=str(producer),
        ),
        reference_path: replace(
            module.load_controller_definition(reference_path),
            flight_parameters_artifact=str(latest),
        ),
    }
    monkeypatch.setattr(module, "load_controller_definition", authored.__getitem__)
    monkeypatch.setenv("ROBOTICS_ALIGNMENT_RUN_DIR", str(run_dir))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "alignment_fit.py",
            "--target",
            str(target_path),
            "--reference",
            str(reference_path),
            "--run-dir",
            str(run_dir),
            "--run-id",
            "prepared-run",
        ],
    )

    def check_definitions(target, reference, /):
        for definition, source_path in ((target, target_path), (reference, reference_path)):
            assert definition.flight_parameters_artifact == str(producer)
            assert definition.definition_fingerprint == authored[source_path].definition_fingerprint
            assert definition.base_settings == authored[source_path].base_settings
            assert definition.image == "captured-image"
        raise RuntimeError("prepared definitions checked")

    monkeypatch.setattr(module, "align", check_definitions)
    with pytest.raises(RuntimeError, match="prepared definitions checked"):
        module.main()
