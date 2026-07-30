"""Host-safe tests for alignment fitting artifact headers."""

import importlib.util
import json
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
