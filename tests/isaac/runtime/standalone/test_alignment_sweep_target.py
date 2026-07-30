"""Host-safe tests for deterministic alignment worker coordination."""

import importlib.util
from types import ModuleType

import pytest

from robotics_experiments.alignment import load_controller_definition
from tests.support.paths import REPO_ROOT


def _load_script() -> ModuleType:
    path = REPO_ROOT / "isaac/runtime/standalone/alignment_sweep_target.py"
    spec = importlib.util.spec_from_file_location("alignment_sweep_target", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _settings(count: int):
    definition = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"
    )
    return tuple(definition.base_settings for _index in range(count))


def test_worker_shards_are_balanced_contiguous_and_deterministic():
    module = _load_script()

    shards = module._partition_settings(_settings(8), 3)  # noqa: SLF001 - focused helper test

    assert [[index for index, _setting in shard] for shard in shards] == [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7],
    ]


def test_worker_results_merge_in_candidate_order():
    module = _load_script()

    merged = module._merge_indexed_results(  # noqa: SLF001 - focused protocol test
        result_count=4,
        shards=[
            ((2, "two"), (3, "three")),
            ((0, "zero"), (1, "one")),
        ],
    )

    assert merged == ("zero", "one", "two", "three")


@pytest.mark.parametrize(
    "shards, message",
    [
        ([((0, "first"), (0, "duplicate"))], "duplicate"),
        ([((0, "first"),)], "omitted"),
    ],
)
def test_worker_merge_rejects_corrupt_protocol(shards, message):
    module = _load_script()

    with pytest.raises(RuntimeError, match=message):
        module._merge_indexed_results(  # noqa: SLF001 - focused protocol test
            result_count=2, shards=shards
        )
