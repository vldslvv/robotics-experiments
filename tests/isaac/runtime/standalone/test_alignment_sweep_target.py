"""Host-safe tests for deterministic alignment worker coordination."""

import importlib.util
import json
from dataclasses import replace
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


def test_candidate_expansions_are_chunked_to_fixed_scene_capacity():
    module = _load_script()

    chunks = module._chunk_settings(_settings(10), 4)  # noqa: SLF001 - focused helper test

    assert [len(chunk) for chunk in chunks] == [4, 4, 2]


def test_smaller_probe_family_reuses_one_fixed_environment_stride():
    module = _load_script()
    traces = [{"profile": "roll"}, {"profile": "pitch"}]

    padded = module._pad_family_traces(  # noqa: SLF001 - fixed-scene contract
        reference_traces=traces,
        environment_traces_per_candidate=3,
        candidate_capacity=2,
    )

    assert padded == [
        {"profile": "roll"},
        {"profile": "pitch"},
        {"profile": "pitch"},
        {"profile": "roll"},
        {"profile": "pitch"},
        {"profile": "pitch"},
    ]

    with pytest.raises(ValueError, match="exceeds"):
        module._pad_family_traces(  # noqa: SLF001 - fixed-scene contract
            reference_traces=traces,
            environment_traces_per_candidate=1,
            candidate_capacity=2,
        )


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


def test_target_candidates_are_streamed_to_atomic_json(tmp_path):
    module = _load_script()
    first = _settings(1)[0]
    second = replace(first, command_delay_s=first.command_delay_s + 0.001)
    trace = module.ProbeTrace(
        family="velocity",
        control_mode="velocity",
        family_metric="translational-controller-quality",
        profile="fit_x",
        split="fit",
        axis="x",
        time_s=(0.0,),
        command_velocity_body_flu_mps=((0.0, 0.0, 0.0),),
        command_yaw_rate_radps=(0.0,),
        command_collective_thrust_normalized=(0.0,),
        command_body_rate_flu_radps=((0.0, 0.0, 0.0),),
        position_enu_m=((0.0, 0.0, 0.0),),
        attitude_flu_enu_wxyz=((1.0, 0.0, 0.0, 0.0),),
        velocity_body_flu_mps=((0.0, 0.0, 0.0),),
        velocity_enu_mps=((0.0, 0.0, 0.0),),
        angular_velocity_body_flu_radps=((0.0, 0.0, 0.0),),
        initial_position_enu_m=(0.0, 0.0, 0.0),
        initial_attitude_flu_enu_wxyz=(1.0, 0.0, 0.0, 0.0),
        initial_linear_velocity_enu_mps=(0.0, 0.0, 0.0),
        initial_estimated_linear_velocity_enu_mps=(0.0, 0.0, 0.0),
        initial_angular_velocity_body_flu_radps=(0.0, 0.0, 0.0),
        initial_estimated_angular_velocity_body_flu_radps=(0.0, 0.0, 0.0),
        completed=True,
    )
    path = tmp_path / "target_candidates.json"

    module._write_target_candidates(  # noqa: SLF001 - focused serializer test
        path=path,
        schema_version=3,
        controller_id="controller",
        definition_fingerprint="definition",
        image="image",
        candidates={first: (trace,), second: (trace,)},
        failures={module._settings_fingerprint(second): "failed"},  # noqa: SLF001
    )

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 3
    assert payload["controller_id"] == "controller"
    assert len(payload["candidates"]) == 2
    assert tuple(payload["failures"].values()) == ("failed",)
    assert not path.with_suffix(".json.tmp").exists()


def test_worker_spools_merge_in_candidate_order_with_effective_aliases(tmp_path):
    module = _load_script()
    first = _settings(1)[0]
    alias = replace(first, hover_thrust_normalized=0.5912500001)
    failed = replace(first, command_delay_s=first.command_delay_s + 0.001)
    trace = {
        "profile": "fit_x",
        "split": "fit",
        "axis": "x",
        "time_s": [0.0],
    }
    spool_path = tmp_path / "worker-0.jsonl"
    with spool_path.open("w") as spool:
        module._write_spooled_trace(  # noqa: SLF001 - focused spool contract test
            stream=spool,
            settings=first,
            traces=[trace],
        )
    scores = (
        module.CandidateScore(0, first, True, {"score": 0.0}, (0.0,), None),
        module.CandidateScore(1, alias, True, {"score": 0.0}, (0.0,), None),
        module.CandidateScore(2, failed, False, {}, (), "failed"),
    )
    path = tmp_path / "target_candidates.json"

    module._merge_target_candidate_spools(  # noqa: SLF001 - focused spool contract test
        path=path,
        controller_id="controller",
        definition_fingerprint="definition",
        image="image",
        scores=scores,
        trace_sources={first: first, alias: first},
        spool_paths=(spool_path,),
    )

    payload = json.loads(path.read_text())
    assert list(payload["candidates"]) == [
        module._settings_fingerprint(first),  # noqa: SLF001
        module._settings_fingerprint(alias),  # noqa: SLF001
    ]
    assert list(payload["candidates"].values()) == [[trace], [trace]]
    assert payload["failures"] == {
        module._settings_fingerprint(failed): "failed"  # noqa: SLF001
    }
