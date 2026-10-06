"""Host-safe tests for deterministic alignment worker coordination."""

import importlib.util
import json
from dataclasses import asdict, replace
from types import ModuleType, SimpleNamespace

import pytest
import torch

from robotics_experiments.alignment import load_controller_definition
from robotics_experiments.config import load_config
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


def test_initial_settings_size_fixed_scene_when_no_staged_parameters():
    module = _load_script()
    definition = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"
    )
    seeds = tuple(
        replace(definition.base_settings, command_delay_s=offset * 0.001) for offset in range(1, 10)
    )
    definition = replace(
        definition,
        parameters=(),
        search=replace(definition.search, initial_settings=seeds),
    )

    assert module._maximum_candidate_expansion(definition) == 10  # noqa: SLF001


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


def test_ctbr_initial_rotor_state_uses_first_physical_trace_command():
    torch = pytest.importorskip("torch")
    module = _load_script()
    traces = [
        {
            "command_collective_thrust_normalized": [0.57],
            "command_body_rate_flu_radps": [[0.2, -0.4, 0.1]],
        },
        {
            "command_collective_thrust_normalized": [0.60],
            "command_body_rate_flu_radps": [[0.0, 0.0, 0.0]],
        },
    ]
    params = SimpleNamespace(
        kind="ctbr",
        hover_thrust_normalized=torch.tensor([0.54, 0.60]),
        max_body_rate_radps=2.0,
    )

    actions = module._initial_trace_actions(  # noqa: SLF001 - replay boundary contract
        traces=traces,
        params=params,
        device="cpu",
    )

    assert actions[:, 0].tolist() == pytest.approx([0.57 / 0.54 - 1.0, 0.0], abs=1e-6)
    assert actions[0, 1:4].tolist() == pytest.approx([0.1, -0.2, 0.05])


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
        source_digest="reference-digest",
        image="image",
        scores=scores,
        trace_sources={first: first, alias: first},
        spool_paths=(spool_path,),
    )

    payload = json.loads(path.read_text())
    assert list(payload["candidates"]) == [
        module._fingerprint(asdict(first)),  # noqa: SLF001
        module._fingerprint(asdict(alias)),  # noqa: SLF001
    ]
    assert list(payload["candidates"].values()) == [[trace], [trace]]
    assert payload["failures"] == {
        module._fingerprint(asdict(failed)): "failed"  # noqa: SLF001
    }


def test_replay_preserves_measurement_offset_and_independent_command_times():
    module = _load_script()
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    elapsed = 0.0
    position = 0.0
    velocity = 0.0
    steps = []

    def step(*, action, physics_steps):
        nonlocal elapsed, position, velocity
        velocity = float(action[0, 0]) * config.rl.action.max_body_velocity_mps
        duration = physics_steps / config.sim.physics_hz
        position += velocity * duration
        elapsed += duration
        steps.append(elapsed)

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        actuation_params=config.rl.action,
        apply_controller_settings_batch=lambda **_kwargs: None,
        set_initial_states=lambda **_kwargs: None,
        step_with_physics_steps=step,
        body_velocities=lambda: [(velocity, 0.0, 0.0)],
        flight_states=lambda: [
            ((position, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (velocity, 0.0, 0.0), (0.0, 0.0, 0.0))
        ],
    )
    source = {
        "profile": "timing",
        "control_mode": "velocity",
        "initial_position_enu_m": (0.0, 0.0, 0.0),
        "time_s": [0.032, 0.080, 0.132],
        "command_time_s": [0.0, 0.060, 0.100],
        "command_velocity_body_flu_mps": [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0)],
        "command_yaw_rate_radps": [0.0, 0.0, 0.0],
    }
    with torch.no_grad():
        result = module._candidate_trace_batch(  # noqa: SLF001 - simulator-free replay timing regression
            env=env,
            config=config,
            settings=_settings(1),
            candidate_capacity=1,
            environment_traces_per_candidate=1,
            reference_traces=[source],
            vehicle_label="x500_01",
        )
    assert not isinstance(result[0], str)
    assert result[0][0]["time_s"] == source["time_s"]
    assert [value[0] for value in result[0][0]["position_enu_m"]] == pytest.approx(
        [0.0, 0.002, 0.0008]
    )
    assert steps == pytest.approx([0.032, 0.060, 0.080, 0.100, 0.132])
