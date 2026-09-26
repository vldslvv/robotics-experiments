"""Curriculum and experiment roots have separate strict schemas."""

import json
import shutil
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from hydra.errors import ConfigCompositionException

from robotics_experiments.config import ExperimentConfig, load_config, load_curriculum_config
from tests.support.paths import REPO_ROOT


def _config(*, tmp_path: Path, **changes) -> Path:
    path = tmp_path / "curriculum.yaml"
    data = {
        "defaults": ["base_curriculum", "_self_"],
        "policy_name": "press",
        "initial_learner": None,
        "stages": [
            {"name": "anything", "config_path": "configs/train/stage/press_hover_quad.yaml"}
        ],
    }
    path.write_text(json.dumps(data | changes))
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"stages": []},
        {"stages": [{"name": "", "config_path": "stage.yaml"}]},
        {"stages": [{"name": "same", "config_path": "a"}, {"name": "same", "config_path": "b"}]},
        {"stages": [{"name": "one", "config_path": ""}]},
        {"policy_name": "../bad"},
        {"initial_learner": ""},
    ],
)
def test_invalid_curriculum_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        load_curriculum_config(_config(tmp_path=tmp_path, **changes))


def test_curriculum_is_immutable_cached_and_not_an_experiment(tmp_path):
    path = _config(tmp_path=tmp_path)
    config = load_curriculum_config(path)
    assert config is load_curriculum_config(path)
    assert isinstance(config.stages, tuple)
    with pytest.raises(FrozenInstanceError):
        config.stages[0].name = "changed"
    with pytest.raises(ValueError, match="experiment"):
        load_config(path)
    with pytest.raises(ValueError, match="curriculum"):
        load_curriculum_config(REPO_ROOT / "configs/train/stage/press_hover_quad.yaml")


@pytest.mark.parametrize(
    "changes",
    [
        {"rl": {}},
        {"curriculum": {}},
        {"stages": [{"name": "one", "config_path": "stage.yaml", "policy_name": "nested"}]},
    ],
)
def test_curriculum_rejects_wrong_fields(tmp_path, changes):
    with pytest.raises(ConfigCompositionException):
        load_curriculum_config(_config(tmp_path=tmp_path, **changes))


def test_shipped_curricula_reference_complete_stage_experiments():
    for path in (REPO_ROOT / "configs").glob("train/*.yaml"):
        curriculum = load_curriculum_config(path)
        for stage in curriculum.stages:
            experiment = load_config(REPO_ROOT / stage.config_path)
            assert isinstance(experiment, ExperimentConfig)


def test_stage_rejects_unknown_fields(tmp_path):
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    path = tmp_path / "configs/train/stage/invalid.yaml"
    path.write_text(
        "# @package _global_\n"
        + json.dumps(
            {
                "defaults": ["/train/stage/press_hover_quad", "_self_"],
                "rl": {"training": {"unknown_field": True}},
            }
        )
    )
    with pytest.raises(ConfigCompositionException, match="unknown_field"):
        load_config(path)
