"""Policy preparation consumes the exact alignment selection and capture."""

import hashlib
import pickle
from types import SimpleNamespace

import pytest

from robotics_experiments.config import load_config
from robotics_experiments.flight_parameters import load_flight_selection
from robotics_experiments.rl import prepare_policy_inputs
from tests.support.flight_parameters import write_hover_dataset
from tests.support.paths import REPO_ROOT
from tests.support.switches import write_switch_asset


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import robotics_experiments.rl.workflow as workflow

    path = write_hover_dataset(tmp_path)
    dataset, _selection = load_flight_selection(path)
    config = load_config(REPO_ROOT / "configs/train/stage/press_hover_quad.yaml")
    controller_path = tmp_path / "controller.json"
    controller_path.write_bytes(b"controller")
    controller = SimpleNamespace(
        vehicle="x500_01",
        airframe="holybro_x500_v2.usda",
        command_rate_hz=20.0,
        certified_control_modes=("velocity",),
        advisory_quality_passed=True,
        advisory_quality_reasons=(),
        flight_parameters_run_id=dataset.run_id,
        flight_parameters_fingerprint=dataset.artifact_digest,
    )
    monkeypatch.setattr(workflow, "load_controller_artifact", lambda _path: controller)
    write_switch_asset(root=tmp_path, model=config.switches.model)
    return dict(
        config=config,
        repo_root=tmp_path,
        controller_artifact=controller_path,
        flight_parameters_artifact=dataset.run_dir,
    ), controller


def test_prepare_and_worker_load_same_inputs(prepared):
    kwargs, _controller = prepared
    inputs = prepare_policy_inputs(**kwargs)
    assert inputs.dataset_path == kwargs["repo_root"] / "dataset.json"
    assert inputs.flight_parameters_path == kwargs["flight_parameters_artifact"]
    assert inputs.controller_digest == hashlib.sha256(b"controller").hexdigest()
    assert len(inputs.hover_reset_samples.position_error_enu_m) == 24
    assert pickle.loads(pickle.dumps(inputs)) == inputs


def test_explicit_capture_must_match_selection(prepared, tmp_path):
    kwargs, _controller = prepared
    other = write_hover_dataset(tmp_path / "other").parent / "capture"
    with pytest.raises(ValueError, match="capture source"):
        prepare_policy_inputs(**{**kwargs, "flight_parameters_artifact": other})


def test_only_procedural_smoke_bypasses_quality_gate(prepared):
    kwargs, controller = prepared
    controller.advisory_quality_passed = False
    controller.advisory_quality_reasons = ("measured response exceeds limit",)
    with pytest.raises(ValueError, match="failed its quality checks"):
        prepare_policy_inputs(**kwargs)
    assert prepare_policy_inputs(**kwargs, procedural_smoke=True).controller is controller
    controller.airframe = "wrong.usda"
    with pytest.raises(ValueError, match="incompatible"):
        prepare_policy_inputs(**kwargs, procedural_smoke=True)
