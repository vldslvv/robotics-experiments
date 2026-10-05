"""Host verification of flight-suite scheduling and body-frame commands."""

import importlib.util
import sys
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from robotics_experiments.commanding import AttitudeRateSetpoint, PolicyInput, VelocitySetpoint
from robotics_experiments.flight_parameters import load_flight_parameters_config
from robotics_experiments.px4_bridge import VehicleState
from tests.support.paths import REPO_ROOT


def _load_script(monkeypatch: pytest.MonkeyPatch, /) -> ModuleType:
    scenario = ModuleType("_px4_takeoff_scenario")
    scenario.Px4TakeoffScenarioOptions = object  # type: ignore[attr-defined] # import-only shell fake
    scenario.run_px4_takeoff_scenario = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    runtime = ModuleType("_runtime")
    runtime.SceneTestContext = object  # type: ignore[attr-defined] # import-only shell fake
    runtime.run_scene_test = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "_px4_takeoff_scenario", scenario)
    monkeypatch.setitem(sys.modules, "_runtime", runtime)
    spec = importlib.util.spec_from_file_location(
        "flight_parameters_capture",
        REPO_ROOT / "isaac/runtime/standalone/tests/flight_parameters_capture.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _input(*, time_us: int, speed: float = 0.0) -> PolicyInput:
    return PolicyInput(
        vehicle_state=VehicleState(
            position_enu=np.array([0.0, 0.0, 2.5]),
            orientation_flu_enu_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity_enu=np.array([speed, 0.0, 0.0]),
            angular_velocity_flu=np.zeros(3),
            sim_time_us=time_us,
        ),
        sim_time_us=time_us,
        task_observation=(),
        reference_position_enu=(0.0, 0.0, 2.5),
    )


def _adapter(*, module, config):
    factory = module._policy_factory(config=config)  # noqa: SLF001 - import-gated shell composition seam
    return factory(artifact="flight-suite", hold_s=None)


@pytest.mark.parametrize("sample_rate", [10.0, 20.0, 50.0])
def test_complete_suite_preserves_one_policy_state_and_fixed_preparations(
    monkeypatch, sample_rate
) -> None:
    module = _load_script(monkeypatch)
    config = load_flight_parameters_config(path=REPO_ROOT / "configs/flight_parameters_x500.yaml")
    config = replace(config, sample_rate_hz=sample_rate)
    adapter = _adapter(module=module, config=config)
    assert adapter.policy_rate_hz == config.command_rate_hz == 20.0
    state = adapter.initial_state
    complete = False
    saw_velocity = saw_ctbr = False
    for time_us in range(0, 800_000_000, 50_000):
        result = adapter.advance(policy_input=_input(time_us=time_us), state=state)
        assert not result.failure_reason
        state = result.state
        saw_velocity |= isinstance(result.setpoint, VelocitySetpoint)
        saw_ctbr |= isinstance(result.setpoint, AttitudeRateSetpoint)
        if result.complete:
            complete = True
            break
    assert complete and saw_velocity and saw_ctbr
    assert len(state.boundaries) == 23
    for boundary, stage in zip(state.boundaries, config.stages[1:], strict=True):
        assert boundary.start_sim_time_us - boundary.preparation_start_us == 20_000_000
        assert boundary.end_sim_time_us - boundary.start_sim_time_us == round(
            sum(c.duration_s for c in stage.commands) * 1e6
        )


def test_early_readiness_keeps_window_fixed_and_late_readiness_fails(monkeypatch) -> None:
    module = _load_script(monkeypatch)
    config = load_flight_parameters_config(path=REPO_ROOT / "configs/flight_parameters_x500.yaml")
    adapter = _adapter(module=module, config=config)
    state = adapter.initial_state
    for time_us in (0, 3_000_000, 3_500_000, 19_950_000):
        result = adapter.advance(policy_input=_input(time_us=time_us), state=state)
        state = result.state
        assert not state.boundaries
    result = adapter.advance(policy_input=_input(time_us=20_000_000), state=state)
    assert result.state.boundaries[0].start_sim_time_us == 20_000_000
    adapter = _adapter(module=module, config=config)
    result = adapter.advance(policy_input=_input(time_us=0, speed=1.0), state=adapter.initial_state)
    result = adapter.advance(policy_input=_input(time_us=20_000_000, speed=1.0), state=result.state)
    assert "did not stabilize" in result.failure_reason


@pytest.mark.parametrize("mode", ["velocity", "ctbr"])
def test_rotational_commands_preserve_body_flu_signs(monkeypatch, mode) -> None:
    module = _load_script(monkeypatch)
    config = load_flight_parameters_config(path=REPO_ROOT / "configs/flight_parameters_x500.yaml")
    stage = next(stage for stage in config.stages[1:] if stage.control_mode == mode)
    command = replace(stage.commands[0], yaw_rate_radps=0.2, body_rate_flu_radps=(0.1, 0.2, 0.3))
    config = replace(config, stages=(config.stages[0], replace(stage, commands=(command,))))
    adapter = _adapter(module=module, config=config)
    state = adapter.initial_state
    for time_us in (0, 3_000_000, 3_500_000, 20_000_000):
        result = adapter.advance(policy_input=_input(time_us=time_us), state=state)
        state = result.state
    assert result.setpoint.yaw_rate_radps == (-0.2 if mode == "velocity" else -0.3)
    if mode == "ctbr":
        assert result.setpoint.pitch_rate_radps == -0.2
        assert result.setpoint.roll_rate_radps == 0.1
