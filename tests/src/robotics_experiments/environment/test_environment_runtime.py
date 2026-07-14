import importlib
import sys
import types
from types import SimpleNamespace

from robotics_experiments.environment.switches import SwitchState


def _import_runtime(monkeypatch):
    records = SimpleNamespace(despawn=[], spawn=[], validate=[], init=[], step=[])
    carb = types.ModuleType("carb")
    carb.log_info = lambda message: None
    carb.log_error = lambda message: None

    stage_module = types.ModuleType("robotics_experiments.environment.stage")
    stage_module.despawn_switches = lambda stage, switches: records.despawn.append(
        (stage, switches)
    )
    stage_module.spawn_switches = lambda **kwargs: records.spawn.append(kwargs) or (
        SwitchState("/Root/Switch", "/Root/Lamp"),
    )
    stage_module.validate_switches = (
        lambda stage, switches: records.validate.append((stage, switches)) or True
    )

    step_module = types.ModuleType("robotics_experiments.environment.step")
    step_module.init_articulations = lambda **kwargs: records.init.append(kwargs) or [
        "articulation"
    ]
    step_module.step_switches = lambda **kwargs: records.step.append(kwargs) or tuple(
        kwargs["switch_state"]
    )

    config_module = types.ModuleType("robotics_experiments.config")
    config_module.LightSwitchConfig = object
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "robotics_experiments.config", config_module)
    monkeypatch.setitem(sys.modules, "robotics_experiments.environment.stage", stage_module)
    monkeypatch.setitem(sys.modules, "robotics_experiments.environment.step", step_module)
    sys.modules.pop("robotics_experiments.environment._runtime", None)
    module = importlib.import_module("robotics_experiments.environment._runtime")
    return module, records, stage_module, step_module


def test_play_spawns_validates_and_enters_arming(monkeypatch):
    module, records, _stage_module, _step_module = _import_runtime(monkeypatch)
    config = object()
    stage = object()
    initial = module.create_switches_runtime(config=config, stage=stage)

    successor = module.play_switches_runtime(state=initial)

    assert initial.phase is module.SwitchesRuntimePhase.IDLE
    assert successor.phase is module.SwitchesRuntimePhase.ARMING
    assert successor.switches == (SwitchState("/Root/Switch", "/Root/Lamp"),)
    assert records.despawn == [(stage, ())]
    assert records.spawn == [{"stage": stage, "config": config}]
    assert records.validate == [(stage, successor.switches)]


def test_validation_failure_despawns_partial_state_and_stays_idle(monkeypatch):
    module, records, _stage_module, _step_module = _import_runtime(monkeypatch)
    monkeypatch.setattr(module, "validate_switches", lambda stage, switches: False)
    state = module.create_switches_runtime(config=object(), stage="stage")

    successor = module.play_switches_runtime(state=state)

    assert successor.phase is module.SwitchesRuntimePhase.IDLE
    assert successor.switches == ()
    assert records.despawn[-1][1] == (SwitchState("/Root/Switch", "/Root/Lamp"),)


def test_articulation_init_retries_then_step_rebinds_switch_state(monkeypatch):
    module, records, _stage_module, _step_module = _import_runtime(monkeypatch)
    state = module.play_switches_runtime(
        state=module.create_switches_runtime(config=object(), stage="stage")
    )
    attempts = iter((None, ["articulation"]))
    monkeypatch.setattr(module, "init_articulations", lambda **kwargs: next(attempts))

    unchanged = module.step_switches_runtime(state=state)
    armed = module.step_switches_runtime(state=unchanged)
    stepped = module.step_switches_runtime(state=armed, on_toggle=lambda index, value: None)

    assert unchanged is state
    assert armed.phase is module.SwitchesRuntimePhase.RUNNING
    assert armed.articulations == ("articulation",)
    assert stepped is not armed
    assert records.step[0]["switch_state"] == armed.switches
    assert records.step[0]["articulations"] == armed.articulations


def test_pause_resume_preserves_resources_and_teardown_clears_them(monkeypatch):
    module, records, _stage_module, _step_module = _import_runtime(monkeypatch)
    state = module.play_switches_runtime(
        state=module.create_switches_runtime(config=object(), stage="stage")
    )
    state = module.step_switches_runtime(state=state)

    paused = module.pause_switches_runtime(state=state)
    resumed = module.play_switches_runtime(state=paused)
    idle = module.teardown_switches_runtime(state=resumed)

    assert paused.phase is module.SwitchesRuntimePhase.PAUSED
    assert paused.switches is state.switches
    assert paused.articulations is state.articulations
    assert resumed.phase is module.SwitchesRuntimePhase.RUNNING
    assert idle.phase is module.SwitchesRuntimePhase.IDLE
    assert idle.switches == () and idle.articulations == ()
    assert records.despawn[-1] == ("stage", resumed.switches)


def test_adopt_stage_retargets_and_clears_runtime(monkeypatch):
    module, records, _stage_module, _step_module = _import_runtime(monkeypatch)
    state = module.play_switches_runtime(
        state=module.create_switches_runtime(config=object(), stage="old-stage")
    )

    adopted = module.adopt_switches_stage(state=state, stage="new-stage")

    assert adopted.stage == "new-stage"
    assert adopted.phase is module.SwitchesRuntimePhase.IDLE
    assert records.despawn[-1] == ("new-stage", state.switches)
