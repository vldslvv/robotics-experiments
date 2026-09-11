"""Host tests for flight-parameters response preconditioning and frame conversion."""

import importlib.util
import json
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from robotics_experiments.alignment import load_controller_definition
from robotics_experiments.alignment._models import ProbeCommand, ProbeFamily, ProbeProfile
from robotics_experiments.commanding import PolicyInput
from robotics_experiments.config import load_config
from robotics_experiments.px4_bridge import VehicleState
from tests.support.paths import REPO_ROOT


def _load_script(
    monkeypatch: pytest.MonkeyPatch, name: str = "flight_parameters_capture_response", /
) -> ModuleType:
    scenario = ModuleType("_px4_takeoff_scenario")
    scenario.Px4TakeoffScenarioOptions = object  # type: ignore[attr-defined]  # dynamic fake
    scenario.run_px4_takeoff_scenario = (  # type: ignore[attr-defined]  # dynamic fake
        lambda *_args, **_kwargs: None
    )
    runtime = ModuleType("_runtime")
    runtime.SceneTestContext = object  # type: ignore[attr-defined]  # dynamic fake
    runtime.run_scene_test = (  # type: ignore[attr-defined]  # dynamic fake
        lambda *_args, **_kwargs: None
    )
    monkeypatch.setitem(sys.modules, "_px4_takeoff_scenario", scenario)
    monkeypatch.setitem(sys.modules, "_runtime", runtime)
    path = REPO_ROOT / f"isaac/runtime/standalone/tests/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _input(
    *,
    sim_time_us: int,
    position=(0.0, 0.0, 2.5),
    linear_velocity=(0.0, 0.0, 0.0),
) -> PolicyInput:
    return PolicyInput(
        vehicle_state=VehicleState(
            position_enu=np.asarray(position, dtype=np.float64),
            orientation_flu_enu_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity_enu=np.asarray(linear_velocity, dtype=np.float64),
            angular_velocity_flu=np.zeros(3),
            sim_time_us=sim_time_us,
        ),
        sim_time_us=sim_time_us,
        task_observation=(),
        reference_position_enu=(0.0, 0.0, 2.5),
    )


def _yaw_definition():
    source = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/px4_sitl_x500_velocity.yaml"
    )
    return replace(
        source,
        probe_families=(
            ProbeFamily(
                name="velocity",
                control_mode="velocity",
                metric="translational-controller-quality",
                profiles=(
                    ProbeProfile(
                        name="yaw",
                        split="fit",
                        axis="yaw",
                        commands=(
                            ProbeCommand(
                                (0.0, 0.0, 0.0),
                                0.2,
                                0.0,
                                (0.0, 0.0, 0.0),
                                1.0,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_capture_waits_for_minimum_and_stable_hold_then_converts_yaw(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script(monkeypatch)
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    adapter = module._policy_factory(  # noqa: SLF001 - focused capture transition test
        definition=_yaw_definition(),
        config=config,
    )(
        artifact="alignment-probe",
        hold_s=None,
    )

    first = adapter.advance(policy_input=_input(sim_time_us=0), state=adapter.initial_state)
    minimum = adapter.advance(policy_input=_input(sim_time_us=3_000_000), state=first.state)
    active = adapter.advance(policy_input=_input(sim_time_us=3_500_000), state=minimum.state)

    assert first.setpoint.yaw_rate_radps == 0.0
    assert minimum.setpoint.yaw_rate_radps == 0.0
    assert active.setpoint.yaw_rate_radps == pytest.approx(-0.2)
    assert active.state.samples[0].command_yaw_rate_radps == pytest.approx(0.2)
    assert active.state.initials[0].sim_time_us == 3_500_000


def test_capture_stability_hold_restarts_after_motion(monkeypatch: pytest.MonkeyPatch):
    module = _load_script(monkeypatch)
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    adapter = module._policy_factory(  # noqa: SLF001 - focused capture transition test
        definition=_yaw_definition(),
        config=config,
    )(
        artifact="alignment-probe",
        hold_s=None,
    )

    state = adapter.advance(policy_input=_input(sim_time_us=0), state=adapter.initial_state).state
    state = adapter.advance(policy_input=_input(sim_time_us=3_000_000), state=state).state
    state = adapter.advance(
        policy_input=_input(sim_time_us=3_250_000, linear_velocity=(0.03, 0.0, 0.0)),
        state=state,
    ).state
    restarted = adapter.advance(policy_input=_input(sim_time_us=3_500_000), state=state)
    active = adapter.advance(policy_input=_input(sim_time_us=4_000_000), state=restarted.state)

    assert restarted.setpoint.yaw_rate_radps == 0.0
    assert active.setpoint.yaw_rate_radps == pytest.approx(-0.2)


def test_capture_recenters_to_reference_between_profiles(monkeypatch: pytest.MonkeyPatch):
    module = _load_script(monkeypatch)
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    adapter = module._policy_factory(  # noqa: SLF001 - focused capture transition test
        definition=_yaw_definition(),
        config=config,
    )(
        artifact="alignment-probe",
        hold_s=None,
    )

    displaced = adapter.advance(
        policy_input=_input(sim_time_us=0, position=(2.0, -2.0, 3.5)),
        state=adapter.initial_state,
    )

    velocity = np.array(
        [
            displaced.setpoint.forward_mps,
            -displaced.setpoint.right_mps,
            -displaced.setpoint.down_mps,
        ]
    )
    assert np.linalg.norm(velocity) == pytest.approx(
        config.alignment.capture.recenter_max_speed_mps
    )
    assert np.dot(velocity, np.array([-2.0, 2.0, -1.0])) > 0.0


def test_capture_reports_preconditioning_timeout(monkeypatch: pytest.MonkeyPatch):
    module = _load_script(monkeypatch)
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    adapter = module._policy_factory(  # noqa: SLF001 - focused capture transition test
        definition=_yaw_definition(),
        config=config,
    )(
        artifact="alignment-probe",
        hold_s=None,
    )

    state = adapter.advance(policy_input=_input(sim_time_us=0), state=adapter.initial_state).state
    failed = adapter.advance(
        policy_input=_input(sim_time_us=12_100_000, linear_velocity=(0.03, 0.0, 0.0)),
        state=state,
    )

    assert failed.failure_reason == "alignment profile 'yaw' did not stabilize within 12.0s"


@pytest.mark.parametrize(("repetition", "seed"), [(1, 1101), (2, 2101), (3, 3101)])
def test_capture_uses_configured_flight_seed_before_starting_scenario(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    repetition,
    seed,
):
    monkeypatch.setenv(
        "FLIGHT_PARAMETERS_CONFIG", str(REPO_ROOT / "configs/flight/holybro_x500_v2_hover.yaml")
    )
    monkeypatch.setenv("FLIGHT_PARAMETERS_RESPONSE_REPETITION", str(repetition))
    module = _load_script(monkeypatch)
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    seeds: list[int] = []
    monkeypatch.setattr(module.np.random, "seed", seeds.append)
    monkeypatch.setattr(module, "Px4TakeoffScenarioOptions", lambda **kwargs: kwargs)
    monkeypatch.setattr(module, "run_px4_takeoff_scenario", lambda *_args: None)

    module._run(  # noqa: SLF001 - focused capture initialization test
        context=SimpleNamespace(config=config),
        definition=_yaw_definition(),
        run_dir=tmp_path,
    )

    assert seeds == [seed]


def test_capture_deploys_the_policy_plant_pusher(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script(monkeypatch)
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    writes: dict[str, float] = {}

    class _Attribute:
        def __init__(self, value: float = 0.0):
            self.value = value

        def Get(self) -> float:
            return self.value

        def Set(self, value: float) -> None:
            self.value = value

    class _Prim:
        def IsValid(self) -> bool:
            return True

    prim = _Prim()
    stage = SimpleNamespace(GetPrimAtPath=lambda _path: prim)
    omni = ModuleType("omni")
    omni_usd = ModuleType("omni.usd")
    omni_usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)  # type: ignore[attr-defined]
    omni.usd = omni_usd  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.usd", omni_usd)

    class _Joint:
        def __init__(self, _prim):
            pass

        def GetUpperLimitAttr(self):
            return _Attribute(0.015)

    drive = SimpleNamespace(
        CreateTargetPositionAttr=lambda value: writes.__setitem__("target", value),
        CreateMaxForceAttr=lambda value: writes.__setitem__("force", value),
    )
    velocity = _Attribute()
    pxr = ModuleType("pxr")
    pxr.UsdPhysics = SimpleNamespace(  # type: ignore[attr-defined]
        PrismaticJoint=_Joint,
        DriveAPI=SimpleNamespace(Get=lambda _prim, _kind: drive),
    )
    pxr.PhysxSchema = SimpleNamespace(  # type: ignore[attr-defined]
        PhysxJointAPI=SimpleNamespace(
            Apply=lambda _prim: SimpleNamespace(CreateMaxJointVelocityAttr=lambda: velocity)
        )
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    reports: list[str] = []

    module._deploy_pusher(  # noqa: SLF001 - focused capture plant-contract test
        SimpleNamespace(config=config, report=reports.append),
        SimpleNamespace(params=SimpleNamespace(articulation_path="/Drone")),
    )

    assert writes == {
        "target": 0.015,
        "force": config.rl.lab_vehicle.pusher_effort_limit_n,
    }
    assert velocity.value == config.rl.lab_vehicle.pusher_velocity_limit_mps
    assert reports == ["alignment plant pusher deployed to 0.015000 m"]


def test_capture_rejects_wrong_body_flu_yaw_response_sign(monkeypatch: pytest.MonkeyPatch):
    module = _load_script(monkeypatch)
    samples = tuple(
        SimpleNamespace(
            control_mode="velocity",
            profile="yaw",
            command_yaw_rate_radps=0.2,
            command_body_rate_flu_radps=(0.0, 0.0, 0.0),
            angular_velocity_body_flu_radps=(0.0, 0.0, -0.1),
        )
        for _index in range(4)
    )

    with pytest.raises(AssertionError, match="sign disagrees"):
        module._assert_rotational_response_sign(  # noqa: SLF001 - focused validation test
            samples=samples
        )


def test_capture_checks_ctbr_roll_pitch_and_yaw_response_signs(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_script(monkeypatch)
    samples = tuple(
        SimpleNamespace(
            control_mode="ctbr",
            profile="rates",
            command_yaw_rate_radps=0.0,
            command_body_rate_flu_radps=(0.2, -0.3, 0.4),
            angular_velocity_body_flu_radps=(0.1, -0.1, 0.2),
        )
        for _index in range(4)
    )

    module._assert_rotational_response_sign(  # noqa: SLF001 - focused validation test
        samples=samples
    )


@pytest.mark.parametrize(
    ("trial", "set_name", "seed"),
    [(1, "fit", 1001), (31, "validation", 2001), (41, "acceptance", 3001)],
)
def test_hover_records_the_same_snapshot_and_world_velocities_in_every_set(
    monkeypatch,
    tmp_path,
    trial,
    set_name,
    seed,
):
    module = _load_script(monkeypatch, "flight_parameters_capture")
    config = load_config(REPO_ROOT / "configs/flight_parameters_x500_hover.yaml")
    monkeypatch.setenv(
        "FLIGHT_PARAMETERS_CONFIG", str(REPO_ROOT / "configs/flight/holybro_x500_v2_hover.yaml")
    )
    monkeypatch.setenv("FLIGHT_PARAMETERS_RUN_ID", "capture-test")
    monkeypatch.setenv("FLIGHT_PARAMETERS_TRIAL_INDEX", str(trial))
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    monkeypatch.setenv("FLIGHT_PARAMETERS_TRIAL_DIR", str(trial_dir))
    debug_log = tmp_path / "bridge.log"
    debug_log.write_text("test bridge")
    seeds = []
    monkeypatch.setattr(module.np.random, "seed", seeds.append)
    monkeypatch.setattr(module, "Px4TakeoffScenarioOptions", lambda **kwargs: kwargs)
    status = SimpleNamespace(sim_time_us=1_000_000, debug_log_path=debug_log)
    flight_runtime = ModuleType("robotics_experiments.runtime")
    monkeypatch.setattr(
        flight_runtime, "read_bridge_host_status", lambda **_kwargs: status, raising=False
    )
    monkeypatch.setitem(sys.modules, "robotics_experiments.runtime", flight_runtime)
    attitude = np.array([np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])

    def read_state(time_us):
        t = time_us / 1_000_000
        return VehicleState(
            position_enu=np.array([t, 0.0, 2.5]),
            orientation_flu_enu_wxyz=attitude,
            linear_velocity_enu=np.array([2.0, t, 4.0]),
            angular_velocity_flu=np.array([t, 0.0, 2.0]),
            sim_time_us=time_us,
        )

    fleet = SimpleNamespace(
        flying=[
            SimpleNamespace(
                driver=SimpleNamespace(routine_state=SimpleNamespace(cursor=2)),
                spawned=SimpleNamespace(
                    host=object(), observer=SimpleNamespace(read_state=read_state)
                ),
            )
        ]
    )

    def run_scenario(context, options):
        assert seeds == [seed]
        for index in range(1020):
            status.sim_time_us = 1_000_000 + index * 50_000
            options["routine_tick_hook"](fleet)
        options["post_routine_hook"](context, fleet)

    monkeypatch.setattr(module, "run_px4_takeoff_scenario", run_scenario)
    module._run(context=SimpleNamespace(config=config))  # noqa: SLF001 - exercises recorder through its host callback boundary
    payload = json.loads((trial_dir / "raw_trace.json").read_text())
    assert (payload["set_name"], payload["seed"]) == (set_name, seed)
    assert len(payload["time_us"]) == 1020
    for index in (0, 100, 1019):
        t = payload["time_us"][index] / 1_000_000
        assert payload["position_enu_m"][index] == [t, 0.0, 2.5]
        assert payload["linear_velocity_enu_mps"][index] == [2.0, t, 4.0]
        assert payload["angular_velocity_enu_radps"][index] == pytest.approx([0.0, t, 2.0])
