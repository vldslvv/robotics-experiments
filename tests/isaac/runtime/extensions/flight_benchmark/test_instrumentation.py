from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.paths import REPO_ROOT


def _load_modules(monkeypatch):
    root = REPO_ROOT / "isaac/runtime/extensions/flight_benchmark/FlightBenchmark_python"
    package = types.ModuleType("FlightBenchmark_python")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "FlightBenchmark_python", package)
    for name in (
        "FlightBenchmark_python._capture",
        "FlightBenchmark_python._instrumentation",
    ):
        sys.modules.pop(name, None)
    capture = importlib.import_module("FlightBenchmark_python._capture")

    class IsaacArticulationIO:
        def read_state(self, sim_time_us):
            return sim_time_us

        def update_rotor_spin(self, rotor_omega_rad_s, dt):
            return rotor_omega_rad_s, dt

        def apply_dynamics(self, dynamics):
            return dynamics

    class PegasusQuadraticThrustModel:
        def commands_to_dynamics(self, commands, *, linear_body_velocity_flu):
            return commands, linear_body_velocity_flu

    class Px4MavlinkPort:
        def __init__(self):
            self.raise_on_poll = False

        def send(self, state):
            return state

        def poll(self):
            if self.raise_on_poll:
                raise RuntimeError("boom")
            return "commands"

    class BridgeHost:
        def __init__(self):
            self.params = SimpleNamespace(vehicle_id=7)
            self.port = Px4MavlinkPort()
            self.io = IsaacArticulationIO()
            self.thrust = PegasusQuadraticThrustModel()

        def send_phase(self, dt):
            self.io.read_state(100)
            return self.port.send(dt)

        def recv_phase(self):
            commands = self.port.poll()
            dynamics = self.thrust.commands_to_dynamics(
                commands,
                linear_body_velocity_flu="velocity",
            )
            self.io.update_rotor_spin("omega", 0.004)
            self.io.apply_dynamics(dynamics)
            return dynamics

    bridge = types.ModuleType("robotics_experiments.px4_bridge")
    bridge.IsaacArticulationIO = IsaacArticulationIO
    bridge.PegasusQuadraticThrustModel = PegasusQuadraticThrustModel
    bridge.Px4MavlinkPort = Px4MavlinkPort
    runtime = types.ModuleType("robotics_experiments.runtime")
    runtime.BridgeHost = BridgeHost
    runtime_extension = types.ModuleType("ExperimentRuntime_python.extension")
    runtime_extension.tick_drivers = lambda *, state, on_error: (state, on_error)
    runtime_package = types.ModuleType("ExperimentRuntime_python")
    runtime_package.__path__ = []
    monkeypatch.setitem(sys.modules, "robotics_experiments.px4_bridge", bridge)
    monkeypatch.setitem(sys.modules, "robotics_experiments.runtime", runtime)
    monkeypatch.setitem(sys.modules, "ExperimentRuntime_python", runtime_package)
    monkeypatch.setitem(sys.modules, "ExperimentRuntime_python.extension", runtime_extension)
    instrumentation = importlib.import_module("FlightBenchmark_python._instrumentation")
    return capture, instrumentation, runtime_extension, BridgeHost


def test_wrappers_measure_real_calls_and_restore(monkeypatch, tmp_path: Path) -> None:
    capture, instrumentation, runtime_extension, bridge_host_type = _load_modules(monkeypatch)
    output = tmp_path / "samples.json"
    recorder = capture.FlightBenchmarkRecorder(
        output_path=output,
        metadata={},
        environment={},
    )
    recorder.transition(phase=capture.FlightPhase.CONNECTED_DISARMED, sim_time_s=1.0)
    recorder.begin_app_frame()
    original_send = bridge_host_type.send_phase
    original_tick = runtime_extension.tick_drivers
    session = instrumentation.InstrumentationSession(recorder=recorder)
    session.start()

    host = bridge_host_type()
    assert host.send_phase(0.004) == 0.004
    assert host.recv_phase()[0] == "commands"
    assert runtime_extension.tick_drivers(state="fleet", on_error=None) == ("fleet", None)
    session.stop()
    recorder.fail(reason="done", sim_time_s=1.1)

    assert bridge_host_type.send_phase is original_send
    assert runtime_extension.tick_drivers is original_tick
    samples = json.loads(output.read_text())["component_samples"]
    metrics = {sample["metric"] for sample in samples}
    assert metrics == {
        "articulation_state_read",
        "bridge_receive",
        "bridge_send",
        "dynamics_apply",
        "mavlink_actuator",
        "mavlink_sensor",
        "rotor_visual_update",
        "routine_driver",
        "thrust_model",
    }
    vehicle_samples = [sample for sample in samples if sample["metric"] != "routine_driver"]
    assert all(sample["entity"] == "vehicle-7" for sample in vehicle_samples)


def test_wrappers_record_and_propagate_exceptions(monkeypatch, tmp_path: Path) -> None:
    capture, instrumentation, _runtime_extension, bridge_host_type = _load_modules(monkeypatch)
    recorder = capture.FlightBenchmarkRecorder(
        output_path=tmp_path / "samples.json",
        metadata={},
        environment={},
    )
    recorder.transition(phase=capture.FlightPhase.CONNECTED_DISARMED, sim_time_s=1.0)
    recorder.begin_app_frame()
    session = instrumentation.InstrumentationSession(recorder=recorder)
    session.start()
    host = bridge_host_type()
    host.port.raise_on_poll = True

    with pytest.raises(RuntimeError, match="boom"):
        host.recv_phase()
    session.stop()
    recorder.fail(reason="boom", sim_time_s=1.1)

    metrics = {
        sample["metric"]
        for sample in json.loads((tmp_path / "samples.json").read_text())["component_samples"]
    }
    assert {"bridge_receive", "mavlink_actuator"} <= metrics
