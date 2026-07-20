from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from tests.support.paths import REPO_ROOT


def _load_capture(monkeypatch):
    root = REPO_ROOT / "isaac/runtime/extensions/flight_benchmark/FlightBenchmark_python"
    package = types.ModuleType("FlightBenchmark_python")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "FlightBenchmark_python", package)
    sys.modules.pop("FlightBenchmark_python._capture", None)
    return importlib.import_module("FlightBenchmark_python._capture")


def _observation(
    capture,
    *,
    sim_time_s: float,
    connected: bool = True,
    any_armed: bool = False,
    takeoff: bool = False,
):
    return capture.FlightPhaseObservation(
        all_connected=connected,
        any_armed=any_armed,
        all_takeoffs_complete=takeoff,
        sim_time_s=sim_time_s,
    )


def test_phase_controller_requires_ordered_fleet_transitions(monkeypatch) -> None:
    capture = _load_capture(monkeypatch)
    state = capture.FlightPhaseController(airborne_duration_s=5.0)
    assert (
        state.observe(observation=_observation(capture, sim_time_s=0.0, connected=False)) == state
    )

    state = state.observe(observation=_observation(capture, sim_time_s=1.0))
    assert state.phase is capture.FlightPhase.CONNECTED_DISARMED
    state = state.observe(observation=_observation(capture, sim_time_s=2.0, any_armed=True))
    assert state.phase is capture.FlightPhase.ARMED_TAKEOFF
    state = state.observe(
        observation=_observation(
            capture,
            sim_time_s=4.0,
            any_armed=True,
            takeoff=True,
        )
    )
    assert state.phase is capture.FlightPhase.AIRBORNE
    assert not state.observe(
        observation=_observation(
            capture,
            sim_time_s=8.99,
            any_armed=True,
            takeoff=True,
        )
    ).complete
    assert state.observe(
        observation=_observation(
            capture,
            sim_time_s=9.0,
            any_armed=True,
            takeoff=True,
        )
    ).complete


def test_recorder_persists_component_exception_sample(monkeypatch, tmp_path: Path) -> None:
    capture = _load_capture(monkeypatch)
    output = tmp_path / "samples.json"
    recorder = capture.FlightBenchmarkRecorder(
        output_path=output,
        metadata={"test": True},
        environment={"test": True},
    )
    recorder.transition(phase=capture.FlightPhase.CONNECTED_DISARMED, sim_time_s=1.0)
    recorder.begin_app_frame()
    recorder.record_duration(metric="bridge_send", entity="vehicle-0", duration_ms=1.5)
    recorder.fail(reason="boom", sim_time_s=1.1)

    payload = json.loads(output.read_text())
    assert payload["status"] == "failed"
    assert payload["component_samples"] == [
        {
            "duration_ms": 1.5,
            "entity": "vehicle-0",
            "frame": 0,
            "metric": "bridge_send",
            "phase": "connected_disarmed",
        }
    ]


def test_recorder_rejects_completion_before_a_phase(monkeypatch, tmp_path: Path) -> None:
    capture = _load_capture(monkeypatch)
    recorder = capture.FlightBenchmarkRecorder(
        output_path=tmp_path / "samples.json",
        metadata={},
        environment={},
    )
    with pytest.raises(RuntimeError, match="before its first measured phase"):
        recorder.finish(sim_time_s=0.0)
