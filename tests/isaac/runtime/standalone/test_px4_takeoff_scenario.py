"""Host checks for the standalone PX4 scenario shell."""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType, SimpleNamespace

import pytest

from robotics_experiments.commanding import DriverPhase
from tests.support.paths import REPO_ROOT


def _load_scenario(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    runtime = ModuleType("_runtime")
    runtime.SceneTestContext = object  # type: ignore[attr-defined]  # dynamic fake
    runtime.TimingReporter = object  # type: ignore[attr-defined]  # dynamic fake
    runtime.format_vehicle_state_vectors = (  # type: ignore[attr-defined]  # dynamic fake
        lambda *_args, **_kwargs: ""
    )
    monkeypatch.setitem(sys.modules, "_runtime", runtime)
    path = REPO_ROOT / "isaac/runtime/standalone/tests/_px4_takeoff_scenario.py"
    spec = importlib.util.spec_from_file_location("_px4_takeoff_scenario_host_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_drive_shell_uses_live_driver_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_scenario(monkeypatch)
    flying = SimpleNamespace(
        driver=SimpleNamespace(phase=DriverPhase.DRIVE, deadline_s=1_200.0),
        context=SimpleNamespace(params=SimpleNamespace(failure_recovery_timeout_s=60.0)),
    )

    active_deadlines = module._active_driver_backstop_deadlines(  # noqa: SLF001
        fleet=SimpleNamespace(flying=(flying,))
    )

    assert active_deadlines == (1_260.0,)


def test_recovery_deadline_already_includes_recovery_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_scenario(monkeypatch)
    flying = SimpleNamespace(
        driver=SimpleNamespace(phase=DriverPhase.RECOVERING, deadline_s=500.0),
        context=SimpleNamespace(params=SimpleNamespace(failure_recovery_timeout_s=60.0)),
    )

    active_deadlines = module._active_driver_backstop_deadlines(  # noqa: SLF001
        fleet=SimpleNamespace(flying=(flying,))
    )

    assert active_deadlines == (500.0,)
