"""Kit-wiring tests for the pure-observer runtime HUD."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

from tests.support.paths import REPO_ROOT


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Label:
    def __init__(self, text, **values):
        self.text = text
        vars(self).update(values)


class _Frame:
    def __init__(self):
        self.build = None
        self.rebuilds = 0

    def set_build_fn(self, build):
        self.build = build

    def rebuild(self):
        self.rebuilds += 1
        self.build()


class _Window:
    def __init__(self, **values):
        vars(self).update(values)
        self.frame = _Frame()


class _EventStream:
    def __init__(self):
        self.callbacks = []

    def create_subscription_to_pop(self, callback, name=None):
        _ = name
        self.callbacks.append(callback)
        return SimpleNamespace()


def _step(*, kind, index):
    return SimpleNamespace(
        index=index,
        total_steps=9,
        kind=kind,
        label=f"{kind}[test]",
        elapsed_s=1.0,
        timeout_s=20.0,
        hold_elapsed_s=0.0,
        hold_required_s=2.0 if kind == "hover" else 0.0,
    )


def _vehicle(*, kind, index, sim_time_us, label="x500_01"):
    return SimpleNamespace(
        label=label,
        sim_time_us=sim_time_us,
        driver_phase="DRIVE",
        routine_step=_step(kind=kind, index=index),
        navigation=None,
        press=None,
        policy=None,
        motion=None,
        failure=None,
        recovery_outcome=None,
    )


def _import_extension(monkeypatch):
    root = REPO_ROOT / "isaac/runtime/extensions/experiment_runtime_hud/ExperimentRuntimeHud_python"
    package = types.ModuleType("ExperimentRuntimeHud_python")
    package.__path__ = [str(root)]
    monkeypatch.setitem(sys.modules, "ExperimentRuntimeHud_python", package)
    for name in list(sys.modules):
        if name.startswith("ExperimentRuntimeHud_python."):
            sys.modules.pop(name, None)

    app = SimpleNamespace(post=_EventStream())
    app.get_post_update_event_stream = lambda: app.post
    settings = SimpleNamespace(get=lambda path: 2)
    carb = types.ModuleType("carb")
    carb.settings = SimpleNamespace(get_settings=lambda: settings)

    omni = types.ModuleType("omni")
    omni.__path__ = []
    omni_ext = types.ModuleType("omni.ext")
    omni_ext.IExt = object
    omni_kit = types.ModuleType("omni.kit")
    omni_kit.__path__ = []
    omni_kit_app = types.ModuleType("omni.kit.app")
    omni_kit_app.get_app = lambda: app
    omni_ui = types.ModuleType("omni.ui")
    omni_ui.Window = lambda **values: _Window(**values)
    omni_ui.DockPreference = SimpleNamespace(RIGHT_BOTTOM="right-bottom")
    omni_ui.ScrollingFrame = _Context
    omni_ui.VStack = lambda **_kwargs: _Context()
    omni_ui.CollapsableFrame = lambda *_args, **_kwargs: _Context()
    omni_ui.Label = _Label
    omni_ui.Separator = lambda *_args, **_kwargs: SimpleNamespace()
    omni.ext = omni_ext
    omni.kit = omni_kit
    omni.kit.app = omni_kit_app

    menu_calls = SimpleNamespace(added=[], removed=[])
    menu_components = types.ModuleType("isaacsim.gui.components.menu")
    menu_components.make_menu_item_description = lambda ext_id, name, callback: SimpleNamespace(
        ext_id=ext_id,
        name=name,
        callback=callback,
    )
    menu_utils = types.ModuleType("omni.kit.menu.utils")
    menu_utils.MenuItemDescription = lambda **values: SimpleNamespace(**values)
    menu_utils.add_menu_items = lambda items, root: menu_calls.added.append((items, root))
    menu_utils.remove_menu_items = lambda items, root: menu_calls.removed.append((items, root))
    isaacsim = types.ModuleType("isaacsim")
    isaacsim.__path__ = []
    isaacsim_gui = types.ModuleType("isaacsim.gui")
    isaacsim_gui.__path__ = []
    isaacsim_components = types.ModuleType("isaacsim.gui.components")
    isaacsim_components.__path__ = []

    runtime = SimpleNamespace(
        status=SimpleNamespace(scene_ready=True, failure=None, vehicles=()),
        reads=0,
        control_effects=0,
    )

    def read_status():
        runtime.reads += 1
        return runtime.status

    runtime.read_status = read_status
    production = types.ModuleType("ExperimentRuntime_python")
    production.get_instance = lambda: runtime

    for name, value in {
        "carb": carb,
        "omni": omni,
        "omni.ext": omni_ext,
        "omni.kit": omni_kit,
        "omni.kit.app": omni_kit_app,
        "omni.ui": omni_ui,
        "omni.kit.menu.utils": menu_utils,
        "isaacsim": isaacsim,
        "isaacsim.gui": isaacsim_gui,
        "isaacsim.gui.components": isaacsim_components,
        "isaacsim.gui.components.menu": menu_components,
        "ExperimentRuntime_python": production,
    }.items():
        monkeypatch.setitem(sys.modules, name, value)
    module = importlib.import_module("ExperimentRuntimeHud_python.extension")
    return module, app, runtime, menu_calls


def _fire_update(app):
    for callback in list(app.post.callbacks):
        callback(SimpleNamespace())


def test_hud_polling_and_visibility_never_enter_the_control_path(monkeypatch):
    module, app, runtime, menu_calls = _import_extension(monkeypatch)
    extension = module.Extension()
    extension.on_startup("experiment_runtime_hud")
    assert extension._window.visible  # noqa: SLF001 - Kit window wiring assertion
    assert menu_calls.added

    runtime.status = SimpleNamespace(
        scene_ready=True,
        failure=None,
        vehicles=(_vehicle(kind="goto", index=3, sim_time_us=1_000_000),),
    )
    _fire_update(app)
    extension._window.visible = False  # noqa: SLF001 - simulate user closing the window
    _fire_update(app)
    extension._show_window()  # noqa: SLF001 - menu action wiring assertion

    assert runtime.reads == 2
    assert runtime.control_effects == 0
    assert extension._window.visible  # noqa: SLF001 - menu action wiring assertion
    assert extension._models["x500_01"].status.text == "NAVIGATION - GOTO"  # noqa: SLF001

    extension.on_shutdown()
    assert menu_calls.removed
    assert extension._update_sub is None  # noqa: SLF001 - subscription lifecycle assertion


def test_hud_keeps_a_two_transition_sliding_window_and_stale_snapshot(monkeypatch):
    module, app, runtime, _menu_calls = _import_extension(monkeypatch)
    extension = module.Extension()
    extension.on_startup("experiment_runtime_hud")

    for kind, index, sim_time_us in (
        ("goto", 3, 1_000_000),
        ("hover", 4, 2_000_000),
        ("policy", 5, 3_000_000),
    ):
        runtime.status = SimpleNamespace(
            scene_ready=True,
            failure=None,
            vehicles=(_vehicle(kind=kind, index=index, sim_time_us=sim_time_us),),
        )
        _fire_update(app)

    history = extension._histories["x500_01"]  # noqa: SLF001 - UI history contract
    assert len(history) == 2
    assert "SETTLING - HOVER" in history[0]
    assert "POLICY - ENTERING" in history[1]

    runtime.status = SimpleNamespace(scene_ready=True, failure=None, vehicles=())
    _fire_update(app)

    history = extension._histories["x500_01"]  # noqa: SLF001 - UI history contract
    assert len(history) == 2
    assert "POLICY - ENTERING" in history[0]
    assert "STALE - TIMELINE STOPPED" in history[1]
    assert extension._models["x500_01"].status.text == "STALE - TIMELINE STOPPED"  # noqa: SLF001


def test_hud_builds_one_full_card_for_every_vehicle_in_fleet_order(monkeypatch):
    module, app, runtime, _menu_calls = _import_extension(monkeypatch)
    extension = module.Extension()
    extension.on_startup("experiment_runtime_hud")

    runtime.status = SimpleNamespace(
        scene_ready=True,
        failure=None,
        vehicles=(
            _vehicle(kind="goto", index=3, sim_time_us=1_000_000, label="quad_01"),
            _vehicle(kind="hover", index=4, sim_time_us=1_000_000, label="quad_02"),
        ),
    )
    _fire_update(app)

    assert list(extension._models) == ["quad_01", "quad_02"]  # noqa: SLF001
    assert extension._models["quad_01"].status.text == "NAVIGATION - GOTO"  # noqa: SLF001
    assert extension._models["quad_02"].status.text == "SETTLING - HOVER"  # noqa: SLF001
