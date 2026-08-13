"""Pure-observer dockable HUD for the live experiment runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import carb
import omni.ext
import omni.kit.app
import omni.ui as ui
from ExperimentRuntime_python import get_instance
from isaacsim.gui.components.menu import make_menu_item_description
from omni.kit.menu.utils import MenuItemDescription, add_menu_items, remove_menu_items

from ExperimentRuntimeHud_python._view import _build_vehicle_view, _HudVehicleView, _semantic

__all__ = ["Extension"]

_TITLE = "Experiment Runtime HUD"
_HISTORY_SETTING = "/exts/experiment_runtime_hud/transition_history_capacity"
_TONE_COLORS = {
    "idle": 0xFFB0B0B0,
    "navigation": 0xFFFFB050,
    "wait": 0xFF4DD7FF,
    "policy": 0xFFFF70FF,
    "complete": 0xFF70E090,
    "failure": 0xFF7070FF,
}


@dataclass
class _VehicleModels:
    status: Any
    transitions: Any
    details: Any
    command: Any


class Extension(omni.ext.IExt):
    """Render immutable runtime snapshots without entering the control path."""

    def on_startup(self, ext_id: str) -> None:
        settings = carb.settings.get_settings()
        self._history_capacity = int(settings.get(_HISTORY_SETTING))
        if self._history_capacity < 1:
            raise ValueError(
                "experiment_runtime_hud transition_history_capacity must be at least 1, "
                f"got {self._history_capacity}"
            )
        self._histories: dict[str, tuple[str, ...]] = {}
        self._semantic_keys: dict[str, str] = {}
        self._last_vehicles: tuple[Any, ...] = ()
        self._views: tuple[_HudVehicleView, ...] = ()
        self._models: dict[str, _VehicleModels] = {}
        self._layout_signature: tuple[tuple[str, str], ...] = ()
        self._had_live_vehicles = False
        self._runtime_message = "Waiting for the runtime to start."
        self._window = ui.Window(
            title=_TITLE,
            width=560,
            height=720,
            visible=True,
            dockPreference=ui.DockPreference.RIGHT_BOTTOM,
        )
        self._window.frame.set_build_fn(self._build_ui)
        self._menu_items = [
            MenuItemDescription(
                name="Robotics Experiments",
                sub_menu=[
                    make_menu_item_description(ext_id, "Runtime HUD", self._show_window),
                ],
            )
        ]
        add_menu_items(self._menu_items, "Window")
        app = omni.kit.app.get_app()
        self._update_sub = app.get_post_update_event_stream().create_subscription_to_pop(
            self._on_update,
            name="experiment_runtime_hud_update",
        )
        self._window.frame.rebuild()

    def on_shutdown(self) -> None:
        self._update_sub = None
        remove_menu_items(self._menu_items, "Window")
        self._window = None
        self._models = {}
        self._views = ()
        self._histories = {}
        self._semantic_keys = {}
        self._last_vehicles = ()

    def _show_window(self) -> None:
        self._window.visible = True

    def _on_update(self, event: Any) -> None:
        _ = event
        status = get_instance().read_status()
        vehicles = tuple(status.vehicles)
        stale = not vehicles and bool(self._last_vehicles)
        if vehicles:
            if not self._had_live_vehicles:
                self._histories = {}
                self._semantic_keys = {}
            self._had_live_vehicles = True
            self._last_vehicles = vehicles
            source = vehicles
            self._runtime_message = ""
        elif stale:
            source = self._last_vehicles
            self._had_live_vehicles = False
            self._runtime_message = "Timeline stopped; showing the last runtime snapshot."
        else:
            source = ()
            self._had_live_vehicles = False
            self._runtime_message = status.failure or "Waiting for Play."

        views = tuple(self._view_for(vehicle=vehicle, stale=stale) for vehicle in source)
        signature = (
            ("__runtime_message__", self._runtime_message),
            *((view.label, view.tone) for view in views),
        )
        self._views = views
        if signature != self._layout_signature:
            self._layout_signature = signature
            self._window.frame.rebuild()
        else:
            self._update_models()

    def _view_for(self, *, vehicle: Any, stale: bool) -> _HudVehicleView:
        semantic_key, status, _tone = _semantic(vehicle=vehicle, stale=stale)
        previous_key = self._semantic_keys.get(vehicle.label)
        if semantic_key != previous_key:
            transition = f"{vehicle.sim_time_us * 1e-6:8.2f} s  {status}"
            history = (*self._histories.get(vehicle.label, ()), transition)
            self._histories[vehicle.label] = history[-self._history_capacity :]
            self._semantic_keys[vehicle.label] = semantic_key
        return _build_vehicle_view(
            vehicle=vehicle,
            transitions=self._histories[vehicle.label],
            stale=stale,
        )

    def _build_ui(self) -> None:
        self._models = {}
        with ui.ScrollingFrame():
            with ui.VStack(spacing=8, height=0):
                ui.Label(
                    _TITLE,
                    style={"font_size": 20, "color": 0xFFFFFFFF},
                    height=28,
                )
                if self._runtime_message:
                    ui.Label(self._runtime_message, word_wrap=True, height=0)
                for view in self._views:
                    with ui.CollapsableFrame(view.label, collapsed=False, height=0):
                        with ui.VStack(spacing=4, height=0):
                            status_label = ui.Label(
                                view.status,
                                style={
                                    "font_size": 18,
                                    "color": _TONE_COLORS[view.tone],
                                },
                                height=26,
                            )
                            ui.Label(
                                "Current and previous transitions",
                                style={"color": 0xFFB0B0B0},
                                height=20,
                            )
                            transition_label = ui.Label(
                                "\n".join(view.transitions),
                                word_wrap=True,
                                height=0,
                            )
                            ui.Separator(height=4)
                            details_label = ui.Label(
                                "\n".join(view.lines),
                                word_wrap=True,
                                height=0,
                            )
                            command_label = ui.Label(
                                view.command_indicator,
                                word_wrap=True,
                                style={"color": _TONE_COLORS["policy"]},
                                height=0,
                            )
                            self._models[view.label] = _VehicleModels(
                                status=status_label,
                                transitions=transition_label,
                                details=details_label,
                                command=command_label,
                            )

    def _update_models(self) -> None:
        for view in self._views:
            models = self._models[view.label]
            models.status.text = view.status
            models.transitions.text = "\n".join(view.transitions)
            models.details.text = "\n".join(view.lines)
            models.command.text = view.command_indicator
