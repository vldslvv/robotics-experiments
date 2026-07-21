"""Live Kit shell hosting the host-agnostic fleet runtime orchestrator.

This is the runtime path for ``make run-all`` / ``make run-isaac``. The extension
owns no flight logic: it builds a
:class:`~robotics_experiments.runtime.FleetState` and wires Kit events to
its transitions. The runtime owns scene-open and sim settings (so the live boot
does not rely on Kit's ``--exec open_stage.py`` or a separate ``sim_settings``
extension):

- ``on_startup`` subscribes to stage + timeline events and arms an app-update
  callback that opens the configured scene once Kit reports the app is ready. The
  stage is adopted and settings applied on the ``OPENED`` event, once the stage
  has finished loading (``lightswitch_spawner`` reacts to the same event).
- Timeline ``PLAY`` spawns the fleet, starts the PX4 lockstep bridge group,
  builds one ``DriverContext`` plus one progress-only ``RoutineDriverState`` per
  bridge-enabled drone, and ticks them from a separate PhysX-step subscription
  (never the bridge's physics legs). Spawn stays gated on PLAY so
  the persisted stage is not mutated with drones until the user presses Play.
- ``STOP`` / ``PAUSE`` and ``on_shutdown`` drop the control tick and tear down.

It is always *forgiving*: a routine ``FAIL``, a PX4 failsafe, or a phase timeout
is logged via ``carb`` and that drone's driver idles (manual control remains); a
control bug is caught per driver so it never spams Kit or stalls the others. The
strict, asserting counterpart is the standalone scenario's adapter.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import carb
import omni.ext
import omni.kit.app
import omni.physx
import omni.timeline
import omni.usd
from omni.timeline import TimelineEventType
from omni.usd import StageEventType

from ExperimentRuntime_python.global_variables import EXTENSION_TITLE
from robotics_experiments.commanding import FailureKind
from robotics_experiments.config import load_config
from robotics_experiments.px4_bridge import BridgeStatus
from robotics_experiments.runtime import (
    FleetDriverEnv,
    adopt_current_stage,
    apply_settings,
    build_drivers,
    create_fleet,
    open_scene,
    read_bridge_host_status,
    spawn_fleet,
    start_bridges,
    teardown,
    tick_drivers,
)

__all__ = [
    "ExperimentRuntimeSnapshot",
    "Extension",
    "VehicleRuntimeSnapshot",
    "get_instance",
]

_LOG_PREFIX = f"[{EXTENSION_TITLE}]"
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_INSTANCE: Extension | None = None


@dataclass(frozen=True)
class VehicleRuntimeSnapshot:
    """Immutable progress and bridge status for one flying vehicle."""

    label: str
    sim_time_us: int
    bridge_status: BridgeStatus | None
    completed_steps: int


@dataclass(frozen=True)
class ExperimentRuntimeSnapshot:
    """Read-only live-extension state exposed to dependent Kit plugins."""

    scene_ready: bool
    failure: str | None
    vehicles: tuple[VehicleRuntimeSnapshot, ...]


class Extension(omni.ext.IExt):
    """Kit lifecycle wiring around the shared frozen fleet state."""

    def on_startup(self, ext_id: str):
        """Build the orchestrator and subscribe to Kit events.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        carb.log_info(f"{_LOG_PREFIX} starting")
        try:
            config_path = os.environ[_CONFIG_PATH_ENV]
        except KeyError as exc:
            raise RuntimeError(f"missing required environment variable {_CONFIG_PATH_ENV}") from exc
        self._fleet = create_fleet(config=load_config(config_path))
        self._scene_opened = False
        self._failure: str | None = None
        self._scene_open_sub: Any = None
        self._physics_sub: Any = None

        usd_context = omni.usd.get_context()
        stage_events = usd_context.get_stage_event_stream()
        self._stage_event_sub = stage_events.create_subscription_to_pop(
            self._on_stage_event, name="experiment_runtime_stage"
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline_events = timeline.get_timeline_event_stream()
        self._timeline_event_sub = timeline_events.create_subscription_to_pop(
            self._on_timeline_event, name="experiment_runtime_timeline"
        )
        app = omni.kit.app.get_app()
        self._scene_open_sub = app.get_update_event_stream().create_subscription_to_pop(
            self._open_scene_when_ready, name="experiment_runtime_open_scene"
        )
        global _INSTANCE
        _INSTANCE = self

    def _open_scene_when_ready(self, event):
        """Open the configured scene once Kit reports the app is ready."""
        if self._scene_opened:
            return
        app = omni.kit.app.get_app()
        if not app.is_app_ready():
            return
        self._scene_opened = True
        self._scene_open_sub = None
        self._fleet = open_scene(state=self._fleet)
        carb.log_info(f"{_LOG_PREFIX} running. Press Play to fly the configured routine.")

    def on_shutdown(self):
        """Drop subscriptions and tear the fleet down."""
        self._scene_open_sub = None
        self._stop_control()
        self._stage_event_sub = None
        self._timeline_event_sub = None
        self._fleet = teardown(state=self._fleet)
        global _INSTANCE
        if _INSTANCE is self:
            _INSTANCE = None

    def read_status(self) -> ExperimentRuntimeSnapshot:
        """Return an immutable snapshot for dependent runtime plugins."""
        vehicles = []
        for flying in self._fleet.flying:
            host = flying.spawned.host
            if host is None:
                raise RuntimeError(
                    f"{_LOG_PREFIX} flying vehicle {flying.spawned.label!r} has no bridge host"
                )
            host_status = read_bridge_host_status(host=host)
            vehicles.append(
                VehicleRuntimeSnapshot(
                    label=flying.spawned.label,
                    sim_time_us=host_status.sim_time_us,
                    bridge_status=host_status.bridge_status,
                    completed_steps=flying.driver.previous_cursor,
                )
            )
        return ExperimentRuntimeSnapshot(
            scene_ready=self._fleet.stage is not None,
            failure=self._failure,
            vehicles=tuple(vehicles),
        )

    def _on_stage_event(self, event):
        """Adopt and configure a stage once it finishes opening.

        Args:
            event: Stage event emitted by ``omni.usd``; only
                ``StageEventType.OPENED`` is handled.
        """
        if event.type == int(StageEventType.OPENED):
            # A new stage replaces any running fleet: stop control and tear down,
            # then adopt the new stage and apply settings to it.
            self._stop_control()
            self._fleet = teardown(state=self._fleet)
            self._fleet = adopt_current_stage(state=self._fleet)
            self._fleet = apply_settings(state=self._fleet)

    def _on_timeline_event(self, event):
        """Spawn the fleet on PLAY; tear it down on STOP/PAUSE.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(TimelineEventType.PLAY):
            self._setup_fleet()
        elif event.type in (int(TimelineEventType.STOP), int(TimelineEventType.PAUSE)):
            self._stop_control()
            self._fleet = teardown(state=self._fleet)

    def _setup_fleet(self):
        """Spawn the fleet, start the bridges, and arm the routine drivers."""
        self._failure = None
        if self._fleet.stage is None:
            carb.log_error(f"{_LOG_PREFIX} no stage adopted yet; cannot spawn fleet")
            return
        # Kit may replace the context's Usd.Stage wrapper while completing a
        # scene/timeline transition. A wrapper retained from OPENED then raises
        # Boost.Python.ArgumentError even though the context has a valid stage.
        # Re-adopt at the imperative PLAY boundary before any stage mutation.
        # Adoption replaces only FleetState.stage, retaining the existing fleet
        # resources; teardown must follow it so despawning uses the valid wrapper.
        self._fleet = adopt_current_stage(state=self._fleet)
        self._fleet = teardown(state=self._fleet)
        self._fleet = spawn_fleet(state=self._fleet)
        self._fleet = start_bridges(state=self._fleet)
        self._fleet = build_drivers(state=self._fleet, make_env=self._make_env)
        self._physics_sub = omni.physx.get_physx_interface().subscribe_physics_step_events(
            self._on_physics_step
        )

    def _stop_control(self):
        """Drop the control tick subscription (the orchestrator owns the ports)."""
        self._physics_sub = None

    def _on_physics_step(self, dt, *_args):
        """Advance the physics-derived fleet driver cadence.

        Forgiving: a control bug in one drone is logged and must not stop the
        others or spam every frame, so each driver tick is guarded.
        """
        self._fleet = tick_drivers(
            state=self._fleet,
            physics_dt_s=float(dt),
            on_error=self._log_tick_error,
        )

    def _log_tick_error(self, flying, exc):
        message = f"[{flying.spawned.label}] control tick failed: {exc}"
        self._failure = message
        carb.log_error(f"{_LOG_PREFIX} {message}")

    def _log_driver_failure(self, kind: FailureKind, message: str) -> None:
        self._failure = f"{message} ({kind.name})"
        carb.log_warn(
            f"{_LOG_PREFIX} {message} ({kind.name}); stopping routine "
            "(manual control remains available)"
        )

    def _make_env(self, spawned, port):
        """Build the forgiving :class:`DriverEnvironment` for one flying drone."""
        label = spawned.label
        return FleetDriverEnv(
            spawned,
            port,
            log=lambda message: carb.log_info(f"{_LOG_PREFIX} {message}"),
            on_failure=self._log_driver_failure,
            on_step_advanced=lambda step_label: carb.log_info(
                f"{_LOG_PREFIX} [{label}] completed step {step_label!r}"
            ),
        )


def get_instance() -> Extension:
    """Return the active live runtime extension for dependent Kit plugins."""
    if _INSTANCE is None:
        raise RuntimeError("experiment_runtime extension is not active")
    return _INSTANCE
