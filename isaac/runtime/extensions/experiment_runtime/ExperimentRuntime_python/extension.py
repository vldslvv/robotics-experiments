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
- Timeline ``PLAY`` spawns the fleet, starts the PX4 lockstep bridge group, and
  -- if any drone has a routine -- builds one ``RoutineDriverState`` per flying drone
  and ticks them from a separate Kit app-update subscription (never the bridge's
  physics legs). Spawn stays gated on PLAY so the persisted stage is not mutated
  with drones until the user presses Play.
- ``STOP`` / ``PAUSE`` and ``on_shutdown`` drop the control tick and tear down.

It is always *forgiving*: a routine ``FAIL``, a PX4 failsafe, or a phase timeout
is logged via ``carb`` and that drone's driver idles (manual control remains); a
control bug is caught per driver so it never spams Kit or stalls the others. The
strict, asserting counterpart is the standalone scenario's adapter.
"""

import os
from typing import Any

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
from omni.timeline import TimelineEventType
from omni.usd import StageEventType

from ExperimentRuntime_python.global_variables import EXTENSION_TITLE
from robotics_experiments.config import load_config
from robotics_experiments.runtime import (
    FleetDriverEnv,
    adopt_current_stage,
    apply_settings,
    build_drivers,
    create_fleet,
    open_scene,
    spawn_fleet,
    start_bridges,
    teardown,
    tick_drivers,
)

__all__ = ["Extension"]

_LOG_PREFIX = f"[{EXTENSION_TITLE}]"
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"


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
        self._scene_open_sub: Any = None
        self._update_sub: Any = None

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
        if self._fleet.stage is None:
            carb.log_error(f"{_LOG_PREFIX} no stage adopted yet; cannot spawn fleet")
            return
        self._fleet = teardown(state=self._fleet)
        self._fleet = spawn_fleet(state=self._fleet)
        self._fleet = start_bridges(state=self._fleet)
        # routine.steps is the per-drone on/off switch: build drivers only when at
        # least one drone has a routine to fly.
        if any(drone.vehicle.routine.steps for drone in self._fleet.drones):
            self._fleet = build_drivers(state=self._fleet, make_env=self._make_env)
            app = omni.kit.app.get_app()
            self._update_sub = app.get_update_event_stream().create_subscription_to_pop(
                self._on_update, name="experiment_runtime_control"
            )

    def _stop_control(self):
        """Drop the control tick subscription (the orchestrator owns the ports)."""
        self._update_sub = None

    def _on_update(self, event):
        """One control tick: advance every flying drone's phase machine.

        Forgiving: a control bug in one drone is logged and must not stop the
        others or spam every frame, so each driver tick is guarded.
        """
        self._fleet = tick_drivers(state=self._fleet, on_error=self._log_tick_error)

    def _log_tick_error(self, flying, exc):
        carb.log_error(f"{_LOG_PREFIX} [{flying.spawned.label}] control tick failed: {exc}")

    def _make_env(self, spawned, port):
        """Build the forgiving :class:`DriverEnvironment` for one flying drone."""
        label = spawned.label
        return FleetDriverEnv(
            spawned,
            port,
            log=lambda message: carb.log_info(f"{_LOG_PREFIX} {message}"),
            on_failure=lambda kind, message: carb.log_warn(
                f"{_LOG_PREFIX} {message} ({kind.name}); stopping routine "
                "(manual control remains available)"
            ),
            on_step_advanced=lambda step_label: carb.log_info(
                f"{_LOG_PREFIX} [{label}] completed step {step_label!r}"
            ),
        )
