"""Live Kit shell hosting the host-agnostic fleet runtime orchestrator.

This is the runtime path for ``make run-all`` / ``make run-isaac``. The extension
owns no flight logic: it builds a
:class:`~robotics_experiments.runtime.FleetOrchestrator` and wires Kit events to
its phases. The orchestrator owns scene-open and sim settings (so the live boot
does not rely on Kit's ``--exec open_stage.py`` or a separate ``sim_settings``
extension):

- ``on_startup`` subscribes to stage + timeline events and arms an app-update
  callback that opens the configured scene once Kit reports the app is ready. The
  stage is adopted and settings applied on the ``OPENED`` event, once the stage
  has finished loading (``lightswitch_spawner`` reacts to the same event).
- Timeline ``PLAY`` spawns the fleet, starts the PX4 lockstep bridge group, and
  -- if any drone has a routine -- builds one ``RoutineDriver`` per flying drone
  and ticks them from a separate Kit app-update subscription (never the bridge's
  physics legs). Spawn stays gated on PLAY so the persisted stage is not mutated
  with drones until the user presses Play.
- ``STOP`` / ``PAUSE`` and ``on_shutdown`` drop the control tick and tear down.

It is always *forgiving*: a routine ``FAIL``, a PX4 failsafe, or a phase timeout
is logged via ``carb`` and that drone's driver idles (manual control remains); a
control bug is caught per driver so it never spams Kit or stalls the others. The
strict, asserting counterpart is the standalone scenario's adapter.
"""

from typing import Any

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
from omni.timeline import TimelineEventType
from omni.usd import StageEventType

from robotics_experiments.config import get_config as _get_experiment_config
from robotics_experiments.runtime import FleetDriverEnv, FleetOrchestrator

from ExperimentRuntime_python.global_variables import EXTENSION_TITLE

__all__ = ["Extension"]

_LOG_PREFIX = f"[{EXTENSION_TITLE}]"


class Extension(omni.ext.IExt):
    """Kit lifecycle wiring around the shared :class:`FleetOrchestrator`."""

    def on_startup(self, ext_id: str):
        """Build the orchestrator and subscribe to Kit events.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        carb.log_info(f"{_LOG_PREFIX} starting")
        self._orchestrator = FleetOrchestrator(_get_experiment_config())
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
        self._orchestrator.open_scene()
        carb.log_info(f"{_LOG_PREFIX} running. Press Play to fly the configured routine.")

    def on_shutdown(self):
        """Drop subscriptions and tear the fleet down."""
        self._scene_open_sub = None
        self._stop_control()
        self._stage_event_sub = None
        self._timeline_event_sub = None
        self._orchestrator.teardown()

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
            self._orchestrator.teardown()
            self._orchestrator.adopt_current_stage()
            self._orchestrator.apply_settings()

    def _on_timeline_event(self, event):
        """Spawn the fleet on PLAY; tear it down on STOP/PAUSE.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(TimelineEventType.PLAY):
            self._setup_fleet()
        elif event.type in (int(TimelineEventType.STOP), int(TimelineEventType.PAUSE)):
            self._stop_control()
            self._orchestrator.teardown()

    def _setup_fleet(self):
        """Spawn the fleet, start the bridges, and arm the routine drivers."""
        if self._orchestrator.stage is None:
            carb.log_error(f"{_LOG_PREFIX} no stage adopted yet; cannot spawn fleet")
            return
        self._orchestrator.teardown()
        self._orchestrator.spawn_fleet()
        self._orchestrator.start_bridges()
        # routine.steps is the per-drone on/off switch: build drivers only when at
        # least one drone has a routine to fly.
        if any(h.vehicle.routine.steps for h in self._orchestrator.drones):
            self._orchestrator.build_drivers(self._make_env)
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
        for flying in self._orchestrator.flying:
            try:
                flying.driver.tick()
            except Exception as exc:  # noqa: BLE001 - keep the sim loop alive
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
