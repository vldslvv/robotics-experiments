import omni.ext
import omni.timeline
import omni.usd
from omni.timeline import TimelineEventType
from omni.usd import StageEventType

from robotics_experiments.config import get_config as _get_experiment_config
from robotics_experiments.drone import despawn_drone, spawn_drone

from .bridge_host import BridgeHost


class Extension(omni.ext.IExt):
    """Isaac Sim extension that spawns a drone into the current stage.

    Subscribes to stage-open and timeline events so the drone is (re)spawned at
    a random pose whenever a stage is opened or playback starts, and despawned
    when playback stops.

    Also owns a :class:`BridgeHost` that, when ``px4.enabled`` is true in the
    experiment config, constructs a :class:`Px4Bridge` on PLAY and tears it
    down on STOP. The host is opt-in so the lower bridge layers can be staged
    independently before the full pipeline is wired.
    """

    def on_startup(self, ext_id: str):
        """Register stage and timeline subscriptions and spawn the initial drone.

        Called by the Kit runtime when the extension is enabled. If a stage is
        already loaded, spawns a drone immediately; otherwise waits for the
        next stage-opened event.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        self._usd_context = omni.usd.get_context()
        stage_events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = stage_events.create_subscription_to_pop(
            self._on_stage_event,
        )
        self._timeline = omni.timeline.get_timeline_interface()
        timeline_events = self._timeline.get_timeline_event_stream()
        self._timeline_event_sub = timeline_events.create_subscription_to_pop(
            self._on_timeline_event,
        )
        self._bridge_host: BridgeHost | None = None
        stage = self._usd_context.get_stage()
        if stage is not None:
            spawn_drone(stage)

    def on_shutdown(self):
        """Release stage and timeline subscriptions and stop the bridge.

        Called by the Kit runtime when the extension is disabled. Clearing the
        subscription references causes the underlying event handles to be
        freed.
        """
        self._stage_event_sub = None
        self._timeline_event_sub = None
        if self._bridge_host is not None:
            self._bridge_host.stop()
            self._bridge_host = None

    def _on_stage_event(self, event):
        """Spawn a drone when a new stage finishes opening.

        Args:
            event: Stage event emitted by ``omni.usd``; only
                ``StageEventType.OPENED`` is handled.
        """
        if event.type == int(StageEventType.OPENED):
            stage = self._usd_context.get_stage()
            if stage is not None:
                spawn_drone(stage)

    def _on_timeline_event(self, event):
        """Spawn or despawn the drone based on timeline transitions.

        Despawns on ``STOP`` so the prim is removed from the persisted stage,
        and respawns at a fresh random pose on ``PLAY``.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        stage = self._usd_context.get_stage()
        if event.type == int(TimelineEventType.STOP):
            if self._bridge_host is not None:
                self._bridge_host.stop()
                self._bridge_host = None
            despawn_drone(stage)
        elif event.type == int(TimelineEventType.PLAY) and stage is not None:
            spawn_drone(stage)
            self._start_bridge_if_enabled(stage)

    def _start_bridge_if_enabled(self, stage):
        """Construct and start the PX4 bridge host when configured.

        No-op when ``px4.enabled`` is false in the experiment config.
        """
        env = _get_experiment_config()
        if not env.px4.enabled:
            return
        drone_cfg = env.drone
        default_prim = stage.GetDefaultPrim()
        parent = default_prim.GetPath().pathString if default_prim else "/World"
        articulation_path = f"{parent}/{drone_cfg.prim_name}"

        if self._bridge_host is not None:
            self._bridge_host.stop()
        self._bridge_host = BridgeHost(articulation_path=articulation_path, config=env.px4)
        self._bridge_host.start()
