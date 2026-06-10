import omni.ext
import omni.timeline
import omni.usd
from omni.timeline import TimelineEventType
from omni.usd import StageEventType

from robotics_experiments.config import get_config as _get_experiment_config
from robotics_experiments.drone import despawn_drone, spawn_drone

from .bridge_host import BridgeHost

__all__ = ["Extension"]


class Extension(omni.ext.IExt):
    """Isaac Sim extension that owns the runtime drone lifecycle.

    Subscribes to stage-open and timeline events. The drone is spawned only
    when playback starts and despawned when playback stops, so opening a stage
    or enabling the extension does not mutate the scene with preview state.

    Also owns a :class:`BridgeHost` that, when ``px4.enabled`` is true in the
    experiment config, constructs a :class:`Px4Bridge` on PLAY and tears it
    down on STOP. The host is opt-in so the lower bridge layers can be staged
    independently before the full pipeline is wired.
    """

    def on_startup(self, ext_id: str):
        """Register stage and timeline subscriptions.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        usd_context = omni.usd.get_context()
        self.stage = usd_context.get_stage()
        stage_events = usd_context.get_stage_event_stream()
        self._stage_event_sub = stage_events.create_subscription_to_pop(
            self._on_stage_event,
        )
        timeline = omni.timeline.get_timeline_interface()
        timeline_events = timeline.get_timeline_event_stream()
        self._timeline_event_sub = timeline_events.create_subscription_to_pop(
            self._on_timeline_event,
        )
        self._bridge_host: BridgeHost | None = None

    def on_shutdown(self):
        """Release subscriptions and clear any runtime drone state.

        Called by the Kit runtime when the extension is disabled. Clearing the
        subscription references causes the underlying event handles to be
        freed.
        """
        self._stage_event_sub = None
        self._timeline_event_sub = None
        self._teardown_drone()

    def _on_stage_event(self, event):
        """Clear runtime state when a new stage finishes opening.

        Args:
            event: Stage event emitted by ``omni.usd``; only
                ``StageEventType.OPENED`` is handled.
        """
        if event.type == int(StageEventType.OPENED):
            self.stage = omni.usd.get_context().get_stage()
            self._teardown_drone()

    def _on_timeline_event(self, event):
        """Spawn or despawn the drone based on timeline transitions.

        Spawns a fresh runtime drone on ``PLAY`` and despawns on ``STOP`` or
        ``PAUSE`` so the prim is removed from the persisted stage.

        Args:
            event: Timeline event emitted by ``omni.timeline``.
        """
        if event.type == int(TimelineEventType.PLAY):
            self._setup_drone()
        elif event.type in (int(TimelineEventType.STOP), int(TimelineEventType.PAUSE)):
            self._teardown_drone()

    def _setup_drone(self):
        """Spawn the runtime drone and start its bridge if configured."""
        if self.stage is None:
            return
        self._teardown_drone()

        env = _get_experiment_config()
        spawn_drone(self.stage, env.drone)
        if not env.px4.enabled:
            return
        drone_cfg = env.drone
        default_prim = self.stage.GetDefaultPrim()
        parent = default_prim.GetPath().pathString if default_prim else "/World"
        articulation_path = f"{parent}/{drone_cfg.prim_name}"
        self._bridge_host = BridgeHost(articulation_path=articulation_path, config=env.px4)
        self._bridge_host.start()

    def _teardown_drone(self):
        """Stop the bridge and then remove the runtime drone from the stage."""
        if self._bridge_host is not None:
            self._bridge_host.stop()
            self._bridge_host = None
        despawn_drone(self.stage)
