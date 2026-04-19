import omni.ext
import omni.timeline
import omni.usd
from drone_config import despawn_drone, spawn_drone
from omni.timeline import TimelineEventType
from omni.usd import StageEventType


class Extension(omni.ext.IExt):
    """Isaac Sim extension that spawns a drone into the current stage.

    Subscribes to stage-open and timeline events so the drone is (re)spawned at
    a random pose whenever a stage is opened or playback starts, and despawned
    when playback stops.
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
        self._stage_event_sub = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event
        )
        self._timeline = omni.timeline.get_timeline_interface()
        self._timeline_event_sub = self._timeline.get_timeline_event_stream().create_subscription_to_pop(
            self._on_timeline_event
        )
        stage = self._usd_context.get_stage()
        if stage is not None:
            spawn_drone(stage)

    def on_shutdown(self):
        """Release stage and timeline subscriptions.

        Called by the Kit runtime when the extension is disabled. Clearing the
        subscription references causes the underlying event handles to be
        freed.
        """
        self._stage_event_sub = None
        self._timeline_event_sub = None

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
            despawn_drone(stage)
        elif event.type == int(TimelineEventType.PLAY) and stage is not None:
            spawn_drone(stage)
