import omni.ext
import omni.timeline
import omni.usd
from drone_config import despawn_drone, spawn_drone
from omni.timeline import TimelineEventType
from omni.usd import StageEventType


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
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
        self._stage_event_sub = None
        self._timeline_event_sub = None

    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            stage = self._usd_context.get_stage()
            if stage is not None:
                spawn_drone(stage)

    def _on_timeline_event(self, event):
        stage = self._usd_context.get_stage()
        if event.type == int(TimelineEventType.STOP):
            despawn_drone(stage)
        elif event.type == int(TimelineEventType.PLAY) and stage is not None:
            spawn_drone(stage)
