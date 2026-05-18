import omni.ext
import omni.usd
from omni.usd import StageEventType

from robotics_experiments.simulation import apply_settings


class Extension(omni.ext.IExt):
    """Isaac Sim extension that applies fixed physics and render settings.

    Settings are re-applied whenever a stage finishes opening so that opening
    a new scene does not silently revert the configured rates or gravity.
    """

    def on_startup(self, ext_id: str):
        """Subscribe to stage-open events and apply settings to any loaded stage.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        self._usd_context = omni.usd.get_context()
        stage_events = self._usd_context.get_stage_event_stream()
        self._stage_event_sub = stage_events.create_subscription_to_pop(
            self._on_stage_event,
        )
        stage = self._usd_context.get_stage()
        if stage is not None:
            apply_settings(stage)

    def on_shutdown(self):
        """Release the stage event subscription."""
        self._stage_event_sub = None

    def _on_stage_event(self, event):
        """Reapply sim settings whenever a stage finishes opening.

        Args:
            event: Stage event emitted by ``omni.usd``; only
                ``StageEventType.OPENED`` is handled.
        """
        if event.type == int(StageEventType.OPENED):
            stage = self._usd_context.get_stage()
            if stage is not None:
                apply_settings(stage)
