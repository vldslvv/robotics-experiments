import omni.ext
import omni.usd
from omni.usd import StageEventType
from sim_config import apply_settings


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        self._usd_context = omni.usd.get_context()
        self._stage_event_sub = self._usd_context.get_stage_event_stream().create_subscription_to_pop(
            self._on_stage_event
        )
        stage = self._usd_context.get_stage()
        if stage is not None:
            apply_settings(stage)

    def on_shutdown(self):
        self._stage_event_sub = None

    def _on_stage_event(self, event):
        if event.type == int(StageEventType.OPENED):
            stage = self._usd_context.get_stage()
            if stage is not None:
                apply_settings(stage)
