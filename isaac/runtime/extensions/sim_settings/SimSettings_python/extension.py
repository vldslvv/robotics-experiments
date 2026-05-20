"""Kit extension workflow for applying shared simulation settings.

The extension does not own Isaac startup or scene loading. It subscribes to
Kit stage events and delegates the actual USD/carb mutations to
``robotics_experiments.simulation.apply_settings(stage, settings)``, which is
the shared implementation used by standalone tests and extension workflows.
"""

import omni.ext
import omni.usd
from omni.usd import StageEventType

from robotics_experiments.simulation import apply_settings

from sim_config import get_config

__all__ = ["Extension"]


class Extension(omni.ext.IExt):
    """Isaac Sim extension that applies fixed physics and render settings.

    Settings are re-applied whenever a stage finishes opening so that opening
    a new scene does not silently revert the configured rates or gravity. Kit
    owns the app and stage lifecycle; this extension only bridges those stage
    events and its Hydra-backed config adapter to the shared ``apply_settings``
    API.
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
            apply_settings(stage, get_config())

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
                apply_settings(stage, get_config())
