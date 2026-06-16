"""Omniverse extension shell.

Per the contract, this extension does not own the scene, the drone, or the
physics callback. The host (``drone_spawner``) constructs the ``Px4Bridge``
and ticks it. This shell exists only so Kit imports the package and the
public API in ``__init__.py`` is available to consumers; it announces its
own load/unload for parity with the other extensions but performs no flight
or scene work.
"""

import carb
import omni.ext

from .global_variables import EXTENSION_TITLE


class Extension(omni.ext.IExt):
    """Load/unload-only extension entrypoint."""

    def on_startup(self, ext_id: str):
        carb.log_info(f"[{EXTENSION_TITLE}] Extension loaded; host owns the bridge lifecycle.")

    def on_shutdown(self):
        carb.log_info(f"[{EXTENSION_TITLE}] Extension unloaded.")
