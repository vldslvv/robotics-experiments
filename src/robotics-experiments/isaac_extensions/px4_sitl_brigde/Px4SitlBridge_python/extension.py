"""Omniverse extension shell.

Per the contract, this extension does not own the scene, the drone, or the
physics callback. The host (``drone_spawner``) constructs the ``Px4Bridge``
and ticks it. This shell exists only so Kit imports the package and the
public API in ``__init__.py`` is available to consumers.
"""

import omni.ext


class Extension(omni.ext.IExt):
    """No-op extension entrypoint."""

    def on_startup(self, ext_id: str):
        pass

    def on_shutdown(self):
        pass
