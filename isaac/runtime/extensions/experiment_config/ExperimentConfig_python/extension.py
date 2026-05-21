import carb
import omni.ext

from robotics_experiments.config import get_config

from .global_variables import EXTENSION_TITLE


class Extension(omni.ext.IExt):
    """Dependency-anchor extension that eagerly composes the experiment config.

    Does no per-frame work. Its sole purpose is to be declared as a Kit
    dependency by the other experiment extensions so the Hydra-composed
    :class:`experiment_config.EnvConfig` is available (and its composition
    errors surface early) before any consumer's ``on_startup`` runs.
    """

    def on_startup(self, ext_id: str):
        """Compose and cache the experiment config on Kit startup.

        Args:
            ext_id: Extension identifier assigned by the Kit runtime.
        """
        cfg = get_config()
        carb.log_info(f"[{EXTENSION_TITLE}] Composed config: {cfg}")

    def on_shutdown(self):
        """No-op; the cached config lives for the Kit process lifetime."""
        pass
