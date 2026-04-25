import omni.ext
from isaacsim.core.utils.stage import add_reference_to_stage


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
        pass

    def on_shutdown(self):
        pass
