import omni.ext
from isaacsim.core.utils.stage import add_reference_to_stage

from .global_variables import DRONE_PRIM_PATH, DRONE_USD_PATH


class Extension(omni.ext.IExt):
    def on_startup(self, ext_id: str):
        add_reference_to_stage(DRONE_USD_PATH, DRONE_PRIM_PATH)

    def on_shutdown(self):
        pass
