# This script's only purpose is to test if LSP symbols are loaded in the editor
# open it and check
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})


from isaacsim.core.prims import Articulation  # noqa: E402

art = Articulation(prim_paths_expr="/World/robot")
