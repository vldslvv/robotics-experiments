# This script's only purpose is to test if LSP symbols are loaded in the editor
# open it and check
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

from isaacsim.core.api import World
from isaacsim.core.api.objects import VisualCuboid

from isaacsim.core.prims import Articulation

art = Articulation(prim_paths_expr="/World/robot")

