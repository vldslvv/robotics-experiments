# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.api.objects import VisualCuboid  # noqa: E402

my_world = World(stage_units_in_meters=1.0)

# Utility to load USD/USDC/USDZ scenes into the current stage via reference
# This uses Isaac Sim's stage utilities to add a reference at a given prim path.
# Note: Referencing is preferred over sublayering for modular scene composition.
try:
    from omni.isaac.core.utils.stage import add_reference_to_stage

    def load_usd_scene(usd_path: str, prim_path: str = "/World/ExternalScene") -> None:
        """Load a USD/USDC/USDZ file into the current stage by adding a reference.

        Args:
            usd_path: Absolute path or Omniverse URL to the USD/USDC/USDZ asset.
            prim_path: Prim path under which the reference will be placed.
        """
        # Add the referenced scene at the desired prim path
        add_reference_to_stage(usd_path, prim_path)

    # Example usage (commented):
    load_usd_scene("/path/to/your_scene.usdc", prim_path="/World/MyUSDCScene")
except Exception as e:
    # If stage utilities are unavailable, we avoid breaking the example and just log.
    # You can remove this try/except if your environment always has omni.isaac available.
    print(f"Warning: Could not import add_reference_to_stage: {e}")

cube_1 = my_world.scene.add(
    VisualCuboid(
        prim_path="/new_cube_1",
        name="visual_cube",
        position=np.array([0, 0, 0.5]),
        size=0.3,
        color=np.array([255, 255, 255]),
    )
)

# cube_2 = my_world.scene.add(
#     DynamicCuboid(
#         prim_path="/new_cube_2",
#         name="cube_1",
#         position=np.array([0, 0, 1.0]),
#         scale=np.array([0.6, 0.5, 0.2]),
#         size=1.0,
#         color=np.array([255, 0, 0]),
#     )
# )
#
# cube_3 = my_world.scene.add(
#     DynamicCuboid(
#         prim_path="/new_cube_3",
#         name="cube_2",
#         position=np.array([0, 0, 3.0]),
#         scale=np.array([0.1, 0.1, 0.1]),
#         size=1.0,
#         color=np.array([0, 0, 255]),
#         linear_velocity=np.array([0, 0, 0.4]),
#     )
# )

my_world.scene.add_default_ground_plane()

for _reset_index in range(5):
    my_world.reset()
    for _i in range(500):
        my_world.step(render=True)
        # print(cube_2.get_angular_velocity())
        # print(cube_2.get_world_pose())

# simulation_app.close()
