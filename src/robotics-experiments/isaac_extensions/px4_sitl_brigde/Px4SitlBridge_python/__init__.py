"""Public API for the PX4 SITL bridge extension.

The bridge is layered so each component is usable on its own — see
``docs/PX4_BRIDGE_CONTRACTS.md`` for the contract these layers implement.

Typical wiring (done by the drone-spawning host, not by this extension):

>>> from Px4SitlBridge_python import (
...     Px4Bridge, Px4MavlinkPort, LinearThrustModel,
... )
>>> port = Px4MavlinkPort(num_rotors=4, px4_autolaunch=False)
>>> model = LinearThrustModel(num_rotors=4, max_force_n=8.0)
>>> bridge = Px4Bridge(
...     articulation_path="/World/DroneSpawned",
...     rotor_link_paths=[...4 paths in PX4 motor order...],
...     thrust_model=model,
...     port=port,
... )
>>> bridge.start()              # at Play
>>> bridge.on_physics_step(dt, sim_time_us)  # every physics tick
>>> bridge.stop()               # at Stop
"""

try:
    from .extension import *  # noqa: F401,F403  -- registers omni.ext entrypoint
except ModuleNotFoundError as exc:
    if exc.name != "omni":
        raise
from .ports import AbstractFlightStackBridge, Px4MavlinkPort
from .px4_bridge import Px4Bridge
from .thrust_model import LinearThrustModel, RotorThrustModel
from .vehicle_state import VehicleState

__all__ = [
    "AbstractFlightStackBridge",
    "LinearThrustModel",
    "Px4Bridge",
    "Px4MavlinkPort",
    "RotorThrustModel",
    "VehicleState",
]
