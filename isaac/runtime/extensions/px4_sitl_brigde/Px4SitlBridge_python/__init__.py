"""Public API for the PX4 SITL bridge extension.

The bridge is layered so each component is usable on its own — see
``docs/PX4_BRIDGE_CONTRACTS.md`` for the contract these layers implement.

Typical wiring (done by the drone-spawning host, not by this extension):

>>> from Px4SitlBridge_python import (
...     Px4Bridge, Px4MavlinkPort, PegasusQuadraticThrustModel,
... )
>>> port = Px4MavlinkPort(num_rotors=4)
>>> model = PegasusQuadraticThrustModel(num_rotors=4)
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
    # ``extension.py`` pulls Kit-only deps (omni, carb) that are absent in
    # pure-Python smoke/unit runs; tolerate those, re-raise anything else.
    if exc.name not in ("omni", "carb"):
        raise

# Re-export the narrowed core bridge API directly; this package is a thin shim
# that lets Kit import the extension while consumers use
# ``robotics_experiments.px4_bridge``.
from robotics_experiments.px4_bridge import (  # noqa: F401
    IsaacArticulationIO,
    LockstepSettings,
    PegasusQuadraticThrustModel,
    Px4Bridge,
    Px4MavlinkPort,
    discover_px4_rotor_mapping,
)

__all__ = [
    "IsaacArticulationIO",
    "LockstepSettings",
    "PegasusQuadraticThrustModel",
    "Px4Bridge",
    "Px4MavlinkPort",
    "discover_px4_rotor_mapping",
]
