"""Public API for the PX4 SITL bridge extension.

The bridge is layered so each component is usable on its own — see
``docs/PX4_BRIDGE_CONTRACTS.md`` for the contract these layers implement.

Typical wiring (done by the drone-spawning host, not by this extension):

>>> from Px4SitlBridge_python import (
...     Px4MavlinkPort, PegasusQuadraticThrustModel, create_px4_bridge,
... )
>>> port = Px4MavlinkPort(num_rotors=4)
>>> model = PegasusQuadraticThrustModel(num_rotors=4)
>>> bridge = create_px4_bridge(
...     articulation_path="/World/DroneSpawned",
...     rotor_link_paths=[...4 paths in PX4 motor order...],
...     thrust_model=model,
...     port=port,
... )
>>> bridge = start_px4_bridge(state=bridge)  # at Play
>>> bridge = tick_px4_bridge(state=bridge, dt=dt, sim_time_us=sim_time_us)
>>> bridge = stop_px4_bridge(state=bridge)   # at Stop
"""

try:
    from Px4SitlBridge_python.extension import *  # noqa: F401,F403  -- registers omni.ext entrypoint
except ModuleNotFoundError as exc:
    # ``extension.py`` pulls Kit-only deps (omni, carb) that are absent in
    # pure-Python smoke/unit runs; tolerate those, re-raise anything else.
    if exc.name not in ("omni", "carb"):
        raise

# Re-export the core bridge API for Kit-side imports.
from robotics_experiments.px4_bridge import (  # noqa: F401  # noqa: F401
    IsaacArticulationIO,  # noqa: F401
    LockstepSettings,  # noqa: F401
    PegasusQuadraticThrustModel,
    Px4BridgeState,  # noqa: F401
    Px4MavlinkPort,  # noqa: F401
    create_px4_bridge,
    discover_px4_rotor_mapping,
    start_px4_bridge,
    stop_px4_bridge,
    tick_px4_bridge,
)

__all__ = [
    "IsaacArticulationIO",
    "LockstepSettings",
    "PegasusQuadraticThrustModel",
    "Px4BridgeState",
    "Px4MavlinkPort",
    "create_px4_bridge",
    "discover_px4_rotor_mapping",
    "start_px4_bridge",
    "stop_px4_bridge",
    "tick_px4_bridge",
]
