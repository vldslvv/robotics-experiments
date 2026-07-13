"""Public contract for the PX4 SITL bridge package.

Code outside this package should import bridge concepts from this package root
instead of depending on the internal module layout. The exports here are the
stable seams used by the runtime, command layer, RL adapters, tests, and Isaac
extension shim.
"""

from robotics_experiments.px4_bridge.asset_diagnostics import (
    RootBodyPoseDiagnosticParams,
    RootBodyPoseDiagnostics,
    compute_root_body_pose_diagnostics,
    root_body_pose_has_mismatch,
    validate_root_body_pose,
)
from robotics_experiments.px4_bridge.bridge import (
    Px4BridgeState,
    create_px4_bridge,
    recv_and_apply,
    send_sensors,
    start_px4_bridge,
    stop_px4_bridge,
    tick_px4_bridge,
)
from robotics_experiments.px4_bridge.contract import (
    ActuatorCommands,
    BridgeStatus,
    LockstepSettings,
)
from robotics_experiments.px4_bridge.frames import (
    DEFAULT_HOME_ALT_M,
    DEFAULT_HOME_LAT_DEG,
    DEFAULT_HOME_LON_DEG,
    DEFAULT_MAG_NED_GAUSS,
    GRAVITY_M_S2,
    body_velocity_flu_from_state,
    normalize_quat_wxyz,
    quat_conjugate_wxyz,
    quat_multiply_wxyz,
    quat_wxyz_to_xyzw,
    rotate_vector_wxyz,
    synth_barometer,
    synth_gps_from_enu,
    synth_magnetometer_body_flu,
    tilt_angle_deg,
    world_enu_to_body_flu,
)
from robotics_experiments.px4_bridge.isaac_articulation import IsaacArticulationIO
from robotics_experiments.px4_bridge.logging import vehicle_log_context
from robotics_experiments.px4_bridge.mavlink import Px4MavlinkPort
from robotics_experiments.px4_bridge.rotor_discovery import (
    DiscoveredRotorMapping,
    discover_px4_rotor_mapping,
)
from robotics_experiments.px4_bridge.sim_clock import (
    SimClockState,
    advance_sim_clock,
    bridge_time_us,
    create_sim_clock,
    sim_clock_is_settling,
)
from robotics_experiments.px4_bridge.thrust_model import (
    PegasusQuadraticThrustModel,
    RotorDynamics,
    RotorThrustModel,
)
from robotics_experiments.px4_bridge.vehicle_state import VehicleState

__all__ = [
    "ActuatorCommands",
    "BridgeStatus",
    "DEFAULT_HOME_ALT_M",
    "DEFAULT_HOME_LAT_DEG",
    "DEFAULT_HOME_LON_DEG",
    "DEFAULT_MAG_NED_GAUSS",
    "DiscoveredRotorMapping",
    "GRAVITY_M_S2",
    "IsaacArticulationIO",
    "LockstepSettings",
    "PegasusQuadraticThrustModel",
    "Px4BridgeState",
    "Px4MavlinkPort",
    "RootBodyPoseDiagnosticParams",
    "RootBodyPoseDiagnostics",
    "RotorDynamics",
    "RotorThrustModel",
    "SimClockState",
    "VehicleState",
    "advance_sim_clock",
    "body_velocity_flu_from_state",
    "bridge_time_us",
    "compute_root_body_pose_diagnostics",
    "create_px4_bridge",
    "create_sim_clock",
    "discover_px4_rotor_mapping",
    "normalize_quat_wxyz",
    "quat_conjugate_wxyz",
    "quat_multiply_wxyz",
    "quat_wxyz_to_xyzw",
    "recv_and_apply",
    "root_body_pose_has_mismatch",
    "rotate_vector_wxyz",
    "send_sensors",
    "sim_clock_is_settling",
    "start_px4_bridge",
    "stop_px4_bridge",
    "synth_barometer",
    "synth_gps_from_enu",
    "synth_magnetometer_body_flu",
    "tilt_angle_deg",
    "tick_px4_bridge",
    "validate_root_body_pose",
    "vehicle_log_context",
    "world_enu_to_body_flu",
]
