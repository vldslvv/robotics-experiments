"""Public command / routine API.

Code outside this package should import only from this root, which exposes the
command payloads, pure routine model, driver seam, and concrete PX4 GCS port
that make up the stable commanding contract.
"""

from robotics_experiments.commanding._routine_builder import build_vehicle_routine
from robotics_experiments.commanding._routine_model import (
    DecisionKind,
    FailureKind,
    PolicyStepFn,
    RoutineDecision,
    RoutineState,
    RoutineStep,
    StreamSetpoint,
    VehicleRoutine,
)
from robotics_experiments.commanding._routine_predicates import (
    EnvelopeGates,
    ack_accepted,
    armed,
    climbed_at_least,
    cross_track_drift_below,
    cross_track_to_segment_below,
    disarmed,
    heading_aligned,
    hovering_stable,
    moved_forward_at_least,
    reached_world_position,
    tilt_below,
)
from robotics_experiments.commanding._routine_stepper import decide
from robotics_experiments.commanding.commands import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    MAV_CMD_DO_REPOSITION,
    MAV_CMD_DO_SET_MODE,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF,
    MAV_DO_REPOSITION_FLAGS_CHANGE_MODE,
    AttitudeRateSetpoint,
    MavlinkCommand,
    PositionSetpoint,
    VelocitySetpoint,
    arm,
    land,
    reposition,
    set_offboard_mode,
    takeoff,
)
from robotics_experiments.commanding.driver import (
    DriverEnvironment,
    DriverParams,
    DriverPhase,
    RoutineDriverState,
    create_routine_driver,
    routine_driver_is_done,
    tick_routine_driver,
)
from robotics_experiments.commanding.observation import (
    MAV_RESULT_ACCEPTED,
    MAV_RESULT_TEMPORARILY_REJECTED,
    CommandAck,
    RoutineObservation,
)
from robotics_experiments.commanding.port import (
    CommandPortStatus,
    MavlinkCommandPort,
    read_command_port_status,
)

__all__ = [
    "AttitudeRateSetpoint",
    "CommandAck",
    "CommandPortStatus",
    "DecisionKind",
    "DriverEnvironment",
    "DriverParams",
    "DriverPhase",
    "EnvelopeGates",
    "FailureKind",
    "MAV_CMD_COMPONENT_ARM_DISARM",
    "MAV_CMD_DO_REPOSITION",
    "MAV_CMD_DO_SET_MODE",
    "MAV_CMD_NAV_LAND",
    "MAV_CMD_NAV_TAKEOFF",
    "MAV_DO_REPOSITION_FLAGS_CHANGE_MODE",
    "MAV_RESULT_ACCEPTED",
    "MAV_RESULT_TEMPORARILY_REJECTED",
    "MavlinkCommandPort",
    "MavlinkCommand",
    "PolicyStepFn",
    "PositionSetpoint",
    "RoutineDecision",
    "RoutineDriverState",
    "RoutineObservation",
    "RoutineState",
    "RoutineStep",
    "StreamSetpoint",
    "VelocitySetpoint",
    "VehicleRoutine",
    "ack_accepted",
    "arm",
    "armed",
    "build_vehicle_routine",
    "climbed_at_least",
    "create_routine_driver",
    "cross_track_drift_below",
    "cross_track_to_segment_below",
    "decide",
    "disarmed",
    "heading_aligned",
    "hovering_stable",
    "land",
    "moved_forward_at_least",
    "reached_world_position",
    "reposition",
    "read_command_port_status",
    "routine_driver_is_done",
    "set_offboard_mode",
    "takeoff",
    "tilt_below",
    "tick_routine_driver",
]
