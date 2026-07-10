"""Compatibility facade for the pure routine contract.

The routine model, stepper, predicates, gates, and descriptor builder live in
private responsibility-focused modules. The commanding package root remains the
only supported public import surface.
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

__all__ = [
    "DecisionKind",
    "EnvelopeGates",
    "FailureKind",
    "PolicyStepFn",
    "RoutineDecision",
    "RoutineState",
    "RoutineStep",
    "StreamSetpoint",
    "VehicleRoutine",
    "ack_accepted",
    "armed",
    "build_vehicle_routine",
    "climbed_at_least",
    "cross_track_drift_below",
    "cross_track_to_segment_below",
    "decide",
    "disarmed",
    "heading_aligned",
    "hovering_stable",
    "moved_forward_at_least",
    "reached_world_position",
    "tilt_below",
]
