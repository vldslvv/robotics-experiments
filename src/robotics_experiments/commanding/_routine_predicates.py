import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from robotics_experiments.commanding._routine_model import (
    _DEFAULT_MIN_HEADING_DIST_M,
    _Invariant,
    _StepPredicate,
)
from robotics_experiments.commanding.observation import RoutineObservation
from robotics_experiments.px4_bridge import rotate_vector_wxyz, tilt_angle_deg

__all__ = [
    "EnvelopeGates",
    "ack_accepted",
    "armed",
    "climbed_at_least",
    "cross_track_drift_below",
    "cross_track_to_segment_below",
    "disarmed",
    "heading_aligned",
    "hovering_stable",
    "moved_forward_at_least",
    "reached_world_position",
    "tilt_below",
]


def _always(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
    """Completion predicate that holds every tick: the step is a pure
    ``hold_s`` wait whose real gate is its continuous invariants (``policy``)."""
    return True


def armed() -> _StepPredicate:
    """Predicate: PX4 currently commands nonzero rotor references (is armed)."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        return bool(observation.bridge_status.is_armed)

    return predicate


def ack_accepted(command_id: int) -> _StepPredicate:
    """Predicate: the last ack this tick accepted ``command_id``."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        ack = observation.last_ack
        return ack is not None and ack.command_id == command_id and ack.accepted

    return predicate


def climbed_at_least(min_climb_m: float) -> _StepPredicate:
    """Predicate: the vehicle climbed at least ``min_climb_m`` above the baseline.

    The baseline is the takeoff step's captured start position (its ground
    altitude), so this is a real airborne check, not just a command ack.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return False
        altitude_m = float(observation.vehicle_state.position_enu[2])
        return altitude_m - float(baseline[2]) >= min_climb_m

    return predicate


def hovering_stable(
    *, max_abs_vertical_speed_m_s: float, max_abs_horizontal_speed_m_s: float
) -> _StepPredicate:
    """Predicate: the vehicle's translational speeds are within hover bounds.

    Combined with a step ``hold_s`` this expresses "held a stable hover for N
    seconds". Reached only after the takeoff step has already verified the climb,
    so a grounded (also slow) vehicle never gets here.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        velocity = np.asarray(observation.vehicle_state.linear_velocity_enu, dtype=np.float64)
        vertical = abs(float(velocity[2]))
        horizontal = float(np.hypot(float(velocity[0]), float(velocity[1])))
        return vertical <= max_abs_vertical_speed_m_s and horizontal <= max_abs_horizontal_speed_m_s

    return predicate


def moved_forward_at_least(distance_m: float) -> _StepPredicate:
    """Predicate: the vehicle moved at least ``distance_m`` along world +X/east."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return False
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        return float(position[0] - baseline[0]) >= distance_m

    return predicate


def disarmed() -> _StepPredicate:
    """Predicate: PX4 no longer commands armed rotor references."""

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        return not bool(observation.bridge_status.is_armed)

    return predicate


def reached_world_position(x_m: float, y_m: float, z_m: float, tol_m: float) -> _StepPredicate:
    """Predicate: the vehicle is within ``tol_m`` (3D) of absolute world ENU ``(x, y, z)``.

    The first *absolute* world-frame completion predicate: it compares the
    observed world position directly to the target, not a baseline-relative
    displacement.
    """
    target = np.array([x_m, y_m, z_m], dtype=np.float64)

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        return float(np.linalg.norm(position - target)) <= tol_m

    return predicate


def heading_aligned(
    x_m: float,
    y_m: float,
    tol_deg: float,
    min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M,
) -> _StepPredicate:
    """Predicate: the body faces the active-step-baseline-to-target heading.

    Trivially satisfied for a near-vertical leg (horizontal
    active-step-baseline-to-target distance below ``min_heading_dist_m``),
    where heading is undefined.
    """

    def predicate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> bool:
        if baseline is None:
            return True
        de = x_m - float(baseline[0])
        dn = y_m - float(baseline[1])
        if math.hypot(de, dn) < min_heading_dist_m:
            return True
        target_heading = math.atan2(dn, de)  # ENU, CCW from +east
        forward_enu = rotate_vector_wxyz(
            observation.vehicle_state.orientation_flu_enu_wxyz,
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        observed = math.atan2(float(forward_enu[1]), float(forward_enu[0]))
        diff = abs((observed - target_heading + math.pi) % (2.0 * math.pi) - math.pi)
        return math.degrees(diff) <= tol_deg

    return predicate


def tilt_below(max_tilt_deg: float) -> _Invariant:
    """Gate: body tilt from upright stays at or below ``max_tilt_deg``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        tilt = float(tilt_angle_deg(observation.vehicle_state.orientation_flu_enu_wxyz))
        if tilt > max_tilt_deg:
            return f"tilt {tilt:.1f}deg exceeds limit {max_tilt_deg:.1f}deg"
        return None

    return gate


def _lateral_drift_below(max_lateral_drift_m: float) -> _Invariant:
    """Gate: horizontal distance from the baseline stays within ``max_lateral_drift_m``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        if baseline is None:
            return None
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        drift = float(np.hypot(float(position[0] - baseline[0]), float(position[1] - baseline[1])))
        if drift > max_lateral_drift_m:
            return f"lateral drift {drift:.2f}m exceeds limit {max_lateral_drift_m:.2f}m"
        return None

    return gate


def cross_track_drift_below(max_cross_track_m: float) -> _Invariant:
    """Gate: world Y/north drift from the baseline stays within ``max_cross_track_m``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        if baseline is None:
            return None
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        drift = abs(float(position[1] - baseline[1]))
        if drift > max_cross_track_m:
            return f"cross-track drift {drift:.2f}m exceeds limit {max_cross_track_m:.2f}m"
        return None

    return gate


def cross_track_to_segment_below(target_x_m: float, target_y_m: float, max_m: float) -> _Invariant:
    """Gate: horizontal distance from the active step's path stays within ``max_m``.

    The direction-agnostic generalization of :func:`cross_track_drift_below`: it
    measures perpendicular deviation from the straight planned path (baseline to
    the absolute target), so along-track progress in any heading is free while a
    sideways excursion off the path trips the envelope.
    """
    target = np.array([target_x_m, target_y_m], dtype=np.float64)

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        if baseline is None:
            return None
        position = np.asarray(observation.vehicle_state.position_enu, dtype=np.float64)
        a = np.array([float(baseline[0]), float(baseline[1])], dtype=np.float64)
        p = position[:2]
        ab = target - a
        ab_len = float(np.linalg.norm(ab))
        if ab_len < 1e-6:
            cross = float(np.linalg.norm(p - a))
        else:
            offset = p - a
            cross = abs(float(ab[0] * offset[1] - ab[1] * offset[0])) / ab_len
        if cross > max_m:
            return f"cross-track {cross:.2f}m exceeds limit {max_m:.2f}m"
        return None

    return gate


def _vertical_speed_within(max_up_m_s: float, max_down_m_s: float) -> _Invariant:
    """Gate: vertical speed stays within (``max_down_m_s``, ``max_up_m_s``).

    ``max_up_m_s`` is the positive upward limit; ``max_down_m_s`` is the negative
    downward limit (e.g. ``-4.0``).
    """

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        vertical = float(observation.vehicle_state.linear_velocity_enu[2])
        if vertical > max_up_m_s:
            return f"vertical speed {vertical:.2f}m/s exceeds up limit {max_up_m_s:.2f}m/s"
        if vertical < max_down_m_s:
            return f"vertical speed {vertical:.2f}m/s exceeds down limit {max_down_m_s:.2f}m/s"
        return None

    return gate


def _angular_speed_below(max_angular_speed_rad_s: float) -> _Invariant:
    """Gate: body angular-speed magnitude stays within ``max_angular_speed_rad_s``."""

    def gate(observation: RoutineObservation, baseline: Optional[np.ndarray]) -> Optional[str]:
        angular = np.asarray(observation.vehicle_state.angular_velocity_flu, dtype=np.float64)
        speed = float(np.linalg.norm(angular))
        if speed > max_angular_speed_rad_s:
            return (
                f"angular speed {speed:.2f}rad/s exceeds limit "
                f"{max_angular_speed_rad_s:.2f}rad/s"
            )
        return None

    return gate


# ---- Config-driven routine assembly ----------------------------------------


@dataclass(frozen=True)
class EnvelopeGates:
    """Flight-envelope thresholds wired onto a routine's flight steps.

    The continuous gates (``max_*``) become per-step :class:`_Invariant`s on the
    takeoff and hover steps; the two ``hover_*`` thresholds parameterize the
    hover step's :func:`hovering_stable` completion predicate. Built by a driver
    from the ``routine`` config and handed to :func:`build_vehicle_routine`.
    """

    max_tilt_deg: float
    max_lateral_drift_m: float
    max_vertical_speed_m_s: float
    max_downward_speed_m_s: float
    max_angular_speed_rad_s: float
    hover_max_abs_vertical_speed_m_s: float
    hover_max_abs_horizontal_speed_m_s: float


def _invariants(gates: EnvelopeGates, lateral: _Invariant) -> tuple[_Invariant, ...]:
    """The four continuous flight-envelope gates, parameterized by the lateral gate.

    Tilt, vertical, and angular gates are shared by every flight step; the lateral
    gate differs per step (baseline drift for takeoff/hover, cross-track for
    forward, cross-track-to-path for goto).
    """
    return (
        tilt_below(gates.max_tilt_deg),
        lateral,
        _vertical_speed_within(gates.max_vertical_speed_m_s, gates.max_downward_speed_m_s),
        _angular_speed_below(gates.max_angular_speed_rad_s),
    )
