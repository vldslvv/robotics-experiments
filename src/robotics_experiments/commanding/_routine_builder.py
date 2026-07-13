import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import numpy as np

from robotics_experiments.commanding._routine_model import (
    _DEFAULT_MIN_HEADING_DIST_M,
    PolicyStepFn,
    RoutineStep,
    StreamSetpoint,
    VehicleRoutine,
    _Invariant,
    _StepPredicate,
    _StreamingSpec,
)
from robotics_experiments.commanding._routine_predicates import (
    EnvelopeGates,
    _always,
    _invariants,
    _lateral_drift_below,
    ack_accepted,
    climbed_at_least,
    cross_track_drift_below,
    cross_track_to_segment_below,
    disarmed,
    heading_aligned,
    hovering_stable,
    moved_forward_at_least,
    reached_world_position,
)
from robotics_experiments.commanding.commands import (
    MAV_CMD_COMPONENT_ARM_DISARM,
    arm,
    land,
    reposition,
    set_offboard_mode,
    takeoff,
)
from robotics_experiments.commanding.observation import RoutineObservation
from robotics_experiments.px4_bridge import synth_gps_from_enu

__all__ = ["build_vehicle_routine"]

_EARTH_RADIUS_M = 6_378_137.0


def _offset_lat_lon(
    latitude_deg: float, longitude_deg: float, *, east_m: float, north_m: float
) -> tuple[float, float]:
    lat_rad = math.radians(float(latitude_deg))
    dlat_deg = (float(north_m) / _EARTH_RADIUS_M) * (180.0 / math.pi)
    dlon_deg = (float(east_m) / (_EARTH_RADIUS_M * math.cos(lat_rad))) * (180.0 / math.pi)
    return float(latitude_deg) + dlat_deg, float(longitude_deg) + dlon_deg


def _build_arm_step(*, command_ack_timeout_s: float) -> RoutineStep:
    return RoutineStep(
        command=arm(),
        is_complete=ack_accepted(MAV_CMD_COMPONENT_ARM_DISARM),
        label="arm",
        timeout_s=command_ack_timeout_s,
        retry_on_temporarily_rejected=True,
    )


def _build_takeoff_step(
    *,
    descriptor: Mapping[str, Any],
    index: int,
    target_latitude_deg: float,
    target_longitude_deg: float,
    target_altitude_m: float,
    takeoff_timeout_s: float,
    flight_invariants: tuple[_Invariant, ...],
) -> tuple[RoutineStep, float]:
    if "climb_m" not in descriptor:
        raise ValueError(f"routine step {index} (takeoff) is missing required 'climb_m'")
    if "min_climb_m" not in descriptor:
        raise ValueError(f"routine step {index} (takeoff) is missing required 'min_climb_m'")
    absolute_altitude_m = float(target_altitude_m) + float(descriptor["climb_m"])
    step = RoutineStep(
        command=takeoff(target_latitude_deg, target_longitude_deg, absolute_altitude_m),
        is_complete=climbed_at_least(float(descriptor["min_climb_m"])),
        label="takeoff",
        timeout_s=takeoff_timeout_s,
        retry_on_temporarily_rejected=True,
        invariants=flight_invariants,
    )
    return step, absolute_altitude_m


def _build_hover_step(
    *,
    descriptor: Mapping[str, Any],
    index: int,
    gates: EnvelopeGates,
    hover_settle_s: float,
    flight_invariants: tuple[_Invariant, ...],
) -> RoutineStep:
    if "hold_s" not in descriptor:
        raise ValueError(f"routine step {index} (hover) is missing required 'hold_s'")
    hold_s = float(descriptor["hold_s"])
    return RoutineStep(
        command=None,
        is_complete=hovering_stable(
            max_abs_vertical_speed_m_s=gates.hover_max_abs_vertical_speed_m_s,
            max_abs_horizontal_speed_m_s=gates.hover_max_abs_horizontal_speed_m_s,
        ),
        label="hover",
        timeout_s=float(hover_settle_s) + hold_s,
        hold_s=hold_s,
        invariants=flight_invariants,
    )


def _build_forward_step(
    *,
    descriptor: Mapping[str, Any],
    index: int,
    target_latitude_deg: float,
    target_longitude_deg: float,
    commanded_altitude_m: float,
    forward_invariants: tuple[_Invariant, ...],
) -> tuple[RoutineStep, float, float]:
    if "distance_m" not in descriptor:
        raise ValueError(f"routine step {index} (forward) is missing required 'distance_m'")
    if "timeout_s" not in descriptor:
        raise ValueError(f"routine step {index} (forward) is missing required 'timeout_s'")
    distance_m = float(descriptor["distance_m"])
    latitude_deg, longitude_deg = _offset_lat_lon(
        target_latitude_deg,
        target_longitude_deg,
        east_m=distance_m,
        north_m=0.0,
    )
    step = RoutineStep(
        command=reposition(latitude_deg, longitude_deg, commanded_altitude_m),
        is_complete=moved_forward_at_least(distance_m),
        label="forward",
        timeout_s=float(descriptor["timeout_s"]),
        retry_on_temporarily_rejected=True,
        invariants=forward_invariants,
    )
    return step, latitude_deg, longitude_deg


def _build_goto_step(
    *,
    descriptor: Mapping[str, Any],
    index: int,
    gates: EnvelopeGates,
    home_lat_deg: float,
    home_lon_deg: float,
    home_alt_m: float,
    goto_position_tol_m: float,
    goto_heading_tol_deg: float,
    goto_cross_track_m: float,
    goto_offboard_prime_ticks: int,
    goto_min_heading_dist_m: float,
) -> RoutineStep:
    for field in ("x", "y", "z", "timeout_s"):
        if field not in descriptor:
            raise ValueError(f"routine step {index} (goto) is missing required {field!r}")
    x_m = float(descriptor["x"])
    y_m = float(descriptor["y"])
    z_m = float(descriptor["z"])
    position_tol_m = float(descriptor.get("position_tol_m", goto_position_tol_m))
    heading_tol_deg = float(descriptor.get("heading_tol_deg", goto_heading_tol_deg))
    gps = synth_gps_from_enu(
        np.array([x_m, y_m, z_m], dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        home_lat_deg,
        home_lon_deg,
        home_alt_m,
    )
    spec = _StreamingSpec(
        target_enu=(x_m, y_m, z_m),
        latitude_deg=float(gps["latitude"]),
        longitude_deg=float(gps["longitude"]),
        altitude_m=float(gps["altitude"]),
        prime_ticks=int(goto_offboard_prime_ticks),
        min_heading_dist_m=float(goto_min_heading_dist_m),
    )
    reached = reached_world_position(x_m, y_m, z_m, position_tol_m)
    aligned = heading_aligned(x_m, y_m, heading_tol_deg, goto_min_heading_dist_m)

    def _reached_and_aligned(
        observation: RoutineObservation,
        baseline: Optional[np.ndarray],
        reached: _StepPredicate = reached,
        aligned: _StepPredicate = aligned,
    ) -> bool:
        return reached(observation, baseline) and aligned(observation, baseline)

    return RoutineStep(
        command=set_offboard_mode(),
        is_complete=_reached_and_aligned,
        label=f"goto[{x_m:.1f},{y_m:.1f},{z_m:.1f}]",
        timeout_s=float(descriptor["timeout_s"]),
        invariants=_invariants(gates, cross_track_to_segment_below(x_m, y_m, goto_cross_track_m)),
        streaming=spec,
    )


def _build_policy_step(
    *,
    descriptor: Mapping[str, Any],
    index: int,
    flight_invariants: tuple[_Invariant, ...],
    goto_offboard_prime_ticks: int,
    policy_fn_factory: Optional[Callable[[Mapping[str, Any]], PolicyStepFn]],
) -> RoutineStep:
    for field in ("artifact", "hold_s", "timeout_s"):
        if field not in descriptor:
            raise ValueError(f"routine step {index} (policy) is missing required {field!r}")
    if policy_fn_factory is None:
        raise ValueError(
            f"routine step {index} (policy) requires an injected policy_fn_factory "
            "(the runtime composes it from robotics_experiments.rl)"
        )
    hold_s = float(descriptor["hold_s"])
    target = descriptor.get("target")
    if target is None:
        target_enu = None
    else:
        try:
            target_values = [float(v) for v in target]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"routine step {index} (policy) target must be a 3-item numeric sequence"
            ) from exc
        if len(target_values) != 3:
            raise ValueError(
                f"routine step {index} (policy) target must be a 3-item numeric sequence"
            )
        target_enu = np.asarray(target_values, dtype=np.float64)
    policy_fn = policy_fn_factory(descriptor)

    def _policy_setpoint(
        observation: RoutineObservation,
        baseline: Optional[np.ndarray],
        policy_fn: PolicyStepFn = policy_fn,
        target_enu: Optional[np.ndarray] = target_enu,
    ) -> StreamSetpoint:
        goal = target_enu if target_enu is not None else baseline
        assert goal is not None  # baseline is captured on the entry tick
        return policy_fn(observation.vehicle_state, goal)

    spec = _StreamingSpec(
        target_enu=(math.nan, math.nan, math.nan),
        latitude_deg=math.nan,
        longitude_deg=math.nan,
        altitude_m=math.nan,
        prime_ticks=int(goto_offboard_prime_ticks),
        setpoint_fn=_policy_setpoint,
    )
    return RoutineStep(
        command=set_offboard_mode(),
        is_complete=_always,
        label=(f"policy[{descriptor['name']}]" if "name" in descriptor else "policy"),
        timeout_s=float(descriptor["timeout_s"]),
        hold_s=hold_s,
        invariants=flight_invariants,
        streaming=spec,
    )


def _build_land_step(*, descriptor: Mapping[str, Any], index: int) -> RoutineStep:
    if "timeout_s" not in descriptor:
        raise ValueError(f"routine step {index} (land) is missing required 'timeout_s'")
    return RoutineStep(
        command=land(),
        is_complete=disarmed(),
        label="land",
        timeout_s=float(descriptor["timeout_s"]),
        retry_on_temporarily_rejected=True,
    )


def build_vehicle_routine(
    steps: Sequence[Mapping[str, Any]],
    *,
    target_latitude_deg: float,
    target_longitude_deg: float,
    target_altitude_m: float,
    command_ack_timeout_s: float,
    gates: EnvelopeGates,
    takeoff_timeout_s: float,
    hover_settle_s: float,
    home_lat_deg: float,
    home_lon_deg: float,
    home_alt_m: float,
    goto_position_tol_m: float,
    goto_heading_tol_deg: float,
    goto_cross_track_m: float,
    goto_offboard_prime_ticks: int,
    goto_min_heading_dist_m: float = _DEFAULT_MIN_HEADING_DIST_M,
    policy_fn_factory: Optional[Callable[[Mapping[str, Any]], PolicyStepFn]] = None,
) -> VehicleRoutine:
    """Build a :class:`VehicleRoutine` from authored step descriptors and a target.

    The single pure mapping from the ``routine.steps`` config (a list of
    ``{kind: ...}`` mappings) to the immutable routine the commander executes,
    shared by every driver (the live extension and the standalone scenario).
    ``target_*`` is the per-drone takeoff target: the drone's current reported
    geographic position and altitude, to which a ``takeoff`` step adds its
    ``climb_m``. ``gates`` are the flight-envelope thresholds wired onto the
    takeoff and hover steps so the routine verifies the flight in every workflow.

    Step kinds:
    - ``arm``: completes on the arm ``COMMAND_ACK``; retries on a transient PX4
      ``TEMPORARILY_REJECTED``.
    - ``takeoff`` (requires ``climb_m`` and ``min_climb_m``): commands takeoff to
      ``target_altitude_m + climb_m`` and completes once the vehicle has actually
      climbed ``min_climb_m`` above its baseline; ``timeout_s`` is
      ``takeoff_timeout_s``; flight gates apply.
    - ``hover`` (requires ``hold_s``): verifies a stable hover for ``hold_s``
      seconds; if it follows a ``goto`` while PX4 is still in OFFBOARD, the
      previous setpoint keeps streaming until the hover advances; ``timeout_s``
      is ``hover_settle_s + hold_s``; flight gates apply.
    - ``forward`` (requires ``distance_m`` and ``timeout_s``): repositions
      ``distance_m`` along world +X/east and completes once that displacement is
      observed from the active step baseline; flight gates apply.
    - ``goto`` (requires ``x``, ``y``, ``z`` and ``timeout_s``; optional
      ``position_tol_m`` / ``heading_tol_deg``): streams OFFBOARD position
      setpoints toward the absolute world-ENU waypoint (converted to a geodetic
      target via ``synth_gps_from_enu`` against ``home_*``, frame-consistent with
      the bridge's synthesized GPS), yawing toward the leg direction; completes
      once within ``goto_position_tol_m`` of the waypoint and heading-aligned. A
      cross-track-to-path gate plus tilt / vertical / angular gates apply.
    - ``policy`` (requires ``artifact``, ``hold_s`` and ``timeout_s``; optional
      ``target`` as world-ENU ``[x, y, z]``): streams OFFBOARD setpoints
      produced by a trained policy each tick -- the injected
      ``policy_fn_factory(descriptor)`` loads the artifact and returns the pure
      :data:`PolicyStepFn` (raising ``ValueError`` for a missing or mismatched
      artifact, so a bad config fails at build time). Reuses the ``goto``
      OFFBOARD prime/engage/stream lifecycle. Without an authored ``target``
      the policy holds the position captured when the step began. Completes
      after PX4 accepts OFFBOARD and then ``hold_s`` of continuous in-envelope
      flight (the flight gates are the invariants, so a bad policy fails as
      ``ENVELOPE``). A routine should follow a policy step with an
      explicit-command step (e.g. ``land``): a command-less verifier would keep
      re-streaming the last policy payload.
    - ``land`` (requires ``timeout_s``): commands a landing at the current
      position and completes once PX4 disarms.

    An empty ``steps`` yields an empty routine. Pure: no Isaac, no pymavlink, no
    sockets.

    Raises:
        ValueError: on an unknown ``kind``, a required descriptor field missing,
            or a ``policy`` step without an injected ``policy_fn_factory``
            (invalid configuration must not pass silently).
    """
    flight_invariants = _invariants(gates, _lateral_drift_below(gates.max_lateral_drift_m))
    forward_invariants = _invariants(gates, cross_track_drift_below(gates.max_lateral_drift_m))
    built: list[RoutineStep] = []
    commanded_altitude_m = float(target_altitude_m)
    for index, descriptor in enumerate(steps):
        kind = descriptor.get("kind")
        if kind == "arm":
            built.append(_build_arm_step(command_ack_timeout_s=command_ack_timeout_s))
        elif kind == "takeoff":
            step, commanded_altitude_m = _build_takeoff_step(
                descriptor=descriptor,
                index=index,
                target_latitude_deg=target_latitude_deg,
                target_longitude_deg=target_longitude_deg,
                target_altitude_m=target_altitude_m,
                takeoff_timeout_s=takeoff_timeout_s,
                flight_invariants=flight_invariants,
            )
            built.append(step)
        elif kind == "hover":
            built.append(
                _build_hover_step(
                    descriptor=descriptor,
                    index=index,
                    gates=gates,
                    hover_settle_s=hover_settle_s,
                    flight_invariants=flight_invariants,
                )
            )
        elif kind == "forward":
            step, target_latitude_deg, target_longitude_deg = _build_forward_step(
                descriptor=descriptor,
                index=index,
                target_latitude_deg=target_latitude_deg,
                target_longitude_deg=target_longitude_deg,
                commanded_altitude_m=commanded_altitude_m,
                forward_invariants=forward_invariants,
            )
            built.append(step)
        elif kind == "goto":
            built.append(
                _build_goto_step(
                    descriptor=descriptor,
                    index=index,
                    gates=gates,
                    home_lat_deg=home_lat_deg,
                    home_lon_deg=home_lon_deg,
                    home_alt_m=home_alt_m,
                    goto_position_tol_m=goto_position_tol_m,
                    goto_heading_tol_deg=goto_heading_tol_deg,
                    goto_cross_track_m=goto_cross_track_m,
                    goto_offboard_prime_ticks=goto_offboard_prime_ticks,
                    goto_min_heading_dist_m=goto_min_heading_dist_m,
                )
            )
        elif kind == "policy":
            built.append(
                _build_policy_step(
                    descriptor=descriptor,
                    index=index,
                    flight_invariants=flight_invariants,
                    goto_offboard_prime_ticks=goto_offboard_prime_ticks,
                    policy_fn_factory=policy_fn_factory,
                )
            )
        elif kind == "land":
            built.append(_build_land_step(descriptor=descriptor, index=index))
        else:
            raise ValueError(f"routine step {index} has unknown kind {kind!r}")
    return VehicleRoutine(steps=tuple(built))
