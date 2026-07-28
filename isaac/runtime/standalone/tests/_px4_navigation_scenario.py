"""Shared standalone Isaac/PX4 absolute-waypoint navigation scenario.

The navigation scenario is the takeoff scenario plus config preconditions: the
flying vehicles must carry a ``goto`` -> ``hover`` transition and exactly one
speed-authored guided leg followed by hover. The speedless transition protects
the point where PX4 will failsafe if its OFFBOARD setpoint stream stops; the
guided transition supplies wall-free speed and braking diagnostics. The flight
itself is driven by the host-agnostic
:class:`~robotics_experiments.runtime.FleetState` lifecycle through the shared
:func:`_px4_takeoff_scenario.run_px4_takeoff_scenario`; this module only adds the
navigation-specific validation (applied per flying vehicle, so a single config may
mix airframes -- e.g. the merged quad + hexacopter multi-drone navigation test)
so the navigation entrypoint stays thin (one config, one scene) over it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from _px4_takeoff_scenario import (
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import SceneTestContext

from robotics_experiments.config import RoutineStepConfig

__all__ = ["run_px4_navigation_scenario"]


def _vector3(values: np.ndarray, /) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class _SpeedSample:
    sim_time_us: int
    cursor: int
    position_enu: tuple[float, float, float]
    linear_velocity_enu: tuple[float, float, float]


@dataclass
class _SpeedLegTrace:
    label: str
    index: int
    hover_index: int
    target_enu: tuple[float, float, float]
    requested_speed_m_s: float
    position_gate_m: float
    cross_track_gate_m: float
    baseline_enu: tuple[float, float, float] | None = None
    completion_position_enu: tuple[float, float, float] | None = None
    samples: list[_SpeedSample] = field(default_factory=list)
    finalized: bool = False


class _SpeedDiagnosticCollector:
    """Collect wall-free guided-goto speed and braking diagnostics."""

    def __init__(self, context: SceneTestContext, test_name: str, /):
        traces: dict[str, _SpeedLegTrace] = {}
        for label, vehicle in context.config.vehicles.items():
            steps = list(vehicle.routine.steps)
            guided = [
                (index, step)
                for index, step in enumerate(steps)
                if _step_kind(step) == "goto" and "ground_speed_m_s" in step
            ]
            if len(guided) != 1:
                raise AssertionError(
                    f"{test_name}: vehicle {vehicle.id} must have exactly one "
                    f"speed-authored goto for diagnostics, got {len(guided)}"
                )
            index, step = guided[0]
            hover_index = index + 1
            if hover_index >= len(steps) or _step_kind(steps[hover_index]) != "hover":
                raise AssertionError(
                    f"{test_name}: vehicle {vehicle.id} speed-authored goto "
                    "must be followed by hover for braking diagnostics"
                )
            traces[label] = _SpeedLegTrace(
                label=label,
                index=index,
                hover_index=hover_index,
                target_enu=_step_waypoint(step),
                requested_speed_m_s=float(step["ground_speed_m_s"]),
                position_gate_m=float(
                    step.get("position_tol_m", vehicle.routine.goto_position_tol_m)
                ),
                cross_track_gate_m=float(vehicle.routine.goto_cross_track_m),
            )
        self._traces = traces
        self._test_name = test_name

    def sample(self, fleet, /) -> None:
        from robotics_experiments.runtime import read_bridge_host_status

        for flying in fleet.flying:
            trace = self._traces[flying.spawned.label]
            if trace.finalized:
                continue
            cursor = flying.driver.routine_state.cursor
            if cursor < trace.index:
                continue
            observer = flying.spawned.observer
            host = flying.spawned.host
            assert observer is not None and host is not None
            state = observer.read_state(sim_time_us=read_bridge_host_status(host=host).sim_time_us)
            sample = _SpeedSample(
                sim_time_us=int(state.sim_time_us),
                cursor=cursor,
                position_enu=_vector3(state.position_enu),
                linear_velocity_enu=_vector3(state.linear_velocity_enu),
            )
            if (
                cursor == trace.index
                and trace.baseline_enu is None
                and flying.driver.routine_state.baseline_position_enu is not None
            ):
                trace.baseline_enu = _vector3(flying.driver.routine_state.baseline_position_enu)
            trace.samples.append(sample)
            if cursor > trace.index and trace.completion_position_enu is None:
                trace.completion_position_enu = sample.position_enu
            if cursor > trace.hover_index:
                trace.finalized = True

    def report(self, context: SceneTestContext, _fleet, /) -> None:
        for trace in self._traces.values():
            if (
                trace.baseline_enu is None
                or trace.completion_position_enu is None
                or not trace.finalized
            ):
                raise AssertionError(f"{self._test_name}: incomplete speed trace for {trace.label}")
            goto_samples = [sample for sample in trace.samples if sample.cursor == trace.index]
            if not goto_samples:
                raise AssertionError(f"{self._test_name}: no guided-goto samples for {trace.label}")

            baseline = np.asarray(trace.baseline_enu, dtype=np.float64)
            target = np.asarray(trace.target_enu, dtype=np.float64)
            leg_xy = target[:2] - baseline[:2]
            leg_length_m = float(np.linalg.norm(leg_xy))
            unit_xy = leg_xy / leg_length_m

            def horizontal_speed(sample: _SpeedSample, /) -> float:
                return float(np.linalg.norm(sample.linear_velocity_enu[:2]))

            def progress(
                sample: _SpeedSample,
                baseline: np.ndarray = baseline,
                unit_xy: np.ndarray = unit_xy,
                /,
            ) -> float:
                return float(
                    np.dot(
                        np.asarray(sample.position_enu[:2], dtype=np.float64) - baseline[:2],
                        unit_xy,
                    )
                )

            peak_sample = max(goto_samples, key=horizontal_speed)
            peak_speed_m_s = horizontal_speed(peak_sample)
            average_speed_m_s = float(
                np.mean([horizontal_speed(sample) for sample in goto_samples])
            )
            at_requested_speed = [
                sample
                for sample in goto_samples
                if horizontal_speed(sample) >= trace.requested_speed_m_s
            ]
            braking_start = at_requested_speed[-1] if at_requested_speed else peak_sample
            braking_samples = [
                sample
                for sample in trace.samples
                if sample.sim_time_us >= braking_start.sim_time_us
            ]
            braking_distance_m = max(
                0.0,
                max(progress(sample) for sample in braking_samples) - progress(braking_start),
            )
            all_progress = [progress(sample) for sample in trace.samples]
            overshoot_m = max(0.0, max(all_progress) - leg_length_m)
            completion_error_m = float(
                np.linalg.norm(np.asarray(trace.completion_position_enu, dtype=np.float64) - target)
            )
            final_error_m = float(
                np.linalg.norm(
                    np.asarray(trace.samples[-1].position_enu, dtype=np.float64) - target
                )
            )
            cross_track_m = max(
                abs(
                    float(
                        unit_xy[0] * (sample.position_enu[1] - baseline[1])
                        - unit_xy[1] * (sample.position_enu[0] - baseline[0])
                    )
                )
                for sample in goto_samples
            )
            if completion_error_m > trace.position_gate_m:
                raise AssertionError(
                    f"{self._test_name}: {trace.label} completed guided goto outside "
                    f"position gate: error={completion_error_m:.6f}m "
                    f"gate={trace.position_gate_m:.6f}m"
                )
            if cross_track_m > trace.cross_track_gate_m:
                raise AssertionError(
                    f"{self._test_name}: {trace.label} breached cross-track gate: "
                    f"observed={cross_track_m:.6f}m gate={trace.cross_track_gate_m:.6f}m"
                )
            payload = {
                "average_horizontal_speed_m_s": average_speed_m_s,
                "braking_distance_m": braking_distance_m,
                "completion_error_m": completion_error_m,
                "cross_track_gate_m": trace.cross_track_gate_m,
                "final_error_m": final_error_m,
                "index": trace.index,
                "label": trace.label,
                "max_cross_track_m": cross_track_m,
                "overshoot_m": overshoot_m,
                "peak_horizontal_speed_m_s": peak_speed_m_s,
                "position_gate_m": trace.position_gate_m,
                "requested_ground_speed_m_s": trace.requested_speed_m_s,
            }
            context.report(
                f"{self._test_name}: GOTO_SPEED_DIAGNOSTIC "
                f"{json.dumps(payload, separators=(',', ':'), sort_keys=True)}"
            )


def _step_kind(step: RoutineStepConfig) -> str:
    return str(step["kind"])


def _step_waypoint(step: RoutineStepConfig) -> tuple[float, float, float]:
    return (float(step["x"]), float(step["y"]), float(step["z"]))


def _goto_waypoints(steps: list[RoutineStepConfig]) -> tuple[tuple[float, float, float], ...]:
    return tuple(_step_waypoint(step) for step in steps if _step_kind(step) == "goto")


def _has_goto_hover_transition(steps: list[RoutineStepConfig]) -> bool:
    kinds = [_step_kind(step) for step in steps]
    return any(
        current == "goto" and following == "hover"
        for current, following in zip(kinds, kinds[1:], strict=False)
    )


def _assert_navigation_routine_config(context: SceneTestContext, test_name: str) -> None:
    env = context.config
    flying = [vehicle for vehicle in env.vehicles.values() if vehicle.routine.steps]
    if not flying:
        raise AssertionError(f"{test_name}: active Hydra config has no flying vehicles")

    for vehicle in flying:
        steps = list(vehicle.routine.steps)
        kinds = tuple(_step_kind(step) for step in steps)
        waypoints = _goto_waypoints(steps)
        if not waypoints:
            raise AssertionError(
                f"{test_name}: vehicle {vehicle.id} has no configured goto steps: {kinds!r}"
            )
        if not _has_goto_hover_transition(steps):
            raise AssertionError(
                f"{test_name}: vehicle {vehicle.id} has no goto -> hover transition: {kinds!r}"
            )

        context.report(
            f"{test_name}: validated vehicle {vehicle.id} navigation routine "
            f"kinds={kinds!r} goto_waypoints={waypoints!r}"
        )


def run_px4_navigation_scenario(
    context: SceneTestContext,
    test_name: str,
    px4_container_log_name: str,
    success_label: str,
) -> None:
    """Validate the navigation routine config, then drive it to completion."""
    _assert_navigation_routine_config(context, test_name)
    speed_diagnostics = _SpeedDiagnosticCollector(context, test_name)
    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name=test_name,
            px4_container_log_name=px4_container_log_name,
            routine_tick_hook=speed_diagnostics.sample,
            post_routine_hook=speed_diagnostics.report,
            success_label=success_label,
        ),
    )
