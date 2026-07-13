"""Shared standalone Isaac/PX4 absolute-waypoint navigation scenario.

The navigation scenario is the takeoff scenario plus a config precondition: the
flying vehicles must carry a ``goto`` leg and a ``goto`` -> ``hover`` transition,
the point where PX4 will failsafe if the OFFBOARD setpoint stream stops while
OFFBOARD is still active. The flight itself is driven by the host-agnostic
:class:`~robotics_experiments.runtime.FleetOrchestrator` through the shared
:func:`_px4_takeoff_scenario.run_px4_takeoff_scenario`; this module only adds the
navigation-specific validation (applied per flying vehicle, so a single config may
mix airframes -- e.g. the merged quad + hexacopter multi-drone navigation test)
so the navigation entrypoint stays thin (one config, one scene) over it.
"""

from __future__ import annotations

from typing import Any

from _px4_takeoff_scenario import (
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import SceneTestContext

__all__ = ["run_px4_navigation_scenario"]


def _step_kind(step: Any) -> str:
    return str(step["kind"])


def _step_waypoint(step: Any) -> tuple[float, float, float]:
    return (float(step["x"]), float(step["y"]), float(step["z"]))


def _goto_waypoints(steps: list[Any]) -> tuple[tuple[float, float, float], ...]:
    return tuple(_step_waypoint(step) for step in steps if _step_kind(step) == "goto")


def _has_goto_hover_transition(steps: list[Any]) -> bool:
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
    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name=test_name,
            px4_container_log_name=px4_container_log_name,
            success_label=success_label,
        ),
    )
