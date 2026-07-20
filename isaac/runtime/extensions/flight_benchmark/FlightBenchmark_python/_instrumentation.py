"""Benchmark-only wrappers around the production flight call graph."""

from __future__ import annotations

import time
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable

from FlightBenchmark_python._capture import FlightBenchmarkRecorder

__all__: list[str] = []

_ENTITY: ContextVar[str] = ContextVar("flight_benchmark_entity", default="fleet")
_SCOPE: ContextVar[str] = ContextVar("flight_benchmark_scope", default="")


@dataclass(frozen=True)
class _Patch:
    owner: Any
    name: str
    original: Callable[..., Any]


class InstrumentationSession:
    """Install and restore timing decorators without changing production code."""

    def __init__(self, *, recorder: FlightBenchmarkRecorder) -> None:
        self._recorder = recorder
        self._patches: list[_Patch] = []

    def start(self) -> None:
        """Wrap the selected production callables before the timeline starts."""
        if self._patches:
            raise RuntimeError("flight benchmark instrumentation is already active")

        import ExperimentRuntime_python.extension as runtime_extension

        from robotics_experiments.px4_bridge import (
            IsaacArticulationIO,
            PegasusQuadraticThrustModel,
            Px4MavlinkPort,
        )
        from robotics_experiments.runtime import BridgeHost

        self._wrap_host(
            owner=BridgeHost,
            name="send_phase",
            metric="bridge_send",
            scope="bridge_send",
        )
        self._wrap_host(
            owner=BridgeHost,
            name="recv_phase",
            metric="bridge_receive",
            scope="bridge_receive",
        )
        self._wrap_in_scope(
            owner=Px4MavlinkPort,
            name="send",
            metric="mavlink_sensor",
            scope="bridge_send",
        )
        self._wrap_in_scope(
            owner=Px4MavlinkPort,
            name="poll",
            metric="mavlink_actuator",
            scope="bridge_receive",
        )
        self._wrap_in_scope(
            owner=IsaacArticulationIO,
            name="read_state",
            metric="articulation_state_read",
            scope="bridge_send",
        )
        self._wrap_in_scope(
            owner=PegasusQuadraticThrustModel,
            name="commands_to_dynamics",
            metric="thrust_model",
            scope="bridge_receive",
        )
        self._wrap_in_scope(
            owner=IsaacArticulationIO,
            name="update_rotor_spin",
            metric="rotor_visual_update",
            scope="bridge_receive",
        )
        self._wrap_in_scope(
            owner=IsaacArticulationIO,
            name="apply_dynamics",
            metric="dynamics_apply",
            scope="bridge_receive",
        )
        self._wrap_callable(
            owner=runtime_extension,
            name="tick_drivers",
            metric="routine_driver",
        )

    def stop(self) -> None:
        """Restore every original callable in reverse installation order."""
        for patch in reversed(self._patches):
            setattr(patch.owner, patch.name, patch.original)
        self._patches.clear()

    def _install(self, *, owner: Any, name: str, wrapped: Callable[..., Any]) -> None:
        original = getattr(owner, name)
        self._patches.append(_Patch(owner=owner, name=name, original=original))
        setattr(owner, name, wrapped)

    def _wrap_host(self, *, owner: Any, name: str, metric: str, scope: str) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(host: Any, *args: Any, **kwargs: Any) -> Any:
            entity = f"vehicle-{host.params.vehicle_id}"
            entity_token = _ENTITY.set(entity)
            scope_token = _SCOPE.set(scope)
            start_ns = time.perf_counter_ns()
            try:
                return original(host, *args, **kwargs)
            finally:
                self._recorder.record_duration(
                    metric=metric,
                    entity=entity,
                    duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
                )
                _SCOPE.reset(scope_token)
                _ENTITY.reset(entity_token)

        self._install(owner=owner, name=name, wrapped=wrapped)

    def _wrap_in_scope(self, *, owner: Any, name: str, metric: str, scope: str) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _SCOPE.get() != scope:
                return original(*args, **kwargs)
            start_ns = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self._recorder.record_duration(
                    metric=metric,
                    entity=_ENTITY.get(),
                    duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
                )

        self._install(owner=owner, name=name, wrapped=wrapped)

    def _wrap_callable(self, *, owner: Any, name: str, metric: str) -> None:
        original = getattr(owner, name)

        @wraps(original)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            entity_token = _ENTITY.set("fleet")
            start_ns = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                self._recorder.record_duration(
                    metric=metric,
                    entity="fleet",
                    duration_ms=(time.perf_counter_ns() - start_ns) / 1_000_000.0,
                )
                _ENTITY.reset(entity_token)

        self._install(owner=owner, name=name, wrapped=wrapped)
