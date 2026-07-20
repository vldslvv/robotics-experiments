"""Raw flight-benchmark capture and its simulator-independent phase machine."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

__all__: list[str] = []


class FlightPhase(str, Enum):
    """Measured phases in causal flight order."""

    CONNECTED_DISARMED = "connected_disarmed"
    ARMED_TAKEOFF = "armed_takeoff"
    AIRBORNE = "airborne"


@dataclass(frozen=True)
class FlightPhaseObservation:
    """One fleet snapshot consumed by the pure phase controller."""

    all_connected: bool
    any_armed: bool
    all_takeoffs_complete: bool
    sim_time_s: float


@dataclass(frozen=True)
class FlightPhaseController:
    """Progress from connection through the configured airborne duration."""

    airborne_duration_s: float
    phase: FlightPhase | None = None
    airborne_start_sim_time_s: float | None = None
    complete: bool = False

    def observe(self, *, observation: FlightPhaseObservation) -> FlightPhaseController:
        """Return the successor phase state for one fleet observation."""
        if self.complete:
            return self
        if not math.isfinite(observation.sim_time_s) or observation.sim_time_s < 0.0:
            raise ValueError(f"invalid flight benchmark sim time {observation.sim_time_s}")

        if self.phase is None:
            if observation.all_connected and not observation.any_armed:
                return FlightPhaseController(
                    phase=FlightPhase.CONNECTED_DISARMED,
                    airborne_duration_s=self.airborne_duration_s,
                )
            return self

        if self.phase is FlightPhase.CONNECTED_DISARMED:
            if observation.any_armed:
                return FlightPhaseController(
                    phase=FlightPhase.ARMED_TAKEOFF,
                    airborne_duration_s=self.airborne_duration_s,
                )
            return self

        if self.phase is FlightPhase.ARMED_TAKEOFF:
            if observation.all_takeoffs_complete:
                return FlightPhaseController(
                    phase=FlightPhase.AIRBORNE,
                    airborne_start_sim_time_s=observation.sim_time_s,
                    airborne_duration_s=self.airborne_duration_s,
                )
            return self

        assert self.airborne_start_sim_time_s is not None
        elapsed_s = observation.sim_time_s - self.airborne_start_sim_time_s
        if elapsed_s >= self.airborne_duration_s:
            return FlightPhaseController(
                phase=self.phase,
                airborne_start_sim_time_s=self.airborne_start_sim_time_s,
                complete=True,
                airborne_duration_s=self.airborne_duration_s,
            )
        return self


class FlightBenchmarkRecorder:
    """Collect raw correlated samples without importing production modules."""

    def __init__(
        self,
        *,
        output_path: Path,
        metadata: dict[str, Any],
        environment: dict[str, Any],
    ) -> None:
        self._output_path = output_path
        self._metadata = metadata
        self._environment = environment
        self._phase: FlightPhase | None = None
        self._phase_start_sim_time_s = 0.0
        self._phase_start_wall_ns = 0
        self._last_app_ns = 0
        self._last_render_ns = 0
        self._frame_index = -1
        self._physics_steps_in_frame = 0
        self._sim_elapsed_in_frame_s = 0.0
        self._phase_records: dict[str, dict[str, Any]] = {}
        self._component_samples: list[dict[str, Any]] = []
        self._resource_samples: list[dict[str, Any]] = []
        self._status = "running"
        self._reason: str | None = None

    @property
    def phase(self) -> FlightPhase | None:
        """Return the active measured phase."""
        return self._phase

    def transition(self, *, phase: FlightPhase, sim_time_s: float) -> None:
        """Close the prior phase and begin ``phase`` at the supplied sim time."""
        if phase is self._phase:
            return
        now_ns = time.perf_counter_ns()
        if self._phase is not None:
            self._close_frame()
            self._close_phase(sim_time_s=sim_time_s, wall_ns=now_ns)
        self._phase = phase
        self._phase_start_sim_time_s = float(sim_time_s)
        self._phase_start_wall_ns = now_ns
        self._phase_records[phase.value] = {
            "start_sim_time_s": float(sim_time_s),
            "end_sim_time_s": None,
            "wall_elapsed_s": None,
            "app_frames": [],
            "render_intervals_ms": [],
        }

    def begin_app_frame(self) -> None:
        """Open a correlated app-frame record and close the preceding frame."""
        now_ns = time.perf_counter_ns()
        if self._phase is None:
            self._last_app_ns = now_ns
            return
        self._close_frame()
        self._frame_index += 1
        interval_ms = 0.0
        if self._last_app_ns:
            interval_ms = (now_ns - self._last_app_ns) / 1_000_000.0
        self._last_app_ns = now_ns
        self._physics_steps_in_frame = 0
        self._sim_elapsed_in_frame_s = 0.0
        self._phase_records[self._phase.value]["app_frames"].append(
            {
                "frame": self._frame_index,
                "interval_ms": interval_ms,
                "physics_steps": 0,
                "sim_elapsed_s": 0.0,
            }
        )

    def record_render_update(self) -> None:
        """Record an interval from the full-GUI application runloop."""
        now_ns = time.perf_counter_ns()
        if self._phase is not None and self._last_render_ns:
            self._phase_records[self._phase.value]["render_intervals_ms"].append(
                (now_ns - self._last_render_ns) / 1_000_000.0
            )
        self._last_render_ns = now_ns

    def record_physics_step(self, *, dt: float) -> None:
        """Associate one physics step and its simulated duration with this frame."""
        if self._phase is None:
            return
        self._physics_steps_in_frame += 1
        self._sim_elapsed_in_frame_s += float(dt)

    def record_duration(self, *, metric: str, duration_ms: float, entity: str) -> None:
        """Append one wrapped-component duration to the active frame and phase."""
        if self._phase is None:
            return
        value = float(duration_ms)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid {metric} duration {duration_ms}")
        self._component_samples.append(
            {
                "phase": self._phase.value,
                "frame": self._frame_index,
                "metric": metric,
                "entity": entity,
                "duration_ms": value,
            }
        )

    def record_resources(self, *, cpu_percent: float, rss_mb: float) -> None:
        """Append process CPU and resident-memory samples for the active phase."""
        if self._phase is None:
            return
        values = (float(cpu_percent), float(rss_mb))
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError(f"invalid process resource sample {values}")
        self._resource_samples.append(
            {
                "phase": self._phase.value,
                "frame": self._frame_index,
                "cpu_percent": values[0],
                "rss_mb": values[1],
            }
        )

    def finish(self, *, sim_time_s: float) -> None:
        """Close capture successfully and atomically write the raw JSON."""
        if self._phase is None:
            raise RuntimeError("flight benchmark finished before its first measured phase")
        self._close_frame()
        self._close_phase(sim_time_s=sim_time_s, wall_ns=time.perf_counter_ns())
        self._status = "complete"
        self._write()

    def fail(self, *, reason: str, sim_time_s: float = 0.0) -> None:
        """Persist partial samples and an explicit failure reason."""
        self._close_frame()
        if self._phase is not None:
            self._close_phase(sim_time_s=sim_time_s, wall_ns=time.perf_counter_ns())
        self._status = "failed"
        self._reason = reason
        self._write()

    def _close_frame(self) -> None:
        if self._phase is None:
            return
        frames = self._phase_records[self._phase.value]["app_frames"]
        if not frames:
            return
        frame = frames[-1]
        if frame["frame"] != self._frame_index:
            return
        frame["physics_steps"] = self._physics_steps_in_frame
        frame["sim_elapsed_s"] = self._sim_elapsed_in_frame_s

    def _close_phase(self, *, sim_time_s: float, wall_ns: int) -> None:
        assert self._phase is not None
        record = self._phase_records[self._phase.value]
        record["end_sim_time_s"] = float(sim_time_s)
        record["wall_elapsed_s"] = (wall_ns - self._phase_start_wall_ns) / 1_000_000_000.0

    def _write(self) -> None:
        payload = {
            "schema_version": 1,
            "status": self._status,
            "reason": self._reason,
            "metadata": self._metadata,
            "environment": self._environment,
            "phases": self._phase_records,
            "component_samples": self._component_samples,
            "resource_samples": self._resource_samples,
        }
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._output_path.with_suffix(self._output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(self._output_path)
