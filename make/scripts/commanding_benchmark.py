#!/usr/bin/env python3
"""Run, compare, report, and accept the isolated commanding benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

from robotics_experiments.commanding import (
    MAV_CMD_DO_SET_MODE,
    CommandAck,
    CommandPortStatus,
    DecisionKind,
    DriverContext,
    DriverParams,
    DriverPhase,
    DriverTick,
    EnvelopeGates,
    FailureKind,
    MavlinkCommandPort,
    PolicyAdapter,
    PolicyResult,
    PositionSetpoint,
    RoutineDecision,
    RoutineDriverState,
    RoutineObservation,
    RoutineState,
    VehicleRoutine,
    VelocitySetpoint,
    build_vehicle_routine,
    decide,
    tick_routine_driver,
)
from robotics_experiments.px4_bridge import BridgeStatus, VehicleState

__all__ = [
    "WorkloadFixture",
    "bootstrap_effect_interval",
    "build_report",
    "build_workload_fixtures",
    "exact_permutation_p_value",
    "holm_adjust",
    "main",
]

_SCHEMA_VERSION = 1
_METHODOLOGY_VERSION = 1
_WORKER_COUNT = 9
_WARMUP_BATCHES = 5
_MEASURED_BATCHES = 12
_TARGET_BATCH_NS = 50_000_000
_BOOTSTRAP_RESAMPLES = 10_000
_FAMILY_ALPHA = 0.05
_PRACTICAL_REGRESSION_PERCENT = 5.0
_WORKLOAD_NAMES = (
    "goto_cached_transmit",
    "policy_cached_transmit",
    "policy_evaluate_transmit",
    "driver_goto_drive",
)
_POLICY = {
    "test": "exact two-sided permutation test of median difference",
    "multiple_testing": "Holm",
    "family_alpha": _FAMILY_ALPHA,
    "practical_regression_percent": _PRACTICAL_REGRESSION_PERCENT,
    "bootstrap_confidence": 0.95,
    "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
}
_GATES = EnvelopeGates(
    max_tilt_deg=35.0,
    max_lateral_drift_m=2.0,
    max_vertical_speed_m_s=8.0,
    max_downward_speed_m_s=-4.0,
    max_angular_speed_rad_s=8.0,
    hover_max_abs_vertical_speed_m_s=0.35,
    hover_max_abs_horizontal_speed_m_s=0.30,
)
_BRIDGE_STATUS = BridgeStatus(
    is_connected=True,
    is_armed=True,
    first_heartbeat_received=True,
    sensor_send_count=1,
    actuator_receive_count=1,
    last_sensor_send_us=0,
    last_actuator_receive_us=0,
    actuator_lag_us=0,
    last_disconnect_reason=None,
)
_COMMAND_STATUS = CommandPortStatus(
    is_handshaked=True,
    failsafe=False,
    failsafe_system_status=4,
    latest_global_position=(47.0, 8.0, 488.0),
)


@dataclass(frozen=True)
class WorkloadFixture:
    """One fixed benchmark snapshot plus its untimed contract validation."""

    name: str
    description: str
    operation: Callable[[], object]
    validate: Callable[[object], None]


class _PrebuiltPolicy:
    def __init__(self, *, result: PolicyResult) -> None:
        self.result = result

    def __call__(self, *, policy_input: object, state: object) -> PolicyResult:
        if state != ():
            raise AssertionError(f"policy benchmark expected immutable () state, got {state!r}")
        return self.result


class _CountingPolicy:
    def __init__(self, *, result: PolicyResult) -> None:
        self.result = result
        self.calls = 0

    def __call__(self, *, policy_input: object, state: object) -> PolicyResult:
        if state != ():
            raise AssertionError(f"policy benchmark expected immutable () state, got {state!r}")
        self.calls += 1
        return self.result


class _InertPort(MavlinkCommandPort):
    """No-I/O command port retaining only the most recent benchmark effect."""

    def __init__(self) -> None:
        self.last_setpoint: object = None

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def pump_heartbeat(self, *, now_s: float, force: bool = False) -> None:
        return None

    def poll_feedback(self) -> Optional[CommandAck]:
        return None

    def read_status(self) -> CommandPortStatus:
        return _COMMAND_STATUS

    def send(self, command: object) -> None:
        raise AssertionError(f"cached accepted goto unexpectedly emitted command {command!r}")

    def send_setpoint(self, setpoint: PositionSetpoint) -> None:
        self.last_setpoint = setpoint

    def send_velocity_setpoint(self, setpoint: VelocitySetpoint) -> None:
        self.last_setpoint = setpoint

    def send_attitude_target(self, setpoint: object) -> None:
        self.last_setpoint = setpoint


class _InertEnvironment:
    def __init__(self, *, observation: RoutineObservation) -> None:
        self.observation = observation

    def is_bridge_connected(self) -> bool:
        return True

    def read_observation(
        self, *, sim_time_us: int, last_ack: Optional[CommandAck]
    ) -> RoutineObservation:
        if sim_time_us != self.observation.sim_time_us or last_ack is not None:
            raise AssertionError("driver benchmark observation changed")
        return self.observation

    def log(self, message: str) -> None:
        return None

    def on_failure(self, kind: FailureKind, message: str) -> None:
        raise AssertionError(f"driver benchmark failed: {kind.name}: {message}")

    def on_step_advanced(self, step_label: str) -> None:
        raise AssertionError(f"driver benchmark unexpectedly advanced {step_label!r}")


def _observation(*, sim_time_us: int) -> RoutineObservation:
    vehicle_state = VehicleState(
        position_enu=np.array([0.0, 0.0, 2.0], dtype=np.float64),
        orientation_flu_enu_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        linear_velocity_enu=np.zeros(3, dtype=np.float64),
        angular_velocity_flu=np.zeros(3, dtype=np.float64),
        sim_time_us=sim_time_us,
    )
    return RoutineObservation(
        vehicle_state=vehicle_state,
        bridge_status=_BRIDGE_STATUS,
        last_ack=None,
        failsafe=False,
        sim_time_us=sim_time_us,
        task_observation=(),
    )


def _routine(
    *,
    kind: str,
    policy_advance: Optional[_PrebuiltPolicy | _CountingPolicy] = None,
    policy_rate_hz: float = 10.0,
) -> VehicleRoutine:
    descriptor: dict[str, object]
    policy_factory: Optional[Callable[..., PolicyAdapter]] = None
    if kind == "goto":
        descriptor = {
            "kind": "goto",
            "x": 3.0,
            "y": 0.0,
            "z": 2.0,
            "timeout_s": 20.0,
        }
    elif kind == "policy":
        descriptor = {
            "kind": "policy",
            "artifact": "benchmark/prebuilt",
            "timeout_s": 20.0,
            "target_enu": [0.0, 0.0, 2.0],
        }
        assert policy_advance is not None

        def make_policy(*, artifact: str, hold_s: Optional[float]) -> PolicyAdapter:
            return PolicyAdapter(
                policy_rate_hz=policy_rate_hz,
                initial_state=(),
                advance=policy_advance,
            )

        policy_factory = make_policy
    else:
        raise ValueError(f"unsupported benchmark routine kind {kind!r}")
    return build_vehicle_routine(
        steps=[descriptor],
        target_latitude_deg=47.0,
        target_longitude_deg=8.0,
        target_altitude_m=488.0,
        command_ack_timeout_s=10.0,
        gates=_GATES,
        takeoff_timeout_s=30.0,
        hover_settle_s=2.0,
        home_lat_deg=47.0,
        home_lon_deg=8.0,
        home_alt_m=488.0,
        goto_position_tol_m=0.5,
        goto_heading_tol_deg=25.0,
        goto_cross_track_m=2.0,
        goto_offboard_prime_ticks=3,
        offboard_setpoint_rate_hz=50.0,
        policy_factory=policy_factory,
    )


def _accepted_goto_snapshot() -> tuple[VehicleRoutine, RoutineState, RoutineObservation]:
    routine = _routine(kind="goto")
    state = RoutineState()
    for sim_time_us in (0, 20_000, 40_000):
        state, decision = decide(routine, state, _observation(sim_time_us=sim_time_us))
        if decision.kind is not DecisionKind.CONTINUE:
            raise AssertionError("goto benchmark preconditioning did not continue")
    accepted = _observation(sim_time_us=60_000)
    accepted = RoutineObservation(
        vehicle_state=accepted.vehicle_state,
        bridge_status=accepted.bridge_status,
        last_ack=CommandAck.from_result(MAV_CMD_DO_SET_MODE, 0),
        failsafe=False,
        sim_time_us=accepted.sim_time_us,
        task_observation=accepted.task_observation,
    )
    state, decision = decide(routine, state, accepted)
    if decision.kind is not DecisionKind.CONTINUE or not state.offboard.accepted:
        raise AssertionError("goto benchmark failed to accept OFFBOARD")
    return routine, state, _observation(sim_time_us=80_000)


def _policy_snapshot(
    *,
    policy_advance: _PrebuiltPolicy | _CountingPolicy,
    sim_time_us: int,
) -> tuple[VehicleRoutine, RoutineState, RoutineObservation]:
    routine = _routine(kind="policy", policy_advance=policy_advance)
    state = RoutineState()
    for tick_us in (0, 20_000, 40_000):
        state, decision = decide(routine, state, _observation(sim_time_us=tick_us))
        if decision.kind is not DecisionKind.CONTINUE:
            raise AssertionError("policy benchmark preconditioning did not continue")
    accepted = _observation(sim_time_us=60_000)
    accepted = RoutineObservation(
        vehicle_state=accepted.vehicle_state,
        bridge_status=accepted.bridge_status,
        last_ack=CommandAck.from_result(MAV_CMD_DO_SET_MODE, 0),
        failsafe=False,
        sim_time_us=accepted.sim_time_us,
        task_observation=accepted.task_observation,
    )
    state, decision = decide(routine, state, accepted)
    if decision.kind is not DecisionKind.CONTINUE or not state.offboard.accepted:
        raise AssertionError("policy benchmark failed to accept OFFBOARD")
    if state.offboard.policy_state != ():
        raise AssertionError("policy benchmark did not retain immutable () state")
    return routine, state, _observation(sim_time_us=sim_time_us)


def _driver_params() -> DriverParams:
    return DriverParams(
        steps=(),
        gates=_GATES,
        command_ack_timeout_s=10.0,
        failure_recovery_timeout_s=30.0,
        bridge_heartbeat_timeout_s=12.0,
        handshake_timeout_s=12.0,
        prearm_settle_s=0.0,
        takeoff_target_timeout_s=30.0,
        takeoff_timeout_s=30.0,
        hover_settle_s=2.0,
        goto_position_tol_m=0.5,
        goto_heading_tol_deg=25.0,
        goto_cross_track_m=2.0,
        goto_offboard_prime_ticks=3,
        offboard_setpoint_rate_hz=50.0,
    )


def build_workload_fixtures() -> tuple[WorkloadFixture, ...]:
    """Build the four fixed public-contract benchmark snapshots."""
    goto_routine, goto_state, goto_observation = _accepted_goto_snapshot()
    goto_cached = goto_state.offboard.cached_setpoint

    def goto_operation() -> object:
        return decide(goto_routine, goto_state, goto_observation)

    def validate_goto(value: object) -> None:
        successor, decision = cast(tuple[RoutineState, RoutineDecision], value)
        assert decision.kind is DecisionKind.CONTINUE
        assert decision.command is None
        assert decision.setpoint is goto_cached
        assert successor.offboard.accepted
        assert successor.offboard.transmissions == goto_state.offboard.transmissions + 1

    cached_result = PolicyResult(
        state=(),
        setpoint=VelocitySetpoint(0.1, 0.0, 0.0, 0.0),
    )
    counting_policy = _CountingPolicy(result=cached_result)
    cached_routine, cached_state, cached_observation = _policy_snapshot(
        policy_advance=counting_policy,
        sim_time_us=80_000,
    )
    cached_calls = counting_policy.calls

    def policy_cached_operation() -> object:
        return decide(cached_routine, cached_state, cached_observation)

    def validate_policy_cached(value: object) -> None:
        successor, decision = cast(tuple[RoutineState, RoutineDecision], value)
        assert decision.kind is DecisionKind.CONTINUE
        assert decision.command is None
        assert decision.setpoint is cached_result.setpoint
        assert successor.offboard.policy_state == ()
        assert successor.offboard.transmissions == cached_state.offboard.transmissions + 1
        assert counting_policy.calls == cached_calls

    evaluation_result = PolicyResult(
        state=(),
        setpoint=VelocitySetpoint(0.2, -0.1, 0.0, 0.05),
    )
    evaluation_policy = _PrebuiltPolicy(result=evaluation_result)
    evaluation_routine, evaluation_state, evaluation_observation = _policy_snapshot(
        policy_advance=evaluation_policy,
        sim_time_us=100_000,
    )

    def policy_evaluation_operation() -> object:
        return decide(evaluation_routine, evaluation_state, evaluation_observation)

    def validate_policy_evaluation(value: object) -> None:
        successor, decision = cast(tuple[RoutineState, RoutineDecision], value)
        assert decision.kind is DecisionKind.CONTINUE
        assert decision.command is None
        assert decision.setpoint is evaluation_result.setpoint
        assert successor.offboard.policy_state == ()
        assert successor.offboard.transmissions == evaluation_state.offboard.transmissions + 1
        assert successor.offboard.policy_cadence != evaluation_state.offboard.policy_cadence

    port = _InertPort()
    environment = _InertEnvironment(observation=goto_observation)
    context = DriverContext(
        params=_driver_params(),
        port=port,
        env=environment,
        label="benchmark",
    )
    driver_state = RoutineDriverState(
        phase=DriverPhase.DRIVE,
        deadline_s=100.0,
        settle_target_us=0,
        target=(47.0, 8.0, 488.0),
        routine=goto_routine,
        routine_state=goto_state,
        previous_cursor=0,
        failure=None,
        recovery_stage=None,
        recovery_ack_deadline_s=0.0,
    )
    driver_tick = DriverTick(control_time_us=80_000, monotonic_s=1.0)

    def driver_operation() -> object:
        return tick_routine_driver(
            context=context,
            state=driver_state,
            tick=driver_tick,
        )

    def validate_driver(value: object) -> None:
        successor = cast(RoutineDriverState, value)
        assert successor.phase is DriverPhase.DRIVE
        assert successor.routine_state.offboard.accepted
        assert (
            successor.routine_state.offboard.transmissions == goto_state.offboard.transmissions + 1
        )
        assert port.last_setpoint is goto_cached

    fixtures = (
        WorkloadFixture(
            name="goto_cached_transmit",
            description="cached accepted goto stepper tick, setpoint due",
            operation=goto_operation,
            validate=validate_goto,
        ),
        WorkloadFixture(
            name="policy_cached_transmit",
            description="policy cached tick, transmission due and inference not due",
            operation=policy_cached_operation,
            validate=validate_policy_cached,
        ),
        WorkloadFixture(
            name="policy_evaluate_transmit",
            description="policy evaluation and transmission both due",
            operation=policy_evaluation_operation,
            validate=validate_policy_evaluation,
        ),
        WorkloadFixture(
            name="driver_goto_drive",
            description="public routine-driver DRIVE tick over inert adapters",
            operation=driver_operation,
            validate=validate_driver,
        ),
    )
    for fixture in fixtures:
        fixture.validate(fixture.operation())
    return fixtures


def _read_json(path: Path, /) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percentile(values: Sequence[float], fraction: float, /) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] * (upper - index) + ordered[upper] * (index - lower))


def _samples(*, values: object, label: str) -> list[float]:
    if not isinstance(values, list) or len(values) != _WORKER_COUNT:
        count = len(values) if isinstance(values, list) else 0
        raise ValueError(f"{label} has {count} worker medians; expected {_WORKER_COUNT}")
    samples = [float(value) for value in values]
    if not all(math.isfinite(value) and value > 0.0 for value in samples):
        raise ValueError(f"{label} contains a non-positive or non-finite worker median")
    return samples


def exact_permutation_p_value(*, baseline: Sequence[float], candidate: Sequence[float]) -> float:
    """Return the exact two-sided permutation p-value for a median difference."""
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("permutation samples must be non-empty and equally sized")
    combined = tuple(float(value) for value in (*baseline, *candidate))
    if not all(math.isfinite(value) for value in combined):
        raise ValueError("permutation samples must be finite")
    group_size = len(baseline)
    observed = abs(statistics.median(candidate) - statistics.median(baseline))
    extreme = 0
    total = 0
    indexes = range(len(combined))
    for candidate_indexes in itertools.combinations(indexes, group_size):
        selected = set(candidate_indexes)
        permuted_candidate = [combined[index] for index in candidate_indexes]
        permuted_baseline = [combined[index] for index in indexes if index not in selected]
        difference = abs(
            statistics.median(permuted_candidate) - statistics.median(permuted_baseline)
        )
        extreme += int(difference >= observed - 1e-12)
        total += 1
    return extreme / total


def holm_adjust(*, p_values: Mapping[str, float]) -> dict[str, float]:
    """Apply Holm's step-down family-wise correction."""
    count = len(p_values)
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p_value) in enumerate(ordered):
        if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"{name} has invalid p-value {p_value}")
        running = max(running, (count - rank) * p_value)
        adjusted[name] = min(1.0, running)
    return adjusted


def bootstrap_effect_interval(
    *,
    baseline: Sequence[float],
    candidate: Sequence[float],
    seed: int,
    resamples: int = _BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return a deterministic percentile interval for median percentage change."""
    if not baseline or not candidate or resamples <= 0:
        raise ValueError("bootstrap samples and resample count must be positive")
    if not all(math.isfinite(value) and value > 0.0 for value in (*baseline, *candidate)):
        raise ValueError("bootstrap samples must be positive and finite")
    generator = random.Random(seed)
    effects = []
    for _ in range(resamples):
        baseline_median = statistics.median(
            generator.choice(baseline) for _ in range(len(baseline))
        )
        candidate_median = statistics.median(
            generator.choice(candidate) for _ in range(len(candidate))
        )
        effects.append(100.0 * (candidate_median - baseline_median) / baseline_median)
    return _percentile(effects, 0.025), _percentile(effects, 0.975)


def _validate_candidate(candidate: Mapping[str, Any], /) -> dict[str, list[float]]:
    if candidate.get("repetitions") != _WORKER_COUNT:
        raise ValueError(f"candidate must contain {_WORKER_COUNT} worker repetitions")
    workloads = candidate.get("workloads")
    if not isinstance(workloads, dict) or set(workloads) != set(_WORKLOAD_NAMES):
        raise ValueError(f"candidate workloads must be exactly {list(_WORKLOAD_NAMES)}")
    return {
        name: _samples(values=workloads[name].get("worker_medians_ns"), label=name)
        for name in _WORKLOAD_NAMES
    }


def _validate_baseline(baseline: Mapping[str, Any], /) -> dict[str, list[float]]:
    if baseline.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("accepted baseline schema differs")
    if baseline.get("policy") != _POLICY:
        raise ValueError("accepted baseline regression policy differs")
    return _validate_candidate(baseline)


def build_report(
    *,
    candidate: Mapping[str, Any],
    baseline: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one candidate and compare it with an optional accepted baseline."""
    candidate_samples = _validate_candidate(candidate)
    report: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": candidate["source_commit"],
        "status": "no_baseline",
        "repetitions": _WORKER_COUNT,
        "policy": _POLICY,
        "fingerprint": candidate["fingerprint"],
        "environment": candidate["environment"],
        "raw_workers": candidate["raw_workers"],
        "workloads": {
            name: {
                "description": candidate["workloads"][name]["description"],
                "worker_medians_ns": candidate_samples[name],
                "median_ns": statistics.median(candidate_samples[name]),
                "worker_range_ns": [
                    min(candidate_samples[name]),
                    max(candidate_samples[name]),
                ],
            }
            for name in _WORKLOAD_NAMES
        },
        "comparisons": {},
    }
    if baseline is None:
        return report
    baseline_samples = _validate_baseline(baseline)
    if baseline["fingerprint"] != candidate["fingerprint"]:
        report["status"] = "incomparable"
        report["reason"] = "hardware or methodology fingerprint differs from the baseline"
        return report

    raw_p_values = {
        name: exact_permutation_p_value(
            baseline=baseline_samples[name],
            candidate=candidate_samples[name],
        )
        for name in _WORKLOAD_NAMES
    }
    adjusted_p_values = holm_adjust(p_values=raw_p_values)
    regression = False
    for index, name in enumerate(_WORKLOAD_NAMES):
        accepted_median = statistics.median(baseline_samples[name])
        candidate_median = statistics.median(candidate_samples[name])
        delta_ns = candidate_median - accepted_median
        delta_percent = 100.0 * delta_ns / accepted_median
        significant = adjusted_p_values[name] < _FAMILY_ALPHA
        failed = significant and delta_percent >= _PRACTICAL_REGRESSION_PERCENT
        regression = regression or failed
        if failed:
            classification = "regression"
        elif significant and delta_ns < 0.0:
            classification = "improvement"
        elif significant:
            classification = "significant_below_threshold"
        elif delta_percent >= _PRACTICAL_REGRESSION_PERCENT:
            classification = "noisy_increase"
        else:
            classification = "no_regression"
        seed = int.from_bytes(
            hashlib.sha256(f"commanding:{name}:{index}".encode()).digest()[:8],
            "big",
        )
        effect_interval = bootstrap_effect_interval(
            baseline=baseline_samples[name],
            candidate=candidate_samples[name],
            seed=seed,
        )
        report["comparisons"][name] = {
            "baseline_median_ns": accepted_median,
            "candidate_median_ns": candidate_median,
            "delta_ns": delta_ns,
            "delta_percent": delta_percent,
            "raw_p_value": raw_p_values[name],
            "adjusted_p_value": adjusted_p_values[name],
            "baseline_worker_range_ns": [
                min(baseline_samples[name]),
                max(baseline_samples[name]),
            ],
            "candidate_worker_range_ns": [
                min(candidate_samples[name]),
                max(candidate_samples[name]),
            ],
            "bootstrap_effect_percent_95": list(effect_interval),
            "classification": classification,
            "failed": failed,
        }
    report["status"] = "fail" if regression else "pass"
    return report


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _cpu_governor(cpu: int, /) -> str:
    path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    return path.read_text(encoding="utf-8").strip() if path.is_file() else "unknown"


def _environment(cpu: int, /) -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0))
    return {
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count(),
        "cpu_governor": _cpu_governor(cpu),
        "affinity": affinity,
        "platform": platform.platform(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def _fingerprint(environment: Mapping[str, Any], /) -> dict[str, Any]:
    return {
        "methodology_version": _METHODOLOGY_VERSION,
        "workloads": list(_WORKLOAD_NAMES),
        "workers": _WORKER_COUNT,
        "warmup_batches": _WARMUP_BATCHES,
        "measured_batches": _MEASURED_BATCHES,
        "target_batch_ns": _TARGET_BATCH_NS,
        "timer": "time.perf_counter_ns",
        "gc": "disabled only during timed batches",
        "cpu_model": environment["cpu_model"],
        "logical_cpus": environment["logical_cpus"],
        "cpu_governor": environment["cpu_governor"],
        "affinity": environment["affinity"],
        "platform": environment["platform"],
        "platform_system": environment["platform_system"],
        "platform_machine": environment["platform_machine"],
        "python_implementation": environment["python_implementation"],
        "python_version": environment["python_version"],
    }


def _execute(operation: Callable[[], object], iterations: int, /) -> object:
    result: object = None
    for _ in range(iterations):
        result = operation()
    return result


def _time_batch(fixture: WorkloadFixture, iterations: int, /) -> int:
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        started_ns = time.perf_counter_ns()
        result = _execute(fixture.operation, iterations)
        elapsed_ns = time.perf_counter_ns() - started_ns
    finally:
        if enabled:
            gc.enable()
    fixture.validate(result)
    return elapsed_ns


def _calibrate(fixture: WorkloadFixture, /) -> tuple[int, int]:
    iterations = 1
    elapsed_ns = _time_batch(fixture, iterations)
    while elapsed_ns < 5_000_000:
        multiplier = max(2, min(100, math.ceil(5_000_000 / max(elapsed_ns, 1))))
        iterations *= multiplier
        elapsed_ns = _time_batch(fixture, iterations)
    for _ in range(3):
        iterations = max(1, round(iterations * _TARGET_BATCH_NS / elapsed_ns))
        elapsed_ns = _time_batch(fixture, iterations)
        if 40_000_000 <= elapsed_ns <= 60_000_000:
            break
    return iterations, elapsed_ns


def _worker(args: argparse.Namespace, /) -> int:
    cpu = int(args.cpu)
    os.sched_setaffinity(0, {cpu})
    if sorted(os.sched_getaffinity(0)) != [cpu]:
        raise ValueError(f"worker could not pin itself to CPU {cpu}")
    fixtures = build_workload_fixtures()
    order_offset = int(args.worker_index) % len(fixtures)
    ordered = fixtures[order_offset:] + fixtures[:order_offset]
    external_modules = (
        "carb",
        "isaacsim",
        "omni",
        "pxr",
    )
    imported = sorted(
        name
        for name in sys.modules
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in external_modules)
    )
    if imported:
        raise ValueError(f"host-only workload imported forbidden modules: {imported}")

    workloads = {}
    for fixture in ordered:
        iterations, calibration_ns = _calibrate(fixture)
        warmup_ns = [_time_batch(fixture, iterations) for _ in range(_WARMUP_BATCHES)]
        measured_ns = [_time_batch(fixture, iterations) for _ in range(_MEASURED_BATCHES)]
        ns_per_operation = [elapsed / iterations for elapsed in measured_ns]
        workloads[fixture.name] = {
            "description": fixture.description,
            "iterations_per_batch": iterations,
            "calibration_batch_ns": calibration_ns,
            "warmup_batch_ns": warmup_ns,
            "measured_batch_ns": measured_ns,
            "ns_per_operation": ns_per_operation,
            "median_ns_per_operation": statistics.median(ns_per_operation),
        }
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "methodology_version": _METHODOLOGY_VERSION,
        "worker_index": int(args.worker_index),
        "workload_order": [fixture.name for fixture in ordered],
        "environment": _environment(cpu),
        "workloads": workloads,
    }
    _write_json(Path(args.output), payload)
    return 0


def _source_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _candidate_from_workers(
    *,
    workers: Sequence[Mapping[str, Any]],
    raw_paths: Sequence[Path],
    run_root: Path,
) -> dict[str, Any]:
    if len(workers) != _WORKER_COUNT:
        raise ValueError(f"run has {len(workers)} workers; expected {_WORKER_COUNT}")
    environments = [worker["environment"] for worker in workers]
    if any(environment != environments[0] for environment in environments[1:]):
        raise ValueError("workers recorded different hardware or runtime environments")
    for index, worker in enumerate(workers):
        if worker.get("worker_index") != index:
            raise ValueError("worker indexes are incomplete or unordered")
        if set(worker.get("workloads", {})) != set(_WORKLOAD_NAMES):
            raise ValueError(f"worker {index} has an unexpected workload set")
    fixtures = {fixture.name: fixture.description for fixture in build_workload_fixtures()}
    return {
        "source_commit": _source_commit(),
        "repetitions": _WORKER_COUNT,
        "fingerprint": _fingerprint(environments[0]),
        "environment": environments[0],
        "raw_workers": [str(path.relative_to(run_root)) for path in raw_paths],
        "workloads": {
            name: {
                "description": fixtures[name],
                "worker_medians_ns": [
                    float(worker["workloads"][name]["median_ns_per_operation"])
                    for worker in workers
                ],
            }
            for name in _WORKLOAD_NAMES
        },
    }


def _markdown(report: Mapping[str, Any], /) -> str:
    lines = [
        "# Isolated commanding performance benchmark",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Source commit: `{report.get('source_commit', 'unknown')}`",
        "",
    ]
    reason = report.get("reason")
    if reason:
        lines.extend([f"Reason: {reason}", ""])
    if report.get("workloads"):
        lines.extend(
            [
                "| Workload | Candidate median (ns) | Worker range (ns) |",
                "| --- | ---: | ---: |",
            ]
        )
        for name in _WORKLOAD_NAMES:
            workload = report["workloads"][name]
            low, high = workload["worker_range_ns"]
            lines.append(f"| `{name}` | {workload['median_ns']:.2f} | {low:.2f}..{high:.2f} |")
        lines.append("")
    comparisons = report.get("comparisons", {})
    if comparisons:
        lines.extend(
            [
                "| Workload | Delta (ns) | Delta | Raw p | Holm p | "
                "Bootstrap effect 95% | Result |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for name in _WORKLOAD_NAMES:
            item = comparisons[name]
            low, high = item["bootstrap_effect_percent_95"]
            lines.append(
                f"| `{name}` | {item['delta_ns']:+.2f} | "
                f"{item['delta_percent']:+.2f}% | {item['raw_p_value']:.6f} | "
                f"{item['adjusted_p_value']:.6f} | {low:+.2f}%..{high:+.2f}% | "
                f"{item['classification']} |"
            )
        lines.append("")
    lines.extend(
        [
            "A workload fails only when the Holm-adjusted p-value is below 0.05 "
            "and median latency is at least 5% above the accepted baseline.",
            "",
        ]
    )
    return "\n".join(lines)


def _invalid_report(*, reason: str) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "invalid",
        "reason": reason,
        "repetitions": 0,
        "policy": _POLICY,
        "workloads": {},
        "comparisons": {},
    }


def _record_report(*, report: Mapping[str, Any], run_root: Path) -> None:
    _write_json(run_root / "report.json", report)
    (run_root / "report.md").write_text(_markdown(report), encoding="utf-8")
    history = run_root.parent / "commanding-history.jsonl"
    with history.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "created_at": report["created_at"],
                    "run_root": run_root.name,
                    "source_commit": report.get("source_commit"),
                    "status": report["status"],
                    "workloads": report.get("workloads", {}),
                },
                sort_keys=True,
            )
            + "\n"
        )


def _run(args: argparse.Namespace, /) -> int:
    output_root = Path(args.output_root).resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_root = output_root / f"commanding.{run_id}"
    raw_root = run_root / "raw"
    raw_root.mkdir(parents=True)
    report: dict[str, Any]
    try:
        available_cpus = sorted(os.sched_getaffinity(0))
        if not available_cpus:
            raise ValueError("process has no available CPU affinity")
        cpu = available_cpus[0]
        raw_paths = [raw_root / f"worker-{index + 1:02d}.json" for index in range(_WORKER_COUNT)]
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = "0"
        for index, raw_path in enumerate(raw_paths):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "worker",
                    "--worker-index",
                    str(index),
                    "--cpu",
                    str(cpu),
                    "--output",
                    str(raw_path),
                ],
                check=True,
                env=environment,
            )
        workers = [_read_json(path) for path in raw_paths]
        candidate = _candidate_from_workers(
            workers=workers,
            raw_paths=raw_paths,
            run_root=run_root,
        )
        baseline_path = Path(args.baseline).resolve()
        baseline = _read_json(baseline_path) if baseline_path.is_file() else None
        report = build_report(candidate=candidate, baseline=baseline)
    except (
        json.JSONDecodeError,
        KeyError,
        OSError,
        subprocess.CalledProcessError,
        TypeError,
        ValueError,
    ) as exc:
        report = _invalid_report(reason=str(exc))
    _record_report(report=report, run_root=run_root)
    print(f"Commanding benchmark report: {run_root / 'report.md'}")
    return 0 if report["status"] == "pass" else 1 if report["status"] == "fail" else 2


def _accept(args: argparse.Namespace, /) -> int:
    result_path = Path(args.result).resolve()
    report = _read_json(result_path)
    status = report.get("status")
    if status not in {"pass", "fail", "no_baseline"}:
        raise ValueError(f"cannot accept a {status!r} benchmark result")
    if report.get("policy") != _POLICY:
        raise ValueError("result regression policy differs from the benchmark contract")
    raw_workers = report.get("raw_workers")
    if (
        not isinstance(raw_workers, list)
        or len(raw_workers) != _WORKER_COUNT
        or len(set(raw_workers)) != _WORKER_COUNT
        or not all(isinstance(path, str) and path for path in raw_workers)
    ):
        raise ValueError(f"benchmark result must reference {_WORKER_COUNT} raw workers")
    for raw_worker in raw_workers:
        raw_path = result_path.parent / raw_worker
        if not raw_path.is_file() or raw_path.stat().st_size == 0:
            raise ValueError(f"benchmark raw worker is missing or empty: {raw_path}")
    candidate = {
        "source_commit": report["source_commit"],
        "repetitions": report["repetitions"],
        "fingerprint": report["fingerprint"],
        "environment": report["environment"],
        "raw_workers": raw_workers,
        "workloads": report["workloads"],
    }
    samples = _validate_candidate(candidate)
    baseline_path = Path(args.baseline).resolve()
    existing = _read_json(baseline_path) if baseline_path.is_file() else None
    if status == "no_baseline" and existing is not None:
        raise ValueError("cannot accept a no_baseline result over an existing baseline")
    if existing is not None:
        _validate_baseline(existing)
        if existing["fingerprint"] != report["fingerprint"]:
            raise ValueError("result fingerprint differs from the existing baseline")
    baseline = {
        "schema_version": _SCHEMA_VERSION,
        "accepted_at": datetime.now(timezone.utc).isoformat(),
        "accepted_from": result_path.parent.name,
        "source_commit": report["source_commit"],
        "repetitions": _WORKER_COUNT,
        "policy": _POLICY,
        "fingerprint": report["fingerprint"],
        "environment": report["environment"],
        "workloads": {
            name: {
                "description": report["workloads"][name]["description"],
                "worker_medians_ns": samples[name],
            }
            for name in _WORKLOAD_NAMES
        },
    }
    _write_json(baseline_path, baseline)
    print(f"Accepted commanding benchmark baseline: {baseline_path}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run nine workers and compare the result")
    run.add_argument("--output-root", required=True)
    run.add_argument("--baseline", required=True)
    run.set_defaults(action=_run)

    accept = subparsers.add_parser("accept", help="replace the committed baseline")
    accept.add_argument("--result", required=True)
    accept.add_argument("--baseline", required=True)
    accept.set_defaults(action=_accept)

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--worker-index", required=True, type=int)
    worker.add_argument("--cpu", required=True, type=int)
    worker.add_argument("--output", required=True)
    worker.set_defaults(action=_worker)
    return parser


def main() -> int:
    """Run the isolated commanding benchmark command-line interface."""
    args = _parser().parse_args()
    try:
        return int(args.action(args))
    except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"commanding benchmark: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
