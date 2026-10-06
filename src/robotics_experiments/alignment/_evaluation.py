"""Pure controller-performance evaluation, independent of search and publication."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from robotics_experiments.alignment._metrics import metric
from robotics_experiments.alignment._models import MetricResult, ProbeTrace, QualityGateResult
from robotics_experiments.flight_parameters import hover_pose_errors

__all__: list[str] = []


@dataclass(frozen=True)
class Evaluation:
    """The worst normalized error decides both ordering and acceptance."""

    checks: tuple[QualityGateResult, ...]
    score: float
    accepted: bool

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(
            f"{check.name}={check.value:.6g} exceeds {check.limit:.6g}"
            for check in self.checks
            if check.value > check.limit
        )


def evaluate_metrics(
    *, measurements: Mapping[str, float], limits: Mapping[str, float]
) -> Evaluation:
    """Join explicit measurements and limits without any I/O or selection policy."""
    if not limits or set(measurements) != set(limits):
        raise ValueError("evaluation needs exactly one measurement for every limit")
    checks = tuple(
        QualityGateResult(name, measurements[name], limit) for name, limit in limits.items()
    )
    if any(
        not math.isfinite(c.value) or c.value < 0 or not math.isfinite(c.limit) or c.limit <= 0
        for c in checks
    ):
        raise ValueError("evaluation requires finite nonnegative errors and positive limits")
    score = max(c.value / c.limit for c in checks)
    return Evaluation(checks=checks, score=score, accepted=score <= 1.0)


def evaluate_response(
    *,
    metric_name: str,
    reference: tuple[ProbeTrace, ...],
    target: tuple[ProbeTrace, ...],
    response_limits: Mapping[str, float],
    hover_limits: Mapping[str, float],
) -> MetricResult:
    """Measure response and every hover stage, then apply the same numerical rule."""
    result = metric(name=metric_name)(reference, target)
    measurements = dict(result.validation_metrics)
    families = {trace.family for trace in reference}
    limits = {
        name: limit for name, limit in response_limits.items() if name.split(".")[0] in families
    }
    hover = [trace for trace in target if trace.profile == "stability-hover-pose"]
    if not hover:
        raise ValueError("alignment requires the full hover response stage")
    hover_measurements = dict.fromkeys(hover_limits, 0.0)
    for trace in hover:
        w, x, y, z = trace.initial_attitude_flu_enu_wxyz
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        target_attitude = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        for position, attitude in zip(
            trace.position_enu_m, trace.attitude_flu_enu_wxyz, strict=True
        ):
            values = hover_pose_errors(
                target_position_enu_m=trace.initial_position_enu_m,
                target_attitude_flu_enu_wxyz=target_attitude,
                position_enu_m=position,
                attitude_flu_enu_wxyz=attitude,
            )
            for name, value in values.items():
                hover_measurements[name] = max(hover_measurements[name], value)
    measurements.update(hover_measurements)
    limits.update(hover_limits)
    evaluated = evaluate_metrics(measurements=measurements, limits=limits)
    return replace(
        result,
        metrics={**result.metrics, **hover_measurements},
        gates=evaluated.checks,
        ranking_values=(evaluated.score,),
        beam_values=tuple(c.value / c.limit for c in evaluated.checks),
        quality_passed=evaluated.accepted,
    )


def evaluate_final(
    *,
    response_checks: tuple[QualityGateResult, ...],
    live_checks: tuple[QualityGateResult, ...],
    replay: Mapping,
) -> Evaluation:
    """Combine frozen-test response, replay and live checks without selection."""
    checks = [*response_checks, *live_checks]
    worst_replay: dict[str, QualityGateResult] = {}
    for flight in replay["flights"]:
        for profile in flight["profiles"]:
            prefix = f"replay.flight-{flight['flight_index']}.{profile['profile']}"
            if (
                profile["physical_state_timestamp_join_error_us"]
                > flight["timestamp_join_limit_us"]
                or profile["actuator_handoff_join_error_us"] > flight["timestamp_join_limit_us"]
            ):
                raise ValueError(f"{prefix}: timestamp association is invalid")
            for name, values, limit_name in (
                ("normalized_motor_rmse", profile["metrics"], "normalized_motor_rmse"),
                ("collective_thrust_rmse_n", profile["metrics"], "collective_thrust_rmse_n"),
                (
                    "physical_normalized_motor_rmse",
                    profile["physical_state_metrics"],
                    "normalized_motor_rmse",
                ),
            ):
                check = QualityGateResult(
                    f"{prefix}.{name}", values[limit_name], flight["limits"][limit_name]
                )
                evaluate_metrics(
                    measurements={check.name: check.value}, limits={check.name: check.limit}
                )
                previous = worst_replay.get(name)
                if previous is None or check.value / check.limit > previous.value / previous.limit:
                    worst_replay[name] = check
    if len(worst_replay) != 3:
        raise ValueError("test replay has no complete performance measurements")
    checks.extend(worst_replay.values())
    if len({c.name for c in checks}) != len(checks):
        raise ValueError("final evaluation has duplicate check identities")
    return evaluate_metrics(
        measurements={c.name: c.value for c in checks}, limits={c.name: c.limit for c in checks}
    )
