"""Frame and sampling tests for the PX4/Isaac controller replay."""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from flight_analysis.tools.controller_conformance import (
    _aligned_trace_diagnostics,
    _fit_lab_command_delay,
    _fit_temporal_response,
    _load_temporal_traces,
    _nearest_indices,
    _px4_attitude_to_isaac,
    _simulate_first_order_response,
    _TemporalTrace,
    _trace_metrics,
)


def test_nearest_indices_selects_edges_and_nearest_interior_sample():
    source = np.array([1.0, 2.0, 4.0])
    query = np.array([0.0, 1.6, 3.7, 5.0])

    assert _nearest_indices(source_t=source, query_t=query).tolist() == [0, 1, 2, 2]


def test_identity_ned_frd_attitude_becomes_north_facing_enu_flu():
    quaternion, yaw = _px4_attitude_to_isaac(quaternion_frd_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))

    assert yaw == math.pi / 2.0
    assert np.allclose(quaternion, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])


def _sample(
    *,
    time_s: float,
    range_m: float,
    velocity_mps: float,
    command_mps: float,
    complete: bool = False,
) -> dict[str, Any]:
    return {
        "sim_time_us": round(time_s * 1_000_000),
        "time_s": time_s,
        "range_m": range_m,
        "closing_speed_mps": velocity_mps,
        "linear_speed_mps": abs(velocity_mps),
        "forward_command_mps": command_mps,
        "observation": [range_m, 0.0, 0.0, *([0.0] * 7), velocity_mps, *([0.0] * 23)],
        "action": [command_mps / 2.0, 0.0, 0.0, 0.0],
        "stable_goal_count": 5 if complete else 0,
        "complete": complete,
    }


def _synthetic_trace(*, entry_range_m: float) -> _TemporalTrace:
    times = np.arange(0.0, 1.25, 0.05)
    commands = np.concatenate(
        (
            np.linspace(0.12, 0.08, 7),
            np.linspace(0.06, -0.04, 8),
            np.linspace(-0.03, 0.0, 10),
        )
    )
    seed = _TemporalTrace(
        label=f"policy[temporal-{entry_range_m:.3f}]",
        nominal_entry_range_m=entry_range_m,
        samples=tuple(
            _sample(
                time_s=float(time_s),
                range_m=entry_range_m,
                velocity_mps=0.005,
                command_mps=float(command),
            )
            for time_s, command in zip(times, commands, strict=True)
        ),
    )
    velocities = _simulate_first_order_response(
        trace=seed,
        command_delay_s=0.08,
        response_time_constant_s=0.20,
    )
    ranges = entry_range_m - np.concatenate(([0.0], np.cumsum(velocities[:-1] * np.diff(times))))
    return _TemporalTrace(
        label=seed.label,
        nominal_entry_range_m=entry_range_m,
        samples=tuple(
            _sample(
                time_s=float(time_s),
                range_m=float(range_m),
                velocity_mps=float(velocity),
                command_mps=float(command),
                complete=index == len(times) - 1,
            )
            for index, (time_s, range_m, velocity, command) in enumerate(
                zip(times, ranges, velocities, commands, strict=True)
            )
        ),
    )


def test_temporal_fit_uses_outer_cases_and_validates_middle_case():
    traces = tuple(_synthetic_trace(entry_range_m=value) for value in (0.460, 0.475, 0.490))

    fit = _fit_temporal_response(traces=traces)

    assert fit["fit_entry_ranges_m"] == [0.460, 0.490]
    assert fit["validation_entry_ranges_m"] == [0.475]
    # The 20 Hz observation cadence can only identify the first 4 ms grid
    # point in the equivalent two-sample delay interval.
    assert math.isclose(float(fit["command_delay_s"]), 0.052, abs_tol=0.004)
    assert math.isclose(float(fit["response_time_constant_s"]), 0.20, abs_tol=0.004)
    assert float(fit["training_velocity_rmse_mps"]) < 1e-9
    assert float(fit["validation_velocity_rmse_mps"]) < 1e-9


def test_temporal_alignment_reports_response_braking_range_and_settling():
    px4 = _synthetic_trace(entry_range_m=0.475)
    lab = _TemporalTrace(
        label=px4.label,
        nominal_entry_range_m=px4.nominal_entry_range_m,
        samples=tuple(
            {
                **sample,
                "range_m": float(sample["range_m"]) + 0.01,
                "closing_speed_mps": float(sample["closing_speed_mps"]) - 0.01,
                "complete": index == len(px4.samples) - 2,
            }
            for index, sample in enumerate(px4.samples)
        ),
    )

    diagnostics = _aligned_trace_diagnostics(px4=px4, lab=lab)

    assert diagnostics["aligned_sample_count"] == len(px4.samples)
    assert math.isclose(float(diagnostics["velocity_response_rmse_mps"]), 0.01)
    assert math.isclose(float(diagnostics["closest_range_delta_m"]), 0.01)
    assert math.isclose(float(diagnostics["settling_time_delta_s"]), -0.05)


def test_lab_delay_fit_uses_outer_cases_and_validates_middle_case():
    px4 = tuple(_synthetic_trace(entry_range_m=value) for value in (0.460, 0.475, 0.490))

    def candidate(*, delay_s: float, velocity_error_mps: float) -> dict[str, Any]:
        traces = []
        for trace in px4:
            error = velocity_error_mps
            if trace.nominal_entry_range_m == 0.475:
                error *= 2.0
            traces.append(
                {
                    "label": trace.label,
                    "nominal_entry_range_m": trace.nominal_entry_range_m,
                    "samples": [
                        {
                            **sample,
                            "closing_speed_mps": float(sample["closing_speed_mps"]) + error,
                        }
                        for sample in trace.samples
                    ],
                }
            )
        return {"controller_command_delay_s": delay_s, "traces": traces}

    fit, selected = _fit_lab_command_delay(
        px4_traces=px4,
        candidates=[
            candidate(delay_s=0.04, velocity_error_mps=0.02),
            candidate(delay_s=0.08, velocity_error_mps=0.005),
            candidate(delay_s=0.12, velocity_error_mps=0.01),
        ],
    )

    assert fit["fit_entry_ranges_m"] == [0.460, 0.490]
    assert fit["validation_entry_ranges_m"] == [0.475]
    assert math.isclose(float(fit["selected_controller_command_delay_s"]), 0.08)
    assert math.isclose(float(fit["training_velocity_rmse_mps"]), 0.005)
    assert math.isclose(float(fit["validation_velocity_rmse_mps"]), 0.01)
    assert [trace.nominal_entry_range_m for trace in selected] == [0.460, 0.475, 0.490]


def test_temporal_log_parser_preserves_exact_actor_cadence(tmp_path: Path):
    path = tmp_path / "runtime.log"
    lines = []
    for nominal in (0.460, 0.475, 0.490):
        label = f"policy[temporal-{nominal:.3f}]"
        lines.append(
            "prefix ROUTINE_STAGE "
            + json.dumps({"event": "start", "kind": "policy", "label": label})
        )
        for index, time_us in enumerate((1_000_000, 1_060_000, 1_100_000)):
            observation = [0.0] * 34
            observation[0] = nominal - index * 0.004
            observation[10] = 0.08
            lines.append(
                "prefix PRESS_POLICY "
                + json.dumps(
                    {
                        "event": "actor_evaluation",
                        "observation": observation,
                        "normalized_action": [0.04, 0.0, 0.0, 0.0],
                        "setpoint": {"forward_mps": 0.08},
                        "sim_time_us": time_us,
                        "stable_goal_count": 5 if index == 2 else 0,
                        "complete": index == 2,
                    }
                )
            )
        lines.append(
            "prefix ROUTINE_STAGE "
            + json.dumps({"event": "complete", "kind": "policy", "label": label})
        )
    path.write_text("\n".join(lines) + "\n")

    traces = _load_temporal_traces(log_path=path)
    metrics = _trace_metrics(trace=traces[0])

    assert np.allclose(
        [sample["time_s"] for sample in traces[0].samples],
        [0.0, 0.06, 0.1],
    )
    assert math.isclose(float(metrics["cadence_mean_s"]), 0.05)
    assert math.isclose(float(metrics["cadence_min_s"]), 0.04)
    assert math.isclose(float(metrics["cadence_max_s"]), 0.06)
