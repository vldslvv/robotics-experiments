"""Frame and sampling tests for the PX4/Isaac controller replay."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flight_analysis.tools.controller_state_replay import (
    _augment_reference_controller_state,
    _bridge_actuator_samples,
    _hold_indices,
    _interpolate_quaternions,
    _linear_interpolate,
    _linear_scale_with_intercept,
    _nearest_indices,
    _normalized_boundary_metrics,
    _profile_topic,
    _profile_windows,
    _px4_attitude_to_isaac,
)


def test_nearest_indices_selects_edges_and_nearest_interior_sample():
    source = np.array([1.0, 2.0, 4.0])
    query = np.array([0.0, 1.6, 3.7, 5.0])

    assert _nearest_indices(source_t=source, query_t=query).tolist() == [0, 1, 2, 2]


def test_hold_indices_preserve_previous_controller_setpoint():
    source = np.array([1.0, 2.0, 4.0])
    query = np.array([0.0, 1.6, 3.7, 5.0])

    assert _hold_indices(source_t=source, query_t=query).tolist() == [0, 0, 1, 2]


def test_linear_interpolation_resamples_controller_state():
    result = _linear_interpolate(
        source_t=np.array([0.0, 1.0]),
        values=np.array([[0.0, 2.0], [2.0, 4.0]]),
        query_t=np.array([0.25, 0.75]),
    )

    assert np.allclose(result, [[0.5, 2.5], [1.5, 3.5]])


def test_quaternion_interpolation_uses_shortest_normalized_path():
    result = _interpolate_quaternions(
        source_t=np.array([0.0, 1.0]),
        quaternions_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [-0.0, -0.0, -0.0, -1.0]]),
        query_t=np.array([0.5]),
    )

    assert np.allclose(np.linalg.norm(result, axis=1), [1.0])
    assert np.allclose(np.abs(result[0]), [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])


def test_identity_ned_frd_attitude_becomes_north_facing_enu_flu():
    quaternion, yaw = _px4_attitude_to_isaac(quaternion_frd_ned_wxyz=np.array([1.0, 0.0, 0.0, 0.0]))

    assert yaw == math.pi / 2.0
    assert np.allclose(quaternion, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])


def test_rate_output_scale_fit_separates_constant_torque_bias():
    controller = np.array([-1.0, 0.0, 1.0, 2.0])
    physical = controller * 6.5 - 0.08

    scale, intercept, rmse = _linear_scale_with_intercept(
        controller_output=controller,
        physical_output=physical,
    )

    assert math.isclose(scale, 6.5)
    assert math.isclose(intercept, -0.08)
    assert rmse < 1e-12


def test_normalized_boundary_reconstructs_unsaturated_collective_and_torque():
    mix = np.array(
        [
            [-0.5, 0.5, 1.0, -1.0],
            [0.5, -0.5, 1.0, -1.0],
            [0.5, 0.5, -1.0, -1.0],
            [-0.5, -0.5, -1.0, -1.0],
        ]
    )
    target_boundary = np.array(
        [
            [0.02, -0.03, 0.01, -0.57],
            [-0.01, 0.02, -0.02, -0.55],
        ]
    )
    target_motor = target_boundary @ mix.T
    px4_motor = np.array(target_motor, copy=True)
    px4_collective = -target_boundary[:, 3]
    px4_torque = target_boundary[:, :3] * np.array([1.0, -1.0, -1.0])

    metrics = _normalized_boundary_metrics(
        target_motor_controls=target_motor,
        px4_motor_controls=px4_motor,
        normalized_mix=mix,
        px4_collective=px4_collective,
        px4_torque_flu=px4_torque,
    )

    assert metrics == {
        "normalized_motor_rmse": 0.0,
        "unsaturated_sample_count": 2,
        "saturated_sample_count": 0,
        "normalized_collective_rmse": pytest.approx(0.0, abs=1e-12),
        "normalized_torque_rmse": pytest.approx([0.0, 0.0, 0.0], abs=1e-12),
    }


def test_normalized_boundary_excludes_saturated_samples_from_upstream_metrics():
    mix = np.eye(4)
    target_motor = np.array([[0.2, 0.3, 0.4, 0.5], [0.0, 0.3, 0.4, 0.5]])

    metrics = _normalized_boundary_metrics(
        target_motor_controls=target_motor,
        px4_motor_controls=np.array(target_motor, copy=True),
        normalized_mix=mix,
        px4_collective=np.array([-0.5, -0.5]),
        px4_torque_flu=np.array([[0.2, -0.3, -0.4], [0.0, -0.3, -0.4]]),
    )

    assert metrics["unsaturated_sample_count"] == 1
    assert metrics["saturated_sample_count"] == 1
    assert metrics["normalized_collective_rmse"] == pytest.approx(0.0)
    assert metrics["normalized_torque_rmse"] == pytest.approx([0.0, 0.0, 0.0])


def test_bridge_actuator_samples_join_dynamics_within_one_tick(tmp_path: Path):
    log = tmp_path / "bridge.log"
    log.write_text(
        "PX4 bridge dynamics: tick=26 sim_time_us=104000 "
        "last_actuator_us=100000 actuator_lag_us=4000 connected=True armed=True "
        "commands=[100.0, 101.0, 102.0, 103.0] "
        "forces_n=[1.0, 1.1, 1.2, 1.3] "
        "body_torque_n_m=[0.0, 0.0, 0.01]\n"
    )

    time_us, rotor_speed, forces, join_and_torque = _bridge_actuator_samples(path=log)

    assert time_us.tolist() == [100000]
    assert rotor_speed.tolist() == [[100.0, 101.0, 102.0, 103.0]]
    assert forces.tolist() == [[1.0, 1.1, 1.2, 1.3]]
    assert join_and_torque.tolist() == [[4000.0, 0.0, 0.0, 0.01]]


def test_profile_topic_uses_absolute_simulation_window():
    topic = {
        "timestamp": np.array([96_000, 100_000, 104_000, 108_000]),
    }
    report = _profile_topic(
        topic=topic,
        values=np.array([[0.0], [1.0], [2.0], [3.0]]),
        start_us=100_000,
        end_us=104_000,
    )

    assert report == {
        "timestamp_us": [100_000, 104_000],
        "values": [[1.0], [2.0]],
    }


def test_profile_windows_use_recorded_absolute_start_and_trace_duration(tmp_path: Path):
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "traces": [
                    {
                        "profile": "validation-combined",
                        "split": "validation",
                        "axis": "combined",
                        "start_sim_time_us": 116_384_000,
                        "time_s": [0.0, 2.5, 5.7],
                    }
                ],
            }
        )
    )

    assert _profile_windows(reference_path=reference) == [
        {
            "profile": "validation-combined",
            "split": "validation",
            "axis": "combined",
            "start_s": 116.384,
            "end_s": 122.084,
        }
    ]


def test_reference_augmentation_records_px4_controller_boundary(tmp_path: Path):
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "traces": [
                    {
                        "profile": "validation",
                        "start_sim_time_us": 104_000,
                    }
                ],
            }
        )
    )
    reference.chmod(0o444)

    def dataset(name, data):
        return SimpleNamespace(name=name, data=data)

    ulog = SimpleNamespace(
        data_list=[
            dataset(
                "vehicle_local_position_setpoint",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "acceleration[0]": np.array([1.0, 3.0]),
                    "acceleration[1]": np.array([2.0, 4.0]),
                    "acceleration[2]": np.array([3.0, 5.0]),
                },
            ),
            dataset(
                "vehicle_local_position",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "ax": np.array([2.0, 4.0]),
                    "ay": np.array([4.0, 8.0]),
                    "az": np.array([6.0, 12.0]),
                },
            ),
            dataset(
                "vehicle_angular_velocity",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "xyz_derivative[0]": np.array([1.0, 3.0]),
                    "xyz_derivative[1]": np.array([2.0, 4.0]),
                    "xyz_derivative[2]": np.array([3.0, 5.0]),
                },
            ),
            dataset(
                "rate_ctrl_status",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "rollspeed_integ": np.array([0.1, 0.3]),
                    "pitchspeed_integ": np.array([0.2, 0.4]),
                    "yawspeed_integ": np.array([0.3, 0.5]),
                },
            ),
        ]
    )

    report = _augment_reference_controller_state(  # noqa: SLF001 - focused diagnostic test
        ulog=ulog,
        reference_path=reference,
    )
    payload = json.loads(reference.read_text())
    trace = payload["traces"][0]

    assert report["method"] == "px4-ulog-controller-boundary-v2"
    assert payload["schema_version"] == 4
    assert trace["initial_controller_state_valid"]
    assert trace["initial_acceleration_setpoint_enu_mps2"] == [3.0, 2.0, -4.0]
    assert trace["initial_linear_acceleration_enu_mps2"] == [6.0, 3.0, -9.0]
    assert trace["initial_angular_acceleration_body_flu_radps2"] == [
        2.0,
        -3.0,
        -4.0,
    ]
    assert trace["initial_rate_integral_body_flu"] == pytest.approx([0.2, -0.3, -0.4])
    assert trace["initial_controller_state_source_age_us"] == {
        "vehicle_local_position_setpoint": 4000,
        "rate_ctrl_status": 4000,
    }
