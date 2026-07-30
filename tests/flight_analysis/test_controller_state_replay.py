"""Frame and sampling tests for the PX4/Isaac controller replay."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from flight_analysis.tools.controller_state_replay import (
    _augment_reference_controller_state,
    _bridge_actuator_samples,
    _hold_indices,
    _interpolate_quaternions,
    _linear_interpolate,
    _linear_scale_with_intercept,
    _nearest_indices,
    _profile_topic,
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
                    "timestamp": np.array([100_000]),
                    "acceleration[0]": np.array([1.0]),
                    "acceleration[1]": np.array([2.0]),
                    "acceleration[2]": np.array([3.0]),
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
                    "timestamp": np.array([100_000]),
                    "rollspeed_integ": np.array([0.1]),
                    "pitchspeed_integ": np.array([0.2]),
                    "yawspeed_integ": np.array([0.3]),
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

    assert report["method"] == "px4-ulog-controller-boundary-v1"
    assert payload["schema_version"] == 4
    assert trace["initial_controller_state_valid"]
    assert trace["initial_acceleration_setpoint_enu_mps2"] == [2.0, 1.0, -3.0]
    assert trace["initial_linear_acceleration_enu_mps2"] == [6.0, 3.0, -9.0]
    assert trace["initial_angular_acceleration_body_flu_radps2"] == [
        2.0,
        -3.0,
        -4.0,
    ]
    assert trace["initial_rate_integral_body_flu"] == [0.1, -0.2, -0.3]
    assert trace["initial_controller_state_source_age_us"] == {
        "vehicle_local_position_setpoint": 4000,
        "rate_ctrl_status": 4000,
    }
