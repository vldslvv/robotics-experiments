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
    _bridge_state_samples,
    _controller_setpoint_metrics,
    _delay_transmitted_actions,
    _delay_velocity_actions,
    _hold_indices,
    _interpolate_quaternions,
    _linear_interpolate,
    _linear_scale_with_intercept,
    _nearest_indices,
    _normalized_boundary_metrics,
    _profile_topic,
    _profile_windows,
    _px4_attitude_to_isaac,
    _px4_vertical_velocity_state,
    _rate_input_metrics,
    _replay_time_grid,
)


def test_nearest_indices_selects_edges_and_nearest_interior_sample():
    source = np.array([1.0, 2.0, 4.0])
    query = np.array([0.0, 1.6, 3.7, 5.0])

    assert _nearest_indices(source_t=source, query_t=query).tolist() == [0, 1, 2, 2]


def test_hold_indices_preserve_previous_controller_setpoint():
    source = np.array([1.0, 2.0, 4.0])
    query = np.array([0.0, 1.6, 3.7, 5.0])

    assert _hold_indices(source_t=source, query_t=query).tolist() == [0, 0, 1, 2]


def test_replay_grid_preserves_exact_relative_command_ticks() -> None:
    absolute_time, relative_time = _replay_time_grid(
        start_s=43.884,
        end_s=45.084,
        controller_dt_s=0.004,
    )
    command_index = _hold_indices(
        source_t=np.array([0.0, 1.0]),
        query_t=relative_time,
    )

    assert relative_time[250] == 1.0
    assert absolute_time[250] == pytest.approx(44.884)
    assert command_index[249:251].tolist() == [0, 1]


def test_velocity_action_delay_shifts_the_complete_command():
    actions = np.arange(20, dtype=np.float64).reshape(5, 4)

    delayed = _delay_velocity_actions(
        actions=actions,
        delay_steps=2,
    )

    assert delayed[:, :3].tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 2.0],
        [4.0, 5.0, 6.0],
        [8.0, 9.0, 10.0],
    ]
    assert delayed[:, 3].tolist() == [0.0, 0.0, 3.0, 7.0, 11.0]


def test_recorded_command_takes_effect_after_its_publishing_tick() -> None:
    _, relative_time = _replay_time_grid(
        start_s=12.0,
        end_s=12.048,
        controller_dt_s=0.004,
    )
    command_index = _hold_indices(
        source_t=np.array([0.0, 0.04]),
        query_t=relative_time,
    )
    sampled_actions = np.array([[0.0, 0.0, 0.1, 0.0], [0.0, 0.0, 0.7, 0.0]])[command_index]

    delayed = _delay_transmitted_actions(actions=sampled_actions)
    velocity_setpoint_ned_z = -(delayed[:, 2] * 2.0)

    assert relative_time[10:12].tolist() == pytest.approx([0.04, 0.044])
    assert delayed[10:12, 2].tolist() == pytest.approx([0.1, 0.7])
    assert velocity_setpoint_ned_z[10:12].tolist() == pytest.approx([-0.2, -1.4])


def test_linear_interpolation_resamples_controller_state():
    result = _linear_interpolate(
        source_t=np.array([0.0, 1.0]),
        values=np.array([[0.0, 2.0], [2.0, 4.0]]),
        query_t=np.array([0.25, 0.75]),
    )

    assert np.allclose(result, [[0.5, 2.5], [1.5, 3.5]])


def test_vertical_velocity_state_matches_px4_velocity_only_blending():
    result = _px4_vertical_velocity_state(
        velocity_setpoint_ned=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -0.35], [0.0, 0.0, 0.8]]),
        velocity_ned=np.array([[1.0, 2.0, 0.1], [1.0, 2.0, 0.2], [1.0, 2.0, 0.3]]),
        position_derivative_z_ned=np.array([0.4, 0.6, 0.9]),
        landing_speed_mps=0.7,
    )

    assert np.allclose(result[:, :2], [[1.0, 2.0]] * 3)
    assert np.allclose(result[:, 2], [0.1, 0.4, 0.9])


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


def test_controller_setpoint_metrics_keep_each_rate_axis_visible():
    metrics = _controller_setpoint_metrics(
        target_body_rate_flu=np.array([[0.2, -0.1, 0.4], [0.0, 0.3, -0.2]]),
        px4_body_rate_flu=np.array([[0.1, -0.3, 0.4], [0.0, 0.1, 0.0]]),
        target_collective=np.array([0.55, 0.60]),
        px4_collective=np.array([0.50, 0.60]),
    )

    assert metrics == {
        "body_rate_setpoint_rmse_radps": pytest.approx(
            [np.sqrt(0.01 / 2.0), 0.2, 0.2 / np.sqrt(2.0)]
        ),
        "collective_setpoint_rmse": pytest.approx(0.05 / np.sqrt(2.0)),
    }


def test_rate_input_metrics_keep_velocity_and_derivative_errors_separate():
    metrics = _rate_input_metrics(
        target_angular_velocity_flu=np.array([[0.1, 0.2, 0.3], [0.3, 0.2, 0.1]]),
        px4_angular_velocity_flu=np.array([[0.1, 0.1, 0.3], [0.1, 0.2, 0.1]]),
        target_angular_acceleration_flu=np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
        px4_angular_acceleration_flu=np.array([[0.0, 2.0, 1.0], [3.0, 1.0, 1.0]]),
    )

    assert metrics == {
        "angular_velocity_input_rmse_radps": pytest.approx(
            [0.2 / np.sqrt(2.0), 0.1 / np.sqrt(2.0), 0.0]
        ),
        "angular_acceleration_input_rmse_radps2": pytest.approx(
            [1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), np.sqrt(2.0)]
        ),
    }


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


def test_bridge_state_samples_preserve_physics_tick_inputs(tmp_path: Path):
    log = tmp_path / "bridge.log"
    log.write_text(
        "PX4 bridge dynamics: tick=26 sim_time_us=104000 "
        "last_actuator_us=104000 actuator_lag_us=0 connected=True armed=True "
        "commands=[100.0, 101.0, 102.0, 103.0] "
        "forces_n=[1.0, 1.1, 1.2, 1.3] "
        "body_torque_n_m=[0.0, 0.0, 0.01] body_drag_n=[0.0, 0.0, 0.0] "
        "position_enu=[1.0, 2.0, 3.0] quat_wxyz=[1.0, 0.0, 0.0, 0.0] "
        "velocity_enu=[0.1, 0.2, 0.3] body_velocity_flu=[0.1, 0.2, 0.3] "
        "angular_velocity_body_flu=[0.01, 0.02, 0.03]\n"
    )

    time_us, position, attitude, velocity, angular_velocity = _bridge_state_samples(path=log)

    assert time_us.tolist() == [104000]
    assert position.tolist() == [[1.0, 2.0, 3.0]]
    assert attitude.tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert velocity.tolist() == [[0.1, 0.2, 0.3]]
    assert angular_velocity.tolist() == [[0.01, 0.02, 0.03]]


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
                "schema_version": 1,
                "traces": [
                    {
                        "family": "velocity",
                        "control_mode": "velocity",
                        "family_metric": "translational-controller-quality",
                        "profile": "validation-combined",
                        "split": "validation",
                        "axis": "combined",
                        "start_sim_time_us": 116_384_000,
                        "time_s": [0.0, 2.5, 5.7],
                        "command_velocity_body_flu_mps": [
                            [0.0, 0.0, 0.0],
                            [0.1, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        "command_yaw_rate_radps": [0.0, 0.2, 0.0],
                        "command_collective_thrust_normalized": [0.0, 0.0, 0.0],
                        "command_body_rate_flu_radps": [[0.0, 0.0, 0.0]] * 3,
                        "attitude_flu_enu_wxyz": [
                            [1.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                        ],
                        "velocity_enu_mps": [
                            [0.0, 0.0, 0.0],
                            [0.1, 0.0, 0.0],
                            [0.0, 0.0, 0.0],
                        ],
                        "angular_velocity_body_flu_radps": [
                            [0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.2],
                            [0.0, 0.0, 0.0],
                        ],
                    }
                ],
            }
        )
    )

    assert _profile_windows(reference_path=reference) == [
        {
            "family": "velocity",
            "control_mode": "velocity",
            "family_metric": "translational-controller-quality",
            "profile": "validation-combined",
            "split": "validation",
            "axis": "combined",
            "start_s": 116.384,
            "end_s": 122.084,
            "command_trace": {
                "time_s": [0.0, 2.5, 5.7],
                "command_velocity_body_flu_mps": [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "command_yaw_rate_radps": [0.0, 0.2, 0.0],
                "command_collective_thrust_normalized": [0.0, 0.0, 0.0],
                "command_body_rate_flu_radps": [[0.0, 0.0, 0.0]] * 3,
                "control_mode": "velocity",
                "attitude_flu_enu_wxyz": [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ],
                "velocity_enu_mps": [
                    [0.0, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                "angular_velocity_body_flu_radps": [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.2],
                    [0.0, 0.0, 0.0],
                ],
            },
        }
    ]


def test_reference_augmentation_records_px4_controller_boundary(tmp_path: Path):
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "schema_version": 1,
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
                    "vx": np.array([0.1, 0.3]),
                    "vy": np.array([0.2, 0.4]),
                    "vz": np.array([0.3, 0.5]),
                },
            ),
            dataset(
                "vehicle_angular_velocity",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "xyz[0]": np.array([0.1, 0.3]),
                    "xyz[1]": np.array([0.2, 0.4]),
                    "xyz[2]": np.array([0.3, 0.5]),
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
            dataset(
                "hover_thrust_estimate",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "hover_thrust": np.array([0.57, 0.574]),
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
    assert payload["schema_version"] == 1
    assert trace["initial_controller_state_valid"]
    assert trace["initial_estimated_linear_velocity_enu_mps"] == pytest.approx([0.3, 0.2, -0.4])
    assert trace["initial_estimated_angular_velocity_body_flu_radps"] == pytest.approx(
        [0.2, -0.3, -0.4]
    )
    assert trace["initial_acceleration_setpoint_enu_mps2"] == pytest.approx([3.0, 2.0, -4.0])
    assert trace["initial_linear_acceleration_enu_mps2"] == [6.0, 3.0, -9.0]
    assert trace["initial_angular_acceleration_body_flu_radps2"] == [
        2.0,
        -3.0,
        -4.0,
    ]
    assert trace["initial_rate_integral_body_flu"] == pytest.approx([0.2, -0.3, -0.4])
    assert trace["initial_hover_thrust_normalized"] == pytest.approx(0.572)
    assert trace["initial_controller_state_source_age_us"] == {
        "vehicle_local_position_setpoint": 4000,
        "vehicle_local_position": 4000,
        "vehicle_angular_velocity": 4000,
        "rate_ctrl_status": 4000,
        "hover_thrust_estimate": 4000,
    }
