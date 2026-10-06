"""Frame and sampling tests for the PX4/Isaac controller replay."""

import gzip
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from flight_analysis.tools.controller_state_replay import (
    _allocator_collective_at_motor_samples,
    _augment_reference_controller_state,
    _bridge_actuator_samples,
    _bridge_state_samples,
    _collective_boundary_metrics,
    _controller_handoff_indices,
    _controller_setpoint_delay_scan,
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
    _velocity_controller_derivative,
    _vertical_acceleration_setpoint_enu,
    _vertical_setpoint_delay_scan,
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


def test_vertical_acceleration_setpoint_is_recovered_from_px4_thrust() -> None:
    hover = np.array([0.57, 0.58])
    acceleration_enu_z = np.array([1.5, -0.7])
    thrust_ned_z = -acceleration_enu_z * hover / 9.81 - hover

    result = _vertical_acceleration_setpoint_enu(
        quaternion_frd_ned_wxyz=np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        thrust_body_z=thrust_ned_z,
        hover_thrust_normalized=hover,
    )

    assert result == pytest.approx(acceleration_enu_z)


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


def test_controller_setpoint_delay_scan_finds_one_tick_handoff() -> None:
    target_rates = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
    target_collective = np.array([0.5, 0.6, 0.7, 0.8])
    replay_indices = np.array([1, 2, 3])

    result = _controller_setpoint_delay_scan(
        target_body_rate_grid_flu=target_rates,
        target_collective_grid=target_collective,
        replay_indices=replay_indices,
        px4_body_rate_flu=target_rates[[0, 1, 2]],
        px4_collective=target_collective[[0, 1, 2]],
        maximum_delay_steps=2,
    )

    assert result[0]["collective_setpoint_rmse"] > 0.0
    assert result[1]["collective_setpoint_rmse"] == 0.0
    assert result[1]["body_rate_setpoint_rmse_radps"] == [0.0, 0.0, 0.0]


def test_allocator_collective_recovers_missing_unsaturated_publications() -> None:
    mix = np.array(
        [
            [-1.0, 1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ]
    )
    controls = np.tile(mix @ np.array([0.01, 0.02, -0.03, -0.57]), (3, 1))
    controls[2, 0] = 1.0

    collective = _allocator_collective_at_motor_samples(
        motor_controls=controls,
        normalized_mix=mix,
        logged_collective=np.array([0.56, 0.56, 0.56]),
        logged_sample_matches=np.array([False, True, False]),
    )

    np.testing.assert_allclose(collective, [0.57, 0.56, 0.56])


def test_controller_handoff_join_reproduces_independent_work_queue_order() -> None:
    joined, attitude_delay, allocator_delay = _controller_handoff_indices(
        replay_indices=np.array([2, 3, 4, 5]),
        current_attitude_collective=np.array([0.6, 0.7, 0.8, 0.9]),
        previous_attitude_collective=np.array([0.5, 0.6, 0.7, 0.8]),
        current_rates_collective=np.array([0.6, 0.6, 0.8, 0.8]),
        previous_rates_collective=np.array([0.5, 0.5, 0.7, 0.7]),
        allocator_collective=np.array([0.6, 0.6, 0.7, 0.7]),
    )

    assert joined.tolist() == [2, 2, 3, 3]
    assert attitude_delay.tolist() == [0, 1, 0, 1]
    assert allocator_delay.tolist() == [0, 0, 1, 1]


def test_vertical_setpoint_delay_scan_finds_two_tick_outer_phase() -> None:
    target = np.column_stack(
        [
            np.zeros(5),
            np.zeros(5),
            np.arange(5, dtype=np.float64),
        ]
    )

    result = _vertical_setpoint_delay_scan(
        target_acceleration_setpoint_grid_enu=target,
        replay_indices=np.array([2, 3, 4]),
        px4_acceleration_setpoint_z_enu=np.array([0.0, 1.0, 2.0]),
        maximum_delay_steps=3,
    )

    assert result[0]["rmse_mps2"] > 0.0
    assert result[2]["rmse_mps2"] == 0.0
    assert result[2]["max_abs_mps2"] == 0.0


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


def test_collective_boundary_metrics_keep_each_handoff_visible() -> None:
    metrics = _collective_boundary_metrics(
        target_desired_collective=np.array([0.50, 0.60]),
        px4_attitude_collective=np.array([0.50, 0.62]),
        px4_rate_collective=np.array([0.51, 0.62]),
        px4_allocator_collective=np.array([0.51, 0.65]),
    )

    assert metrics == {
        "desired_to_attitude_collective_rmse": pytest.approx(0.02 / np.sqrt(2.0)),
        "attitude_to_rate_collective_rmse": pytest.approx(0.01 / np.sqrt(2.0)),
        "rate_to_allocator_collective_rmse": pytest.approx(0.03 / np.sqrt(2.0)),
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


@pytest.mark.parametrize("compressed", [False, True])
def test_bridge_actuator_samples_join_dynamics_within_one_tick(tmp_path: Path, compressed: bool):
    log = tmp_path / "bridge.log"
    log.write_text(
        "PX4 bridge dynamics: tick=26 sim_time_us=104000 "
        "last_actuator_us=100000 actuator_lag_us=4000 connected=True armed=True "
        "commands=[100.0, 101.0, 102.0, 103.0] "
        "forces_n=[1.0, 1.1, 1.2, 1.3] "
        "body_torque_n_m=[0.0, 0.0, 0.01]\n"
    )

    if compressed:
        # Consumers may retain their existing logical filename via a symlink.
        packed = log.with_suffix(".log.gz")
        packed.write_bytes(gzip.compress(log.read_bytes()))
        log.unlink()
        log.symlink_to(packed.name)
    time_us, rotor_speed, forces, join_and_torque = _bridge_actuator_samples(path=log)

    assert time_us.tolist() == [100000]
    assert rotor_speed.tolist() == [[100.0, 101.0, 102.0, 103.0]]
    assert forces.tolist() == [[1.0, 1.1, 1.2, 1.3]]
    assert join_and_torque.tolist() == [[4000.0, 0.0, 0.0, 0.01]]


@pytest.mark.parametrize("compressed", [False, True])
def test_bridge_state_samples_preserve_physics_tick_inputs(tmp_path: Path, compressed: bool):
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

    if compressed:
        # Consumers may retain their existing logical filename via a symlink.
        packed = log.with_suffix(".log.gz")
        packed.write_bytes(gzip.compress(log.read_bytes()))
        log.unlink()
        log.symlink_to(packed.name)
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
                        "command_time_s": [0.0, 2.5, 5.7],
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

    assert _profile_windows(reference=json.loads(reference.read_text())) == [
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


@pytest.mark.parametrize("cutoff_hz", [0.0, 5.0])
def test_velocity_controller_derivative_filters_velocity_and_resets_invalid_axes(cutoff_hz):
    ulog = SimpleNamespace(
        initial_parameters={"MPC_VEL_LP": 0.0, "MPC_VEL_NF_FRQ": 0.0, "MPC_VELD_LP": cutoff_hz},
        changed_parameters=[],
    )
    local = {
        "timestamp_sample": np.arange(6) * 8_000,
        "vx": np.array([0.0, 0.008, 0.016, 0.024, np.nan, 0.0]),
        "vy": np.arange(6) * 0.016,
        "vz": np.arange(6) * 0.024,
        "v_xy_valid": np.ones(6, dtype=bool),
        "v_z_valid": np.ones(6, dtype=bool),
    }

    actual = _velocity_controller_derivative(ulog=ulog, local_position=local)

    alpha = 1.0 if cutoff_hz == 0.0 else 0.008 / (0.008 + 1.0 / (2.0 * math.pi * cutoff_hz))
    assert actual[3] == pytest.approx(np.array([1.0, 2.0, 3.0]) * (1.0 - (1.0 - alpha) ** 3))
    assert np.all(np.isnan(actual[4, :2]))
    assert actual[5, 0] == 0.0
    assert actual[5, 2] == pytest.approx(3.0 * (1.0 - (1.0 - alpha) ** 5))


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
        initial_parameters={"MPC_VEL_LP": 0.0, "MPC_VEL_NF_FRQ": 0.0, "MPC_VELD_LP": 5.0},
        changed_parameters=[],
        data_list=[
            dataset(
                "vehicle_local_position_setpoint",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "acceleration[0]": np.array([1.0, 3.0]),
                    "acceleration[1]": np.array([2.0, 4.0]),
                    "acceleration[2]": np.array([3.0, 5.0]),
                    "vx": np.array([0.01, 0.03]),
                    "vy": np.array([0.02, 0.04]),
                    "vz": np.array([0.03, 0.05]),
                },
            ),
            dataset(
                "vehicle_local_position",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "timestamp_sample": np.array([100_000, 108_000]),
                    "v_xy_valid": np.ones(2, dtype=bool),
                    "v_z_valid": np.ones(2, dtype=bool),
                    "ax": np.array([2.0, 4.0]),
                    "ay": np.array([4.0, 8.0]),
                    "az": np.array([6.0, 12.0]),
                    "vx": np.array([0.1, 0.3]),
                    "vy": np.array([0.2, 0.4]),
                    "vz": np.array([0.3, 0.5]),
                },
            ),
            dataset(
                "vehicle_attitude_setpoint",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "q_d[0]": np.array([1.0, 1.0]),
                    "q_d[1]": np.array([0.0, 0.0]),
                    "q_d[2]": np.array([0.0, 0.0]),
                    "q_d[3]": np.array([0.0, 0.0]),
                    "yaw_sp_move_rate": np.array([0.2, 0.4]),
                    "thrust_body[2]": np.array([-0.56, -0.58]),
                },
            ),
            dataset(
                "vehicle_rates_setpoint",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "roll": np.array([0.1, 0.3]),
                    "pitch": np.array([0.2, 0.4]),
                    "yaw": np.array([0.3, 0.5]),
                    "thrust_body[2]": np.array([-0.55, -0.57]),
                },
            ),
            dataset(
                "vehicle_torque_setpoint",
                {
                    "timestamp": np.array([100_000, 108_000]),
                    "xyz[0]": np.array([0.02, 0.04]),
                    "xyz[1]": np.array([0.03, 0.05]),
                    "xyz[2]": np.array([0.04, 0.06]),
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
        ],
    )

    report = _augment_reference_controller_state(  # noqa: SLF001 - focused diagnostic test
        ulog=ulog,
        reference_path=reference,
    )
    payload = json.loads(reference.read_text())
    trace = payload["traces"][0]

    assert (
        report["method"] == payload["controller_state_method"] == "px4-ulog-controller-boundary-v2"
    )
    assert payload["schema_version"] == 1
    assert trace["initial_controller_state_valid"]
    assert trace["initial_estimated_linear_velocity_enu_mps"] == pytest.approx([0.3, 0.2, -0.4])
    assert trace["initial_velocity_setpoint_enu_mps"] == pytest.approx([0.02, 0.01, -0.03])
    assert trace["initial_estimated_angular_velocity_body_flu_radps"] == pytest.approx(
        [0.2, -0.3, -0.4]
    )
    assert trace["initial_acceleration_setpoint_enu_mps2"] == pytest.approx([2.0, 1.0, -3.0])
    alpha = 0.008 / (0.008 + 1.0 / (2.0 * math.pi * 5.0))
    first_derivative = alpha * np.array([0.2, 0.1, -0.3]) / 0.008
    second_derivative = first_derivative + alpha * (
        np.array([0.2, 0.2, -0.2]) / 0.008 - first_derivative
    )
    assert trace["initial_linear_acceleration_enu_mps2"] == pytest.approx(
        (first_derivative + second_derivative) / 2.0
    )
    assert trace["initial_angular_acceleration_body_flu_radps2"] == [
        2.0,
        -3.0,
        -4.0,
    ]
    assert trace["initial_rate_integral_body_flu"] == pytest.approx([0.1, -0.2, -0.3])
    assert trace["initial_hover_thrust_normalized"] == pytest.approx(0.57)
    assert trace["initial_desired_attitude_flu_enu_wxyz"] == pytest.approx(
        [2**-0.5, 0.0, 0.0, 2**-0.5]
    )
    assert trace["initial_desired_yaw_rate_radps"] == pytest.approx(-0.2)
    assert trace["initial_desired_collective_thrust_normalized"] == pytest.approx(0.56)
    assert trace["initial_collective_thrust_normalized"] == pytest.approx(0.55)
    assert trace["initial_body_rate_setpoint_flu_radps"] == pytest.approx([0.1, -0.2, -0.3])
    assert trace["initial_filtered_yaw_torque_normalized"] == pytest.approx(-0.04)
    assert trace["initial_controller_state_source_age_us"] == {
        "vehicle_local_position_setpoint": 4000,
        "vehicle_local_position": 4000,
        "vehicle_attitude_setpoint": 4000,
        "vehicle_rates_setpoint": 4000,
        "vehicle_torque_setpoint": 4000,
        "vehicle_angular_velocity": 4000,
        "rate_ctrl_status": 4000,
        "hover_thrust_estimate": 4000,
    }
