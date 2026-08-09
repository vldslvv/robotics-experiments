"""Pure unit tests for the RL actuation seam (velocity + ctbr).

The oracle is the PX4 bridge's own ``PegasusQuadraticThrustModel``: an action is
mapped to per-rotor omega, fed through the thrust model, and the realized body
wrench is reconstructed with the same reduction the Isaac Lab force sink uses.
Assertions are on that public realized wrench, not model internals.
"""

import math
from dataclasses import replace

import numpy as np
import pytest

from robotics_experiments.config import RlActionConfig
from robotics_experiments.controller import (
    ActuationState,
    ControllerSettings,
    MixerGeometry,
    build_actuation_params,
    controller_command_to_rotor_omega,
    hover_action,
    initial_actuation_controller_state,
    normalized_action_to_controller_command,
    rotor_forces_to_body_wrench,
    seed_actuation_controller_state,
    update_hover_thrust_estimate,
)
from robotics_experiments.controller.actuation import (
    _effective_velocity_proportional_gain,
    _normalized_controls_to_omega,
)
from robotics_experiments.px4_bridge import (
    GRAVITY_M_S2,
    ActuatorCommands,
    PegasusQuadraticThrustModel,
    create_pegasus_thrust_model,
)

_MASS_KG = 1.5
_ROTOR_CONSTANT = 8.54858e-6
_ROLLING_MOMENT = 1e-6
_MAX_OMEGA = 1100.0
_ARM_RADIUS = 0.25
_ROT_DIR = [-1.0, -1.0, 1.0, 1.0]


def _step_controller(params, controller, action, state, dt_s, /):
    command = normalized_action_to_controller_command(
        params,
        action,
        state.attitude_flu_enu_wxyz,
    )
    return controller_command_to_rotor_omega(params, controller, command, state, dt_s)


def _quad_geometry() -> MixerGeometry:
    # Generic Quad X order FR, RL, FL, RR in body-FLU (matches rotor_discovery).
    component = _ARM_RADIUS / math.sqrt(2.0)
    positions = np.array(
        [
            [component, -component, 0.0],  # FR (+x, -y)
            [-component, component, 0.0],  # RL (-x, +y)
            [component, component, 0.0],  # FL (+x, +y)
            [-component, -component, 0.0],  # RR (-x, -y)
        ]
    )
    ones = np.ones(4)
    return MixerGeometry(
        positions_body=positions,
        rot_dir=np.array(_ROT_DIR),
        rotor_constant=_ROTOR_CONSTANT * ones,
        rolling_moment_coefficient=_ROLLING_MOMENT * ones,
        min_rotor_velocity=np.zeros(4),
        max_rotor_velocity=_MAX_OMEGA * ones,
        input_offset=np.zeros(4),
        input_scaling=_MAX_OMEGA * ones,
        zero_position_armed=np.zeros(4),
    )


def _thrust_model() -> PegasusQuadraticThrustModel:
    return create_pegasus_thrust_model(
        num_rotors=4,
        rotor_constant=(_ROTOR_CONSTANT,) * 4,
        rolling_moment_coefficient=(_ROLLING_MOMENT,) * 4,
        rot_dir=tuple(_ROT_DIR),
        min_rotor_velocity=(0.0,) * 4,
        max_rotor_velocity=(_MAX_OMEGA,) * 4,
        linear_drag=(0.5, 0.3, 0.0),
    )


def test_velocity_action_is_converted_once_at_command_arrival():
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=_controller_settings(),
        mass=_MASS_KG,
    )
    action = np.array([0.5, 0.0, 0.0, 0.25])
    level = np.array([1.0, 0.0, 0.0, 0.0])
    yaw_90 = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])

    held_command = normalized_action_to_controller_command(params, action, level)
    next_command = normalized_action_to_controller_command(params, action, yaw_90)

    assert held_command == pytest.approx([1.5, 0.0, 0.0, 0.375])
    assert next_command == pytest.approx([0.0, 1.5, 0.0, 0.375], abs=1e-12)


def _action_cfg(kind: str) -> RlActionConfig:
    return RlActionConfig(
        kind=kind,
        action_dim=4,
        max_body_velocity_mps=3.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
    )


def _controller_settings() -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=0.0,
        velocity_controller_rate_hz=125.0,
        kp_vel=(4.0, 4.0, 5.0),
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        ki_vel=(0.4, 0.4, 2.0),
        kd_vel=(0.5, 0.5, 0.5),
        velocity_observation_scale=(1.0, 1.0, 1.0),
        velocity_observation_bias_mps=(0.0, 0.0, 0.0),
        velocity_estimator_bias_decay_hz=0.0,
        velocity_filter_hz=10.0,
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 4.0),
        hover_thrust_normalized=0.57,
        kp_att=(8.0, 8.0, 3.0),
        attitude_yaw_weight=0.4,
        kp_rate=(0.3, 0.3, 0.1),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.003, 0.003, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        angular_velocity_filter_hz=0.0,
        rate_derivative_filter_hz=30.0,
        yaw_torque_filter_hz=2.0,
        rate_integral_limit=(0.3, 0.3, 0.2),
        max_tilt_cmd_deg=30.0,
        min_collective_thrust_fraction=0.05,
        max_collective_thrust_fraction=1.0,
        horizontal_thrust_margin_fraction=0.3,
        thruster_tau_inc_s=0.0,
        thruster_tau_dec_s=0.0,
        thruster_max_thrust_rate_nps=100000.0,
    )


def _params(kind: str):
    return build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg(kind),
        controller_settings=_controller_settings(),
        mass=_MASS_KG,
    )


def _level_rest_state():
    return ActuationState(
        attitude_flu_enu_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_enu=np.zeros(3),
        angular_velocity_body_flu=np.zeros(3),
    )


def _realized_wrench(params, action, state):
    geometry = _quad_geometry()
    _controller_state, omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        action,
        state,
        0.004,
    )
    dynamics = _thrust_model().commands_to_dynamics(ActuatorCommands(controls=omega))
    return rotor_forces_to_body_wrench(
        geometry.positions_body,
        dynamics.forces_n,
        dynamics.body_torque_n_m,
        dynamics.body_drag_n,
    )


def test_velocity_hover_action_uses_normalized_controller_collective():
    params = _params("velocity")
    force_body, torque_body = _realized_wrench(params, hover_action(params), _level_rest_state())

    expected = float(
        np.sum(
            params.geometry.rotor_constant
            * np.square(params.geometry.input_scaling * params.hover_thrust_normalized)
        )
    )
    assert force_body[2] == pytest.approx(expected, rel=1e-6)
    assert abs(force_body[0]) < 1e-9
    assert abs(force_body[1]) < 1e-9
    assert np.allclose(torque_body, 0.0, atol=1e-9)


def test_velocity_normalized_hover_sets_collective_feed_forward():
    settings = replace(
        _controller_settings(),
        kp_vel=(0.0, 0.0, 0.0),
        ki_vel=(0.0, 0.0, 0.0),
        kd_vel=(0.0, 0.0, 0.0),
    )
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=settings,
        mass=_MASS_KG,
    )

    force_body, torque_body = _realized_wrench(params, np.zeros(4), _level_rest_state())

    expected = float(
        np.sum(
            params.geometry.rotor_constant
            * np.square(params.geometry.input_scaling * params.hover_thrust_normalized)
        )
    )
    assert force_body[2] == pytest.approx(expected)
    assert np.allclose(torque_body, 0.0, atol=1e-9)


def test_horizontal_velocity_gain_schedule_is_neutral_smooth_and_saturating():
    settings = replace(
        _controller_settings(),
        horizontal_velocity_error_gain_scale=2.0,
        horizontal_velocity_error_gain_start_mps=0.1,
        horizontal_velocity_error_gain_full_mps=0.3,
    )
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=settings,
        mass=_MASS_KG,
    )

    start = _effective_velocity_proportional_gain(params, np.array([0.1, 0.0, 1.0]))
    midpoint = _effective_velocity_proportional_gain(params, np.array([0.2, 0.0, 1.0]))
    full = _effective_velocity_proportional_gain(params, np.array([0.3, 0.0, 1.0]))

    assert start == pytest.approx([4.0, 4.0, 5.0])
    assert midpoint == pytest.approx([6.0, 6.0, 5.0])
    assert full == pytest.approx([8.0, 8.0, 5.0])


def test_normalized_hover_maps_through_bridge_affine_rotor_curve():
    geometry = replace(
        _quad_geometry(),
        input_scaling=np.full(4, 475.0),
        zero_position_armed=np.full(4, 50.0),
        max_rotor_velocity=np.full(4, 525.0),
    )
    settings = replace(
        _controller_settings(),
        hover_thrust_normalized=0.57,
        kp_vel=(0.0, 0.0, 0.0),
        ki_vel=(0.0, 0.0, 0.0),
        kd_vel=(0.0, 0.0, 0.0),
    )
    params = build_actuation_params(
        geometry=geometry,
        action_cfg=_action_cfg("velocity"),
        controller_settings=settings,
        mass=_MASS_KG,
    )

    _controller, omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        np.zeros(4),
        _level_rest_state(),
        0.004,
    )
    max_total_thrust = float(
        np.sum(geometry.rotor_constant * np.square(geometry.max_rotor_velocity))
    )
    physical_collective_fraction = float(
        np.sum(geometry.rotor_constant * np.square(omega)) / max_total_thrust
    )

    assert np.allclose(omega, 50.0 + 475.0 * 0.57)
    assert physical_collective_fraction == pytest.approx((320.75 / 525.0) ** 2)


def test_collective_preserves_desired_thrust_magnitude_during_attitude_error():
    settings = replace(
        _controller_settings(),
        hover_thrust_normalized=0.57,
        kp_vel=(0.0, 0.0, 0.0),
        ki_vel=(0.0, 0.0, 0.0),
        kd_vel=(0.0, 0.0, 0.0),
    )
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=settings,
        mass=_MASS_KG,
    )
    tilt_rad = math.radians(8.0)
    tilted = replace(
        _level_rest_state(),
        attitude_flu_enu_wxyz=np.array(
            [math.cos(tilt_rad / 2.0), math.sin(tilt_rad / 2.0), 0.0, 0.0]
        ),
    )

    _level_controller, level_omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        np.zeros(4),
        _level_rest_state(),
        0.004,
    )
    _tilted_controller, tilted_omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        np.zeros(4),
        tilted,
        0.004,
    )

    assert float(np.mean(tilted_omega)) == pytest.approx(float(np.mean(level_omega)))


def test_normalized_allocator_matches_px4_mix_and_sequential_desaturation():
    settings = replace(_controller_settings(), hover_thrust_normalized=0.57)
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=settings,
        mass=_MASS_KG,
    )

    unsaturated = _normalized_controls_to_omega(
        params,
        0.57,
        np.array([0.1, 0.0, 0.0]),
    )
    saturated = _normalized_controls_to_omega(
        params,
        0.9,
        np.array([1.0, 0.0, 0.0]),
    )

    assert unsaturated / _MAX_OMEGA == pytest.approx(
        [0.4992893219, 0.6407106781, 0.6407106781, 0.4992893219]
    )
    assert saturated / _MAX_OMEGA == pytest.approx([0.0, 1.0, 1.0, 0.0], abs=1e-12)


def test_normalized_velocity_anti_windup_blocks_vertical_saturation():
    settings = replace(
        _controller_settings(),
        hover_thrust_normalized=0.57,
        kp_vel=(0.0, 0.0, 10.0),
        ki_vel=(0.0, 0.0, 5.0),
        kd_vel=(0.0, 0.0, 0.0),
        max_collective_thrust_fraction=0.4,
    )
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=settings,
        mass=_MASS_KG,
    )
    controller = initial_actuation_controller_state()
    for _step in range(100):
        controller, omega = _step_controller(
            params,
            controller,
            np.array([0.0, 0.0, 1.0, 0.0]),
            _level_rest_state(),
            0.004,
        )

    assert controller.velocity_integral_enu[2] == pytest.approx(0.0)
    assert np.all((omega >= 0.0) & (omega <= _MAX_OMEGA))


def test_ctbr_hover_action_uses_normalized_controller_collective():
    params = _params("ctbr")
    force_body, torque_body = _realized_wrench(params, hover_action(params), _level_rest_state())

    expected = float(
        np.sum(
            params.geometry.rotor_constant
            * np.square(params.geometry.input_scaling * params.hover_thrust_normalized)
        )
    )
    assert force_body[2] == pytest.approx(expected, rel=1e-6)
    assert np.allclose(torque_body, 0.0, atol=1e-9)


def test_velocity_forward_command_tilts_toward_plus_x():
    params = _params("velocity")
    forward_action = np.array([1.0, 0.0, 0.0, 0.0])
    force_body, torque_body = _realized_wrench(params, forward_action, _level_rest_state())

    # Forward (+x) motion requires a nose-down pitch = a positive body torque
    # about +y; roll and yaw stay ~0, and collective still holds weight.
    assert torque_body[1] > 1e-3
    assert abs(torque_body[0]) < 1e-4
    assert abs(torque_body[2]) < 1e-4
    assert force_body[2] > 0.0


def test_velocity_left_command_tilts_toward_plus_y():
    params = _params("velocity")
    left_action = np.array([0.0, 1.0, 0.0, 0.0])
    _force_body, torque_body = _realized_wrench(params, left_action, _level_rest_state())

    # Left (+y) motion requires a roll toward -x about ... a negative body torque
    # about +x (right-hand rule: +x torque tilts +z toward -y).
    assert torque_body[0] < -1e-3
    assert abs(torque_body[1]) < 1e-4


def test_velocity_yaw_rate_command_produces_positive_yaw_torque():
    params = _params("velocity")
    yaw_action = np.array([0.0, 0.0, 0.0, 0.5])
    _force_body, torque_body = _realized_wrench(
        params,
        yaw_action,
        _level_rest_state(),
    )

    assert torque_body[2] > 1e-4


def test_rate_integral_affects_the_following_control_cycle():
    settings = replace(
        _controller_settings(),
        kp_rate=(0.0, 0.0, 0.0),
        ki_rate=(1.0, 0.0, 0.0),
        kd_rate=(0.0, 0.0, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
    )
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("ctbr"),
        controller_settings=settings,
        mass=_MASS_KG,
    )
    action = np.array([0.0, 0.1, 0.0, 0.0])
    controller, first_omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        action,
        _level_rest_state(),
        0.004,
    )
    _controller, second_omega = _step_controller(
        params,
        controller,
        action,
        _level_rest_state(),
        0.004,
    )

    assert np.allclose(first_omega, first_omega[0])
    assert not np.allclose(second_omega, second_omega[0])


def test_velocity_integral_back_calculates_against_thrust_saturation():
    settings = replace(
        _controller_settings(),
        ki_vel=(5.0, 5.0, 5.0),
        horizontal_velocity_error_gain_scale=8.0 / 7.0,
        horizontal_velocity_error_gain_start_mps=0.1,
        horizontal_velocity_error_gain_full_mps=0.3,
        max_collective_thrust_fraction=0.4,
        horizontal_thrust_margin_fraction=0.1,
    )
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=settings,
        mass=_MASS_KG,
    )
    controller_state = initial_actuation_controller_state()
    action = np.array([1.0, 0.0, 0.0, 0.0])
    for _step in range(100):
        controller_state, _omega = _step_controller(
            params,
            controller_state,
            action,
            _level_rest_state(),
            0.004,
        )

    assert controller_state.velocity_integral_enu[0] < 0.0
    assert (
        abs(controller_state.velocity_integral_enu[0]) <= settings.velocity_integral_limit_mps2[0]
    )


def test_controller_filters_are_sequential_and_fresh_state_resets_them():
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(
            _controller_settings(),
            hover_thrust_normalized=0.57,
            velocity_controller_rate_hz=250.0,
        ),
        mass=_MASS_KG,
    )
    controller_state, _omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        np.zeros(4),
        _level_rest_state(),
        0.004,
    )
    moving = replace(
        _level_rest_state(),
        linear_velocity_enu=np.array([1.0, 0.0, 0.0]),
    )
    controller_state, _omega = _step_controller(
        params,
        controller_state,
        np.zeros(4),
        moving,
        0.004,
    )

    assert 0.0 < controller_state.filtered_velocity_enu[0] < 1.0
    assert controller_state.acceleration_enu[0] > 0.0
    reset = initial_actuation_controller_state()
    assert not reset.initialized
    assert np.array_equal(reset.filtered_velocity_enu, np.zeros(3))
    assert np.array_equal(reset.velocity_estimator_bias_enu, np.zeros(3))


def test_zero_velocity_filter_cutoff_disables_filtering():
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(
            _controller_settings(),
            velocity_controller_rate_hz=250.0,
            velocity_filter_hz=0.0,
        ),
        mass=_MASS_KG,
    )
    controller, _omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        np.zeros(4),
        _level_rest_state(),
        0.004,
    )
    moving = replace(
        _level_rest_state(),
        linear_velocity_enu=np.array([1.0, 0.0, 0.0]),
    )
    controller, _omega = _step_controller(
        params,
        controller,
        np.zeros(4),
        moving,
        0.004,
    )

    assert controller.filtered_velocity_enu[0] == pytest.approx(1.0)


def test_velocity_outer_loop_holds_between_declared_rate_updates():
    params = _params("velocity")
    controller, _omega = _step_controller(
        params,
        initial_actuation_controller_state(),
        np.zeros(4),
        _level_rest_state(),
        0.004,
    )
    moving = replace(_level_rest_state(), linear_velocity_enu=np.array([1.0, 0.0, 0.0]))

    held, _omega = _step_controller(params, controller, np.zeros(4), moving, 0.004)
    updated, _omega = _step_controller(params, held, np.zeros(4), moving, 0.004)

    assert np.array_equal(held.filtered_velocity_enu, controller.filtered_velocity_enu)
    assert not np.array_equal(updated.filtered_velocity_enu, controller.filtered_velocity_enu)


def test_hover_estimate_update_preserves_thrust_with_integrator_correction():
    params = _params("velocity")
    state = seed_actuation_controller_state(
        params=params,
        controller_dt_s=0.004,
        physical_linear_velocity_enu=np.zeros(3),
        filtered_linear_velocity_enu=np.zeros(3),
        angular_velocity_body_flu=np.zeros(3),
        acceleration_setpoint_enu_mps2=np.array([0.0, 0.0, 0.9]),
        linear_acceleration_enu_mps2=np.zeros(3),
        angular_acceleration_body_flu_radps2=np.zeros(3),
        rate_integral_body_flu=np.zeros(3),
        hover_thrust_normalized=0.55,
        outer_loop_elapsed_s=0.0,
    )

    updated = update_hover_thrust_estimate(
        params=params,
        controller_state=state,
        hover_thrust_normalized=0.56,
    )

    expected_delta = (0.9 + GRAVITY_M_S2) * (0.55 / 0.56 - 1.0)
    assert updated.velocity_integral_enu[2] == pytest.approx(
        state.velocity_integral_enu[2] + expected_delta
    )
    assert updated.hover_thrust_normalized == 0.56


@pytest.mark.parametrize("kind", ["velocity", "ctbr"])
def test_seed_controller_state_matches_neutral_acceleration_boundary(kind: str):
    params = _params(kind)
    filtered_velocity = np.array([0.08, -0.18, 0.04])
    acceleration = np.array([0.3, -0.4, 0.2])
    acceleration_setpoint = np.array([0.7, -0.8, 0.9])
    angular_velocity = np.array([0.01, -0.02, 0.03])
    angular_acceleration = np.array([0.1, -0.2, 0.3])
    rate_integral = np.array([0.04, -0.05, 0.06])

    state = seed_actuation_controller_state(
        params=params,
        controller_dt_s=0.004,
        physical_linear_velocity_enu=filtered_velocity,
        filtered_linear_velocity_enu=filtered_velocity,
        angular_velocity_body_flu=angular_velocity,
        acceleration_setpoint_enu_mps2=acceleration_setpoint,
        linear_acceleration_enu_mps2=acceleration,
        angular_acceleration_body_flu_radps2=angular_acceleration,
        rate_integral_body_flu=rate_integral,
        hover_thrust_normalized=0.572,
        outer_loop_elapsed_s=-0.004,
    )
    reconstructed = (
        params.kp_vel * -filtered_velocity
        + state.velocity_integral_enu
        - params.kd_vel * acceleration
    )

    assert state.initialized
    assert state.filtered_velocity_enu == pytest.approx(filtered_velocity)
    assert reconstructed == pytest.approx(acceleration_setpoint)
    assert state.previous_angular_velocity_body_flu == pytest.approx(angular_velocity)
    assert state.angular_acceleration_body_flu == pytest.approx(angular_acceleration)
    assert state.rate_integral_body_flu == pytest.approx(rate_integral)
    assert state.hover_thrust_normalized == pytest.approx(0.572)
    assert state.outer_loop_elapsed_s == pytest.approx(-0.004)


def test_seeded_estimator_bias_decays_at_velocity_controller_rate():
    steady_bias = np.array([0.001, -0.003, 0.002])
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(
            _controller_settings(),
            velocity_observation_bias_mps=tuple(steady_bias),
            velocity_estimator_bias_decay_hz=2.0,
        ),
        mass=_MASS_KG,
    )
    zero = np.zeros(3)
    controller = seed_actuation_controller_state(
        params=params,
        controller_dt_s=0.004,
        physical_linear_velocity_enu=zero,
        filtered_linear_velocity_enu=np.array([0.1, -0.2, 0.3]),
        angular_velocity_body_flu=zero,
        acceleration_setpoint_enu_mps2=zero,
        linear_acceleration_enu_mps2=zero,
        angular_acceleration_body_flu_radps2=zero,
        rate_integral_body_flu=zero,
        hover_thrust_normalized=0.57,
        outer_loop_elapsed_s=0.0,
    )
    initial_bias = np.array(controller.velocity_estimator_bias_enu)
    assert initial_bias == pytest.approx(np.array([0.1, -0.2, 0.3]) - steady_bias)

    controller, _omega = _step_controller(
        params, controller, np.zeros(4), _level_rest_state(), 0.004
    )
    controller, _omega = _step_controller(
        params, controller, np.zeros(4), _level_rest_state(), 0.004
    )

    expected_decay = math.exp(-2.0 * math.pi * 2.0 / 125.0)
    assert controller.velocity_estimator_bias_enu == pytest.approx(initial_bias * expected_decay)


def test_body_rate_input_filter_uses_the_seeded_estimator_state():
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(
            _controller_settings(),
            angular_velocity_filter_hz=10.0,
        ),
        mass=_MASS_KG,
    )
    zero = np.zeros(3)
    controller = seed_actuation_controller_state(
        params=params,
        controller_dt_s=0.004,
        physical_linear_velocity_enu=zero,
        filtered_linear_velocity_enu=zero,
        angular_velocity_body_flu=zero,
        acceleration_setpoint_enu_mps2=zero,
        linear_acceleration_enu_mps2=zero,
        angular_acceleration_body_flu_radps2=zero,
        rate_integral_body_flu=zero,
        hover_thrust_normalized=0.57,
        outer_loop_elapsed_s=0.0,
    )
    rotating = replace(
        _level_rest_state(),
        angular_velocity_body_flu=np.ones(3),
    )

    controller, _omega = _step_controller(params, controller, np.zeros(4), rotating, 0.004)

    assert np.all(controller.previous_angular_velocity_body_flu > 0.0)
    assert np.all(controller.previous_angular_velocity_body_flu < 1.0)


def test_seeded_hover_estimate_controls_neutral_collective() -> None:
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(
            _controller_settings(),
            hover_thrust_normalized=0.57,
        ),
        mass=_MASS_KG,
    )
    zero = np.zeros(3)

    def neutral_omega(hover: float) -> np.ndarray:
        controller = seed_actuation_controller_state(
            params=params,
            controller_dt_s=0.004,
            physical_linear_velocity_enu=zero,
            filtered_linear_velocity_enu=zero,
            angular_velocity_body_flu=zero,
            acceleration_setpoint_enu_mps2=zero,
            linear_acceleration_enu_mps2=zero,
            angular_acceleration_body_flu_radps2=zero,
            rate_integral_body_flu=zero,
            hover_thrust_normalized=hover,
            outer_loop_elapsed_s=0.0,
        )
        _successor, omega = _step_controller(
            params,
            controller,
            np.zeros(4),
            _level_rest_state(),
            0.004,
        )
        return omega

    assert np.all(neutral_omega(0.572) > neutral_omega(0.57))


def test_lab_body_wrench_translates_origin_torque_to_com():
    torch = pytest.importorskip("torch")
    from robotics_experiments.workflows.isaac_lab.tasks.flight import apply_body_wrench

    class Articulation:
        def set_external_force_and_torque(
            self,
            *,
            forces,
            torques,
            body_ids,
        ):
            self.arguments = forces, torques, body_ids

    articulation = Articulation()
    com_position = torch.tensor([[0.25, 0.0, 0.0]])
    force = torch.tensor([[0.0, 0.0, 3.0]])
    torque = torch.tensor([[4.0, 5.0, 6.0]])

    apply_body_wrench(
        articulation=articulation,
        body_id=2,
        body_com_position_body=com_position,
        force_body=force,
        torque_body=torque,
    )

    forces, torques, body_ids = articulation.arguments
    assert torch.equal(forces, force.unsqueeze(1))
    assert torch.equal(torques, torch.tensor([[[4.0, 5.75, 6.0]]]))
    assert body_ids == [2]


def test_airborne_thruster_reset_primes_hover_and_preserves_fitted_time_constants():
    torch = pytest.importorskip("torch")
    from robotics_experiments.workflows.isaac_lab.tasks.flight import (
        _reset_thruster_for_airborne_state,
    )

    class Thruster:
        def __init__(self):
            self.tau_inc_s = torch.tensor([[0.01], [0.02], [0.04]])
            self.tau_dec_s = torch.tensor([[0.08], [0.04], [0.02]])
            self.curr_thrust = torch.ones((3, 1))

        def reset_idx(self, env_ids, /):
            self.tau_inc_s[env_ids] = -1.0
            self.tau_dec_s[env_ids] = -1.0
            self.curr_thrust[env_ids] = 0.0

    thruster = Thruster()

    _reset_thruster_for_airborne_state(
        thruster=thruster,
        env_ids=torch.tensor([0, 2]),
        hover_thrust_per_rotor_n=5.75,
    )

    assert torch.equal(thruster.tau_inc_s, torch.tensor([[0.01], [0.02], [0.04]]))
    assert torch.equal(thruster.tau_dec_s, torch.tensor([[0.08], [0.04], [0.02]]))
    assert torch.equal(thruster.curr_thrust, torch.tensor([[5.75], [1.0], [5.75]]))


@pytest.mark.parametrize("kind", ["velocity", "ctbr"])
def test_seeded_controller_primes_nonzero_rotor_thrust_without_advancing_state(kind: str):
    torch = pytest.importorskip("torch")
    from robotics_experiments.workflows.isaac_lab.tasks.flight import (
        _initial_rotor_thrust_from_controller_state,
        build_flight_actuation_params,
        initial_flight_controller_state,
        seed_flight_controller_state,
    )

    geometry = replace(
        _quad_geometry(),
        input_scaling=np.full(4, 475.0),
        zero_position_armed=np.full(4, 50.0),
        max_rotor_velocity=np.full(4, 525.0),
    )
    settings = replace(
        _controller_settings(),
        hover_thrust_normalized=0.57,
        velocity_estimator_bias_decay_hz=0.02,
        angular_velocity_filter_hz=10.0,
        horizontal_velocity_error_gain_scale=8.0 / 7.0,
        horizontal_velocity_error_gain_start_mps=0.1,
        horizontal_velocity_error_gain_full_mps=0.3,
    )
    params = build_flight_actuation_params(
        kind=kind,
        positions_body=geometry.positions_body,
        rot_dir=geometry.rot_dir,
        rotor_constant=geometry.rotor_constant,
        rolling_moment_coefficient=geometry.rolling_moment_coefficient,
        min_rotor_velocity=geometry.min_rotor_velocity,
        max_rotor_velocity=geometry.max_rotor_velocity,
        input_offset=geometry.input_offset,
        input_scaling=geometry.input_scaling,
        zero_position_armed=geometry.zero_position_armed,
        action_cfg=_action_cfg(kind),
        controller_settings=settings,
        policy_contract=None,
        mass=_MASS_KG,
        controller_dt_s=0.004,
        device="cpu",
    )
    zero = torch.zeros((1, 3))
    state = seed_flight_controller_state(
        state=initial_flight_controller_state(num_envs=1, device="cpu"),
        params=params,
        physical_linear_velocity_world=zero,
        filtered_linear_velocity_world=zero,
        filtered_angular_velocity_body=zero,
        acceleration_setpoint_world=zero,
        linear_acceleration_world=zero,
        angular_acceleration_body=zero,
        rate_integral_body=zero,
        hover_thrust_normalized=torch.tensor([0.572]),
        outer_loop_elapsed_s=torch.tensor([-0.004]),
        valid=torch.tensor([True]),
    )
    assert state.outer_loop_elapsed_s.item() == pytest.approx(-0.004)
    velocity_integral_before = state.velocity_integral_world.clone()

    thrust = _initial_rotor_thrust_from_controller_state(
        params=params,
        state=state,
        action=torch.zeros((1, 4)),
        attitude_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        linear_velocity_world=zero,
        angular_velocity_body=zero,
    )

    assert thrust.shape == (1, 4)
    assert torch.all(thrust > 0.0)
    assert torch.equal(state.velocity_integral_world, velocity_integral_before)


def test_numpy_and_torch_cascades_match_over_a_stateful_sequence():
    torch = pytest.importorskip("torch")
    from robotics_experiments.workflows.isaac_lab.tasks.flight import (
        _normalized_action_to_controller_command as torch_action_to_command,
    )
    from robotics_experiments.workflows.isaac_lab.tasks.flight import (
        build_flight_actuation_params,
        initial_flight_controller_state,
        seed_flight_controller_state,
    )
    from robotics_experiments.workflows.isaac_lab.tasks.flight import (
        controller_command_to_rotor_omega as torch_controller_step,
    )

    geometry = replace(
        _quad_geometry(),
        input_scaling=np.full(4, 475.0),
        zero_position_armed=np.full(4, 50.0),
        max_rotor_velocity=np.full(4, 525.0),
    )
    settings = replace(
        _controller_settings(),
        hover_thrust_normalized=0.57,
        velocity_observation_bias_mps=(0.001, -0.003, 0.002),
    )
    action_cfg = _action_cfg("velocity")
    numpy_params = build_actuation_params(
        geometry=geometry,
        action_cfg=action_cfg,
        controller_settings=settings,
        mass=_MASS_KG,
    )
    torch_params = build_flight_actuation_params(
        kind="velocity",
        positions_body=geometry.positions_body,
        rot_dir=geometry.rot_dir,
        rotor_constant=geometry.rotor_constant,
        rolling_moment_coefficient=geometry.rolling_moment_coefficient,
        min_rotor_velocity=geometry.min_rotor_velocity,
        max_rotor_velocity=geometry.max_rotor_velocity,
        input_offset=geometry.input_offset,
        input_scaling=geometry.input_scaling,
        zero_position_armed=geometry.zero_position_armed,
        action_cfg=action_cfg,
        controller_settings=settings,
        policy_contract=None,
        mass=_MASS_KG,
        controller_dt_s=0.004,
        device="cpu",
    )
    physical_seed_velocity = np.array([0.16, -0.11, 0.015])
    seed_velocity = np.array([0.18, -0.12, 0.01])
    seed_angular_velocity = np.array([0.01, -0.015, 0.02])
    seed_acceleration_setpoint = np.array([0.04, -0.03, 0.02])
    seed_acceleration = np.array([0.05, -0.04, 0.03])
    seed_angular_acceleration = np.array([0.1, -0.2, 0.3])
    seed_rate_integral = np.array([0.01, -0.02, 0.03])
    numpy_controller = seed_actuation_controller_state(
        params=numpy_params,
        controller_dt_s=0.004,
        physical_linear_velocity_enu=physical_seed_velocity,
        filtered_linear_velocity_enu=seed_velocity,
        angular_velocity_body_flu=seed_angular_velocity,
        acceleration_setpoint_enu_mps2=seed_acceleration_setpoint,
        linear_acceleration_enu_mps2=seed_acceleration,
        angular_acceleration_body_flu_radps2=seed_angular_acceleration,
        rate_integral_body_flu=seed_rate_integral,
        hover_thrust_normalized=0.572,
        outer_loop_elapsed_s=-0.004,
    )
    torch_controller = seed_flight_controller_state(
        state=initial_flight_controller_state(num_envs=1, device="cpu"),
        params=torch_params,
        physical_linear_velocity_world=torch.as_tensor(
            physical_seed_velocity,
            dtype=torch.float32,
        ).reshape(1, 3),
        filtered_linear_velocity_world=torch.as_tensor(
            seed_velocity,
            dtype=torch.float32,
        ).reshape(1, 3),
        filtered_angular_velocity_body=torch.as_tensor(
            seed_angular_velocity,
            dtype=torch.float32,
        ).reshape(1, 3),
        acceleration_setpoint_world=torch.as_tensor(
            seed_acceleration_setpoint,
            dtype=torch.float32,
        ).reshape(1, 3),
        linear_acceleration_world=torch.as_tensor(
            seed_acceleration,
            dtype=torch.float32,
        ).reshape(1, 3),
        angular_acceleration_body=torch.as_tensor(
            seed_angular_acceleration,
            dtype=torch.float32,
        ).reshape(1, 3),
        rate_integral_body=torch.as_tensor(
            seed_rate_integral,
            dtype=torch.float32,
        ).reshape(1, 3),
        hover_thrust_normalized=torch.tensor([0.572]),
        outer_loop_elapsed_s=torch.tensor([-0.004]),
        valid=torch.tensor([True]),
    )
    actions = (
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.4, -0.2, 0.1, 0.3]),
        np.array([-0.1, 0.3, -0.2, -0.4]),
        np.zeros(4),
    )
    attitudes = tuple(
        value / np.linalg.norm(value)
        for value in (
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 0.03, -0.02, 0.05]),
            np.array([1.0, -0.04, 0.02, -0.03]),
            np.array([1.0, 0.01, 0.04, 0.02]),
        )
    )
    for index, action in enumerate(actions):
        state = ActuationState(
            attitude_flu_enu_wxyz=attitudes[index],
            linear_velocity_enu=np.array([0.05 * index, -0.02 * index, 0.01 * index]),
            angular_velocity_body_flu=np.array([0.01 * index, -0.015 * index, 0.02 * index]),
        )
        numpy_controller, numpy_omega = _step_controller(
            numpy_params,
            numpy_controller,
            action,
            state,
            0.004,
        )
        torch_action = torch.as_tensor(action, dtype=torch.float32).reshape(1, 4)
        torch_attitude = torch.as_tensor(
            state.attitude_flu_enu_wxyz,
            dtype=torch.float32,
        ).reshape(1, 4)
        torch_command = torch_action_to_command(
            torch_params,
            torch_action,
            torch_attitude,
        )
        torch_controller, torch_omega = torch_controller_step(
            torch_params,
            torch_controller,
            torch_command,
            torch_attitude,
            torch.as_tensor(
                state.linear_velocity_enu,
                dtype=torch.float32,
            ).reshape(1, 3),
            torch.as_tensor(
                state.angular_velocity_body_flu,
                dtype=torch.float32,
            ).reshape(1, 3),
            0.004,
        )
        assert np.allclose(
            numpy_omega,
            torch_omega[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        assert np.allclose(
            numpy_controller.filtered_velocity_enu,
            torch_controller.filtered_velocity_world[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        assert np.allclose(
            numpy_controller.velocity_estimator_bias_enu,
            torch_controller.velocity_estimator_bias_world[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        assert np.allclose(
            numpy_controller.previous_angular_velocity_body_flu,
            torch_controller.previous_angular_velocity_body[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        assert np.allclose(
            numpy_controller.acceleration_enu,
            torch_controller.acceleration_world[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        assert numpy_controller.hover_thrust_normalized == pytest.approx(
            float(torch_controller.hover_thrust_normalized[0])
        )
        assert np.allclose(
            numpy_controller.velocity_integral_enu,
            torch_controller.velocity_integral_world[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )
        assert np.allclose(
            numpy_controller.angular_acceleration_body_flu,
            torch_controller.angular_acceleration_body[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
        )


def test_build_actuation_params_rejects_unimplemented_kind():
    with pytest.raises(ValueError, match="attitude"):
        build_actuation_params(
            geometry=_quad_geometry(),
            action_cfg=_action_cfg("attitude"),
            controller_settings=_controller_settings(),
            mass=_MASS_KG,
        )


def test_built_actuation_parameter_arrays_are_read_only_and_non_aliased():
    geometry = _quad_geometry()
    params = build_actuation_params(
        geometry=geometry,
        action_cfg=_action_cfg("velocity"),
        controller_settings=_controller_settings(),
        mass=_MASS_KG,
    )
    arrays = (
        params.geometry.positions_body,
        params.geometry.rot_dir,
        params.geometry.rotor_constant,
        params.geometry.rolling_moment_coefficient,
        params.geometry.min_rotor_velocity,
        params.geometry.max_rotor_velocity,
        params.geometry.input_offset,
        params.geometry.input_scaling,
        params.geometry.zero_position_armed,
        params.normalized_allocation_mix,
        params.kp_rate,
        params.ki_rate,
        params.kd_rate,
        params.kff_rate,
        params.rate_integral_limit,
        params.kp_vel,
        params.ki_vel,
        params.kd_vel,
        params.velocity_integral_limit_mps2,
        params.kp_att,
    )

    geometry.positions_body[0, 0] = 99.0
    assert params.geometry.positions_body[0, 0] != 99.0
    for array in arrays:
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 1.0
