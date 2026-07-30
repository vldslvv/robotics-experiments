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

from robotics_experiments.alignment import ControllerSettings
from robotics_experiments.config import RlActionConfig
from robotics_experiments.px4_bridge import (
    ActuatorCommands,
    PegasusQuadraticThrustModel,
    create_pegasus_thrust_model,
)
from robotics_experiments.rl import (
    ActuationState,
    MixerGeometry,
    action_to_rotor_omega,
    build_actuation_params,
    hover_action,
    initial_actuation_controller_state,
    rotor_forces_to_body_wrench,
    seed_actuation_controller_state,
)
from robotics_experiments.rl.actuation import _normalized_controls_to_omega

_MASS_KG = 1.5
_GRAVITY = 9.81
_ROTOR_CONSTANT = 8.54858e-6
_ROLLING_MOMENT = 1e-6
_MAX_OMEGA = 1100.0
_ARM_RADIUS = 0.25
_ROT_DIR = [-1.0, -1.0, 1.0, 1.0]


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


def _action_cfg(kind: str) -> RlActionConfig:
    return RlActionConfig(
        kind=kind,
        action_dim=4,
        max_body_velocity_mps=3.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
    )


def _controller_settings() -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=0.0,
        kp_vel=(4.0, 4.0, 5.0),
        ki_vel=(0.4, 0.4, 2.0),
        kd_vel=(0.5, 0.5, 0.5),
        velocity_filter_hz=10.0,
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 4.0),
        hover_thrust_fraction=0.0,
        hover_thrust_normalized=0.0,
        kp_att=(8.0, 8.0, 3.0),
        attitude_yaw_weight=0.4,
        kp_rate=(0.3, 0.3, 0.1),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.003, 0.003, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        rate_output_scale_nm=(1.0, 1.0, 1.0),
        rate_derivative_filter_hz=30.0,
        rate_integral_limit_nm=(0.3, 0.3, 0.2),
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
    _controller_state, omega = action_to_rotor_omega(
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


def test_velocity_hover_action_cancels_weight():
    params = _params("velocity")
    force_body, torque_body = _realized_wrench(params, hover_action(params), _level_rest_state())

    assert force_body[2] == pytest.approx(_MASS_KG * _GRAVITY, rel=1e-6)
    assert abs(force_body[0]) < 1e-9
    assert abs(force_body[1]) < 1e-9
    assert np.allclose(torque_body, 0.0, atol=1e-9)


def test_velocity_px4_hover_fraction_sets_collective_feed_forward():
    settings = replace(
        _controller_settings(),
        hover_thrust_fraction=0.57,
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

    assert force_body[2] == pytest.approx(0.57 * params.max_total_thrust)
    assert np.allclose(torque_body, 0.0, atol=1e-9)


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

    _controller, omega = action_to_rotor_omega(
        params,
        initial_actuation_controller_state(),
        np.zeros(4),
        _level_rest_state(),
        0.004,
    )
    physical_collective_fraction = float(
        np.sum(geometry.rotor_constant * np.square(omega)) / params.max_total_thrust
    )

    assert np.allclose(omega, 50.0 + 475.0 * 0.57)
    assert physical_collective_fraction == pytest.approx((320.75 / 525.0) ** 2)


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
        controller, omega = action_to_rotor_omega(
            params,
            controller,
            np.array([0.0, 0.0, 1.0, 0.0]),
            _level_rest_state(),
            0.004,
        )

    assert controller.velocity_integral_enu[2] == pytest.approx(0.0)
    assert np.all((omega >= 0.0) & (omega <= _MAX_OMEGA))


def test_ctbr_hover_action_cancels_weight():
    params = _params("ctbr")
    force_body, torque_body = _realized_wrench(params, hover_action(params), _level_rest_state())

    assert force_body[2] == pytest.approx(_MASS_KG * _GRAVITY, rel=1e-6)
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
    assert force_body[2] == pytest.approx(_MASS_KG * _GRAVITY, rel=1e-3)


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


def test_rate_output_scale_converts_normalized_pid_output_to_physical_torque():
    base = _controller_settings()
    action = np.array([0.0, 0.1, 0.0, 0.0])
    unit_params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("ctbr"),
        controller_settings=base,
        mass=_MASS_KG,
    )
    scaled_params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("ctbr"),
        controller_settings=replace(base, rate_output_scale_nm=(2.0, 1.0, 1.0)),
        mass=_MASS_KG,
    )

    _force, unit_torque = _realized_wrench(unit_params, action, _level_rest_state())
    _force, scaled_torque = _realized_wrench(scaled_params, action, _level_rest_state())

    assert scaled_torque[0] == pytest.approx(2.0 * unit_torque[0])


def test_normalized_velocity_rate_pid_ignores_legacy_physical_torque_scale():
    base = replace(_controller_settings(), hover_thrust_normalized=0.57)
    action = np.array([0.0, 0.0, 0.0, 0.5])
    unit_params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=base,
        mass=_MASS_KG,
    )
    scaled_params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(base, rate_output_scale_nm=(20.0, 20.0, 20.0)),
        mass=_MASS_KG,
    )

    _controller, unit_omega = action_to_rotor_omega(
        unit_params,
        initial_actuation_controller_state(),
        action,
        _level_rest_state(),
        0.004,
    )
    _controller, scaled_omega = action_to_rotor_omega(
        scaled_params,
        initial_actuation_controller_state(),
        action,
        _level_rest_state(),
        0.004,
    )

    assert np.allclose(scaled_omega, unit_omega)


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
    controller, first_omega = action_to_rotor_omega(
        params,
        initial_actuation_controller_state(),
        action,
        _level_rest_state(),
        0.004,
    )
    _controller, second_omega = action_to_rotor_omega(
        params,
        controller,
        action,
        _level_rest_state(),
        0.004,
    )

    assert np.allclose(first_omega, first_omega[0])
    assert not np.allclose(second_omega, second_omega[0])


def test_velocity_integral_does_not_wind_up_against_thrust_saturation():
    settings = replace(
        _controller_settings(),
        ki_vel=(5.0, 5.0, 5.0),
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
        controller_state, _omega = action_to_rotor_omega(
            params,
            controller_state,
            action,
            _level_rest_state(),
            0.004,
        )

    assert controller_state.velocity_integral_enu[0] == pytest.approx(0.0)


def test_controller_filters_are_sequential_and_fresh_state_resets_them():
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(
            _controller_settings(),
            hover_thrust_normalized=0.57,
        ),
        mass=_MASS_KG,
    )
    controller_state, _omega = action_to_rotor_omega(
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
    controller_state, _omega = action_to_rotor_omega(
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


def test_zero_velocity_filter_cutoff_disables_filtering():
    params = build_actuation_params(
        geometry=_quad_geometry(),
        action_cfg=_action_cfg("velocity"),
        controller_settings=replace(_controller_settings(), velocity_filter_hz=0.0),
        mass=_MASS_KG,
    )
    controller, _omega = action_to_rotor_omega(
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
    controller, _omega = action_to_rotor_omega(
        params,
        controller,
        np.zeros(4),
        moving,
        0.004,
    )

    assert controller.filtered_velocity_enu[0] == pytest.approx(1.0)


def test_seed_controller_state_matches_neutral_acceleration_boundary():
    params = _params("velocity")
    velocity = np.array([0.1, -0.2, 0.05])
    acceleration = np.array([0.3, -0.4, 0.2])
    acceleration_setpoint = np.array([0.7, -0.8, 0.9])
    angular_velocity = np.array([0.01, -0.02, 0.03])
    angular_acceleration = np.array([0.1, -0.2, 0.3])
    rate_integral = np.array([0.04, -0.05, 0.06])

    state = seed_actuation_controller_state(
        params=params,
        linear_velocity_enu=velocity,
        angular_velocity_body_flu=angular_velocity,
        acceleration_setpoint_enu_mps2=acceleration_setpoint,
        linear_acceleration_enu_mps2=acceleration,
        angular_acceleration_body_flu_radps2=angular_acceleration,
        rate_integral_body_flu=rate_integral,
    )
    reconstructed = (
        params.kp_vel * -velocity + state.velocity_integral_enu - params.kd_vel * acceleration
    )

    assert state.initialized
    assert reconstructed == pytest.approx(acceleration_setpoint)
    assert state.previous_angular_velocity_body_flu == pytest.approx(angular_velocity)
    assert state.angular_acceleration_body_flu == pytest.approx(angular_acceleration)
    assert state.rate_integral_body_flu == pytest.approx(rate_integral)


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

    apply_body_wrench(articulation, 2, com_position, force, torque)

    forces, torques, body_ids = articulation.arguments
    assert torch.equal(forces, force.unsqueeze(1))
    assert torch.equal(torques, torch.tensor([[[4.0, 5.75, 6.0]]]))
    assert body_ids == [2]


def test_numpy_and_torch_cascades_match_over_a_stateful_sequence():
    torch = pytest.importorskip("torch")
    from robotics_experiments.workflows.isaac_lab.tasks.flight import (
        action_to_rotor_omega as torch_action_to_rotor_omega,
    )
    from robotics_experiments.workflows.isaac_lab.tasks.flight import (
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
        hover_thrust_fraction=0.375,
        hover_thrust_normalized=0.57,
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
    seed_velocity = np.array([0.03, -0.02, 0.01])
    seed_angular_velocity = np.array([0.01, -0.015, 0.02])
    seed_acceleration_setpoint = np.array([0.04, -0.03, 0.02])
    seed_acceleration = np.array([0.05, -0.04, 0.03])
    seed_angular_acceleration = np.array([0.1, -0.2, 0.3])
    seed_rate_integral = np.array([0.01, -0.02, 0.03])
    numpy_controller = seed_actuation_controller_state(
        params=numpy_params,
        linear_velocity_enu=seed_velocity,
        angular_velocity_body_flu=seed_angular_velocity,
        acceleration_setpoint_enu_mps2=seed_acceleration_setpoint,
        linear_acceleration_enu_mps2=seed_acceleration,
        angular_acceleration_body_flu_radps2=seed_angular_acceleration,
        rate_integral_body_flu=seed_rate_integral,
    )
    torch_controller = seed_flight_controller_state(
        state=initial_flight_controller_state(num_envs=1, device="cpu"),
        params=torch_params,
        linear_velocity_world=torch.as_tensor(seed_velocity, dtype=torch.float32).reshape(1, 3),
        angular_velocity_body=torch.as_tensor(
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
        numpy_controller, numpy_omega = action_to_rotor_omega(
            numpy_params,
            numpy_controller,
            action,
            state,
            0.004,
        )
        torch_controller, torch_omega = torch_action_to_rotor_omega(
            torch_params,
            torch_controller,
            torch.as_tensor(action, dtype=torch.float32).reshape(1, 4),
            torch.as_tensor(
                state.attitude_flu_enu_wxyz,
                dtype=torch.float32,
            ).reshape(1, 4),
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
            numpy_controller.acceleration_enu,
            torch_controller.acceleration_world[0].detach().numpy(),
            rtol=2e-5,
            atol=2e-5,
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
        params.allocation_pinv,
        params.normalized_allocation_mix,
        params.kp_rate,
        params.ki_rate,
        params.kd_rate,
        params.kff_rate,
        params.rate_output_scale_nm,
        params.rate_integral_limit_nm,
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
