"""Pure unit tests for the RL actuation seam (velocity + ctbr).

The oracle is the PX4 bridge's own ``PegasusQuadraticThrustModel``: an action is
mapped to per-rotor omega, fed through the thrust model, and the realized body
wrench is reconstructed with the same reduction the Isaac Lab force sink uses.
Assertions are on that public realized wrench, not model internals.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from robotics_experiments.px4_bridge import ActuatorCommands, PegasusQuadraticThrustModel
from robotics_experiments.rl import (
    ActuationState,
    MixerGeometry,
    action_to_rotor_omega,
    build_actuation_params,
    hover_action,
    rotor_forces_to_body_wrench,
)

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
    )


def _thrust_model() -> PegasusQuadraticThrustModel:
    return PegasusQuadraticThrustModel(
        num_rotors=4,
        rotor_constant=(_ROTOR_CONSTANT,) * 4,
        rolling_moment_coefficient=(_ROLLING_MOMENT,) * 4,
        rot_dir=tuple(_ROT_DIR),
        min_rotor_velocity=(0.0,) * 4,
        max_rotor_velocity=(_MAX_OMEGA,) * 4,
        linear_drag=(0.5, 0.3, 0.0),
    )


def _action_cfg(kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        kind=kind,
        action_dim=4,
        max_body_velocity_mps=3.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        max_tilt_cmd_deg=30.0,
        hover_thrust_fraction=0.4,
        controller_kp_vel=[4.0, 4.0, 5.0],
        controller_kd_vel=[0.5, 0.5, 0.5],
        controller_kp_att=[8.0, 8.0, 3.0],
        controller_kp_rate=[0.3, 0.3, 0.1],
    )


def _params(kind: str):
    return build_actuation_params(_quad_geometry(), _action_cfg(kind), _MASS_KG)


def _level_rest_state():
    return ActuationState(
        attitude_flu_enu_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_enu=np.zeros(3),
        angular_velocity_body_flu=np.zeros(3),
    )


def _realized_wrench(params, action, state):
    geometry = _quad_geometry()
    omega = action_to_rotor_omega(params, action, state)
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


def test_build_actuation_params_rejects_unimplemented_kind():
    with pytest.raises(ValueError, match="attitude"):
        build_actuation_params(_quad_geometry(), _action_cfg("attitude"), _MASS_KG)
