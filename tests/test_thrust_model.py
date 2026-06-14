import numpy as np
import pytest

from robotics_experiments.px4_bridge.thrust_model import (
    LinearThrustModel,
    PegasusQuadraticThrustModel,
)


def test_linear_thrust_model_clips_and_scales_commands():
    model = LinearThrustModel(num_rotors=4, max_force_n=8.0)

    forces = model.commands_to_forces(np.array([-1.0, 0.25, 1.0, 2.0]))

    np.testing.assert_allclose(forces, np.array([0.0, 2.0, 8.0, 8.0]))


def test_linear_thrust_model_rejects_wrong_shape():
    model = LinearThrustModel(num_rotors=4, max_force_n=8.0)

    with pytest.raises(ValueError):
        model.commands_to_forces(np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    "commands",
    [
        np.array([0.0, 100.0, 500.0, 1200.0]),
        np.array([75.0, 250.0, 875.0, 1100.0]),
        np.array([-50.0, 0.0, 900.0, 1500.0]),
    ],
)
def test_pegasus_quadratic_model_matches_rotor_force_curve(commands):
    model = PegasusQuadraticThrustModel(num_rotors=4)

    forces = model.commands_to_forces(commands)

    expected = 8.54858e-6 * np.square(np.clip(commands, 0.0, 1100.0))
    np.testing.assert_allclose(forces, expected)


@pytest.mark.parametrize(
    "commands",
    [
        np.array([100.0, 100.0, 300.0, 300.0]),
        np.array([0.0, 500.0, 1000.0, 1200.0]),
        np.array([800.0, 200.0, 200.0, 800.0]),
    ],
)
def test_pegasus_quadratic_model_matches_reaction_torque_formula(commands):
    model = PegasusQuadraticThrustModel(num_rotors=4)

    dynamics = model.commands_to_dynamics(
        commands,
        linear_body_velocity_flu=np.array([2.0, -3.0, 1.0]),
    )

    omega = np.clip(commands, 0.0, 1100.0)
    expected_yaw_torque = 1e-6 * np.sum(
        np.square(omega) * np.array([-1.0, -1.0, 1.0, 1.0])
    )
    np.testing.assert_allclose(dynamics.body_torque_n_m, [0.0, 0.0, expected_yaw_torque])
    np.testing.assert_allclose(dynamics.body_drag_n, [-1.0, 0.9, 0.0])


def test_pegasus_quadratic_model_drag_opposes_body_velocity():
    model = PegasusQuadraticThrustModel(num_rotors=4)

    dynamics = model.commands_to_dynamics(
        np.zeros(4),
        linear_body_velocity_flu=np.array([4.0, 5.0, -6.0]),
    )

    np.testing.assert_allclose(dynamics.body_drag_n, [-2.0, -1.5, 0.0])
