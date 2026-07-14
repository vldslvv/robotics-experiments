import numpy as np
import pytest

from robotics_experiments.px4_bridge import ActuatorCommands, create_pegasus_thrust_model


@pytest.mark.parametrize(
    "commands",
    [
        np.array([0.0, 100.0, 500.0, 1200.0]),
        np.array([75.0, 250.0, 875.0, 1100.0]),
        np.array([-50.0, 0.0, 900.0, 1500.0]),
    ],
)
def test_pegasus_quadratic_model_matches_rotor_force_curve(commands):
    model = create_pegasus_thrust_model(num_rotors=4)

    forces = model.commands_to_forces(ActuatorCommands(commands))

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
    model = create_pegasus_thrust_model(num_rotors=4)

    dynamics = model.commands_to_dynamics(
        ActuatorCommands(commands),
        linear_body_velocity_flu=np.array([2.0, -3.0, 1.0]),
    )

    omega = np.clip(commands, 0.0, 1100.0)
    expected_yaw_torque = 1e-6 * np.sum(np.square(omega) * np.array([-1.0, -1.0, 1.0, 1.0]))
    np.testing.assert_allclose(dynamics.body_torque_n_m, [0.0, 0.0, expected_yaw_torque])
    np.testing.assert_allclose(dynamics.body_drag_n, [-1.0, 0.9, 0.0])


def test_pegasus_quadratic_model_reports_clipped_omega():
    model = create_pegasus_thrust_model(num_rotors=4)

    commands = np.array([-50.0, 500.0, 1100.0, 1500.0])
    dynamics = model.commands_to_dynamics(ActuatorCommands(commands))

    # rotor_omega_rad_s is the clipped angular-velocity reference used to produce
    # forces, ordered by motor index, and drives only the cosmetic spin.
    expected_omega = np.clip(commands, 0.0, 1100.0)
    np.testing.assert_allclose(dynamics.rotor_omega_rad_s, expected_omega)
    np.testing.assert_allclose(dynamics.forces_n, 8.54858e-6 * np.square(expected_omega))
    for array in (
        dynamics.forces_n,
        dynamics.body_torque_n_m,
        dynamics.body_drag_n,
        dynamics.rotor_omega_rad_s,
    ):
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 1.0


def test_pegasus_quadratic_model_drag_opposes_body_velocity():
    model = create_pegasus_thrust_model(num_rotors=4)

    dynamics = model.commands_to_dynamics(
        ActuatorCommands(np.zeros(4)),
        linear_body_velocity_flu=np.array([4.0, 5.0, -6.0]),
    )

    np.testing.assert_allclose(dynamics.body_drag_n, [-2.0, -1.5, 0.0])


@pytest.mark.parametrize(
    ("argument", "values", "message"),
    [
        ("rotor_constant", [1.0, 1.0, 1.0], r"rotor_constant expects shape \(4,\)"),
        ("linear_drag", [0.5, 0.3], r"linear_drag expects shape \(3,\)"),
    ],
)
def test_pegasus_quadratic_model_rejects_invalid_coefficient_shapes(argument, values, message):
    with pytest.raises(ValueError, match=message):
        create_pegasus_thrust_model(num_rotors=4, **{argument: values})


def test_pegasus_quadratic_model_copies_coefficient_inputs():
    rotor_constant = np.full(4, 8.54858e-6)
    linear_drag = np.array([0.5, 0.3, 0.0])
    model = create_pegasus_thrust_model(
        num_rotors=4,
        rotor_constant=rotor_constant,
        linear_drag=linear_drag,
    )
    commands = ActuatorCommands(np.full(4, 500.0))
    baseline = model.commands_to_dynamics(
        commands,
        linear_body_velocity_flu=np.array([2.0, -3.0, 1.0]),
    )

    rotor_constant[:] = 1.0
    linear_drag[:] = 100.0
    after_mutation = model.commands_to_dynamics(
        commands,
        linear_body_velocity_flu=np.array([2.0, -3.0, 1.0]),
    )

    np.testing.assert_array_equal(after_mutation.forces_n, baseline.forces_n)
    np.testing.assert_array_equal(after_mutation.body_drag_n, baseline.body_drag_n)
