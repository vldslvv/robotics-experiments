import numpy as np
import pytest

from robotics_experiments.domain.thrust_model import LinearThrustModel


def test_linear_thrust_model_clips_and_scales_commands():
    model = LinearThrustModel(num_rotors=4, max_force_n=8.0)

    forces = model.commands_to_forces(np.array([-1.0, 0.25, 1.0, 2.0]))

    np.testing.assert_allclose(forces, np.array([0.0, 2.0, 8.0, 8.0]))


def test_linear_thrust_model_rejects_wrong_shape():
    model = LinearThrustModel(num_rotors=4, max_force_n=8.0)

    with pytest.raises(ValueError):
        model.commands_to_forces(np.array([0.0, 1.0]))
