from pathlib import Path

import pytest

from robotics_experiments.config import load_config
from robotics_experiments.drone._airframe import (
    compute_body_mass_properties,
    load_builder_config,
)
from tests.support.paths import REPO_ROOT

_CONFIG_DIR = REPO_ROOT / "configs/usd_builder"


def test_airframe_config_parsing_is_independent_of_usd_bindings() -> None:
    config = load_builder_config(config_dir=_CONFIG_DIR)

    assert config.airframe.name == "holybro_x500_v2"
    assert config.airframe.output_filename == "holybro_x500_v2.usda"
    assert Path(config.output_dir) == Path("isaac/assets/LightSwitchWarehouse")


def test_airframe_mass_properties_are_computed_from_components() -> None:
    airframe = load_builder_config(config_dir=_CONFIG_DIR).airframe

    total_mass, center_of_mass, diagonal_inertia = compute_body_mass_properties(airframe)

    assert airframe.pusher is not None
    assert total_mass == pytest.approx(2.3372)
    assert total_mass + airframe.pusher.pad.mass == pytest.approx(2.3472)
    assert center_of_mass[1] == pytest.approx(0.0, abs=1e-6)
    assert diagonal_inertia[2] > diagonal_inertia[0]
    assert diagonal_inertia[2] > diagonal_inertia[1]


def test_x500_composed_actuator_curve_requires_point_375_physical_hover() -> None:
    airframe = load_builder_config(config_dir=_CONFIG_DIR).airframe
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    bridge = config.vehicles["x500_01"].bridge
    body_mass, _center_of_mass, _diagonal_inertia = compute_body_mass_properties(airframe)

    assert airframe.pusher is not None
    mass_kg = body_mass + airframe.pusher.pad.mass
    max_thrust_n = sum(
        rotor_constant * max_omega**2
        for rotor_constant, max_omega in zip(
            bridge.rotor_constant,
            bridge.max_rotor_velocity,
            strict=True,
        )
    )
    hover_fraction = mass_kg * 9.80665 / max_thrust_n
    hover_rotor_speed = (
        mass_kg * 9.80665 / (len(bridge.rotor_constant) * bridge.rotor_constant[0])
    ) ** 0.5
    hover_motor_control = (
        hover_rotor_speed - bridge.zero_position_armed[0]
    ) / bridge.input_scaling[0] - bridge.input_offset[0]

    assert mass_kg == pytest.approx(2.3472)
    assert max_thrust_n == pytest.approx(61.389629)
    assert bridge.input_scaling == (475.0,) * 4
    assert bridge.zero_position_armed == (50.0,) * 4
    assert bridge.input_offset == (0.0,) * 4
    assert hover_motor_control == pytest.approx(0.5715, abs=0.0002)
    assert hover_fraction == pytest.approx(0.375, abs=0.0001)
