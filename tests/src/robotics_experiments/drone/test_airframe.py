from pathlib import Path

import pytest

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
