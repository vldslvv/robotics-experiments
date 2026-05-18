import pytest


def test_default_config_loads_from_core_package():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import EnvConfig, get_config, reset_config

    reset_config()

    config = get_config()

    assert isinstance(config, EnvConfig)
    assert config.sim.physics_hz > 0
    assert config.drone.prim_name
    reset_config()
