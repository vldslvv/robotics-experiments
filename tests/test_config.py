import pytest


def _write_yaml(path, lines):
    path.write_text("\n".join([*lines, ""]), encoding="utf-8")


def test_default_config_loads_from_root_experiment_config():
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import EnvConfig, get_config, reset_config

    reset_config()

    config = get_config()

    assert isinstance(config, EnvConfig)
    assert config.sim.physics_hz > 0
    assert config.drone.prim_name
    reset_config()


def test_experiment_config_dir_override_loads(tmp_path, monkeypatch):
    pytest.importorskip("hydra")
    pytest.importorskip("omegaconf")
    from robotics_experiments.config import EnvConfig, get_config, reset_config

    conf_dir = tmp_path / "experiment"
    conf_dir.mkdir()
    (conf_dir / "sim").mkdir()
    (conf_dir / "drone").mkdir()
    (conf_dir / "switches").mkdir()
    (conf_dir / "px4").mkdir()
    _write_yaml(
        conf_dir / "config.yaml",
        [
            "defaults:",
            "  - base_env",
            "  - sim: default",
            "  - drone: default",
            "  - switches: default",
            "  - px4: default",
            "  - _self_",
        ],
    )
    _write_yaml(
        conf_dir / "sim" / "default.yaml",
        ["defaults:", "  - base_sim", "  - _self_", "", "physics_hz: 123.0"],
    )
    _write_yaml(
        conf_dir / "drone" / "default.yaml",
        ["defaults:", "  - base_drone", "  - _self_", "", "prim_name: OverrideDrone"],
    )
    _write_yaml(
        conf_dir / "switches" / "default.yaml",
        ["defaults:", "  - base_switches", "  - _self_"],
    )
    _write_yaml(
        conf_dir / "px4" / "default.yaml",
        ["defaults:", "  - base_px4", "  - _self_"],
    )

    monkeypatch.setenv("EXPERIMENT_CONFIG_DIR", str(conf_dir))
    reset_config()

    config = get_config()

    assert isinstance(config, EnvConfig)
    assert config.sim.physics_hz == 123.0
    assert config.drone.prim_name == "OverrideDrone"
    reset_config()
