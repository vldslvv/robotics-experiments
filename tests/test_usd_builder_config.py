import argparse
import importlib
from pathlib import Path

import pytest

pytest.importorskip("hydra")
pytest.importorskip("omegaconf")
pytest.importorskip("pxr")

from pxr import Usd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
_BUILDER_PATH = _REPO_ROOT / "isaac/runtime/standalone/build_copter_usd_standalone.py"
_CONFIG_DIR = _REPO_ROOT / "configs/usd_builder"


def _load_builder_module():
    return importlib.import_module("isaac.runtime.standalone.build_copter_usd_standalone")


def test_usd_builder_default_hydra_config_loads_quadcopter():
    builder = _load_builder_module()

    config = builder.load_builder_config(config_dir=_CONFIG_DIR)

    assert config.output_dir == "isaac/assets/LightSwitchWarehouse"
    assert config.pusher_drive.travel == 0.015
    assert config.drone.name == "quadcopter_500mm"
    assert config.drone.output_filename == "quadcopter_500mm.usda"


def test_usd_builder_hydra_override_loads_hexacopter():
    builder = _load_builder_module()

    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=["drone=hexacopter_s550"],
    )

    assert config.drone.name == "hexacopter_s550"
    assert config.drone.prim_name == "hexacopter"
    assert len(config.drone.motors) == 6


def test_usd_builder_quad_px4_motor_metadata_matches_generic_quad_x_order():
    builder = _load_builder_module()
    config = builder.load_builder_config(config_dir=_CONFIG_DIR)

    assert [motor.px4_motor_index for motor in config.drone.motors] == [2, 1, 3, 0]
    assert [motor.spin_dir for motor in config.drone.motors] == ["CCW", "CW", "CCW", "CW"]


def test_usd_builder_positional_cli_shorthand_maps_to_hydra_override():
    builder = _load_builder_module()
    args = argparse.Namespace(
        config_name="hexacopter_s550",
        overrides=["pusher_drive.stiffness=900.0"],
        output_dir="/tmp/usd-out",
    )

    assert builder._hydra_overrides_from_args(args) == [
        "drone=hexacopter_s550",
        "pusher_drive.stiffness=900.0",
        "output_dir=/tmp/usd-out",
    ]


def test_usd_builder_hydra_override_cli_does_not_force_default_drone():
    builder = _load_builder_module()
    args = argparse.Namespace(
        config_name=None,
        overrides=["drone=hexacopter_s550"],
        output_dir=None,
    )

    assert builder._hydra_overrides_from_args(args) == ["drone=hexacopter_s550"]


def test_usd_builder_generates_quad_with_rotor_metadata(tmp_path):
    builder = _load_builder_module()
    config = builder.load_builder_config(
        config_dir=_CONFIG_DIR,
        overrides=[f"output_dir={tmp_path}"],
    )

    stage_path = builder.build_drone_usd(
        config.drone,
        output_dir=config.output_dir,
        pusher_drive=config.pusher_drive,
    )
    stage = Usd.Stage.Open(stage_path)

    assert stage.GetDefaultPrim().GetName() == "quadcopter"
    expected = {
        "/quadcopter/body/propulsion/motor_4/prop": (0, "CW"),
        "/quadcopter/body/propulsion/motor_2/prop": (1, "CW"),
        "/quadcopter/body/propulsion/motor_1/prop": (2, "CCW"),
        "/quadcopter/body/propulsion/motor_3/prop": (3, "CCW"),
    }
    for path, (px4_index, spin_dir) in expected.items():
        prim = stage.GetPrimAtPath(path)
        assert prim.GetAttribute("drone:rotorMarker").Get() is True
        assert prim.GetAttribute("drone:px4MotorIndex").Get() == px4_index
        assert prim.GetAttribute("drone:spinDir").Get() == spin_dir
