import json
import subprocess
from pathlib import Path

from tests.support.paths import REPO_ROOT

_ROOT = REPO_ROOT
_GENERATOR = _ROOT / "px4/scripts/generate_px4_airframes.py"
_AIRFRAME_CONFIG_DIR = _ROOT / "configs/airframes"


def _run_generator(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", str(_GENERATOR), *(str(arg) for arg in args)],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )


def test_generator_derives_px4_rotor_allocation_from_airframe_geometry():
    result = _run_generator("--config-dir", _AIRFRAME_CONFIG_DIR, "--print-manifest")
    manifest = json.loads(result.stdout)
    airframes = {airframe["source"]: airframe for airframe in manifest["airframes"]}
    quad = airframes["holybro_x500_v2"]
    hexa = airframes["hexacopter_s550"]

    assert (
        quad["script_name"],
        quad["sim_model"],
        quad["sys_autostart"],
        quad["mav_type"],
    ) == ("4019_none_x500_v2", "none_x500_v2", 4019, 2)
    # X500 V2 500mm diagonal -> motor radius 0.25 -> CA_ROTOR +/-0.25*cos(45).
    assert [(r["index"], r["px"], r["py"], r["km"]) for r in quad["rotors"]] == [
        (0, 0.176777, 0.176777, 0.05),
        (1, -0.176777, -0.176777, 0.05),
        (2, 0.176777, -0.176777, -0.05),
        (3, -0.176777, 0.176777, -0.05),
    ]

    assert (
        hexa["script_name"],
        hexa["sim_model"],
        hexa["sys_autostart"],
        hexa["mav_type"],
    ) == ("6001_none_hexa_x", "none_hexa_x", 6001, 13)
    assert [(r["index"], r["px"], r["py"], r["km"]) for r in hexa["rotors"]] == [
        (0, 0.275, 0.0, -0.05),
        (1, 0.1375, -0.238157, 0.05),
        (2, -0.1375, -0.238157, -0.05),
        (3, -0.275, 0.0, 0.05),
        (4, -0.1375, 0.238157, -0.05),
        (5, 0.1375, 0.238157, 0.05),
    ]


def test_generator_writes_scripts_manifest_and_registers_cmake(tmp_path):
    output_dir = tmp_path / "airframes"
    manifest = tmp_path / "px4-airframes.json"
    cmake_list = tmp_path / "CMakeLists.txt"
    cmake_list.write_text("px4_add_romfs_files(\n\t10016_none_iris\n)\n")

    _run_generator(
        "--config-dir",
        _AIRFRAME_CONFIG_DIR,
        "--output-dir",
        output_dir,
        "--cmake-list",
        cmake_list,
        "--manifest",
        manifest,
    )

    quad_script = (output_dir / "4019_none_x500_v2").read_text()
    hexa_script = (output_dir / "6001_none_hexa_x").read_text()
    patched_cmake = cmake_list.read_text()
    manifest_text = manifest.read_text()

    assert "param set-default CA_ROTOR_COUNT 4" in quad_script
    assert "# @type Quadrotor x" in quad_script
    assert "param set-default MAV_TYPE 2" in quad_script
    assert "param set-default BAT1_SOURCE 1" in quad_script
    assert "param set-default PWM_MAIN_FUNC4 104" in quad_script
    assert "param set-default CA_ROTOR_COUNT 6" in hexa_script
    assert "# @type Hexarotor x" in hexa_script
    assert "param set-default MAV_TYPE 13" in hexa_script
    assert "param set-default BAT1_SOURCE 1" in hexa_script
    assert "param set-default PWM_MAIN_FUNC6 106" in hexa_script
    assert "\t4019_none_x500_v2\n" in patched_cmake
    assert "\t6001_none_hexa_x\n" in patched_cmake
    assert '"script_name": "4019_none_x500_v2"' in manifest_text
    assert '"script_name": "6001_none_hexa_x"' in manifest_text


_QUAD_4001_CONFIG = """
name: collide_quad
output_filename: collide_quad.usda
arm_geometry:
  motor:
    position: [0.25, 0.0, 0.0]
arms:
  - azimuth_deg: 45.0
    spin_dir: CCW
    px4_motor_index: 0
  - azimuth_deg: -45.0
    spin_dir: CW
    px4_motor_index: 1
px4:
  sitl:
    sim_model: none_quad_x
    sys_autostart: 4001
    mav_type: 2
    ca_rotor_km_abs: 0.05
"""


def _run_generator_unchecked(*args: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", str(_GENERATOR), *(str(arg) for arg in args)],
        cwd=_ROOT,
        text=True,
        capture_output=True,
    )


def test_generator_rejects_airframe_id_colliding_with_stock_file(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "collide_quad.yaml").write_text(_QUAD_4001_CONFIG)
    output_dir = tmp_path / "airframes"
    output_dir.mkdir()
    (output_dir / "4001_gz_x500").write_text("# stock gazebo airframe\n")
    cmake_list = output_dir / "CMakeLists.txt"
    cmake_list.write_text("px4_add_romfs_files(\n\t4001_gz_x500\n\t10016_none_iris\n)\n")

    result = _run_generator_unchecked(
        "--config-dir", config_dir, "--output-dir", output_dir, "--cmake-list", cmake_list
    )

    assert result.returncode != 0
    assert "conflicting" in result.stderr
    assert "4001_gz_x500" in result.stderr
    # The stock file is left intact when not pruning.
    assert (output_dir / "4001_gz_x500").exists()


def test_generator_prunes_conflicting_stock_airframe_with_flag(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "collide_quad.yaml").write_text(_QUAD_4001_CONFIG)
    output_dir = tmp_path / "airframes"
    output_dir.mkdir()
    (output_dir / "4001_gz_x500").write_text("# stock gazebo airframe\n")
    cmake_list = output_dir / "CMakeLists.txt"
    cmake_list.write_text("px4_add_romfs_files(\n\t4001_gz_x500\n\t10016_none_iris\n)\n")

    _run_generator(
        "--config-dir",
        config_dir,
        "--output-dir",
        output_dir,
        "--cmake-list",
        cmake_list,
        "--prune-conflicts",
    )

    patched_cmake = cmake_list.read_text()
    assert not (output_dir / "4001_gz_x500").exists()
    assert (output_dir / "4001_none_quad_x").exists()
    assert "4001_gz_x500" not in patched_cmake
    assert "\t4001_none_quad_x\n" in patched_cmake


def test_generator_rejects_non_contiguous_px4_motor_indices(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        """
name: bad_quad
output_filename: bad_quad.usda
arm_geometry:
  motor:
    position: [1.0, 0.0, 0.0]
arms:
  - azimuth_deg: 0.0
    spin_dir: CW
    px4_motor_index: 0
  - azimuth_deg: 90.0
    spin_dir: CCW
    px4_motor_index: 2
px4:
  sitl:
    sim_model: none_bad_quad
    sys_autostart: 4999
    mav_type: 2
    ca_rotor_km_abs: 0.05
"""
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(_GENERATOR),
            "--config-dir",
            str(tmp_path),
            "--print-manifest",
        ],
        cwd=_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "contiguous" in result.stderr
