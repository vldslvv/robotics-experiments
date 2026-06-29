from pathlib import Path

PX4_BRIDGE_EXTENSION_PATH = "isaac/runtime/extensions/px4_sitl_brigde"


def test_px4_standalone_scenarios_request_bridge_extension_python_path():
    root = Path(__file__).resolve().parents[1]
    scripts = (
        root / "isaac/runtime/standalone/tests/test_px4_takeoff_stability.py",
        root / "isaac/runtime/standalone/tests/test_px4_takeoff_instability_detection.py",
        root / "isaac/runtime/standalone/tests/test_px4_multi_drone_takeoff.py",
        root / "isaac/runtime/standalone/tests/test_px4_navigation_routine.py",
        root / "isaac/runtime/standalone/tests/test_px4_hexacopter_navigation_routine.py",
    )

    for script in scripts:
        source = script.read_text()
        assert f'PX4_BRIDGE_EXTENSION_PATH = "{PX4_BRIDGE_EXTENSION_PATH}"' in source
        assert "extra_python_paths=[PX4_BRIDGE_EXTENSION_PATH]" in source
