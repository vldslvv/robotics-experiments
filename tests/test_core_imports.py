import importlib


def test_role_packages_import_without_simulator_dependencies():
    importlib.import_module("robotics_experiments.environment")
    importlib.import_module("robotics_experiments.drone")
    importlib.import_module("robotics_experiments.px4_bridge")


def test_px4_bridge_orchestrator_import_without_isaac():
    importlib.import_module("robotics_experiments.px4_bridge.bridge")
