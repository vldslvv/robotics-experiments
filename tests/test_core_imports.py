import importlib


def test_domain_and_app_import_without_simulator_dependencies():
    importlib.import_module("robotics_experiments.domain")
    importlib.import_module("robotics_experiments.app")


def test_px4_bridge_orchestrator_import_without_isaac():
    importlib.import_module("robotics_experiments.app.px4_bridge")
