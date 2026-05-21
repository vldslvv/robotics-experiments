import importlib
import importlib.abc
import sys
import tomllib
from pathlib import Path


class _BlockedDependencyFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_roots: set[str]):
        self._module_roots = module_roots

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", maxsplit=1)[0] in self._module_roots:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def _drop_modules(*prefixes: str) -> None:
    for name in list(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            sys.modules.pop(name, None)


def test_role_packages_import_without_simulator_dependencies():
    importlib.import_module("robotics_experiments.environment")
    importlib.import_module("robotics_experiments.drone")
    importlib.import_module("robotics_experiments.px4_bridge")


def test_px4_bridge_orchestrator_import_without_isaac():
    importlib.import_module("robotics_experiments.px4_bridge.bridge")


def test_lightweight_config_and_simulation_import_without_hydra(monkeypatch):
    _drop_modules(
        "hydra",
        "omegaconf",
        "robotics_experiments.config",
        "robotics_experiments.simulation",
    )
    blocker = _BlockedDependencyFinder({"hydra", "omegaconf"})
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

    config = importlib.import_module("robotics_experiments.config")
    assert config.SimSettings.__name__ == "SimSettings"

    simulation = importlib.import_module("robotics_experiments.simulation")
    assert simulation.__all__ == ["apply_settings"]

    try:
        config.get_config
    except ModuleNotFoundError as exc:
        assert exc.name == "hydra" or exc.name.startswith("hydra.")
    else:
        raise AssertionError("Expected lazy get_config access to require hydra")


def test_sim_settings_manifest_exposes_extension_module():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "isaac/runtime/extensions/sim_settings/config/extension.toml"
    )
    manifest = tomllib.loads(manifest_path.read_text())
    module_names = {module["name"] for module in manifest["python"]["module"]}

    assert module_names == {"SimSettings_python"}
