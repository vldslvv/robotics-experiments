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


def test_role_packages_import_without_simulator_dependencies(monkeypatch):
    # Re-importing the package tree under a blocker creates fresh module objects,
    # so snapshot sys.modules and restore it afterwards to avoid leaking that
    # state into other tests (some mock module-level state by import).
    saved_modules = dict(sys.modules)
    try:
        _drop_modules("carb", "pxr", "isaacsim", "torch", "robotics_experiments")
        blocker = _BlockedDependencyFinder({"carb", "pxr", "isaacsim", "torch"})
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

        importlib.import_module("robotics_experiments")
        importlib.import_module("robotics_experiments.environment")
        importlib.import_module("robotics_experiments.drone")
        importlib.import_module("robotics_experiments.simulation")
        importlib.import_module("robotics_experiments.px4_bridge")
        # The fleet runtime is an Isaac-coupled leaf package; its lazy __init__
        # must import without carb/omni present.
        importlib.import_module("robotics_experiments.runtime")
        # The pure press layer (actuation + deploy mapping) must import with both
        # Isaac and torch absent: stdlib + numpy only at module scope.
        importlib.import_module("robotics_experiments.rl.press")
        # Shared RL adapters are lazy: the package and adapter facade must not
        # import torch/rsl_rl/isaaclab on the core path.
        importlib.import_module("robotics_experiments.rl")
        importlib.import_module("robotics_experiments.rl.training")

        # The Isaac-only / torch leaf modules must stay off the core import path:
        # the lazy package __init__ shields (drone/environment __getattr__,
        # simulation re-export, runtime __getattr__) and the torch env leaves must
        # not eager-import them.
        for leaf in (
            "robotics_experiments.drone.spawn",
            "robotics_experiments.environment.stage",
            "robotics_experiments.environment.step",
            "robotics_experiments.simulation._isaac_settings",
            "robotics_experiments.rl.press._isaac_provider",
            "robotics_experiments.rl.training._rsl_rl",
            "robotics_experiments.runtime._bridge_host",
            "robotics_experiments.runtime.orchestrator",
            "robotics_experiments.workflows.isaac_lab.tasks._flight",
            "robotics_experiments.workflows.isaac_lab.tasks._drone_cfg",
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env",
        ):
            assert leaf not in sys.modules
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_px4_bridge_orchestrator_import_without_isaac():
    importlib.import_module("robotics_experiments.px4_bridge.bridge")


def test_commanding_imports_without_isaac_or_pymavlink(monkeypatch):
    # The command layer is a core domain leaf: importing it (including the
    # concrete MavlinkCommandPort) must not pull in Isaac or pymavlink, which
    # the port imports only inside its methods.
    saved_modules = dict(sys.modules)
    try:
        _drop_modules("carb", "pxr", "isaacsim", "pymavlink", "robotics_experiments.commanding")
        blocker = _BlockedDependencyFinder({"carb", "pxr", "isaacsim", "pymavlink"})
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

        commanding = importlib.import_module("robotics_experiments.commanding")
        # Constructing the MavlinkCommandPort must not import pymavlink either;
        # only connect()/send() do, inside their bodies.
        commanding.MavlinkCommandPort(gcs_endpoint="udp:127.0.0.1:14550", heartbeat_period_s=1.0)
        # The config-driven routine builder is pure too (it is what the live
        # extension driver calls), so it must build a routine with Isaac and
        # pymavlink blocked.
        from robotics_experiments.commanding.routine import EnvelopeGates, build_vehicle_routine

        gates = EnvelopeGates(
            max_tilt_deg=35.0,
            max_lateral_drift_m=2.0,
            max_vertical_speed_m_s=8.0,
            max_downward_speed_m_s=-4.0,
            max_angular_speed_rad_s=8.0,
            hover_max_abs_vertical_speed_m_s=0.35,
            hover_max_abs_horizontal_speed_m_s=0.30,
        )
        routine = build_vehicle_routine(
            [
                {"kind": "arm"},
                {"kind": "takeoff", "climb_m": 1.0, "min_climb_m": 0.8},
                {"kind": "hover", "hold_s": 2.0},
            ],
            target_latitude_deg=47.0,
            target_longitude_deg=8.0,
            target_altitude_m=488.0,
            command_ack_timeout_s=10.0,
            gates=gates,
            takeoff_timeout_s=30.0,
            hover_settle_s=2.0,
            home_lat_deg=47.0,
            home_lon_deg=8.0,
            home_alt_m=488.0,
            goto_position_tol_m=0.5,
            goto_heading_tol_deg=25.0,
            goto_cross_track_m=2.0,
            goto_offboard_prime_ticks=3,
        )
        assert [s.label for s in routine.steps] == ["arm", "takeoff", "hover"]
        # The per-drone routine driver is pure too (its phase machine imports no
        # carb/omni/pymavlink); it must be importable with the deps blocked.
        assert hasattr(commanding, "RoutineDriver")
        assert "pymavlink" not in sys.modules
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)


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
    assert config.__all__ == [
        "ResolvedVehicle",
        "Px4SitlLaunchConfig",
        "get_enabled_px4_instance_indices",
        "get_enabled_px4_launch_configs",
        "get_config",
        "resolve_drone_bridge_config",
    ]
    assert config.ResolvedVehicle.__name__ == "ResolvedVehicle"

    simulation = importlib.import_module("robotics_experiments.simulation")
    assert simulation.__all__ == ["apply_settings"]

    try:
        _ = config.get_config
    except ModuleNotFoundError as exc:
        assert exc.name == "hydra" or exc.name.startswith("hydra.")
    else:
        raise AssertionError("Expected lazy get_config access to require hydra")


def test_experiment_runtime_manifest_exposes_extension_module():
    manifest_path = (
        Path(__file__).resolve().parents[1]
        / "isaac/runtime/extensions/experiment_runtime/config/extension.toml"
    )
    manifest = tomllib.loads(manifest_path.read_text())
    module_names = {module["name"] for module in manifest["python"]["module"]}

    assert module_names == {"ExperimentRuntime_python"}
