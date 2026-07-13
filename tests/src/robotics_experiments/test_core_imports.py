import importlib
import importlib.abc
import sys
import tomllib

from tests.support.paths import REPO_ROOT


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
        environment = importlib.import_module("robotics_experiments.environment")
        assert environment.__all__ == [
            "SwitchState",
            "apply_button_sample",
            "despawn_switches",
            "init_articulations",
            "spawn_switches",
            "step_switches",
            "validate_switches",
        ]
        drone = importlib.import_module("robotics_experiments.drone")
        assert drone.__all__ == ["spawn_drone", "despawn_drone"]
        simulation = importlib.import_module("robotics_experiments.simulation")
        assert simulation.__all__ == ["apply_settings"]
        importlib.import_module("robotics_experiments.px4_bridge")
        # The fleet runtime is an Isaac-coupled leaf package; its lazy __init__
        # must import without carb/omni present.
        runtime = importlib.import_module("robotics_experiments.runtime")
        assert runtime.__all__ == [
            "BridgeHost",
            "BridgeHostGroup",
            "BridgeHostParams",
            "FleetDriverEnv",
            "FleetOrchestrator",
            "SpawnedDrone",
        ]
        # The RL package root is the public facade and must import with Isaac,
        # torch, rsl_rl, and Isaac Lab absent.
        rl = importlib.import_module("robotics_experiments.rl")
        assert rl.__all__ == [
            "AbstractPoseProvider",
            "ActuationParams",
            "ActuationState",
            "GroundTruthPoseProvider",
            "HOVER_OBSERVATION_DIM",
            "HOVER_OBSERVATION_SCHEMA_VERSION",
            "HandoffGate",
            "HandoffReason",
            "HandoffResult",
            "METADATA_FILENAME",
            "MixerGeometry",
            "OBSERVATION_SCHEMA_VERSION",
            "POLICY_FILENAME",
            "PoseSample",
            "PressObservation",
            "TrainingResult",
            "action_to_rotor_omega",
            "build_actuation_params",
            "build_policy_step_fn",
            "build_runner",
            "export_policy",
            "handoff_ready",
            "hover_action",
            "hover_observation_from_state",
            "pack_hover_observation",
            "policy_metadata",
            "press_action_to_offboard",
            "rotor_forces_to_body_wrench",
            "train",
            "to_vector",
            "world_switch_to_contact_frame",
        ]
        press = importlib.import_module("robotics_experiments.rl.press")
        assert press.__all__ == [
            "AbstractPoseProvider",
            "ActuationParams",
            "ActuationState",
            "GroundTruthPoseProvider",
            "HOVER_OBSERVATION_DIM",
            "HOVER_OBSERVATION_SCHEMA_VERSION",
            "HandoffGate",
            "HandoffReason",
            "HandoffResult",
            "METADATA_FILENAME",
            "MixerGeometry",
            "OBSERVATION_SCHEMA_VERSION",
            "POLICY_FILENAME",
            "PoseSample",
            "PressObservation",
            "action_to_rotor_omega",
            "build_actuation_params",
            "build_policy_step_fn",
            "handoff_ready",
            "hover_action",
            "hover_observation_from_state",
            "pack_hover_observation",
            "policy_metadata",
            "press_action_to_offboard",
            "rotor_forces_to_body_wrench",
            "to_vector",
            "world_switch_to_contact_frame",
        ]
        training = importlib.import_module("robotics_experiments.rl.training")
        assert training.__all__ == ["TrainingResult", "build_runner", "export_policy", "train"]
        isaac_lab = importlib.import_module("robotics_experiments.workflows.isaac_lab")
        assert isaac_lab.__all__ == [
            "DRONE_HOVER_TASK_ID",
            "DroneHoverEnv",
            "DroneHoverEnvCfg",
            "compute_lab_rotor_omega",
            "register_isaac_lab_tasks",
        ]

        # The Isaac-only / torch leaf modules must stay off the core import path:
        # lazy package initializers must not eager-import them.
        for leaf in (
            "robotics_experiments.drone.spawn",
            "robotics_experiments.environment.stage",
            "robotics_experiments.environment.step",
            "robotics_experiments.simulation._isaac_settings",
            "robotics_experiments.rl.press.isaac_provider",
            "robotics_experiments.rl.training.rsl_rl",
            "robotics_experiments.runtime.bridge_host",
            "robotics_experiments.runtime.orchestrator",
            "robotics_experiments.workflows.isaac_lab.tasks.flight",
            "robotics_experiments.workflows.isaac_lab.tasks._drone_cfg",
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env",
        ):
            assert leaf not in sys.modules
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_px4_bridge_orchestrator_import_without_isaac():
    importlib.import_module("robotics_experiments.px4_bridge")


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
        gates = commanding.EnvelopeGates(
            max_tilt_deg=35.0,
            max_lateral_drift_m=2.0,
            max_vertical_speed_m_s=8.0,
            max_downward_speed_m_s=-4.0,
            max_angular_speed_rad_s=8.0,
            hover_max_abs_vertical_speed_m_s=0.35,
            hover_max_abs_horizontal_speed_m_s=0.30,
        )
        routine = commanding.build_vehicle_routine(
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


def test_commanding_public_api_contract_is_explicit():
    commanding = importlib.import_module("robotics_experiments.commanding")

    assert commanding.__all__ == [
        "AttitudeRateSetpoint",
        "CommandAck",
        "DecisionKind",
        "DriverEnvironment",
        "DriverParams",
        "DriverPhase",
        "EnvelopeGates",
        "FailureKind",
        "MAV_CMD_COMPONENT_ARM_DISARM",
        "MAV_CMD_DO_REPOSITION",
        "MAV_CMD_DO_SET_MODE",
        "MAV_CMD_NAV_LAND",
        "MAV_CMD_NAV_TAKEOFF",
        "MAV_DO_REPOSITION_FLAGS_CHANGE_MODE",
        "MAV_RESULT_ACCEPTED",
        "MAV_RESULT_TEMPORARILY_REJECTED",
        "MavlinkCommandPort",
        "MavlinkCommand",
        "PolicyStepFn",
        "PositionSetpoint",
        "RoutineDecision",
        "RoutineDriver",
        "RoutineObservation",
        "RoutineState",
        "RoutineStep",
        "StreamSetpoint",
        "VelocitySetpoint",
        "VehicleRoutine",
        "ack_accepted",
        "arm",
        "armed",
        "build_vehicle_routine",
        "climbed_at_least",
        "cross_track_drift_below",
        "cross_track_to_segment_below",
        "decide",
        "disarmed",
        "heading_aligned",
        "hovering_stable",
        "land",
        "moved_forward_at_least",
        "reached_world_position",
        "reposition",
        "set_offboard_mode",
        "takeoff",
        "tilt_below",
    ]


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
        "BoundingBox",
        "DriverConfig",
        "DroneSpawnConfig",
        "ExperimentConfig",
        "LightSwitchConfig",
        "LightSwitchStatePublisherConfig",
        "LockstepConfig",
        "PressActionConfig",
        "PressHandoffConfig",
        "PressHoverTaskConfig",
        "PressTaskConfig",
        "Px4BridgeConfig",
        "Px4SitlLaunchConfig",
        "RlConfig",
        "RlTrainingConfig",
        "RosQosConfig",
        "RosVehicleConfig",
        "RoutineConfig",
        "RoutineStepConfig",
        "SimSettings",
        "VehicleConfig",
        "get_config",
        "get_enabled_px4_launch_configs",
        "load_config",
        "reset_config_cache",
    ]

    simulation = importlib.import_module("robotics_experiments.simulation")
    assert simulation.__all__ == ["apply_settings"]

    try:
        _ = config.get_config
    except ModuleNotFoundError as exc:
        assert exc.name == "hydra" or exc.name.startswith("hydra.")
    else:
        raise AssertionError("Expected lazy get_config access to require hydra")


def test_experiment_runtime_manifest_exposes_extension_module():
    manifest_path = REPO_ROOT / "isaac/runtime/extensions/experiment_runtime/config/extension.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    module_names = {module["name"] for module in manifest["python"]["module"]}

    assert module_names == {"ExperimentRuntime_python"}
