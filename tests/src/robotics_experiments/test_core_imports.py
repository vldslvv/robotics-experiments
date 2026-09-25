import ast
import importlib
import importlib.abc
import json
import subprocess
import sys
import tomllib

from tests.support.paths import REPO_ROOT


def _project_imports(path, /):
    imported = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _package_imports(relative, /):
    package = REPO_ROOT / "src" / "robotics_experiments" / relative
    return {
        imported
        for path in package.rglob("*.py")
        for imported in _project_imports(path)
        if imported.startswith("robotics_experiments.")
    }


class _BlockedDependencyFinder(importlib.abc.MetaPathFinder):
    def __init__(self, module_roots: set[str]):
        self._module_roots = module_roots

    def find_spec(self, fullname, path, target=None):
        if fullname.split(".", maxsplit=1)[0] in self._module_roots:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


def test_controller_alignment_rl_dependency_direction():
    controller_imports = _package_imports("controller")
    assert not any(
        imported.startswith(
            (
                "robotics_experiments.alignment",
                "robotics_experiments.rl",
                "robotics_experiments.workflows",
                "robotics_experiments.runtime",
            )
        )
        for imported in controller_imports
    )
    flight_parameter_imports = _package_imports("flight_parameters")
    assert not any(
        imported.startswith(
            (
                "robotics_experiments.alignment",
                "robotics_experiments.controller",
                "robotics_experiments.rl",
                "robotics_experiments.workflows",
                "robotics_experiments.runtime",
            )
        )
        for imported in flight_parameter_imports
    )

    for package in ("rl", "workflows/isaac_lab"):
        assert not any(
            imported.startswith("robotics_experiments.alignment")
            for imported in _package_imports(package)
        )

    deployment_imports = _project_imports(
        REPO_ROOT / "src" / "robotics_experiments" / "runtime" / "_policy.py"
    )
    assert not any(
        imported.startswith(("robotics_experiments.alignment", "robotics_experiments.controller"))
        for imported in deployment_imports
    )


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
        _drop_modules("carb", "omni", "pxr", "isaacsim", "isaaclab", "robotics_experiments")
        blocker = _BlockedDependencyFinder({"carb", "omni", "pxr", "isaacsim", "isaaclab"})
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

        importlib.import_module("robotics_experiments")
        environment = importlib.import_module("robotics_experiments.environment")
        assert environment.__all__ == [
            "SwitchState",
            "SwitchesRuntimePhase",
            "SwitchesRuntimeState",
            "adopt_switches_stage",
            "apply_button_sample",
            "create_switches_runtime",
            "despawn_switches",
            "init_articulations",
            "pause_switches_runtime",
            "play_switches_runtime",
            "spawn_switches",
            "step_switches",
            "step_switches_runtime",
            "teardown_switches_runtime",
            "validate_switches",
        ]
        drone = importlib.import_module("robotics_experiments.drone")
        assert drone.__all__ == ["spawn_drone", "despawn_drone"]
        simulation = importlib.import_module("robotics_experiments.simulation")
        assert simulation.__all__ == ["apply_settings"]
        importlib.import_module("robotics_experiments.px4_bridge")
        controller = importlib.import_module("robotics_experiments.controller")
        assert "ControllerSettings" in controller.__all__
        assert "build_actuation_params" in controller.__all__
        alignment = importlib.import_module("robotics_experiments.alignment")
        assert alignment.__all__ == [
            "AlignmentResult",
            "ControllerDefinition",
            "align",
            "load_controller_definition",
            "prepare_alignment_reference",
        ]
        flight_parameters = importlib.import_module("robotics_experiments.flight_parameters")
        assert flight_parameters.__all__ == [
            "FlightParametersConfig",
            "FlightParameters",
            "FlightParametersBundle",
            "FlightTrace",
            "HoverEvaluation",
            "ResponseTraceEvidence",
            "SignalStatistics",
            "derive_flight_parameters",
            "evaluate_hover_pose",
            "finalize_flight_parameters_bundle",
            "flight_parameters_from_payload",
            "flight_trace_from_payload",
            "load_flight_parameters_config",
            "load_flight_parameters",
            "load_flight_parameters_bundle",
            "load_flight_trace",
            "publish_flight_parameters",
            "summarize_signal",
            "validate_flight_parameters_bundle",
            "write_flight_parameters",
        ]
        # The fleet runtime is an Isaac-coupled leaf package; its lazy __init__
        # must import without carb/omni present.
        runtime = importlib.import_module("robotics_experiments.runtime")
        assert runtime.__all__ == [
            "BridgeHost",
            "BridgeHostGroup",
            "BridgeHostParams",
            "BridgeHostStatus",
            "FleetDriverEnv",
            "FleetState",
            "FlyingDrone",
            "SpawnedDrone",
            "SwitchToggleEvent",
            "adopt_current_stage",
            "apply_settings",
            "build_drivers",
            "create_fleet",
            "fleet_is_done",
            "open_scene",
            "read_bridge_host_status",
            "spawn_fleet",
            "start_bridges",
            "teardown",
            "tick_drivers",
        ]
        # RL owning packages eagerly import ordinary dependencies but must still
        # import with Isaac Sim, Kit, and Isaac Lab absent.
        rl = importlib.import_module("robotics_experiments.rl")
        assert "robotics_experiments.rl.press" not in sys.modules
        assert rl.__all__ == [
            "HOVER_OBSERVATION_DIM",
            "HOVER_OBSERVATION_SCHEMA_VERSION",
            "EvaluationMetric",
            "HoverEnvelopeEvaluation",
            "HoverResetSamples",
            "LoadedPolicy",
            "METADATA_FILENAME",
            "POLICY_FILENAME",
            "PolicyContract",
            "PolicyInterface",
            "PolicyMetadata",
            "PolicyInputs",
            "PolicyEvaluation",
            "RewardMean",
            "evaluate_hover_envelope",
            "hover_observation_from_state",
            "load_hover_reset_samples",
            "load_policy",
            "load_policy_metadata",
            "pack_hover_observation",
            "pack_pose_observation",
            "policy_action_to_offboard",
            "policy_metadata",
            "prepare_policy_inputs",
            "validate_policy_artifact",
        ]
        press = importlib.import_module("robotics_experiments.rl.press")
        assert press.__all__ == [
            "AbstractPoseProvider",
            "AcceptanceCheck",
            "AcceptanceResult",
            "GroundTruthPoseProvider",
            "HandoffGate",
            "HandoffReason",
            "HandoffResult",
            "OBSERVATION_SCHEMA_VERSION",
            "PRESS_OBSERVATION_DIM",
            "PoseSample",
            "PressAcceptanceCriteria",
            "PressApproachExecutionParams",
            "PressApproachExecutionResult",
            "PressApproachExecutionState",
            "PressApproachInput",
            "PressApproachResult",
            "PressApproachRewardTerms",
            "PressApproachState",
            "PressApproachTerminationFlags",
            "PressObservation",
            "PressObservationMask",
            "advance_press_approach_execution",
            "create_press_acceptance_criteria",
            "evaluate_press_acceptance",
            "handoff_ready",
            "initial_press_approach_execution_state",
            "initial_press_approach_state",
            "pack_press_observation",
            "press_approach_target_velocity",
            "press_policy_contract",
            "quaternion_geodesic_error_rad",
            "reset_press_approach_state",
            "step_press_approach",
            "to_vector",
            "world_switch_to_contact_frame",
        ]
        training = importlib.import_module("robotics_experiments.rl.training")
        assert training.__all__ == []
        isaac_lab = importlib.import_module("robotics_experiments.workflows.isaac_lab")
        assert isaac_lab.__all__ == [
            "DRONE_HOVER_TASK_ID",
            "DroneHoverEnv",
            "DroneHoverEnvCfg",
            "PressApproachEpisodeMetrics",
            "PressApproachEvaluationResult",
            "PressApproachEvaluationSummary",
            "SWITCH_PRESS_TASK_ID",
            "SwitchPressEnv",
            "SwitchPressEnvCfg",
            "compute_lab_rotor_omega",
            "evaluate_press_approach_policy",
            "make_drone_hover_env_cfg",
            "make_isaac_lab_env",
            "make_switch_press_env_cfg",
            "press_policy_evaluation",
            "register_isaac_lab_tasks",
            "summarize_press_approach_episodes",
        ]

        # Isaac-only leaf modules must stay off the core import path:
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
            "robotics_experiments.workflows.isaac_lab.tasks._flight_env",
            "robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env",
            "robotics_experiments.workflows.isaac_lab.tasks.switch_press_env",
        ):
            assert leaf not in sys.modules
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_px4_bridge_orchestrator_import_without_isaac():
    px4_bridge = importlib.import_module("robotics_experiments.px4_bridge")
    assert px4_bridge.__all__ == [
        "ActuatorCommands",
        "BridgeStatus",
        "DEFAULT_HOME_ALT_M",
        "DEFAULT_HOME_LAT_DEG",
        "DEFAULT_HOME_LON_DEG",
        "DEFAULT_MAG_NED_GAUSS",
        "DiscoveredRotorMapping",
        "GRAVITY_M_S2",
        "IsaacArticulationIO",
        "LockstepSettings",
        "PegasusQuadraticThrustModel",
        "Px4BridgeState",
        "Px4MavlinkPort",
        "RootBodyPoseDiagnosticParams",
        "RootBodyPoseDiagnostics",
        "RotorDynamics",
        "RotorThrustModel",
        "SimClockState",
        "VehicleState",
        "advance_sim_clock",
        "body_velocity_flu_from_state",
        "bridge_time_us",
        "compute_root_body_pose_diagnostics",
        "create_px4_bridge",
        "create_pegasus_thrust_model",
        "create_sim_clock",
        "discover_px4_rotor_mapping",
        "normalize_quat_wxyz",
        "quat_conjugate_wxyz",
        "quat_multiply_wxyz",
        "quat_wxyz_to_xyzw",
        "recv_and_apply",
        "root_body_pose_has_mismatch",
        "rotate_vector_wxyz",
        "send_sensors",
        "sim_clock_is_settling",
        "start_px4_bridge",
        "stop_px4_bridge",
        "synth_barometer",
        "synth_gps_from_enu",
        "synth_magnetometer_body_flu",
        "tilt_angle_deg",
        "tick_px4_bridge",
        "validate_root_body_pose",
        "vehicle_log_context",
        "world_enu_to_body_flu",
    ]


def test_commanding_imports_without_isaac(monkeypatch):
    # The command layer eagerly owns pymavlink but remains independent of Isaac.
    saved_modules = dict(sys.modules)
    try:
        _drop_modules("carb", "omni", "pxr", "isaacsim", "robotics_experiments.commanding")
        blocker = _BlockedDependencyFinder({"carb", "omni", "pxr", "isaacsim"})
        monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])

        commanding = importlib.import_module("robotics_experiments.commanding")
        commanding.MavlinkCommandPort(gcs_endpoint="udp:127.0.0.1:14550", heartbeat_period_s=1.0)
        # The config-driven routine builder is pure too (it is what the live
        # extension driver calls), so it must build a routine with Isaac and
        # Isaac blocked.
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
            steps=[
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
            offboard_setpoint_rate_hz=50.0,
        )
        assert [s.label for s in routine.steps] == ["arm", "takeoff", "hover"]
        # The per-drone routine driver is pure too (its phase machine imports no
        # carb/omni); it must be importable with the simulator deps blocked.
        assert hasattr(commanding, "RoutineDriverState")
        assert "pymavlink" in sys.modules
    finally:
        sys.modules.clear()
        sys.modules.update(saved_modules)


def test_commanding_public_api_contract_is_explicit():
    commanding = importlib.import_module("robotics_experiments.commanding")

    assert commanding.__all__ == [
        "AttitudeRateSetpoint",
        "CommandAck",
        "CommandPortStatus",
        "DecisionKind",
        "DriverContext",
        "DriverEnvironment",
        "DriverFailure",
        "DriverParams",
        "DriverPhase",
        "DriverTick",
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
        "PolicyAdapter",
        "PolicyInput",
        "PolicyResult",
        "PositionSetpoint",
        "RecoveryOutcome",
        "RecoveryStage",
        "RoutineDecision",
        "RoutineDriverState",
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
        "create_routine_driver",
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
        "routine_driver_is_done",
        "set_offboard_mode",
        "takeoff",
        "tilt_below",
        "tick_routine_driver",
    ]


def test_config_eagerly_imports_hydra() -> None:
    # Import in a fresh process so Hydra's singleton registries remain intact
    # for tests that already hold references to configuration modules.
    code = """
import json
import sys
sys.path.insert(0, "src")
import robotics_experiments.config as config
assert "hydra" in sys.modules
assert "omegaconf" in sys.modules
print(json.dumps(config.__all__))
"""
    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True, cwd=REPO_ROOT
    )
    assert json.loads(result.stdout) == [
        "CurriculumConfig",
        "CurriculumStageConfig",
        "load_curriculum_config",
        "BoundingBox",
        "DroneSpawnConfig",
        "ExperimentConfig",
        "HoverTaskConfig",
        "IsaacLabVehicleConfig",
        "LightSwitchConfig",
        "PressApproachAcceptanceConfig",
        "PressApproachEvaluationConfig",
        "PressApproachGoalConfig",
        "PressApproachRewardConfig",
        "PressApproachTerminationConfig",
        "PressApproachVelocityReferenceConfig",
        "PressTaskConfig",
        "PressResetSourceConfig",
        "PressHandoffConfig",
        "PressResetConfig",
        "Px4BridgeConfig",
        "RlActionConfig",
        "RlInputsConfig",
        "RlPolicyActorConfig",
        "RlPolicyConfig",
        "RlPolicyCriticConfig",
        "RlPlateauCriteriaConfig",
        "RlTrainingPhaseConfig",
        "RlTrainingConfig",
        "RoutineStepConfig",
        "SimSettings",
        "VehicleConfig",
        "load_config",
    ]


def test_simulation_package_remains_lazy_without_isaac(monkeypatch):
    _drop_modules("carb", "omni", "pxr", "isaacsim", "robotics_experiments.simulation")
    blocker = _BlockedDependencyFinder({"carb", "omni", "pxr", "isaacsim"})
    monkeypatch.setattr(sys, "meta_path", [blocker, *sys.meta_path])
    simulation = importlib.import_module("robotics_experiments.simulation")
    assert simulation.__all__ == ["apply_settings"]


def test_ordinary_dependencies_load_eagerly_from_owning_modules():
    code = """
import sys
sys.path.insert(0, "src")
import robotics_experiments.commanding
import robotics_experiments.rl
import robotics_experiments.rl.press
import robotics_experiments.rl.training.rsl_rl
import robotics_experiments.workflows.isaac_lab.registration
required = {"gymnasium", "pymavlink", "rsl_rl", "torch"}
missing = required - sys.modules.keys()
raise SystemExit(f"ordinary dependencies not loaded: {sorted(missing)}" if missing else 0)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_experiment_runtime_manifest_exposes_extension_module():
    manifest_path = REPO_ROOT / "isaac/runtime/extensions/experiment_runtime/config/extension.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    module_names = {module["name"] for module in manifest["python"]["module"]}

    assert module_names == {"ExperimentRuntime_python"}
