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
        # RL owning packages must import with Isaac, torch, rsl_rl, and Isaac
        # Lab absent; the parent package provides no forwarding facade.
        rl = importlib.import_module("robotics_experiments.rl")
        assert rl.__all__ == []
        press = importlib.import_module("robotics_experiments.rl.press")
        assert press.__all__ == [
            "AbstractPoseProvider",
            "ActuationParams",
            "ActuationState",
            "GroundTruthPoseProvider",
            "HOVER_OBSERVATION_DIM",
            "HOVER_OBSERVATION_SCHEMA_VERSION",
            "HOVER_POLICY_INTERFACE",
            "HandoffGate",
            "HandoffReason",
            "HandoffResult",
            "LoadedVectorPolicy",
            "METADATA_FILENAME",
            "MixerGeometry",
            "OBSERVATION_SCHEMA_VERSION",
            "PRESS_OBSERVATION_DIM",
            "POLICY_FILENAME",
            "PRESS_POLICY_INTERFACE",
            "PolicyContract",
            "PoseSample",
            "PressDeploymentDecision",
            "PressDeploymentState",
            "PressObservation",
            "PressObservationMask",
            "PressRewardTerms",
            "PressStepMetrics",
            "PressTaskInput",
            "PressTaskResult",
            "PressTaskState",
            "PressTerminationFlags",
            "action_to_rotor_omega",
            "build_actuation_params",
            "build_policy_step_fn",
            "handoff_ready",
            "hover_action",
            "hover_observation_from_state",
            "initial_press_task_state",
            "initial_press_deployment_state",
            "load_policy_contract",
            "load_policy_metadata",
            "load_vector_policy",
            "pack_hover_observation",
            "pack_press_observation",
            "policy_metadata",
            "press_action_to_offboard",
            "reset_press_task_state",
            "rotor_forces_to_body_wrench",
            "step_press_task",
            "step_press_deployment",
            "to_vector",
            "validate_policy_artifact",
            "world_switch_to_contact_frame",
        ]
        training = importlib.import_module("robotics_experiments.rl.training")
        assert training.__all__ == []
        isaac_lab = importlib.import_module("robotics_experiments.workflows.isaac_lab")
        assert isaac_lab.__all__ == [
            "DRONE_HOVER_TASK_ID",
            "DroneHoverEnv",
            "DroneHoverEnvCfg",
            "SWITCH_PRESS_TASK_ID",
            "SwitchPressEnv",
            "SwitchPressEnvCfg",
            "compute_lab_rotor_omega",
            "make_drone_hover_env_cfg",
            "make_isaac_lab_env",
            "make_switch_press_env_cfg",
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
        # carb/omni/pymavlink); it must be importable with the deps blocked.
        assert hasattr(commanding, "RoutineDriverState")
        assert "pymavlink" not in sys.modules
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
        "PolicyStepFn",
        "PolicyStepResult",
        "PositionSetpoint",
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
        "DroneSpawnConfig",
        "ExperimentConfig",
        "IsaacLabVehicleConfig",
        "LightSwitchConfig",
        "PressActionConfig",
        "PressEvaluationConfig",
        "PressResetConfig",
        "PressRewardConfig",
        "PressTerminationConfig",
        "Px4BridgeConfig",
        "SimSettings",
        "VehicleConfig",
        "load_config",
    ]

    simulation = importlib.import_module("robotics_experiments.simulation")
    assert simulation.__all__ == ["apply_settings"]

    try:
        _ = config.load_config
    except ModuleNotFoundError as exc:
        assert exc.name == "hydra" or exc.name.startswith("hydra.")
    else:
        raise AssertionError("Expected lazy load_config access to require hydra")


def test_experiment_runtime_manifest_exposes_extension_module():
    manifest_path = REPO_ROOT / "isaac/runtime/extensions/experiment_runtime/config/extension.toml"
    manifest = tomllib.loads(manifest_path.read_text())
    module_names = {module["name"] for module in manifest["python"]["module"]}

    assert module_names == {"ExperimentRuntime_python"}
