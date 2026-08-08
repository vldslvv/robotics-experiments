"""Capture task-neutral body-FLU velocity probes through PX4 SITL OFFBOARD."""

from __future__ import annotations

import json
import math
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from _px4_takeoff_scenario import (
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import SceneTestContext, run_scene_test

from robotics_experiments.alignment import load_controller_definition
from robotics_experiments.alignment._models import ControllerDefinition
from robotics_experiments.commanding import (
    PolicyAdapter,
    PolicyInput,
    PolicyResult,
    VelocitySetpoint,
)
from robotics_experiments.config import ExperimentConfig
from robotics_experiments.px4_bridge import body_velocity_flu_from_state

_REFERENCE_ENV = "ALIGNMENT_REFERENCE"
_RUN_DIR_ENV = "ROBOTICS_ALIGNMENT_RUN_DIR"
_DEFAULT_REFERENCE = "configs/alignment/controllers/px4_sitl_x500_velocity.yaml"
_TEST_NAME = "alignment_capture_reference"


@dataclass(frozen=True)
class _InitialState:
    profile: str
    sim_time_us: int
    position_enu_m: tuple[float, float, float]
    attitude_flu_enu_wxyz: tuple[float, float, float, float]
    linear_velocity_enu_mps: tuple[float, float, float]
    angular_velocity_body_flu_radps: tuple[float, float, float]


@dataclass(frozen=True)
class _Sample:
    profile: str
    sim_time_us: int
    command_velocity_body_flu_mps: tuple[float, float, float]
    command_yaw_rate_radps: float
    position_enu_m: tuple[float, float, float]
    attitude_flu_enu_wxyz: tuple[float, float, float, float]
    velocity_body_flu_mps: tuple[float, float, float]
    velocity_enu_mps: tuple[float, float, float]
    angular_velocity_body_flu_radps: tuple[float, float, float]


@dataclass(frozen=True)
class _ProbeState:
    profile_index: int
    command_index: int
    phase: str
    phase_started_us: int
    stable_started_us: int
    command_started_us: int
    initials: tuple[_InitialState, ...]
    samples: tuple[_Sample, ...]
    complete: bool


def _float3(values, /) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _float4(values, /) -> tuple[float, float, float, float]:
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
    )


def _initial(*, profile: str, policy_input: PolicyInput) -> _InitialState:
    vehicle = policy_input.vehicle_state
    return _InitialState(
        profile=profile,
        sim_time_us=policy_input.sim_time_us,
        position_enu_m=_float3(vehicle.position_enu),
        attitude_flu_enu_wxyz=_float4(vehicle.orientation_flu_enu_wxyz),
        linear_velocity_enu_mps=_float3(vehicle.linear_velocity_enu),
        angular_velocity_body_flu_radps=_float3(vehicle.angular_velocity_flu),
    )


def _is_stable(*, policy_input: PolicyInput, config: ExperimentConfig) -> bool:
    vehicle = policy_input.vehicle_state
    linear_speed = math.sqrt(sum(float(value) ** 2 for value in vehicle.linear_velocity_enu))
    angular_speed = math.sqrt(sum(float(value) ** 2 for value in vehicle.angular_velocity_flu))
    w, x, y, _z = (float(value) for value in vehicle.orientation_flu_enu_wxyz)
    body_z_alignment = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    tilt_deg = math.degrees(math.acos(body_z_alignment))
    capture = config.alignment.capture
    return (
        linear_speed <= capture.max_linear_speed_mps
        and angular_speed <= capture.max_angular_speed_radps
        and tilt_deg <= capture.max_tilt_deg
        and math.isfinite(w)
    )


def _advance_factory(*, definition: ControllerDefinition, config: ExperimentConfig):
    profiles = definition.probes

    def advance(*, policy_input: PolicyInput, state: object) -> PolicyResult:
        if not isinstance(state, _ProbeState):
            raise TypeError(f"alignment probe state has unexpected type {type(state).__name__}")
        profile_index = state.profile_index
        command_index = state.command_index
        phase = state.phase
        phase_started_us = state.phase_started_us
        stable_started_us = state.stable_started_us
        command_started_us = state.command_started_us
        initials = state.initials
        now_us = policy_input.sim_time_us
        capture = config.alignment.capture
        neutral = VelocitySetpoint(0.0, 0.0, 0.0, 0.0)
        if phase == "precondition":
            if phase_started_us < 0:
                phase_started_us = now_us
            elapsed_us = now_us - phase_started_us
            if elapsed_us > round(capture.timeout_s * 1_000_000):
                return PolicyResult(
                    state=state,
                    setpoint=neutral,
                    failure_reason=(
                        f"alignment profile {profiles[profile_index].name!r} did not "
                        f"stabilize within {capture.timeout_s:.1f}s"
                    ),
                )
            minimum_elapsed = elapsed_us >= round(capture.minimum_neutral_s * 1_000_000)
            if minimum_elapsed and _is_stable(policy_input=policy_input, config=config):
                if stable_started_us < 0:
                    stable_started_us = now_us
            else:
                stable_started_us = -1
            stable_elapsed = stable_started_us >= 0 and (
                now_us - stable_started_us >= round(capture.stable_hold_s * 1_000_000)
            )
            if not stable_elapsed:
                return PolicyResult(
                    state=_ProbeState(
                        profile_index=profile_index,
                        command_index=0,
                        phase="precondition",
                        phase_started_us=phase_started_us,
                        stable_started_us=stable_started_us,
                        command_started_us=-1,
                        initials=initials,
                        samples=state.samples,
                        complete=False,
                    ),
                    setpoint=neutral,
                )
            initials = (
                *initials,
                _initial(profile=profiles[profile_index].name, policy_input=policy_input),
            )
            phase = "probe"
            command_index = 0
            command_started_us = now_us
        while profile_index < len(profiles):
            command = profiles[profile_index].commands[command_index]
            duration_us = int(round(command.duration_s * 1_000_000))
            if now_us - command_started_us < duration_us:
                break
            command_started_us += duration_us
            command_index += 1
            if command_index < len(profiles[profile_index].commands):
                continue
            profile_index += 1
            command_index = 0
            if profile_index < len(profiles):
                return PolicyResult(
                    state=_ProbeState(
                        profile_index=profile_index,
                        command_index=0,
                        phase="precondition",
                        phase_started_us=now_us,
                        stable_started_us=-1,
                        command_started_us=-1,
                        initials=initials,
                        samples=state.samples,
                        complete=False,
                    ),
                    setpoint=neutral,
                )
        if profile_index == len(profiles):
            successor = _ProbeState(
                profile_index=profile_index,
                command_index=0,
                phase="complete",
                phase_started_us=phase_started_us,
                stable_started_us=stable_started_us,
                command_started_us=command_started_us,
                initials=initials,
                samples=state.samples,
                complete=True,
            )
            return PolicyResult(
                state=successor,
                setpoint=VelocitySetpoint(0.0, 0.0, 0.0, 0.0),
                complete=True,
            )
        probe_command = profiles[profile_index].commands[command_index]
        command = probe_command.velocity_body_flu_mps
        vehicle = policy_input.vehicle_state
        velocity = body_velocity_flu_from_state(policy_input.vehicle_state)
        sample = _Sample(
            profile=profiles[profile_index].name,
            sim_time_us=policy_input.sim_time_us,
            command_velocity_body_flu_mps=command,
            command_yaw_rate_radps=probe_command.yaw_rate_radps,
            position_enu_m=_float3(vehicle.position_enu),
            attitude_flu_enu_wxyz=_float4(vehicle.orientation_flu_enu_wxyz),
            velocity_body_flu_mps=_float3(velocity),
            velocity_enu_mps=_float3(vehicle.linear_velocity_enu),
            angular_velocity_body_flu_radps=_float3(vehicle.angular_velocity_flu),
        )
        successor = _ProbeState(
            profile_index=profile_index,
            command_index=command_index,
            phase=phase,
            phase_started_us=phase_started_us,
            stable_started_us=stable_started_us,
            command_started_us=command_started_us,
            initials=initials,
            samples=(*state.samples, sample),
            complete=False,
        )
        return PolicyResult(
            state=successor,
            setpoint=VelocitySetpoint(
                forward_mps=command[0],
                right_mps=-command[1],
                down_mps=-command[2],
                yaw_rate_radps=-probe_command.yaw_rate_radps,
            ),
        )

    return advance


def _policy_factory(*, definition: ControllerDefinition, config: ExperimentConfig):
    def factory(*, artifact: str, hold_s: float | None) -> PolicyAdapter:
        if artifact != "alignment-probe" or hold_s is not None:
            raise ValueError("alignment capture requires its injected alignment-probe step")
        return PolicyAdapter(
            policy_rate_hz=definition.command_rate_hz,
            initial_state=_ProbeState(
                profile_index=0,
                command_index=0,
                phase="precondition",
                phase_started_us=-1,
                stable_started_us=-1,
                command_started_us=-1,
                initials=(),
                samples=(),
                complete=False,
            ),
            advance=_advance_factory(definition=definition, config=config),
        )

    return factory


def _deploy_pusher(*, context: SceneTestContext, host) -> None:
    """Match the deployed-pusher plant used by Lab training and policy runtime."""
    if not context.config.alignment.capture.pusher_deployed:
        return
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    joint_path = f"{host.params.articulation_path}/pusher_joint"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim.IsValid():
        raise RuntimeError(f"alignment plant has no pusher joint at {joint_path!r}")
    joint = UsdPhysics.PrismaticJoint(joint_prim)
    upper_limit = float(joint.GetUpperLimitAttr().Get())
    drive = UsdPhysics.DriveAPI.Get(joint_prim, "linear")
    drive.CreateTargetPositionAttr(upper_limit)
    drive.CreateMaxForceAttr(float(context.config.rl.lab_vehicle.pusher_effort_limit_n))
    PhysxSchema.PhysxJointAPI.Apply(joint_prim).CreateMaxJointVelocityAttr().Set(
        float(context.config.rl.lab_vehicle.pusher_velocity_limit_mps)
    )
    context.report(f"alignment plant pusher deployed to {upper_limit:.6f} m")


def _assert_yaw_response_sign(*, samples: tuple[_Sample, ...]) -> None:
    segment: list[_Sample] = []
    previous_command = 0.0

    def validate(values: list[_Sample], command: float) -> None:
        if not values or abs(command) < 1e-9:
            return
        settled = values[len(values) // 2 :]
        mean_rate = sum(sample.angular_velocity_body_flu_radps[2] for sample in settled) / len(
            settled
        )
        if command * mean_rate <= 0.0:
            raise AssertionError(
                "PX4 alignment yaw response sign disagrees with the body-FLU "
                f"command: command={command:.3f}rad/s response={mean_rate:.3f}rad/s"
            )

    for sample in samples:
        command = sample.command_yaw_rate_radps
        if segment and not math.isclose(command, previous_command, abs_tol=1e-9):
            validate(segment, previous_command)
            segment = []
        segment.append(sample)
        previous_command = command
    validate(segment, previous_command)


def _write_capture(
    *,
    bridge_log_path: Path,
    definition: ControllerDefinition,
    run_dir: Path,
    state: _ProbeState,
) -> None:
    if not state.complete or len(state.initials) != len(definition.probes):
        raise AssertionError("PX4 alignment probe adapter did not complete every profile")
    _assert_yaw_response_sign(samples=state.samples)
    traces = []
    for profile, initial in zip(definition.probes, state.initials, strict=True):
        samples = tuple(sample for sample in state.samples if sample.profile == profile.name)
        if not samples:
            raise AssertionError(f"PX4 alignment profile {profile.name!r} produced no samples")
        traces.append(
            {
                "profile": profile.name,
                "split": profile.split,
                "axis": profile.axis,
                "start_sim_time_us": initial.sim_time_us,
                "time_s": [(sample.sim_time_us - initial.sim_time_us) * 1e-6 for sample in samples],
                "command_velocity_body_flu_mps": [
                    list(sample.command_velocity_body_flu_mps) for sample in samples
                ],
                "command_yaw_rate_radps": [sample.command_yaw_rate_radps for sample in samples],
                "position_enu_m": [list(sample.position_enu_m) for sample in samples],
                "attitude_flu_enu_wxyz": [list(sample.attitude_flu_enu_wxyz) for sample in samples],
                "velocity_body_flu_mps": [list(sample.velocity_body_flu_mps) for sample in samples],
                "velocity_enu_mps": [list(sample.velocity_enu_mps) for sample in samples],
                "angular_velocity_body_flu_radps": [
                    list(sample.angular_velocity_body_flu_radps) for sample in samples
                ],
                "initial_position_enu_m": list(initial.position_enu_m),
                "initial_attitude_flu_enu_wxyz": list(initial.attitude_flu_enu_wxyz),
                "initial_linear_velocity_enu_mps": list(initial.linear_velocity_enu_mps),
                "initial_angular_velocity_body_flu_radps": list(
                    initial.angular_velocity_body_flu_radps
                ),
                "completed": True,
                "failure_reason": None,
            }
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "reference.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "controller_id": definition.controller_id,
                "definition_fingerprint": definition.definition_fingerprint,
                "image": os.environ.get(
                    "PX4_BENCHMARK_IMAGE",
                    definition.image,
                ),
                "traces": traces,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    shutil.copy2(bridge_log_path, run_dir / "bridge.log")
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifacts": {
                    "reference_trace": "reference.json",
                    "px4_ulog": "reference.ulg",
                    "bridge_log": "bridge.log",
                    "actuator_audit": "actuator_audit.json",
                    "target_candidates": "target_candidates.json",
                    "gpu_telemetry": "gpu_utilization.json",
                    "controller_artifact": "controller.json",
                    "controller_boundary_replay": "controller_boundary_replay.json",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _run(*, context: SceneTestContext, definition: ControllerDefinition, run_dir: Path) -> None:
    np.random.seed(context.config.alignment.capture.sensor_seed)

    def post_routine(_context: SceneTestContext, fleet) -> None:
        state = fleet.flying[0].driver.routine_state.last_policy_state
        if not isinstance(state, _ProbeState):
            raise AssertionError("alignment routine retained no probe state")
        from robotics_experiments.runtime import read_bridge_host_status

        host = fleet.flying[0].spawned.host
        if host is None:
            raise AssertionError("alignment capture retained no bridge host")
        debug_log_path = read_bridge_host_status(host=host).debug_log_path
        if debug_log_path is None:
            raise AssertionError("alignment capture has no bridge debug log")
        _write_capture(
            bridge_log_path=Path(debug_log_path),
            definition=definition,
            run_dir=run_dir,
            state=state,
        )

    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name=_TEST_NAME,
            px4_container_log_name="alignment_capture_reference_px4.log",
            post_takeoff_hook=_deploy_pusher,
            post_routine_hook=post_routine,
            policy_factory=_policy_factory(definition=definition, config=context.config),
            success_label="PX4 controller alignment probes",
        ),
    )


def main() -> None:
    definition = load_controller_definition(os.environ.get(_REFERENCE_ENV, _DEFAULT_REFERENCE))
    try:
        run_dir = Path(os.environ[_RUN_DIR_ENV])
    except KeyError as exc:
        raise RuntimeError(f"missing required environment variable {_RUN_DIR_ENV}") from exc
    run_scene_test(
        test_name=_TEST_NAME,
        default_scene=None,
        manage_stage=False,
        extra_python_paths=["isaac/runtime/extensions/px4_sitl_brigde"],
        test_body=lambda context: _run(
            context=context,
            definition=definition,
            run_dir=run_dir,
        ),
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
