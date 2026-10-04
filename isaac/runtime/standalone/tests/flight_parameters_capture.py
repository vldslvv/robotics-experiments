"""Capture one independent flight: native hover followed by the response suite."""

from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from _px4_takeoff_scenario import Px4TakeoffScenarioOptions, run_px4_takeoff_scenario
from _runtime import SceneTestContext, run_scene_test

from robotics_experiments.commanding import (
    AttitudeRateSetpoint,
    PolicyAdapter,
    PolicyInput,
    PolicyResult,
    VelocitySetpoint,
)
from robotics_experiments.config import RoutineStepConfig
from robotics_experiments.flight_parameters import (
    FlightParametersConfig,
    load_flight_parameters_config,
)
from robotics_experiments.px4_bridge import body_velocity_flu_from_state, world_enu_to_body_flu
from robotics_experiments.px4_bridge.frames import rotate_vector_wxyz

_TEST_NAME = "flight_parameters_capture"


@dataclass(frozen=True)
class _Boundary:
    stage_index: int
    preparation_start_us: int
    start_sim_time_us: int
    end_sim_time_us: int
    initial_position_enu_m: tuple[float, ...]
    initial_attitude_flu_enu_wxyz: tuple[float, ...]
    initial_linear_velocity_enu_mps: tuple[float, ...]
    initial_angular_velocity_body_flu_radps: tuple[float, ...]


@dataclass(frozen=True)
class _ProbeState:
    stage_index: int = 1
    command_index: int = 0
    phase: str = "precondition"
    phase_started_us: int = -1
    stable_started_us: int = -1
    command_started_us: int = -1
    boundaries: tuple[_Boundary, ...] = ()
    requested: object = None
    requested_at_us: int = -1
    complete: bool = False


def _plain(value: Any, /) -> Any:
    if not isinstance(value, type) and is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(item) for item in value]
    return value


def _routine_steps(*, config: FlightParametersConfig) -> tuple[RoutineStepConfig, ...]:
    hover = config.hover
    hold_s = hover.mode_transition_timeout_s + hover.settle_s + hover.measure_s
    response_s = sum(
        config.preparation.timeout_s + sum(command.duration_s for command in stage.commands)
        for stage in config.stages[1:]
    )
    return (
        {"kind": "arm"},
        {
            "kind": "takeoff",
            "climb_m": hover.target_position_enu_m[2],
            "min_climb_m": config.takeoff_min_climb_m,
        },
        {"kind": "hover", "hold_s": hold_s, "completion": "elapsed"},
        {
            "kind": "policy",
            "artifact": "flight-suite",
            "label": "flight-response-stages",
            "timeout_s": response_s + config.preparation.timeout_s,
        },
    )


def _position_setpoint(
    *, policy_input: PolicyInput, target: tuple[float, ...], config: FlightParametersConfig
) -> VelocitySetpoint:
    vehicle = policy_input.vehicle_state
    desired = (np.asarray(target) - vehicle.position_enu) * config.preparation.recenter_gain_hz
    speed = float(np.linalg.norm(desired))
    if speed > config.preparation.recenter_max_speed_mps:
        desired *= config.preparation.recenter_max_speed_mps / speed
    body = world_enu_to_body_flu(vehicle.orientation_flu_enu_wxyz, desired)
    return VelocitySetpoint(float(body[0]), -float(body[1]), -float(body[2]), 0.0)


def _is_stable(*, policy_input: PolicyInput, config: FlightParametersConfig) -> bool:
    vehicle = policy_input.vehicle_state
    w, x, y, _z = vehicle.orientation_flu_enu_wxyz
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))))
    limits = config.preparation
    return (
        np.linalg.norm(vehicle.linear_velocity_enu) <= limits.max_linear_speed_mps
        and np.linalg.norm(vehicle.angular_velocity_flu) <= limits.max_angular_speed_radps
        and tilt <= limits.max_tilt_deg
        and math.isfinite(w)
    )


def _advance_factory(*, config: FlightParametersConfig):
    def advance(*, policy_input: PolicyInput, state: object) -> PolicyResult:
        if not isinstance(state, _ProbeState):
            raise TypeError("flight suite policy state has an unexpected type")
        now = policy_input.sim_time_us
        neutral = _position_setpoint(
            policy_input=policy_input,
            target=tuple(policy_input.reference_position_enu),
            config=config,
        )
        if state.phase == "precondition":
            if state.phase_started_us < 0:
                state = replace(state, phase_started_us=now)
            elapsed = now - state.phase_started_us
            stable = elapsed >= round(config.preparation.minimum_neutral_s * 1e6) and _is_stable(
                policy_input=policy_input, config=config
            )
            since = (
                state.stable_started_us
                if stable and state.stable_started_us >= 0
                else now
                if stable
                else -1
            )
            state = replace(state, stable_started_us=since)
            if elapsed < round(config.preparation.timeout_s * 1e6):
                return PolicyResult(
                    state=replace(state, requested=neutral, requested_at_us=now), setpoint=neutral
                )
            if since < 0 or now - since < round(config.preparation.stable_hold_s * 1e6):
                return PolicyResult(
                    state=state,
                    setpoint=neutral,
                    failure_reason=(
                        f"stage {config.stages[state.stage_index].name!r} did not stabilize "
                        f"within {config.preparation.timeout_s}s"
                    ),
                )
            vehicle = policy_input.vehicle_state
            boundary = _Boundary(
                state.stage_index,
                state.phase_started_us,
                now,
                -1,
                tuple(float(v) for v in vehicle.position_enu),
                tuple(float(v) for v in vehicle.orientation_flu_enu_wxyz),
                tuple(float(v) for v in vehicle.linear_velocity_enu),
                tuple(float(v) for v in vehicle.angular_velocity_flu),
            )
            state = replace(
                state,
                phase="measurement",
                phase_started_us=now,
                command_started_us=now,
                boundaries=(*state.boundaries, boundary),
            )
        stage = config.stages[state.stage_index]
        while now - state.command_started_us >= round(
            stage.commands[state.command_index].duration_s * 1e6
        ):
            next_start = state.command_started_us + round(
                stage.commands[state.command_index].duration_s * 1e6
            )
            if state.command_index + 1 < len(stage.commands):
                state = replace(
                    state, command_index=state.command_index + 1, command_started_us=next_start
                )
                continue
            boundaries = (
                *state.boundaries[:-1],
                replace(state.boundaries[-1], end_sim_time_us=now),
            )
            complete = state.stage_index + 1 == len(config.stages)
            successor = replace(
                state,
                stage_index=state.stage_index + 1,
                command_index=0,
                phase="complete" if complete else "precondition",
                phase_started_us=now,
                stable_started_us=-1,
                command_started_us=-1,
                boundaries=boundaries,
                requested=neutral,
                requested_at_us=now,
                complete=complete,
            )
            return PolicyResult(state=successor, setpoint=neutral, complete=complete)
        command = stage.commands[state.command_index]
        if stage.control_mode == "ctbr":
            roll, pitch, yaw = command.body_rate_flu_radps
            setpoint = AttitudeRateSetpoint(
                command.collective_thrust_normalized, roll, -pitch, -yaw
            )
        elif stage.hold_position:
            setpoint = _position_setpoint(
                policy_input=policy_input,
                target=state.boundaries[-1].initial_position_enu_m,
                config=config,
            )
        else:
            forward, left, up = command.velocity_body_flu_mps
            setpoint = VelocitySetpoint(forward, -left, -up, -command.yaw_rate_radps)
        return PolicyResult(
            state=replace(state, requested=setpoint, requested_at_us=now), setpoint=setpoint
        )

    return advance


def _policy_factory(*, config: FlightParametersConfig):
    def factory(*, artifact: str, hold_s: float | None) -> PolicyAdapter:
        if artifact != "flight-suite" or hold_s is not None:
            raise ValueError("flight capture requires the injected flight-suite policy")
        return PolicyAdapter(
            policy_rate_hz=config.command_rate_hz,
            initial_state=_ProbeState(),
            advance=_advance_factory(config=config),
        )

    return factory


def _deploy_pusher(*, context: SceneTestContext, host, config: FlightParametersConfig) -> None:
    if not config.preparation.pusher_deployed:
        return
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    stage = omni.usd.get_context().get_stage()
    path = f"{host.params.articulation_path}/pusher_joint"
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"flight plant has no pusher joint at {path!r}")
    joint = UsdPhysics.PrismaticJoint(prim)
    drive = UsdPhysics.DriveAPI.Get(prim, "linear")
    drive.CreateTargetPositionAttr(float(joint.GetUpperLimitAttr().Get()))
    drive.CreateMaxForceAttr(float(context.config.vehicles[config.vehicle].pusher.effort_limit_n))
    PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr().Set(
        float(context.config.vehicles[config.vehicle].pusher.velocity_limit_mps)
    )


def _run(context: SceneTestContext, /) -> None:
    config = load_flight_parameters_config(
        path=os.environ["FLIGHT_PARAMETERS_CONFIG"],
        flight_count=int(os.environ["FLIGHT_COUNT"]) if os.environ.get("FLIGHT_COUNT") else None,
    )
    index = int(os.environ["FLIGHT_PARAMETERS_FLIGHT_INDEX"])
    seed = config.start_seed + index
    np.random.seed(seed)
    directory = Path(os.environ["FLIGHT_PARAMETERS_TRIAL_DIR"])
    if tuple(context.config.vehicles) != (config.vehicle,):
        raise ValueError("flight capture requires exactly its configured vehicle")
    vehicle = context.config.vehicles[config.vehicle]
    if (
        vehicle.bridge.sensor_noise_enabled
        or vehicle.bridge.sensor_excitation_mode != "deterministic"
    ):
        raise ValueError("flight calibration requires noise-free deterministic sensor excitation")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("flight capture directory must be empty")
    next_sample_us = -1
    sample_count = 0
    hover_start_us = -1
    hover_end_us = -1
    period_us = round(1e6 / config.sample_rate_hz)

    with (directory / "motion.jsonl").open("w") as stream:

        def physics_tick(fleet) -> None:
            nonlocal next_sample_us, sample_count, hover_start_us, hover_end_us
            from robotics_experiments.runtime import read_bridge_host_status

            flying = fleet.flying[0]
            host, observer = flying.spawned.host, flying.spawned.observer
            if host is None or observer is None:
                raise AssertionError("capture requires a live bridge observer")
            now = read_bridge_host_status(host=host).sim_time_us
            progress = flying.driver.routine_state
            if progress.cursor == 2 and progress.sent and hover_start_us < 0:
                hover_start_us = progress.step_started_us
                hover_end_us = hover_start_us + round(
                    (
                        config.hover.mode_transition_timeout_s
                        + config.hover.settle_s
                        + config.hover.measure_s
                    )
                    * 1e6
                )
                next_sample_us = now
            if next_sample_us < 0:
                next_sample_us = now
            if now < next_sample_us:
                return
            next_sample_us += period_us
            state = observer.read_state(now)
            policy = progress.offboard.policy_state or progress.last_policy_state
            phase = "startup"
            stage_index = -1
            if hover_start_us >= 0 and now < hover_end_us:
                stage_index = 0
                elapsed = (now - hover_start_us) * 1e-6
                phase = (
                    "transition"
                    if elapsed < config.hover.mode_transition_timeout_s
                    else "settle"
                    if elapsed < config.hover.mode_transition_timeout_s + config.hover.settle_s
                    else "measurement"
                )
            elif isinstance(policy, _ProbeState):
                stage_index, phase = policy.stage_index, policy.phase
            row = {
                "sim_time_us": now,
                "stage_index": stage_index,
                "phase": phase,
                "position_enu_m": _plain(state.position_enu),
                "attitude_flu_enu_wxyz": _plain(state.orientation_flu_enu_wxyz),
                "linear_velocity_enu_mps": _plain(state.linear_velocity_enu),
                "velocity_body_flu_mps": _plain(body_velocity_flu_from_state(state)),
                "angular_velocity_body_flu_radps": _plain(state.angular_velocity_flu),
                "angular_velocity_enu_radps": _plain(
                    rotate_vector_wxyz(state.orientation_flu_enu_wxyz, state.angular_velocity_flu)
                ),
                "requested_command": _plain(policy.requested)
                if isinstance(policy, _ProbeState)
                else None,
                "requested_at_us": policy.requested_at_us
                if isinstance(policy, _ProbeState)
                else None,
                "transmitted_command": _plain(progress.offboard.transmitted_setpoint),
                "transmitted_at_us": progress.offboard.transmitted_setpoint_sim_time_us,
                "policy_generation": progress.offboard.policy_generation,
                "transmitted_policy_generation": progress.offboard.transmitted_policy_generation,
            }
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
            sample_count += 1

        def post_routine(_context: SceneTestContext, fleet) -> None:
            from robotics_experiments.runtime import read_bridge_host_status

            flying = fleet.flying[0]
            state = flying.driver.routine_state.last_policy_state
            if (
                not isinstance(state, _ProbeState)
                or not state.complete
                or len(state.boundaries) != len(config.stages) - 1
            ):
                raise AssertionError("flight did not finish every configured stage")
            host = flying.spawned.host
            if host is None:
                raise AssertionError("capture retained no bridge host")
            log = read_bridge_host_status(host=host).debug_log_path
            if log is None:
                raise AssertionError("capture retained no bridge log")
            stream.flush()
            boundaries = [
                {
                    "stage_index": 0,
                    "preparation_start_us": hover_start_us,
                    "start_sim_time_us": hover_start_us
                    + round((config.hover.mode_transition_timeout_s + config.hover.settle_s) * 1e6),
                    "end_sim_time_us": hover_end_us,
                },
                *(asdict(boundary) for boundary in state.boundaries),
            ]
            payload = {
                "artifact_kind": "robotics-experiments-flight-recording",
                "schema_version": 1,
                "flight_parameters_id": config.flight_parameters_id,
                "run_id": os.environ["FLIGHT_PARAMETERS_RUN_ID"],
                "flight_index": index,
                "seed": seed,
                "state_source": config.state_source,
                "source_backend": config.source_backend,
                "preparation": asdict(config.preparation),
                "sample_rate_hz": config.sample_rate_hz,
                "command_rate_hz": config.command_rate_hz,
                "sample_count": sample_count,
                "routine_steps": _plain(_routine_steps(config=config)),
                "px4_home_lat_lon_alt": [
                    vehicle.bridge.home_lat_deg,
                    vehicle.bridge.home_lon_deg,
                    vehicle.bridge.home_alt_m,
                ],
                "stages": [
                    {**asdict(stage), **boundary}
                    for stage, boundary in zip(config.stages, boundaries, strict=True)
                ],
                "completed": True,
            }
            (directory / "raw_flight.json").write_text(
                json.dumps(payload, allow_nan=False, indent=2) + "\n"
            )
            shutil.copy2(log, directory / "bridge.log")

        run_px4_takeoff_scenario(
            context,
            Px4TakeoffScenarioOptions(
                test_name=_TEST_NAME,
                px4_container_log_name="flight_parameters_capture_px4.log",
                routine_steps=_routine_steps(config=config),
                physics_tick_hook=physics_tick,
                post_takeoff_hook=lambda context, host: _deploy_pusher(
                    context=context, host=host, config=config
                ),
                post_routine_hook=post_routine,
                policy_factory=_policy_factory(config=config),
                success_label="24-stage flight parameter suite",
            ),
        )


def main() -> None:
    run_scene_test(
        test_name=_TEST_NAME,
        default_scene=None,
        manage_stage=False,
        extra_python_paths=["isaac/runtime/extensions/px4_sitl_brigde"],
        test_body=_run,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
