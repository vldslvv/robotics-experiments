"""Capture task-neutral body-FLU velocity probes through PX4 SITL OFFBOARD."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

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
    velocity_body_flu_mps: tuple[float, float, float]


@dataclass(frozen=True)
class _ProbeState:
    profile_index: int
    command_index: int
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


def _advance_factory(*, definition: ControllerDefinition):
    profiles = definition.probes

    def advance(*, policy_input: PolicyInput, state: object) -> PolicyResult:
        if not isinstance(state, _ProbeState):
            raise TypeError(f"alignment probe state has unexpected type {type(state).__name__}")
        profile_index = state.profile_index
        command_index = state.command_index
        command_started_us = state.command_started_us
        initials = state.initials
        if command_started_us < 0:
            command_started_us = policy_input.sim_time_us
            initials = (_initial(profile=profiles[0].name, policy_input=policy_input),)
        while profile_index < len(profiles):
            command = profiles[profile_index].commands[command_index]
            duration_us = int(round(command.duration_s * 1_000_000))
            if policy_input.sim_time_us - command_started_us < duration_us:
                break
            command_started_us += duration_us
            command_index += 1
            if command_index < len(profiles[profile_index].commands):
                continue
            profile_index += 1
            command_index = 0
            if profile_index < len(profiles):
                initials = (
                    *initials,
                    _initial(
                        profile=profiles[profile_index].name,
                        policy_input=policy_input,
                    ),
                )
        if profile_index == len(profiles):
            successor = _ProbeState(
                profile_index=profile_index,
                command_index=0,
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
        command = profiles[profile_index].commands[command_index].velocity_body_flu_mps
        velocity = body_velocity_flu_from_state(policy_input.vehicle_state)
        sample = _Sample(
            profile=profiles[profile_index].name,
            sim_time_us=policy_input.sim_time_us,
            command_velocity_body_flu_mps=command,
            velocity_body_flu_mps=_float3(velocity),
        )
        successor = _ProbeState(
            profile_index=profile_index,
            command_index=command_index,
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
                yaw_rate_radps=0.0,
            ),
        )

    return advance


def _policy_factory(*, definition: ControllerDefinition):
    def factory(*, artifact: str, hold_s: float | None) -> PolicyAdapter:
        if artifact != "alignment-probe" or hold_s is not None:
            raise ValueError("alignment capture requires its injected alignment-probe step")
        return PolicyAdapter(
            policy_rate_hz=definition.command_rate_hz,
            initial_state=_ProbeState(
                profile_index=0,
                command_index=0,
                command_started_us=-1,
                initials=(),
                samples=(),
                complete=False,
            ),
            advance=_advance_factory(definition=definition),
        )

    return factory


def _write_capture(
    *,
    definition: ControllerDefinition,
    run_dir: Path,
    state: _ProbeState,
) -> None:
    if not state.complete or len(state.initials) != len(definition.probes):
        raise AssertionError("PX4 alignment probe adapter did not complete every profile")
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
                "time_s": [(sample.sim_time_us - initial.sim_time_us) * 1e-6 for sample in samples],
                "command_velocity_body_flu_mps": [
                    list(sample.command_velocity_body_flu_mps) for sample in samples
                ],
                "velocity_body_flu_mps": [list(sample.velocity_body_flu_mps) for sample in samples],
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


def _run(*, context: SceneTestContext, definition: ControllerDefinition, run_dir: Path) -> None:
    def post_routine(_context: SceneTestContext, fleet) -> None:
        state = fleet.flying[0].driver.routine_state.last_policy_state
        if not isinstance(state, _ProbeState):
            raise AssertionError("alignment routine retained no probe state")
        _write_capture(definition=definition, run_dir=run_dir, state=state)

    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name=_TEST_NAME,
            px4_container_log_name="alignment_capture_reference_px4.log",
            post_routine_hook=post_routine,
            policy_factory=_policy_factory(definition=definition),
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
