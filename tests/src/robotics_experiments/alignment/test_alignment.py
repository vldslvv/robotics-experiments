"""Host tests for controller definition, fitting, artifacts, and publication."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

import robotics_experiments.alignment._flight_parameters as flight_parameters_module
from robotics_experiments.alignment import (
    ControllerDefinition,
    align,
    load_controller_definition,
)
from robotics_experiments.alignment._align import (
    _module_fingerprint,
    _resolved_experiment_fingerprint,
    _staged_beam_search,
)
from robotics_experiments.alignment._artifact import (
    _require_passed_diagnostic,
    finalize_alignment_bundle,
    load_selection,
    publish_controller,
    validate_alignment_bundle,
    write_alignment_result,
)
from robotics_experiments.alignment._evaluation import evaluate_metrics, evaluate_response
from robotics_experiments.alignment._flight_parameters import (
    _split_flights as split_flights,
)
from robotics_experiments.alignment._flight_parameters import (
    load_alignment_reference,
    training_hover_limits,
)
from robotics_experiments.alignment._io import (
    _index_candidate_diagnostics,
    _trace,
    _TraceFileEvaluator,
)
from robotics_experiments.alignment._metrics import (
    _braking_events,
    register_metric,
)
from robotics_experiments.alignment._metrics import (
    metric as measured_metric,
)
from robotics_experiments.alignment._models import (
    MetricResult,
    ParameterSpec,
    ProbeCommand,
    ProbeFamily,
    ProbeProfile,
    ProbeTrace,
    SearchSpec,
)
from robotics_experiments.config import load_config
from robotics_experiments.controller import ControllerSettings, load_controller_artifact
from robotics_experiments.flight_parameters import (
    flight_stage_views,
    hover_pose_errors,
    load_flight_dataset,
)
from tests.support.paths import REPO_ROOT

_ZERO3 = (0.0, 0.0, 0.0)
_IDENTITY = (1.0, 0.0, 0.0, 0.0)


def metric(*, name):
    measure = measured_metric(name=name)

    def evaluated(reference, target, /):
        result = measure(reference, target)
        config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
        if name == "controller-quality":
            return evaluate_response(
                metric_name=name,
                reference=reference,
                target=target,
                response_limits=config.alignment.evaluation.response_limits,
                hover_limits=_hover_limits(),
            )
        prefix = "ctbr." if name == "ctbr-controller-quality" else "velocity."
        limits = {
            key.removeprefix(prefix): value
            for key, value in config.alignment.evaluation.response_limits.items()
            if key.startswith(prefix)
        }
        evaluation = evaluate_metrics(measurements=result.validation_metrics, limits=limits)
        return replace(
            result,
            gates=evaluation.checks,
            quality_passed=evaluation.accepted,
            ranking_values=(evaluation.score,),
            beam_values=tuple(c.value / c.limit for c in evaluation.checks),
        )

    return evaluated


def _hover_limits():
    return {
        name: 0.005 if "position" in name else math.radians(0.25)
        for name in hover_pose_errors(
            target_position_enu_m=_ZERO3,
            target_attitude_flu_enu_wxyz=_IDENTITY,
            position_enu_m=_ZERO3,
            attitude_flu_enu_wxyz=_IDENTITY,
        )
    }


def _settings(
    *,
    delay: float = 0.05,
    kp: tuple[float, float, float] = (2.0, 3.0, 4.0),
    kd: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> ControllerSettings:
    return ControllerSettings(
        command_delay_s=delay,
        velocity_controller_rate_hz=125.0,
        attitude_controller_rate_hz=187.5,
        kp_vel=kp,
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        horizontal_velocity_error_gain_end_mps=0.80,
        horizontal_velocity_braking_gain_scale=1.0,
        horizontal_velocity_braking_gain_start_mps=0.05,
        horizontal_velocity_braking_gain_full_mps=0.07,
        horizontal_velocity_braking_gain_end_mps=0.10,
        ki_vel=(0.4, 0.4, 0.4),
        kd_vel=kd,
        velocity_observation_scale=(1.0, 1.0, 1.0),
        velocity_observation_hold_scale=(1.0, 1.0, 1.0),
        velocity_observation_bias_mps=(0.0, 0.0, 0.0),
        velocity_observation_bias_command_start_mps=0.0,
        velocity_observation_bias_command_full_mps=0.01,
        velocity_estimator_bias_decay_hz=0.0,
        velocity_estimator_bias_hold_decay_hz=(0.0, 0.0, 0.0),
        velocity_filter_hz=(10.0, 10.0, 10.0),
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 2.0),
        hover_thrust_normalized=0.57,
        ctbr_effective_collective_scale=1.0,
        kp_att=(6.0, 6.0, 2.0),
        attitude_yaw_weight=0.4,
        kp_rate=(1.2, 1.2, 0.4),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.01, 0.01, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        angular_velocity_filter_hz=0.0,
        rate_derivative_filter_hz=30.0,
        yaw_torque_filter_hz=2.0,
        rate_integral_limit=(0.3, 0.3, 0.2),
        max_tilt_cmd_deg=30.0,
        min_collective_thrust_fraction=0.05,
        max_collective_thrust_fraction=1.0,
        horizontal_thrust_margin_fraction=0.3,
        thruster_tau_inc_s=0.0,
        thruster_tau_dec_s=0.0,
        thruster_max_thrust_rate_nps=100000.0,
    )


def _profiles() -> tuple[ProbeProfile, ...]:
    profiles = []
    for split, amplitude in (("fit", 1.0), ("validation", 0.7)):
        for axis_index, axis in enumerate(("x", "y", "z")):
            positive = [0.0, 0.0, 0.0]
            negative = [0.0, 0.0, 0.0]
            positive[axis_index] = amplitude
            negative[axis_index] = -amplitude * 0.5
            profiles.append(
                ProbeProfile(
                    name=f"{split}-{axis}",
                    split=split,
                    axis=axis,
                    commands=(
                        ProbeCommand(_ZERO3, 0.0, 0.0, _ZERO3, 0.1),
                        ProbeCommand(tuple(positive), 0.0, 0.0, _ZERO3, 0.15),
                        ProbeCommand(tuple(negative), 0.0, 0.0, _ZERO3, 0.1),
                        ProbeCommand(_ZERO3, 0.0, 0.0, _ZERO3, 0.1),
                    ),
                )
            )
            if split == "validation":
                profiles.append(
                    ProbeProfile(
                        name=f"validation-{axis}-envelope",
                        split=split,
                        axis=axis,
                        commands=profiles[-1].commands,
                    )
                )
        profiles.extend(
            (
                ProbeProfile(
                    name=f"{split}-yaw",
                    split=split,
                    axis="yaw",
                    commands=(
                        ProbeCommand(_ZERO3, 0.0, 0.0, _ZERO3, 0.1),
                        ProbeCommand(_ZERO3, amplitude * 0.2, 0.0, _ZERO3, 0.15),
                        ProbeCommand(_ZERO3, 0.0, 0.0, _ZERO3, 0.1),
                    ),
                ),
                ProbeProfile(
                    name=f"{split}-combined",
                    split=split,
                    axis="combined",
                    commands=(
                        ProbeCommand(_ZERO3, 0.0, 0.0, _ZERO3, 0.1),
                        ProbeCommand((amplitude, amplitude * 0.5, 0.0), 0.2, 0.0, _ZERO3, 0.15),
                        ProbeCommand(_ZERO3, 0.0, 0.0, _ZERO3, 0.1),
                    ),
                ),
            )
        )
    profiles.append(
        ProbeProfile(
            name="stability-hover-pose",
            split="stability",
            axis="combined",
            commands=(ProbeCommand(_ZERO3, 0.0, 0.0, _ZERO3, 0.1),),
        )
    )
    return tuple(profiles)


def _command_samples(profile: ProbeProfile, /) -> tuple[tuple[float, float, float], ...]:
    values = []
    for command in profile.commands:
        values.extend([command.velocity_body_flu_mps] * round(command.duration_s / 0.05))
    return tuple(values)


def _yaw_samples(profile: ProbeProfile, /) -> tuple[float, ...]:
    values = []
    for command in profile.commands:
        values.extend([command.yaw_rate_radps] * round(command.duration_s / 0.05))
    return tuple(values)


def _response_traces(
    *,
    profiles: tuple[ProbeProfile, ...],
    settings: ControllerSettings,
) -> tuple[ProbeTrace, ...]:
    traces = []
    delay_steps = round(settings.command_delay_s / 0.05)
    for profile in profiles:
        commands = _command_samples(profile)
        yaw_commands = _yaw_samples(profile)
        velocities = []
        angular_velocities = []
        positions = []
        attitudes = []
        position = [0.0, 0.0, 0.0]
        yaw = 0.0
        for index in range(len(commands)):
            delayed = _ZERO3 if index < delay_steps else commands[index - delay_steps]
            delayed_yaw = 0.0 if index < delay_steps else yaw_commands[index - delay_steps]
            velocity = tuple(
                delayed[axis] * settings.kp_vel[axis] / settings.kd_vel[axis] for axis in range(3)
            )
            position = [value + velocity[axis] * 0.05 for axis, value in enumerate(position)]
            yaw += delayed_yaw * 0.05
            velocities.append(velocity)
            positions.append(tuple(position))
            attitudes.append((math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)))
            angular_velocities.append((0.0, 0.0, delayed_yaw))
        traces.append(
            ProbeTrace(
                flight_index=0,
                family="velocity",
                control_mode="velocity",
                family_metric="translational-controller-quality",
                profile=profile.name,
                split=profile.split,
                axis=profile.axis,
                time_s=tuple(index * 0.05 for index in range(len(commands))),
                command_time_s=tuple(index * 0.05 for index in range(len(commands))),
                command_velocity_body_flu_mps=commands,
                command_yaw_rate_radps=yaw_commands,
                command_collective_thrust_normalized=tuple(0.0 for _ in commands),
                command_body_rate_flu_radps=tuple(_ZERO3 for _ in commands),
                position_enu_m=tuple(positions),
                attitude_flu_enu_wxyz=tuple(attitudes),
                velocity_body_flu_mps=tuple(velocities),
                velocity_enu_mps=tuple(velocities),
                angular_velocity_body_flu_radps=tuple(angular_velocities),
                initial_position_enu_m=_ZERO3,
                initial_attitude_flu_enu_wxyz=_IDENTITY,
                initial_linear_velocity_enu_mps=_ZERO3,
                initial_estimated_linear_velocity_enu_mps=_ZERO3,
                initial_angular_velocity_body_flu_radps=_ZERO3,
                initial_estimated_angular_velocity_body_flu_radps=_ZERO3,
                completed=True,
            )
        )
    return tuple(traces)


def _ctbr_response_traces(*, family: ProbeFamily) -> tuple[ProbeTrace, ...]:
    traces = []
    for profile in family.profiles:
        collective_commands = []
        body_rate_commands = []
        for command in profile.commands:
            count = round(command.duration_s / 0.05)
            collective_commands.extend([command.collective_thrust_normalized] * count)
            body_rate_commands.extend([command.body_rate_flu_radps] * count)
        position = np.zeros(3)
        angles = np.zeros(3)
        vertical_velocity = 0.0
        positions = []
        attitudes = []
        velocities = []
        for collective, body_rate in zip(collective_commands, body_rate_commands, strict=True):
            vertical_velocity += (collective - 0.57) * 20.0 * 0.05
            position[2] += vertical_velocity * 0.05
            angles += np.asarray(body_rate) * 0.05
            roll, pitch, yaw = angles / 2.0
            attitudes.append(
                (
                    math.cos(roll) * math.cos(pitch) * math.cos(yaw)
                    + math.sin(roll) * math.sin(pitch) * math.sin(yaw),
                    math.sin(roll) * math.cos(pitch) * math.cos(yaw)
                    - math.cos(roll) * math.sin(pitch) * math.sin(yaw),
                    math.cos(roll) * math.sin(pitch) * math.cos(yaw)
                    + math.sin(roll) * math.cos(pitch) * math.sin(yaw),
                    math.cos(roll) * math.cos(pitch) * math.sin(yaw)
                    - math.sin(roll) * math.sin(pitch) * math.cos(yaw),
                )
            )
            positions.append(tuple(float(value) for value in position))
            velocities.append((0.0, 0.0, vertical_velocity))
        sample_count = len(collective_commands)
        traces.append(
            ProbeTrace(
                flight_index=0,
                family=family.name,
                control_mode=family.control_mode,
                family_metric=family.metric,
                profile=profile.name,
                split=profile.split,
                axis=profile.axis,
                time_s=tuple(index * 0.05 for index in range(sample_count)),
                command_time_s=tuple(index * 0.05 for index in range(sample_count)),
                command_velocity_body_flu_mps=tuple(_ZERO3 for _ in range(sample_count)),
                command_yaw_rate_radps=tuple(0.0 for _ in range(sample_count)),
                command_collective_thrust_normalized=tuple(collective_commands),
                command_body_rate_flu_radps=tuple(body_rate_commands),
                position_enu_m=tuple(positions),
                attitude_flu_enu_wxyz=tuple(attitudes),
                velocity_body_flu_mps=tuple(velocities),
                velocity_enu_mps=tuple(velocities),
                angular_velocity_body_flu_radps=tuple(body_rate_commands),
                initial_position_enu_m=_ZERO3,
                initial_attitude_flu_enu_wxyz=_IDENTITY,
                initial_linear_velocity_enu_mps=_ZERO3,
                initial_estimated_linear_velocity_enu_mps=_ZERO3,
                initial_angular_velocity_body_flu_radps=_ZERO3,
                initial_estimated_angular_velocity_body_flu_radps=_ZERO3,
                completed=True,
            )
        )
    return tuple(traces)


class _Evaluator:
    def for_set(self, set_name, /):
        return self

    def __init__(self, *, reference_settings: ControllerSettings | None = None) -> None:
        self._reference_settings = reference_settings

    def evaluate(self, *, definition, settings, reference_traces):
        active = self._reference_settings or settings
        return _response_traces(profiles=definition.profiles, settings=active)


class _BatchEvaluator(_Evaluator):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def evaluate_batch(self, *, definition, settings, reference_traces):
        self.batch_sizes.append(len(settings))
        return tuple(
            self.evaluate(
                definition=definition,
                settings=value,
                reference_traces=reference_traces,
            )
            for value in settings
        )


def _definition(
    *,
    backend: str,
    evaluator: _Evaluator,
    parameters: tuple[ParameterSpec, ...] = (),
    metric_name: str = "controller-quality",
) -> ControllerDefinition:
    experiment = REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml"
    return ControllerDefinition(
        alignment_id="synthetic",
        controller_id=f"{backend}-controller",
        controller_schema_version=1,
        backend=backend,
        experiment_config=str(experiment),
        flight_parameters_artifact=str(REPO_ROOT / "tests/fixtures/flight_parameters"),
        vehicle="x500_01",
        airframe="holybro_x500_v2.usda",
        command_rate_hz=20.0,
        base_settings=_settings(delay=0.0, kp=(1.0, 1.0, 1.0)),
        probe_families=(
            ProbeFamily(
                name="velocity",
                control_mode="velocity",
                metric="translational-controller-quality",
                profiles=_profiles(),
            ),
        ),
        parameters=parameters,
        search=SearchSpec(
            strategy="coarse-to-fine-beam",
            beam_width=8,
            passes_per_stage=2,
        ),
        selection_metric=metric_name,
        publication_metrics=(),
        evaluator=evaluator,
        definition_fingerprint=f"{backend}-fingerprint",
        image=f"{backend}-image",
    )


def _search(**kwargs):
    config = load_config(REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml")
    return _staged_beam_search(
        **kwargs,
        response_limits=config.alignment.evaluation.response_limits,
        hover_limits=_hover_limits(),
    )


def _raw_dataset(root: Path, count: int = 3, /) -> Path:
    root.mkdir(parents=True)
    definition = _definition(backend="px4-sitl", evaluator=_Evaluator())
    traces = _response_traces(profiles=definition.profiles, settings=definition.base_settings)

    def record(path):
        return {
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    flights = []
    for index in range(count):
        directory = root / "flights" / f"flight-{index:03d}"
        directory.mkdir(parents=True)
        rows: list[dict] = []
        hover = {
            "kind": "hover",
            "name": "hover",
            "stage_index": 0,
            "start_sim_time_us": 1000000,
            "end_sim_time_us": 1100000,
            "sample_start": 0,
            "sample_stop": 2,
            "effective_target_position_enu_m": _ZERO3,
            "effective_target_attitude_flu_enu_wxyz": _IDENTITY,
        }
        stages = [hover]
        for stage_index, trace in enumerate((traces[-1], *traces)):
            start = 1000000 + stage_index * 1000000
            if stage_index:
                profile = definition.profiles[stage_index - 1]
                metadata = {
                    **asdict(trace),
                    "name": trace.profile,
                    "kind": "controller-response",
                    "stage_index": stage_index,
                    "sample_start": len(rows),
                    "sample_stop": len(rows) + len(trace.time_s),
                    "start_sim_time_us": start,
                    "end_sim_time_us": start + round((trace.time_s[-1] + 0.05) * 1e6),
                    "hold_position": profile.hold_position,
                    "commands": [asdict(c) for c in profile.commands],
                    "initial_controller_state_valid": True,
                    "initial_hover_thrust_normalized": 0.57,
                }
                stages.append(metadata)
            for j, time_s in enumerate(trace.time_s):
                command = trace.command_velocity_body_flu_mps[j]
                rows.append(
                    {
                        "sim_time_us": start + round(time_s * 1e6),
                        "requested_at_us": start + round(time_s * 1e6),
                        "stage_index": stage_index,
                        "phase": "measurement",
                        "position_enu_m": trace.position_enu_m[j],
                        "attitude_flu_enu_wxyz": trace.attitude_flu_enu_wxyz[j],
                        "linear_velocity_enu_mps": trace.velocity_enu_mps[j],
                        "velocity_body_flu_mps": trace.velocity_body_flu_mps[j],
                        "angular_velocity_body_flu_radps": trace.angular_velocity_body_flu_radps[j],
                        "angular_velocity_enu_radps": trace.angular_velocity_body_flu_radps[j],
                        "requested_command": {
                            "forward_mps": command[0],
                            "right_mps": -command[1],
                            "down_mps": -command[2],
                            "yaw_rate_radps": -trace.command_yaw_rate_radps[j],
                        },
                    }
                )
        metadata = {
            "flight_index": index,
            "seed": 1001 + index,
            "run_id": root.name,
            "completed": True,
            "sample_count": len(rows),
            "sample_rate_hz": 20.0,
            "stages": stages,
            "controller_state_method": "px4-ulog-controller-boundary-v2",
        }
        (directory / "flight.json").write_text(json.dumps(metadata))
        (directory / "motion.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        checkpoint = {
            "flight": {
                "index": index,
                "seed": 1001 + index,
                "directory": directory.relative_to(root).as_posix(),
            },
            "completed": True,
            "files": [record(directory / name) for name in ("flight.json", "motion.jsonl")],
        }
        (directory / "complete.json").write_text(json.dumps(checkpoint))
        flights.append(
            {
                **record(directory / "complete.json"),
                "path": (directory / "complete.json").relative_to(root).as_posix(),
            }
        )
    (root / "definition.json").write_text(
        json.dumps(
            {
                "flight_parameters_id": "test",
                "start_seed": 1001,
                "flight_count": count,
                "stages": ["hover", *[profile.name for profile in definition.profiles]],
                "vehicle": "x500_01",
                "command_rate_hz": 20.0,
            }
        )
    )
    (root / "flight_parameters.json").write_text(
        json.dumps(
            {
                "artifact_kind": "robotics-experiments-flight-dataset",
                "schema_version": 1,
                "run_id": root.name,
                "completed": True,
                "flight_parameters_id": "test",
                "definition": record(root / "definition.json"),
                "flights": flights,
            }
        )
    )
    return root


@pytest.fixture(autouse=True)
def prepared_dataset(tmp_path, monkeypatch):
    source = _raw_dataset(tmp_path / "source")
    run = tmp_path / "prepared"
    flight_parameters_module.prepare_alignment_reference(
        flight_parameters_artifact=source,
        definition=_definition(backend="px4-sitl", evaluator=_Evaluator()),
        run_dir=run,
    )
    monkeypatch.setenv("ROBOTICS_ALIGNMENT_RUN_DIR", str(run))
    return run


def test_controller_yaml_loads_and_generic_expansion_accepts_another_field() -> None:
    target = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"
    )
    assert target.backend == "isaac-lab"
    assert target.alignment_id == "x500-px4-v1"
    assert target.controller_schema_version == 1
    assert target.base_settings.command_delay_s == pytest.approx(0.0)
    assert target.base_settings.attitude_controller_rate_hz == pytest.approx(204.0)
    assert target.base_settings.kp_vel == pytest.approx((1.8, 1.8, 4.0))
    assert target.base_settings.horizontal_velocity_error_gain_scale == pytest.approx(1.23)
    assert target.base_settings.horizontal_velocity_error_gain_full_mps == pytest.approx(0.30)
    assert target.base_settings.horizontal_velocity_error_gain_end_mps == pytest.approx(0.50)
    assert target.base_settings.horizontal_velocity_braking_gain_scale == pytest.approx(0.50)
    assert target.base_settings.horizontal_velocity_braking_gain_full_mps == pytest.approx(0.070)
    assert target.base_settings.horizontal_velocity_braking_gain_end_mps == pytest.approx(0.085)
    assert target.base_settings.velocity_observation_bias_command_start_mps == pytest.approx(0.040)
    assert target.base_settings.velocity_observation_bias_command_full_mps == pytest.approx(0.060)
    assert target.base_settings.ki_vel == pytest.approx((0.4, 0.20, 2.0))
    assert target.base_settings.kd_vel == pytest.approx((0.2, 0.2, 0.0))
    assert target.base_settings.velocity_integral_limit_mps2 == pytest.approx((0.35, 0.35, 4.0))
    assert target.search.strategy == "coarse-to-fine-beam"
    assert target.search.beam_width == 256
    assert target.search.passes_per_stage == 1
    assert target.search.refinement_levels == 0
    assert target.search.initial_settings == ()
    assert tuple(
        (family.name, family.control_mode, family.metric) for family in target.probe_families
    ) == (
        ("velocity", "velocity", "translational-controller-quality"),
        ("ctbr", "ctbr", "ctbr-controller-quality"),
    )
    assert {profile.axis for profile in target.probe_families[0].profiles} >= {
        "x",
        "y",
        "z",
    }
    assert {profile.axis for profile in target.probe_families[1].profiles} >= {
        "roll",
        "pitch",
        "yaw",
        "collective",
    }
    assert tuple(parameter.field for parameter in target.parameters) == (
        "attitude_controller_rate_hz",
        "velocity_estimator_bias_decay_hz",
        "velocity_estimator_bias_hold_decay_hz",
        "velocity_observation_bias_mps",
        "velocity_filter_hz",
        "velocity_observation_scale",
        "velocity_observation_hold_scale",
        "ki_vel",
        "kd_vel",
        "horizontal_velocity_error_gain_scale",
        "horizontal_velocity_error_gain_start_mps",
        "horizontal_velocity_braking_gain_scale",
        "horizontal_velocity_braking_gain_start_mps",
        "horizontal_velocity_braking_gain_full_mps",
        "horizontal_velocity_braking_gain_end_mps",
        "angular_velocity_filter_hz",
        "kp_vel",
        "velocity_integral_limit_mps2",
        "ctbr_effective_collective_scale",
        "velocity_observation_bias_command_start_mps",
        "velocity_observation_bias_command_full_mps",
    )
    assert next(
        profile
        for profile in target.probe_families[0].profiles
        if profile.name == "stability-hover-pose"
    ).hold_position
    extended = replace(
        target,
        parameters=(
            *target.parameters,
            ParameterSpec("max_tilt_cmd_deg", 2, (20.0, 30.0)),
        ),
    )
    assert extended.parameters[-1].stage == 2
    assert extended.parameters[-1].candidates == (20.0, 30.0)


def test_trace_requires_complete_controller_boundary() -> None:
    raw = json.loads(
        json.dumps(
            asdict(
                _response_traces(
                    profiles=_profiles(),
                    settings=_settings(),
                )[0]
            )
        )
    )
    for field in (
        "initial_controller_state_valid",
        "initial_acceleration_setpoint_enu_mps2",
        "initial_linear_acceleration_enu_mps2",
        "initial_angular_acceleration_body_flu_radps2",
        "initial_rate_integral_body_flu",
        "initial_hover_thrust_normalized",
    ):
        raw.pop(field)

    with pytest.raises(ValueError, match="fields"):
        _trace(value=raw, name="trace")  # noqa: SLF001 - trace contract


def test_trace_rejects_out_of_range_initial_hover_thrust() -> None:
    raw = json.loads(
        json.dumps(asdict(_response_traces(profiles=_profiles(), settings=_settings())[0]))
    )
    raw["initial_hover_thrust_normalized"] = 1.01

    with pytest.raises(ValueError, match="must not exceed 1"):
        _trace(value=raw, name="trace")  # noqa: SLF001 - trace validation contract


def test_ctbr_family_has_independent_roll_pitch_yaw_and_collective_gates() -> None:
    definition = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"
    )
    family = next(family for family in definition.probe_families if family.name == "ctbr")
    reference = _ctbr_response_traces(family=family)

    passing = metric(name=family.metric)(reference, reference)

    assert passing.quality_passed
    assert {gate.name for gate in passing.gates} == {
        *(f"{axis}_body_rate_nrmse" for axis in ("roll", "pitch", "yaw")),
        *(f"{axis}_envelope_body_rate_nrmse" for axis in ("roll", "pitch", "yaw")),
        *(f"{axis}_attitude_error_deg" for axis in ("roll", "pitch", "yaw")),
        "collective_velocity_nrmse",
        "collective_acceleration_nrmse",
    }
    roll_index = next(
        index
        for index, trace in enumerate(reference)
        if trace.profile == "validation-roll-envelope"
    )
    roll = reference[roll_index]
    modified = replace(
        roll,
        angular_velocity_body_flu_radps=tuple(
            (value[0] * 0.5, value[1], value[2]) for value in roll.angular_velocity_body_flu_radps
        ),
    )
    target = (*reference[:roll_index], modified, *reference[roll_index + 1 :])

    failed = metric(name=family.metric)(reference, target)

    assert not failed.quality_passed
    assert {gate.name for gate in failed.gates if gate.value > gate.limit} == {
        "roll_body_rate_nrmse",
        "roll_envelope_body_rate_nrmse",
    }


def test_ctbr_rotation_profiles_have_zero_net_commanded_rotation() -> None:
    definition = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/px4_sitl_x500_velocity.yaml"
    )
    family = next(family for family in definition.probe_families if family.name == "ctbr")

    for profile in family.profiles:
        commanded_rotation = tuple(
            sum(
                command.body_rate_flu_radps[axis] * command.duration_s
                for command in profile.commands
            )
            for axis in range(3)
        )
        assert commanded_rotation == pytest.approx(_ZERO3, abs=1e-12), profile.name


def test_align_recovers_known_delay_and_per_axis_gain_on_validation(
    tmp_path: Path,
) -> None:
    known = _settings(delay=0.05, kp=(2.0, 3.0, 4.0))
    parameters = (
        ParameterSpec("command_delay_s", 1, (0.0, 0.05, 0.1)),
        ParameterSpec(
            "kp_vel",
            1,
            ((1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (4.0, 3.0, 2.0)),
        ),
    )
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
        parameters=parameters,
    )
    result = align(target, reference)
    selected = result.candidates[result.selected_candidate_index]

    assert result.controller.selected_settings.command_delay_s == 0.05
    assert result.controller.selected_settings.kp_vel == (2.0, 3.0, 4.0)
    assert selected.metrics["velocity.x_velocity_nrmse"] == 0.0
    assert result.controller.advisory_quality_passed


def test_staged_beam_search_is_deterministic() -> None:
    known = _settings(delay=0.05, kp=(2.0, 3.0, 4.0))
    parameters = (
        ParameterSpec("command_delay_s", 1, (0.0, 0.05, 0.1)),
        ParameterSpec(
            "kp_vel",
            1,
            ((1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (4.0, 3.0, 2.0)),
        ),
        ParameterSpec(
            "kd_vel",
            1,
            ((0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)),
        ),
        ParameterSpec("max_tilt_cmd_deg", 2, (20.0, 30.0, 40.0)),
    )
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
        parameters=parameters,
    )

    first = align(target, reference)
    second = align(target, reference)

    assert [candidate.settings for candidate in first.candidates] == [
        candidate.settings for candidate in second.candidates
    ]
    assert first.controller.selected_settings == second.controller.selected_settings


def test_staged_beam_search_reports_initial_candidate_failure() -> None:
    class FailedEvaluator(_Evaluator):
        def evaluate_scored_batch(self, *, definition, settings, reference_traces):
            del definition, reference_traces
            return tuple("ctbr trace failed" for _settings in settings)

    target = _definition(backend="isaac-lab", evaluator=FailedEvaluator())

    with pytest.raises(RuntimeError, match="unusable: ctbr trace failed"):
        _search(target=target, reference_traces=())


def test_beam_retains_per_objective_champions() -> None:
    name = "test-diverse-beam-v1"

    def diverse_metric(reference, target, /):
        del reference
        peak = max(abs(value[0]) for trace in target for value in trace.velocity_body_flu_mps)
        return MetricResult(
            metrics={"velocity_rmse_mps": 0.0},
            ranking_values=(abs(peak - 2.0),),
            fit_metrics={"velocity_rmse_mps": 0.0},
            validation_metrics={
                "velocity.x_velocity_nrmse": abs(peak - 1.0),
                "velocity.y_velocity_nrmse": abs(peak - 3.0),
            },
        )

    register_metric(name=name, metric=diverse_metric)
    parameters = (
        ParameterSpec(
            "kp_vel",
            1,
            ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0), (3.0, 1.0, 1.0)),
        ),
        ParameterSpec("max_tilt_cmd_deg", 2, (20.0, 40.0)),
    )
    target = replace(
        _definition(
            backend="isaac-lab",
            evaluator=_Evaluator(),
            parameters=parameters,
            metric_name=name,
        ),
        search=SearchSpec(
            strategy="coarse-to-fine-beam",
            beam_width=2,
            passes_per_stage=1,
        ),
    )
    reference_traces = _response_traces(
        profiles=target.profiles,
        settings=_settings(kp=(1.0, 1.0, 1.0)),
    )

    outcome = _staged_beam_search(
        response_limits={"velocity.x_velocity_nrmse": 0.1, "velocity.y_velocity_nrmse": 0.1},
        hover_limits=_hover_limits(),
        target=target,
        reference_traces=reference_traces,
    )

    stage_two_x_gains = {
        score.settings.kp_vel[0]
        for score in outcome.candidate_scores
        if score.settings.max_tilt_cmd_deg != 30.0
    }
    assert {1.0, 3.0} <= stage_two_x_gains


def test_staged_beam_search_batches_without_changing_deterministic_order() -> None:
    known = _settings(delay=0.05, kp=(2.0, 3.0, 4.0))
    parameters = (
        ParameterSpec("command_delay_s", 1, (0.0, 0.05, 0.1)),
        ParameterSpec(
            "kp_vel",
            1,
            ((1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (4.0, 3.0, 2.0)),
        ),
    )
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    serial = align(
        _definition(
            backend="isaac-lab",
            evaluator=_Evaluator(),
            parameters=parameters,
        ),
        reference,
    )
    batch_evaluator = _BatchEvaluator()
    batched = align(
        _definition(
            backend="isaac-lab",
            evaluator=batch_evaluator,
            parameters=parameters,
        ),
        reference,
    )

    assert [candidate.settings for candidate in batched.candidates] == [
        candidate.settings for candidate in serial.candidates
    ]
    assert [candidate.ranking_values for candidate in batched.candidates] == [
        candidate.ranking_values for candidate in serial.candidates
    ]
    assert batched.controller.selected_settings == serial.controller.selected_settings
    assert max(batch_evaluator.batch_sizes) > 1


def test_alignment_loads_qualified_traces_for_each_recording_set() -> None:
    class ScoredEvaluator(_Evaluator):
        def __init__(self) -> None:
            super().__init__()
            self.trace_loads = 0

        def evaluate(self, *, definition, settings, reference_traces):
            del reference_traces
            self.trace_loads += 1
            return _response_traces(profiles=definition.profiles, settings=settings)

        def evaluate_scored_batch(self, *, definition, settings, reference_traces):
            assert reference_traces is not None
            score = metric(name=definition.selection_metric)
            return tuple(
                score(
                    reference_traces,
                    _response_traces(profiles=definition.profiles, settings=value),
                )
                for value in settings
            )

    evaluator = ScoredEvaluator()
    target = _definition(
        backend="isaac-lab",
        evaluator=evaluator,
        parameters=(ParameterSpec("command_delay_s", 1, (0.0, 0.05)),),
    )
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=target.base_settings),
    )

    result = align(target, reference)

    assert result.controller.selected_settings == target.base_settings
    assert evaluator.trace_loads == 3


def test_alignment_selects_response_candidate_inside_flight_parameters() -> None:
    class EnvelopeChoiceEvaluator(_Evaluator):
        def evaluate(self, *, definition, settings, reference_traces):
            del reference_traces
            traces = _response_traces(profiles=definition.profiles, settings=settings)
            if settings.command_delay_s != 0.0:
                return traces
            index = next(
                index
                for index, trace in enumerate(traces)
                if trace.profile == "stability-hover-pose"
            )
            escaped = replace(
                traces[index],
                position_enu_m=tuple((1.0, 0.0, 0.0) for _value in traces[index].position_enu_m),
            )
            return (*traces[:index], escaped, *traces[index + 1 :])

        def evaluate_scored_batch(self, *, definition, settings, reference_traces):
            del definition, reference_traces
            return tuple(
                MetricResult(
                    metrics={},
                    ranking_values=(value.command_delay_s,),
                    fit_metrics={},
                    validation_metrics={},
                    quality_passed=True,
                )
                for value in settings
            )

    target = _definition(
        backend="isaac-lab",
        evaluator=EnvelopeChoiceEvaluator(),
        parameters=(ParameterSpec("command_delay_s", 1, (0.0, 0.05)),),
    )
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(
            reference_settings=replace(target.base_settings, command_delay_s=0.05)
        ),
    )

    result = align(target, reference)

    assert result.controller.selected_settings.command_delay_s == pytest.approx(0.05)
    assert result.controller.advisory_quality_passed


def test_effective_key_aliases_float32_equivalent_settings_and_stops_fixed_point() -> None:
    evaluator = _BatchEvaluator()
    target = replace(
        _definition(
            backend="isaac-lab",
            evaluator=evaluator,
            parameters=(
                ParameterSpec(
                    "hover_thrust_normalized",
                    1,
                    (0.5700000001, 0.5700000002),
                ),
            ),
        ),
        search=SearchSpec(
            strategy="coarse-to-fine-beam",
            beam_width=8,
            passes_per_stage=3,
        ),
    )
    reference_traces = _response_traces(profiles=target.profiles, settings=target.base_settings)

    outcome = _search(target=target, reference_traces=reference_traces)

    assert len(outcome.candidate_scores) == 3
    assert outcome.statistics["submitted_unique_candidates"] == 1
    assert outcome.statistics["effective_key_aliases"] == 2
    assert outcome.statistics["skipped_fixed_point_passes"] == 1
    assert evaluator.batch_sizes == [1]


def test_coarse_to_fine_search_refines_only_surviving_intervals() -> None:
    known = _settings(delay=0.25)
    target = replace(
        _definition(
            backend="isaac-lab",
            evaluator=_BatchEvaluator(),
            parameters=(ParameterSpec("command_delay_s", 1, (0.0, 1.0)),),
        ),
        base_settings=_settings(delay=0.5),
        search=SearchSpec(
            strategy="coarse-to-fine-beam",
            beam_width=2,
            passes_per_stage=1,
            refinement_levels=1,
            min_refinement_survivors=1,
        ),
    )
    reference_traces = _response_traces(profiles=target.profiles, settings=known)

    outcome = _search(target=target, reference_traces=reference_traces)

    assert outcome.selected.settings.command_delay_s == pytest.approx(0.25)
    assert any(
        record["field"] == "command_delay_s" and record["candidate"] == pytest.approx(0.25)
        for record in outcome.generated_refinements
    )


def test_search_rejects_invalid_cross_field_combinations_before_evaluation() -> None:
    target = replace(
        _definition(
            backend="isaac-lab",
            evaluator=_BatchEvaluator(),
            parameters=(
                ParameterSpec(
                    "horizontal_velocity_error_gain_start_mps",
                    1,
                    (0.2, 0.4),
                ),
            ),
        ),
        base_settings=replace(
            _settings(),
            horizontal_velocity_error_gain_start_mps=0.1,
            horizontal_velocity_error_gain_full_mps=0.3,
            horizontal_velocity_error_gain_end_mps=0.8,
        ),
    )
    reference_traces = _response_traces(
        profiles=target.profiles,
        settings=target.base_settings,
    )

    outcome = _search(target=target, reference_traces=reference_traces)

    assert outcome.statistics["rejected_invalid_settings"] >= 1
    assert all(
        candidate.settings.horizontal_velocity_error_gain_start_mps
        < candidate.settings.horizontal_velocity_error_gain_full_mps
        < candidate.settings.horizontal_velocity_error_gain_end_mps
        for candidate in outcome.candidate_scores
    )


def test_coarse_to_fine_search_reseeds_from_global_archive() -> None:
    class ArchiveEvaluator(_Evaluator):
        def evaluate_scored_batch(self, *, definition, settings, reference_traces):
            del definition, reference_traces
            results = []
            for value in settings:
                delay = value.command_delay_s
                velocity_filter = value.velocity_filter_hz
                tilt = value.max_tilt_cmd_deg
                if delay == 1.0 and tilt == 40.0:
                    ranking, beam = 0.0, (0.0, 0.0)
                elif delay == 1.0:
                    ranking, beam = 1.0, (5.0, 5.0)
                elif delay == 2.0 and velocity_filter == (20.0, 20.0, 20.0):
                    ranking, beam = 9.0, (0.0, 0.0)
                elif delay == 2.0:
                    ranking, beam = 10.0, (0.0, 10.0)
                elif delay == 3.0:
                    ranking, beam = 10.0, (10.0, 0.0)
                else:
                    ranking, beam = 20.0, (20.0, 20.0)
                results.append(
                    MetricResult(
                        metrics={"velocity_rmse_mps": ranking},
                        ranking_values=(ranking,),
                        fit_metrics={"velocity_rmse_mps": ranking},
                        validation_metrics={"worst_axis_velocity_nrmse": ranking},
                        beam_values=beam,
                    )
                )
            return tuple(results)

    target = replace(
        _definition(
            backend="isaac-lab",
            evaluator=ArchiveEvaluator(),
            parameters=(
                ParameterSpec("command_delay_s", 1, (1.0, 2.0, 3.0)),
                ParameterSpec(
                    "velocity_filter_hz",
                    1,
                    ((10.0, 10.0, 10.0), (20.0, 20.0, 20.0)),
                ),
                ParameterSpec("max_tilt_cmd_deg", 1, (30.0, 40.0)),
            ),
        ),
        search=SearchSpec(
            strategy="coarse-to-fine-beam",
            beam_width=2,
            passes_per_stage=1,
        ),
    )

    outcome = _search(target=target, reference_traces=())

    assert outcome.selected.settings.command_delay_s == 1.0
    assert outcome.selected.settings.max_tilt_cmd_deg == 40.0


def test_coarse_to_fine_search_expands_each_initial_setting() -> None:
    base = _settings(delay=0.0)
    seed = replace(base, command_delay_s=0.1, velocity_filter_hz=(20.0, 20.0, 20.0))
    target = replace(
        _definition(
            backend="isaac-lab",
            evaluator=_BatchEvaluator(),
            parameters=(ParameterSpec("max_tilt_cmd_deg", 1, (40.0,)),),
        ),
        base_settings=base,
        search=SearchSpec(
            strategy="coarse-to-fine-beam-v3",
            beam_width=8,
            passes_per_stage=1,
            initial_settings=(seed,),
        ),
    )
    reference_traces = _response_traces(profiles=target.profiles, settings=base)

    outcome = _search(target=target, reference_traces=reference_traces)

    assert any(
        score.settings.command_delay_s == 0.1
        and score.settings.velocity_filter_hz == (20.0, 20.0, 20.0)
        and score.settings.max_tilt_cmd_deg == 40.0
        for score in outcome.candidate_scores
    )


@pytest.mark.parametrize("indent", [None, 2])
def test_file_evaluator_indexes_candidate_diagnostics_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indent: int | None,
) -> None:
    settings = _settings()
    definition = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
    )
    encoded = json.dumps(
        asdict(settings),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    key = hashlib.sha256(encoded).hexdigest()
    path = tmp_path / "target_candidates.json"
    (tmp_path / "reference.json").write_text('{"dataset":"dataset.json","set_name":"train"}')
    source_digest = "producer-digest"
    (tmp_path / "dataset.json").write_text(json.dumps({"source_digest": source_digest}))
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "set_name": "train",
                "source_digest": source_digest,
                "controller_id": definition.controller_id,
                "definition_fingerprint": definition.definition_fingerprint,
                "image": definition.image,
                "candidates": {
                    key: [
                        asdict(trace)
                        for trace in _response_traces(
                            profiles=definition.profiles,
                            settings=settings,
                        )
                    ]
                },
                "failures": {},
            },
            indent=indent,
        )
    )
    original_index = _index_candidate_diagnostics
    index_count = 0

    def counted_index(
        *,
        path: Path,
    ) -> tuple[dict[str, object], dict[str, tuple[int, int]], dict[str, str]]:
        nonlocal index_count
        index_count += 1
        return original_index(path=path)

    monkeypatch.setenv("ROBOTICS_ALIGNMENT_RUN_DIR", str(tmp_path))
    monkeypatch.setattr(
        "robotics_experiments.alignment._io._index_candidate_diagnostics",
        counted_index,
    )
    evaluator = _TraceFileEvaluator(backend="isaac-lab")

    first = evaluator.evaluate(
        definition=definition,
        settings=settings,
        reference_traces=None,
    )
    second = evaluator.evaluate(
        definition=definition,
        settings=settings,
        reference_traces=None,
    )

    assert first == second
    assert index_count == 1
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    (validation_dir / "reference.json").write_text(
        '{"dataset":"../dataset.json","set_name":"validation"}'
    )
    payload = json.loads(path.read_text())
    payload["set_name"] = "validation"
    payload["candidates"][key][0]["start_sim_time_us"] = 123456
    validation_path = validation_dir / path.name
    validation_path.write_text(json.dumps(payload))
    validation_evaluator = evaluator.for_set("validation")
    held_out = validation_evaluator.evaluate(
        definition=definition,
        settings=settings,
        reference_traces=None,
    )
    assert held_out[0].start_sim_time_us == 123456
    assert held_out != first
    assert (
        evaluator.evaluate(definition=definition, settings=settings, reference_traces=None) == first
    )
    assert index_count == 2
    payload["set_name"] = "train"
    validation_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="different recording set"):
        evaluator.for_set("validation").evaluate(
            definition=definition,
            settings=settings,
            reference_traces=None,
        )

    payload["set_name"] = "validation"
    payload["source_digest"] = "different-producer"
    validation_path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="different reference recording"):
        evaluator.for_set("validation").evaluate(
            definition=definition,
            settings=settings,
            reference_traces=None,
        )


def test_controller_quality_reports_braking_distance_and_residual_speed_error() -> None:
    settings = _settings(kp=(1.0, 1.0, 1.0))
    reference = _response_traces(profiles=_profiles(), settings=settings)
    trace_index = next(
        index for index, trace in enumerate(reference) if trace.profile == "validation-x"
    )
    source = reference[trace_index]
    transition = next(
        index
        for index in range(1, len(source.command_velocity_body_flu_mps))
        if source.command_velocity_body_flu_mps[index - 1][0] != 0.0
        and source.command_velocity_body_flu_mps[index][0] == 0.0
    )
    target_velocity = list(source.velocity_body_flu_mps)
    target_world_velocity = list(source.velocity_enu_mps)
    target_position = list(source.position_enu_m)
    target_velocity[transition] = (0.02, 0.0, 0.0)
    target_world_velocity[transition] = (0.02, 0.0, 0.0)
    target_position[transition + 1] = (
        target_position[transition][0] + 0.02,
        target_position[transition][1],
        target_position[transition][2],
    )
    modified = replace(
        source,
        velocity_body_flu_mps=tuple(target_velocity),
        velocity_enu_mps=tuple(target_world_velocity),
        position_enu_m=tuple(target_position),
    )
    target = (*reference[:trace_index], modified, *reference[trace_index + 1 :])

    result = metric(name="controller-quality")(
        reference,
        target,
    )

    assert result.validation_metrics["velocity.x_braking_distance_error_m"] > 0.0
    assert result.validation_metrics["velocity.x_residual_speed_error_mps"] >= 0.0


def test_vertical_envelope_has_a_dedicated_gate_without_redefining_braking() -> None:
    settings = _settings(kp=(1.0, 1.0, 1.0))
    reference = _response_traces(profiles=_profiles(), settings=settings)
    trace_index = next(
        index for index, trace in enumerate(reference) if trace.profile == "validation-z-envelope"
    )
    source = reference[trace_index]
    modified = replace(
        source,
        position_enu_m=tuple(
            (position[0], position[1], position[2] * 10.0) for position in source.position_enu_m
        ),
    )
    target = (*reference[:trace_index], modified, *reference[trace_index + 1 :])

    result = metric(name="controller-quality")(reference, target)

    assert result.validation_metrics["velocity.z_braking_distance_error_m"] == 0.0


def test_braking_stop_time_preserves_exact_trace_grid_boundary() -> None:
    source = _response_traces(
        profiles=_profiles(),
        settings=_settings(kp=(1.0, 1.0, 1.0)),
    )[0]
    time_s = tuple(1_000_000.0 + index * 0.05 for index in range(6))
    command = ((1.0, 0.0, 0.0),) * 2 + (_ZERO3,) * 4
    reference_velocity = (
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.6, 0.0, 0.0),
        _ZERO3,
        _ZERO3,
        _ZERO3,
    )
    target_velocity = (
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.6, 0.0, 0.0),
        (0.02, 0.0, 0.0),
        (0.02, 0.0, 0.0),
        _ZERO3,
    )
    positions = tuple((index * 0.05, 0.0, 0.0) for index in range(6))
    reference = replace(
        source,
        time_s=time_s,
        command_velocity_body_flu_mps=command,
        velocity_body_flu_mps=reference_velocity,
        position_enu_m=positions,
    )
    target = replace(
        reference,
        velocity_body_flu_mps=target_velocity,
    )

    events = _braking_events(reference=reference, target=target, axis=0)

    assert events[0][-1] == 0.1


def test_braking_ignores_transition_before_reference_reaches_half_command() -> None:
    source = _response_traces(
        profiles=_profiles(),
        settings=_settings(kp=(1.0, 1.0, 1.0)),
    )[0]
    command = ((0.04, 0.0, 0.0),) * 2 + (_ZERO3,) * 2
    reference_velocity = (
        _ZERO3,
        (0.009, 0.0, 0.0),
        (0.011, 0.0, 0.0),
        _ZERO3,
    )
    reference = replace(
        source,
        time_s=(0.0, 0.05, 0.10, 0.15),
        command_velocity_body_flu_mps=command,
        velocity_body_flu_mps=reference_velocity,
        position_enu_m=((0.0, 0.0, 0.0),) * 4,
    )

    assert _braking_events(reference=reference, target=reference, axis=0) == []


def test_registered_metric_can_change_selection_without_changing_aligner(
    tmp_path: Path,
) -> None:
    name = "test-highest-x-response-v1"

    def highest_x(reference, target, /):
        peak = max(abs(value[0]) for trace in target for value in trace.velocity_body_flu_mps)
        return MetricResult(
            metrics={"negative_peak": -peak},
            ranking_values=(-peak,),
            fit_metrics={"velocity_rmse_mps": 0.0},
            validation_metrics={
                key: (abs(3.0 - peak) if key == "velocity.x_velocity_nrmse" else 0.0)
                for key in load_config(
                    REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml"
                ).alignment.evaluation.response_limits
                if key.startswith("velocity.")
            },
        )

    register_metric(name=name, metric=highest_x)
    parameters = (
        ParameterSpec(
            "kp_vel",
            1,
            ((1.0, 1.0, 1.0), (3.0, 1.0, 1.0)),
        ),
    )
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=_settings(kp=(1.0, 1.0, 1.0))),
        metric_name=name,
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
        parameters=parameters,
        metric_name=name,
    )

    assert align(target, reference).controller.selected_settings.kp_vel == (3.0, 1.0, 1.0)


def test_experiment_fingerprint_includes_capture_but_excludes_execution(
    tmp_path: Path,
) -> None:
    definition = _definition(backend="isaac-lab", evaluator=_Evaluator())
    fingerprints = []
    for name, old, new in (
        ("baseline", "", ""),
        ("execution", "worker_count: 3", "worker_count: 2"),
        ("capture", "stable_hold_s: 0.5", "stable_hold_s: 0.6"),
    ):
        root = tmp_path / name / "configs"
        shutil.copytree(REPO_ROOT / "configs", root)
        base = root / "base.yaml"
        if old:
            base.write_text(base.read_text().replace(old, new))
        fingerprints.append(
            _resolved_experiment_fingerprint(
                target=replace(
                    definition,
                    experiment_config=str(root / "alignment_px4_x500_velocity.yaml"),
                ),
                repo_root=REPO_ROOT,
            )
        )

    assert fingerprints[0] == fingerprints[1]
    assert fingerprints[0] != fingerprints[2]


def test_module_fingerprint_changes_with_controller_or_metric_source(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text("contract = 1\n")
    first = _module_fingerprint(str(source))
    source.write_text("contract = 2\n")

    assert _module_fingerprint(str(source)) != first


def test_controller_and_selection_artifacts_are_compact_and_exact(tmp_path: Path) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    result = align(
        _definition(
            backend="isaac-lab",
            evaluator=_Evaluator(),
            parameters=(
                ParameterSpec(
                    "kp_vel",
                    1,
                    ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0)),
                ),
            ),
        ),
        _definition(
            backend="px4-sitl",
            evaluator=_Evaluator(reference_settings=known),
        ),
    )
    run_dir = tmp_path / "run"
    controller_path, selection_path = write_alignment_result(result=result, run_dir=run_dir)

    payload = json.loads(controller_path.read_text())
    assert controller_path.stat().st_size < 16 * 1024
    assert (
        not {
            "candidates",
            "declared_parameters",
            "selection_metric",
            "ranking_values",
            "metrics",
        }
        & payload.keys()
    )
    controller = load_controller_artifact(run_dir)
    assert controller.certified_control_modes == ("velocity",)
    assert controller.flight_parameters_run_id == "source"
    reconstructed = load_selection(run_dir=run_dir, controller=controller)
    assert reconstructed.candidates == result.candidates
    assert reconstructed.selected_candidate_index == result.selected_candidate_index
    assert selection_path.stat().st_size > 0


def test_controller_loader_is_strict_and_rejects_legacy(tmp_path: Path) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    result = align(
        _definition(backend="isaac-lab", evaluator=_Evaluator()),
        _definition(
            backend="px4-sitl",
            evaluator=_Evaluator(reference_settings=known),
        ),
    )
    run_dir = tmp_path / "run"
    controller_path, _selection_path = write_alignment_result(result=result, run_dir=run_dir)
    payload = json.loads(controller_path.read_text())
    attenuated_settings = {
        **payload["selected_settings"],
        "horizontal_velocity_error_gain_scale": 0.8,
    }
    payload["selected_settings"] = attenuated_settings
    payload["settings_fingerprint"] = hashlib.sha256(
        json.dumps(
            attenuated_settings,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    controller_path.write_text(json.dumps(payload))
    attenuated = load_controller_artifact(run_dir)
    assert attenuated.selected_settings.horizontal_velocity_error_gain_scale == pytest.approx(0.8)

    scalar_filter_payload = {
        **payload,
        "selected_settings": {
            **payload["selected_settings"],
            "velocity_filter_hz": 10.0,
        },
    }
    controller_path.write_text(json.dumps(scalar_filter_payload))
    with pytest.raises(ValueError, match="must contain three finite values"):
        load_controller_artifact(run_dir)

    invalid_bias_transition = {
        **payload,
        "selected_settings": {
            **payload["selected_settings"],
            "velocity_observation_bias_command_start_mps": 0.02,
            "velocity_observation_bias_command_full_mps": 0.02,
        },
    }
    controller_path.write_text(json.dumps(invalid_bias_transition))
    with pytest.raises(ValueError, match="observation-bias command transition"):
        load_controller_artifact(run_dir)

    payload["unexpected"] = True
    controller_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fields mismatch"):
        load_controller_artifact(run_dir)

    legacy = tmp_path / "alignment.json"
    legacy.write_text("{}")
    with pytest.raises(ValueError, match="fields mismatch"):
        load_controller_artifact(legacy)


def test_selection_serialization_is_deterministic(tmp_path: Path) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    result = align(
        _definition(backend="isaac-lab", evaluator=_Evaluator()),
        _definition(
            backend="px4-sitl",
            evaluator=_Evaluator(reference_settings=known),
        ),
    )
    _first_controller, first_selection = write_alignment_result(
        result=result,
        run_dir=tmp_path / "first",
    )
    _second_controller, second_selection = write_alignment_result(
        result=result,
        run_dir=tmp_path / "second",
    )
    assert first_selection.read_bytes() == second_selection.read_bytes()


def test_failed_boundary_diagnostic_blocks_acceptance(tmp_path: Path) -> None:
    report = tmp_path / "diagnostic.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "passed": False,
                "failures": ["vertical collective mismatch"],
            }
        )
    )

    with pytest.raises(ValueError, match="vertical collective mismatch"):
        _require_passed_diagnostic(path=report, name="controller boundary replay")


@pytest.mark.parametrize(
    ("count", "expected"), [(3, (1, 1, 1)), (10, (8, 1, 1)), (100, (80, 10, 10)), (7, (5, 1, 1))]
)
def test_whole_flight_percentages_are_complete_disjoint_and_reproducible(count, expected):
    arguments = dict(
        flight_indices=tuple(range(count)), train_percent=80, validation_percent=10, seed=1001
    )
    split = split_flights(**arguments)
    assert split == split_flights(**arguments)
    assert tuple(map(len, split.values())) == expected
    assert sorted(index for indices in split.values() for index in indices) == list(range(count))
    assert split != split_flights(**{**arguments, "seed": 1002})


@pytest.mark.parametrize(
    ("value", "score", "accepted"), [(0.09, 0.9, True), (0.1, 1.0, True), (0.11, 1.1, False)]
)
def test_numerical_evaluation_threshold(value, score, accepted):
    result = evaluate_metrics(
        measurements={"response": value, "hover": 0.001}, limits={"response": 0.1, "hover": 0.005}
    )
    assert result.score == pytest.approx(score)
    assert result.accepted is accepted
    assert [check.name for check in result.checks] == ["response", "hover"]


def test_worst_check_controls_ranking_and_acceptance():
    worse = evaluate_metrics(
        measurements={"x": 1.2, "y": 0.0, "z": 0.0}, limits=dict.fromkeys(("x", "y", "z"), 1.0)
    )
    better = evaluate_metrics(
        measurements={"x": 1.1, "y": 1.1, "z": 1.1}, limits=dict.fromkeys(("x", "y", "z"), 1.0)
    )
    assert not worse.accepted and not better.accepted
    assert better.score < worse.score


def test_reference_views_preserve_source_samples_and_minimal_membership(prepared_dataset):
    metadata = json.loads((prepared_dataset / "dataset.json").read_text())
    assert set(metadata) == {"source", "source_digest", "train", "validation", "test"}
    dataset = load_flight_dataset(prepared_dataset / metadata["source"])
    before = {p: p.read_bytes() for p in dataset.run_dir.rglob("*") if p.is_file()}
    for role in ("train", "validation", "test"):
        directory = prepared_dataset if role == "train" else prepared_dataset / role
        reference = load_alignment_reference(directory / "reference.json")
        assert {trace["flight_index"] for trace in reference["traces"]} == set(metadata[role])
        assert "traces" not in json.loads((directory / "reference.json").read_text())
        for trace in reference["traces"]:
            flight = next(f for f in dataset.flights if f.flight_index == trace["flight_index"])
            view = next(
                view for view in flight_stage_views(flight) if view["name"] == trace["profile"]
            )
            assert trace["position_enu_m"] == view["position_enu_m"]
            assert trace["time_s"] == view["time_s"]
            assert trace["initial_rate_integral_body_flu"] == view["initial_rate_integral_body_flu"]
            _trace(value=trace, name="source view")
    assert before == {p: p.read_bytes() for p in before}


def test_repeated_profile_names_are_distinct_flights():
    reference = _response_traces(profiles=_profiles(), settings=_settings())
    repeated = (*reference, *(replace(trace, flight_index=1) for trace in reference))
    assert metric(name="controller-quality")(repeated, repeated).ranking_values == (0.0,)
    perturbed = tuple(
        replace(
            trace,
            velocity_body_flu_mps=tuple(
                (v[0] + 0.1, v[1], v[2]) for v in trace.velocity_body_flu_mps
            ),
        )
        if trace.flight_index == 1
        else trace
        for trace in repeated
    )
    assert metric(name="controller-quality")(repeated, perturbed).ranking_values[0] > 0
    with pytest.raises(ValueError, match="duplicate"):
        metric(name="controller-quality")((*reference, *reference), (*reference, *reference))


def test_hover_bounds_read_training_windows_only(prepared_dataset, monkeypatch):
    dataset, membership = flight_parameters_module.load_alignment_dataset(
        prepared_dataset / "dataset.json"
    )
    reader = flight_stage_views
    read = []

    def traced_reader(flight, /):
        read.append(flight.flight_index)
        return reader(flight)

    monkeypatch.setattr(flight_parameters_module, "flight_stage_views", traced_reader)
    bounds = training_hover_limits(
        dataset=dataset,
        membership=membership,
        position_margin_m=0.005,
        attitude_margin_rad=math.radians(0.25),
    )
    assert read == membership["train"]
    assert bounds["hover.position"] == 0.005
    assert bounds["hover.orientation"] == math.radians(0.25)


@pytest.mark.parametrize("test_gain", [1.0, 4.0])
def test_validation_selects_and_test_cannot_change_selection(test_gain):
    calls = []

    class SetEvaluator(_Evaluator):
        def __init__(self, role="train"):
            super().__init__()
            self.role = role

        def for_set(self, set_name, /):
            return SetEvaluator(set_name)

        def evaluate(self, *, definition, settings, reference_traces):
            calls.append((definition.backend, self.role, settings.kp_vel))
            active = settings
            if definition.backend == "px4-sitl":
                gain = {"train": 2.0, "validation": 1.0, "test": test_gain}[self.role]
                active = replace(settings, kp_vel=(gain, 1.0, 1.0))
            return _response_traces(profiles=definition.profiles, settings=active)

    target = _definition(
        backend="isaac-lab",
        evaluator=SetEvaluator(),
        parameters=(ParameterSpec("kp_vel", 1, ((1.0, 1.0, 1.0), (2.0, 1.0, 1.0))),),
    )
    reference = _definition(backend="px4-sitl", evaluator=SetEvaluator())
    result = align(target, reference)
    assert result.controller.selected_settings.kp_vel == (1.0, 1.0, 1.0)
    assert result.controller.advisory_quality_passed == (test_gain == 1.0)
    assert [entry[2] for entry in calls if entry[:2] == ("isaac-lab", "test")] == [(1.0, 1.0, 1.0)]


def _completed_run(*, root, name, result, score):
    run = root / name
    result = replace(result, controller=replace(result.controller, run_id=name))
    write_alignment_result(result=result, run_dir=run)
    (run / "dataset.json").write_bytes(
        (Path(os.environ["ROBOTICS_ALIGNMENT_RUN_DIR"]) / "dataset.json").read_bytes()
    )
    identity = {
        "schema_version": 1,
        "completed": True,
        "settings_fingerprint": result.controller.settings_fingerprint,
        "flight_parameters_fingerprint": result.controller.flight_parameters_fingerprint,
    }
    (run / "qualification.json").write_text(
        json.dumps(
            {
                **identity,
                "set_name": "test",
                "checks": [{"name": "response", "value": score, "limit": 1.0}],
            }
        )
    )
    (run / "integration.json").write_text(
        json.dumps({**identity, "checks": [{"name": "live", "value": 0.0, "limit": 1.0}]})
    )
    for role in ("train", "validation", "test"):
        directory = run if role == "train" else run / role
        directory.mkdir(exist_ok=True)
        (directory / "actuator_audit.json").write_text('{"schema_version":1,"passed":true}')
        (directory / "target_candidates.json").write_text("{}")
        (directory / "controller_boundary_replay.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "set_name": role,
                    "settings_fingerprint": result.controller.settings_fingerprint,
                    "source_digest": result.controller.flight_parameters_fingerprint,
                    "flights": [
                        {
                            "flight_index": 0,
                            "profiles": [
                                {
                                    "profile": "probe",
                                    "physical_state_timestamp_join_error_us": 0,
                                    "actuator_handoff_join_error_us": 0,
                                    "metrics": {
                                        "normalized_motor_rmse": 0.0,
                                        "collective_thrust_rmse_n": 0.0,
                                    },
                                    "physical_state_metrics": {"normalized_motor_rmse": 0.0},
                                }
                            ],
                            "limits": {
                                "normalized_motor_rmse": 0.01,
                                "collective_thrust_rmse_n": 0.05,
                            },
                            "timestamp_join_limit_us": 4000,
                        }
                    ],
                }
            )
        )
    return run


def test_completed_rejected_runs_publish_and_best_uses_the_final_score(tmp_path):
    result = align(
        _definition(backend="isaac-lab", evaluator=_Evaluator()),
        _definition(backend="px4-sitl", evaluator=_Evaluator()),
    )
    runs = []
    for name, score in (
        ("first", 3.0),
        ("better", 2.0),
        ("tied", 2.0),
        ("accepted", 1.0),
        ("worse", 4.0),
    ):
        run = _completed_run(root=tmp_path, name=name, result=result, score=score)
        finalize_alignment_bundle(run_dir=run)
        validate_alignment_bundle(run)
        publish_controller(run_dir=run)
        assert (tmp_path / "latest").resolve() == run
        assert load_controller_artifact(run).advisory_quality_passed == (score <= 1.0)
        assert (run / "target_candidates.json.zst").exists()
        assert not (run / "target_candidates.json").exists()
        runs.append(run)
        best_index = (0, 1, 1, 3, 3)[len(runs) - 1]
        assert (tmp_path / "best").resolve() == runs[best_index]
    (runs[-1] / "untracked.log").write_text("untracked")
    with pytest.raises(ValueError, match="inventory every payload"):
        validate_alignment_bundle(runs[-1])


def test_incomplete_or_mismatched_reports_cannot_finalize(tmp_path):
    result = align(
        _definition(backend="isaac-lab", evaluator=_Evaluator()),
        _definition(backend="px4-sitl", evaluator=_Evaluator()),
    )
    run = _completed_run(root=tmp_path, name="incomplete", result=result, score=2.0)
    report = json.loads((run / "integration.json").read_text())
    for change in ({"completed": False}, {"settings_fingerprint": "wrong"}):
        (run / "integration.json").write_text(json.dumps({**report, **change}))
        with pytest.raises(ValueError, match="did not complete for the selected"):
            finalize_alignment_bundle(run_dir=run)
        assert not (run / "manifest.json").exists()


def test_best_starts_again_when_comparison_inputs_change(tmp_path):
    result = align(
        _definition(backend="isaac-lab", evaluator=_Evaluator()),
        _definition(backend="px4-sitl", evaluator=_Evaluator()),
    )
    original = _completed_run(root=tmp_path, name="group-one-best", result=result, score=2.0)
    finalize_alignment_bundle(run_dir=original)
    publish_controller(run_dir=original)
    other = replace(result, controller=replace(result.controller, comparison_fingerprint="a" * 64))
    separate = _completed_run(root=tmp_path, name="group-two", result=other, score=0.1)
    finalize_alignment_bundle(run_dir=separate)
    publish_controller(run_dir=separate)
    assert (tmp_path / "best").resolve() == separate
    worse = _completed_run(root=tmp_path, name="group-one-new", result=result, score=3.0)
    finalize_alignment_bundle(run_dir=worse)
    publish_controller(run_dir=worse)
    assert (tmp_path / "best").resolve() == worse
    assert (tmp_path / "latest").resolve() == worse
