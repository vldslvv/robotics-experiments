"""Host tests for controller definition, fitting, artifacts, and publication."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

import robotics_experiments.alignment._artifact as artifact_module
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
    _compress_file,
    _require_passed_diagnostic,
    finalize_alignment_bundle,
    load_selection,
    validate_alignment_bundle,
    write_alignment_result,
)
from robotics_experiments.alignment._io import (
    _index_candidate_diagnostics,
    _trace,
    _TraceFileEvaluator,
)
from robotics_experiments.alignment._metrics import (
    _braking_events,
    metric,
    register_metric,
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
from robotics_experiments.controller import ControllerSettings, load_controller_artifact
from tests.support.paths import REPO_ROOT

_ZERO3 = (0.0, 0.0, 0.0)
_IDENTITY = (1.0, 0.0, 0.0, 0.0)


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
        ki_vel=(0.4, 0.4, 0.4),
        kd_vel=kd,
        velocity_observation_scale=(1.0, 1.0, 1.0),
        velocity_observation_bias_mps=(0.0, 0.0, 0.0),
        velocity_estimator_bias_decay_hz=0.0,
        velocity_filter_hz=(10.0, 10.0, 10.0),
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 2.0),
        hover_thrust_normalized=0.57,
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
                family="velocity",
                control_mode="velocity",
                family_metric="translational-controller-quality",
                profile=profile.name,
                split=profile.split,
                axis=profile.axis,
                time_s=tuple(index * 0.05 for index in range(len(commands))),
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
                family=family.name,
                control_mode=family.control_mode,
                family_metric=family.metric,
                profile=profile.name,
                split=profile.split,
                axis=profile.axis,
                time_s=tuple(index * 0.05 for index in range(sample_count)),
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


def test_controller_yaml_loads_and_generic_expansion_accepts_another_field() -> None:
    target = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"
    )
    assert target.backend == "isaac-lab"
    assert target.alignment_id == "x500-px4-v1"
    assert target.controller_schema_version == 1
    assert target.base_settings.command_delay_s == pytest.approx(0.0)
    assert target.base_settings.attitude_controller_rate_hz == pytest.approx(187.5)
    assert target.base_settings.kp_vel == pytest.approx((1.8, 1.8, 4.20))
    assert target.base_settings.horizontal_velocity_error_gain_scale == pytest.approx(1.0)
    assert target.base_settings.horizontal_velocity_error_gain_full_mps == pytest.approx(0.30)
    assert target.base_settings.ki_vel == pytest.approx((0.4, 0.4, 2.1))
    assert target.base_settings.kd_vel == pytest.approx((0.2, 0.2, 0.0))
    assert target.base_settings.velocity_integral_limit_mps2 == pytest.approx((0.35, 0.35, 4.0))
    assert target.search.strategy == "coarse-to-fine-beam"
    assert target.search.beam_width == 32
    assert target.search.passes_per_stage == 1
    assert target.search.refinement_levels == 2
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
        "velocity_observation_bias_mps",
        "velocity_filter_hz",
        "velocity_observation_scale",
        "velocity_integral_limit_mps2",
    )
    assert target.parameters[0].candidates == pytest.approx(
        (125.0, 142.8571, 150.0, 160.0, 166.6667, 187.5, 200.0, 250.0)
    )
    assert len(target.parameters[2].candidates) == 63
    assert target.parameters[2].candidates[0] == pytest.approx((-0.040, -0.030, -0.0039))
    assert target.parameters[2].candidates[-1] == pytest.approx((0.040, 0.030, -0.0039))
    assert target.parameters[2].selection_regularization_scale == pytest.approx(1.0)
    assert target.parameters[3].candidates[0] == pytest.approx((0.0, 0.0, 0.0))
    assert target.parameters[3].candidates[3] == pytest.approx((0.0, 0.375, 0.0))
    assert target.parameters[3].candidates[-1] == pytest.approx((0.0, 1.0, 0.0))
    assert target.parameters[4].candidates[0] == pytest.approx((0.98, 0.9800, 1.0))
    assert target.parameters[4].candidates[-1] == pytest.approx((0.98, 1.0425, 1.0))
    assert target.parameters[-1].candidates == (pytest.approx((0.35, 0.35, 4.0)),)
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
        _retain_traces = False

        def evaluate_scored_batch(self, *, definition, settings, reference_traces):
            del definition, reference_traces
            return tuple("ctbr trace failed" for _settings in settings)

    target = _definition(backend="isaac-lab", evaluator=FailedEvaluator())

    with pytest.raises(RuntimeError, match="unusable: ctbr trace failed"):
        _staged_beam_search(target=target, reference_traces=())


def test_beam_retains_per_objective_champions() -> None:
    name = "test-diverse-beam-v1"

    def diverse_metric(reference, target, /):
        del reference
        peak = max(abs(value[0]) for trace in target for value in trace.velocity_body_flu_mps)
        return MetricResult(
            metrics={"velocity_rmse_mps": 0.0},
            ranking_values=(abs(peak - 2.0),),
            fit_metrics={"velocity_rmse_mps": 0.0},
            validation_metrics={"worst_axis_velocity_nrmse": 0.0},
            beam_values=(abs(peak - 1.0), abs(peak - 3.0)),
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

    outcome = _staged_beam_search(target=target, reference_traces=reference_traces)

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

    outcome = _staged_beam_search(target=target, reference_traces=reference_traces)

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
        ),
    )
    reference_traces = _response_traces(
        profiles=target.profiles,
        settings=target.base_settings,
    )

    outcome = _staged_beam_search(target=target, reference_traces=reference_traces)

    assert outcome.statistics["rejected_invalid_settings"] >= 1
    assert all(
        candidate.settings.horizontal_velocity_error_gain_start_mps
        < candidate.settings.horizontal_velocity_error_gain_full_mps
        for candidate in outcome.candidate_scores
    )


def test_coarse_to_fine_search_reseeds_from_global_archive() -> None:
    class ArchiveEvaluator(_Evaluator):
        _retain_traces = False

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

    outcome = _staged_beam_search(target=target, reference_traces=())

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

    outcome = _staged_beam_search(target=target, reference_traces=reference_traces)

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
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
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
            validation_metrics={"worst_axis_velocity_nrmse": 0.0},
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


def test_regularized_parameter_prefers_smallest_quality_passing_correction() -> None:
    name = "test-regularized-highest-x-response-v1"

    def highest_x(reference, target, /):
        peak = max(abs(value[0]) for trace in target for value in trace.velocity_body_flu_mps)
        return MetricResult(
            metrics={"negative_peak": -peak},
            ranking_values=(-peak,),
            fit_metrics={"velocity_rmse_mps": 0.0},
            validation_metrics={"worst_axis_velocity_nrmse": 0.0},
        )

    register_metric(name=name, metric=highest_x)
    parameters = (
        ParameterSpec(
            "kp_vel",
            1,
            ((1.0, 1.0, 1.0), (3.0, 1.0, 1.0)),
            selection_regularization_scale=1.0,
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

    result = align(target, reference)

    assert result.controller.selected_settings.kp_vel == (1.0, 1.0, 1.0)


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


def test_bundle_finalization_compresses_and_inventories_every_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    result = align(
        _definition(backend="isaac-lab", evaluator=_Evaluator()),
        _definition(
            backend="px4-sitl",
            evaluator=_Evaluator(reference_settings=known),
        ),
    )
    result = replace(
        result,
        controller=replace(
            result.controller,
            advisory_quality_passed=True,
            advisory_quality_reasons=(),
        ),
    )
    run_dir = tmp_path / "run"
    write_alignment_result(result=result, run_dir=run_dir)
    for name, payload in {
        "reference.json": "{}\n",
        "reference.ulg": "ulog",
        "bridge.log": "bridge",
        "target_candidates.json": "{}\n",
        "actuator_audit.json": '{"schema_version":1,"passed":true}\n',
        "controller_boundary_replay.json": '{"schema_version":1,"passed":true}\n',
        "gpu_utilization.json": "{}\n",
        "search_realization.json": "{}\n",
    }.items():
        (run_dir / name).write_text(payload)

    compress_file = _compress_file
    compression_arguments_seen: list[tuple[str, ...]] = []

    def record_compression(
        *,
        source: Path,
        destination: Path,
        compression_arguments: tuple[str, ...] = ("-19", "-T1"),
    ) -> None:
        compression_arguments_list = tuple(compression_arguments)
        compression_arguments_seen.append(compression_arguments_list)
        compress_file(
            source=source,
            destination=destination,
            compression_arguments=compression_arguments_list,
        )

    monkeypatch.setattr(artifact_module, "_compress_file", record_compression)

    manifest_path = finalize_alignment_bundle(run_dir=run_dir)

    validate_alignment_bundle(run_dir)
    manifest = json.loads(manifest_path.read_text())
    paths = {entry["path"] for entry in manifest["files"]}
    assert "controller.json" in paths
    assert "selection.json.zst" in paths
    assert "evidence/target_traces.json.zst" in paths
    assert "diagnostics/stages/search_realization.json" in paths
    assert not (run_dir / "target_candidates.json").exists()
    assert compression_arguments_seen == [("-8", "--long=27", "-T0")]

    (run_dir / "diagnostics" / "untracked.log").write_text("untracked")
    with pytest.raises(ValueError, match="inventory every payload"):
        validate_alignment_bundle(run_dir)


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
