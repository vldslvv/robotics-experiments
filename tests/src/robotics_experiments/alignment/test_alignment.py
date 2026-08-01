"""Host tests for controller definition, fitting, artifacts, and publication."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from robotics_experiments.alignment import (
    ControllerDefinition,
    ControllerSettings,
    align,
    load_alignment_artifact,
    load_controller_definition,
)
from robotics_experiments.alignment._align import (
    _resolved_experiment_fingerprint,
    _staged_beam_search,
)
from robotics_experiments.alignment._io import (
    _index_candidate_diagnostics,
    _trace,
    _TraceFileEvaluator,
    publish_alignment,
    write_alignment_artifact,
)
from robotics_experiments.alignment._metrics import (
    _braking_events,
    _feasibility_metric_result,
    metric,
    register_metric,
)
from robotics_experiments.alignment._models import (
    CommandInterface,
    MetricResult,
    ParameterSpec,
    ProbeCommand,
    ProbeProfile,
    ProbeTrace,
    SearchSpec,
)
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
        kp_vel=kp,
        horizontal_velocity_error_gain_scale=1.0,
        horizontal_velocity_error_gain_start_mps=0.15,
        horizontal_velocity_error_gain_full_mps=0.30,
        ki_vel=(0.4, 0.4, 0.4),
        kd_vel=kd,
        velocity_filter_hz=10.0,
        velocity_derivative_filter_hz=5.0,
        velocity_integral_limit_mps2=(2.0, 2.0, 2.0),
        hover_thrust_fraction=0.5,
        hover_thrust_normalized=0.57,
        kp_att=(6.0, 6.0, 2.0),
        attitude_yaw_weight=0.4,
        kp_rate=(1.2, 1.2, 0.4),
        ki_rate=(0.2, 0.2, 0.1),
        kd_rate=(0.01, 0.01, 0.0),
        kff_rate=(0.0, 0.0, 0.0),
        rate_output_scale_nm=(1.0, 1.0, 1.0),
        rate_derivative_filter_hz=30.0,
        rate_integral_limit_nm=(0.3, 0.3, 0.2),
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
                        ProbeCommand(_ZERO3, 0.0, 0.1),
                        ProbeCommand(tuple(positive), 0.0, 0.15),
                        ProbeCommand(tuple(negative), 0.0, 0.1),
                        ProbeCommand(_ZERO3, 0.0, 0.1),
                    ),
                )
            )
        profiles.extend(
            (
                ProbeProfile(
                    name=f"{split}-yaw",
                    split=split,
                    axis="yaw",
                    commands=(
                        ProbeCommand(_ZERO3, 0.0, 0.1),
                        ProbeCommand(_ZERO3, amplitude * 0.2, 0.15),
                        ProbeCommand(_ZERO3, 0.0, 0.1),
                    ),
                ),
                ProbeProfile(
                    name=f"{split}-combined",
                    split=split,
                    axis="combined",
                    commands=(
                        ProbeCommand(_ZERO3, 0.0, 0.1),
                        ProbeCommand((amplitude, amplitude * 0.5, 0.0), 0.2, 0.15),
                        ProbeCommand(_ZERO3, 0.0, 0.1),
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
                profile=profile.name,
                split=profile.split,
                axis=profile.axis,
                time_s=tuple(index * 0.05 for index in range(len(commands))),
                command_velocity_body_flu_mps=commands,
                command_yaw_rate_radps=yaw_commands,
                position_enu_m=tuple(positions),
                attitude_flu_enu_wxyz=tuple(attitudes),
                velocity_body_flu_mps=tuple(velocities),
                velocity_enu_mps=tuple(velocities),
                angular_velocity_body_flu_radps=tuple(angular_velocities),
                initial_position_enu_m=_ZERO3,
                initial_attitude_flu_enu_wxyz=_IDENTITY,
                initial_linear_velocity_enu_mps=_ZERO3,
                initial_angular_velocity_body_flu_radps=_ZERO3,
                completed=True,
            )
        )
    return tuple(traces)


class _Evaluator:
    def __init__(self, *, reference_settings: ControllerSettings | None = None) -> None:
        self._reference_settings = reference_settings

    def evaluate(self, *, definition, settings, reference_traces):
        active = self._reference_settings or settings
        return _response_traces(profiles=definition.probes, settings=active)


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
    metric_name: str = "worst-axis-validation-velocity-nrmse-v1",
) -> ControllerDefinition:
    experiment = REPO_ROOT / "configs/alignment_px4_x500_velocity.yaml"
    return ControllerDefinition(
        alignment_id="synthetic",
        controller_id=f"{backend}-controller",
        controller_schema_version=6,
        backend=backend,
        experiment_config=str(experiment),
        vehicle="x500_01",
        airframe="holybro_x500_v2.usda",
        command_rate_hz=20.0,
        command_interface=CommandInterface(
            kind="body-flu-velocity-yaw-rate-v1",
            dimension=4,
            frame="body-flu",
            linear_scale_mps=3.0,
            yaw_rate_scale_radps=1.5,
        ),
        base_settings=_settings(delay=0.0, kp=(1.0, 1.0, 1.0)),
        probes=_profiles(),
        parameters=parameters,
        search=SearchSpec(
            strategy="staged-beam-v1",
            beam_width=8,
            passes_per_stage=2,
        ),
        selection_metric=metric_name,
        publication_metrics=("velocity_rmse_mps",),
        evaluator=evaluator,
        definition_fingerprint=f"{backend}-fingerprint",
        image=f"{backend}-image",
    )


def test_controller_yaml_loads_and_generic_expansion_accepts_another_field() -> None:
    target = load_controller_definition(
        REPO_ROOT / "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"
    )
    assert target.backend == "isaac-lab"
    assert target.alignment_id == "x500-velocity-px4-v7"
    assert target.controller_schema_version == 6
    assert target.base_settings.command_delay_s == pytest.approx(0.06)
    assert target.base_settings.kp_vel == pytest.approx((1.8, 1.8, 4.0))
    assert target.base_settings.horizontal_velocity_error_gain_scale == pytest.approx(9.0 / 7.0)
    assert target.base_settings.kd_vel == pytest.approx((0.375, 0.3, 0.0))
    assert target.search.strategy == "coarse-to-fine-beam-v3"
    assert target.search.beam_width == 32
    assert target.search.passes_per_stage == 1
    assert target.search.refinement_levels == 2
    assert [parameter.field for parameter in target.parameters[5:8]] == [
        "horizontal_velocity_error_gain_scale",
        "horizontal_velocity_error_gain_start_mps",
        "horizontal_velocity_error_gain_full_mps",
    ]
    assert [(parameter.field, parameter.stage) for parameter in target.parameters[-3:]] == [
        ("thruster_tau_inc_s", 3),
        ("thruster_tau_dec_s", 3),
        ("thruster_max_thrust_rate_nps", 3),
    ]
    extended = replace(
        target,
        parameters=(
            *target.parameters,
            ParameterSpec("max_tilt_cmd_deg", 2, (20.0, 30.0)),
        ),
    )
    assert extended.parameters[-1].stage == 2
    assert extended.parameters[-1].candidates == (20.0, 30.0)


def test_legacy_trace_loads_with_zero_controller_boundary() -> None:
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

    trace = _trace(value=raw, name="legacy")  # noqa: SLF001 - legacy trace contract

    assert not trace.initial_controller_state_valid
    assert trace.initial_acceleration_setpoint_enu_mps2 == _ZERO3
    assert trace.initial_linear_acceleration_enu_mps2 == _ZERO3
    assert trace.initial_angular_acceleration_body_flu_radps2 == _ZERO3
    assert trace.initial_rate_integral_body_flu == _ZERO3
    assert trace.initial_hover_thrust_normalized == 0.0


def test_trace_rejects_out_of_range_initial_hover_thrust() -> None:
    raw = json.loads(
        json.dumps(asdict(_response_traces(profiles=_profiles(), settings=_settings())[0]))
    )
    raw["initial_hover_thrust_normalized"] = 1.01

    with pytest.raises(ValueError, match="must not exceed 1"):
        _trace(value=raw, name="trace")  # noqa: SLF001 - trace validation contract


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
    artifact = align(target, reference)

    assert artifact.selected_settings.command_delay_s == 0.05
    assert artifact.selected_settings.kp_vel == (2.0, 3.0, 4.0)
    assert artifact.validation_metrics["worst_axis_velocity_nrmse"] == 0.0
    assert artifact.advisory_quality_passed


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
    assert first.selected_settings == second.selected_settings


def test_staged_beam_v2_retains_per_objective_champions() -> None:
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
            strategy="staged-beam-v2",
            beam_width=2,
            passes_per_stage=1,
        ),
    )
    reference_traces = _response_traces(
        profiles=target.probes,
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
    assert batched.selected_settings == serial.selected_settings
    assert max(batch_evaluator.batch_sizes) > 1


def test_effective_key_aliases_schema_inactive_settings_and_stops_fixed_point() -> None:
    evaluator = _BatchEvaluator()
    target = replace(
        _definition(
            backend="isaac-lab",
            evaluator=evaluator,
            parameters=(ParameterSpec("hover_thrust_fraction", 1, (0.4, 0.6)),),
        ),
        search=SearchSpec(
            strategy="staged-beam-v2",
            beam_width=8,
            passes_per_stage=3,
        ),
    )
    reference_traces = _response_traces(profiles=target.probes, settings=target.base_settings)

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
            strategy="coarse-to-fine-beam-v3",
            beam_width=2,
            passes_per_stage=1,
            refinement_levels=1,
            min_refinement_survivors=1,
        ),
    )
    reference_traces = _response_traces(profiles=target.probes, settings=known)

    outcome = _staged_beam_search(target=target, reference_traces=reference_traces)

    assert outcome.selected.settings.command_delay_s == pytest.approx(0.25)
    assert any(
        record["field"] == "command_delay_s" and record["candidate"] == pytest.approx(0.25)
        for record in outcome.generated_refinements
    )


def test_file_evaluator_indexes_candidate_diagnostics_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
                "schema_version": 2,
                "controller_id": definition.controller_id,
                "definition_fingerprint": definition.definition_fingerprint,
                "image": definition.image,
                "candidates": {
                    key: [
                        asdict(trace)
                        for trace in _response_traces(
                            profiles=definition.probes,
                            settings=settings,
                        )
                    ]
                },
                "failures": {},
            },
            indent=2,
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


def test_v2_metric_reports_braking_distance_and_residual_speed_error() -> None:
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

    result = metric(name="policy-execution-controller-quality-v2")(
        reference,
        target,
    )

    assert result.validation_metrics["braking_distance_error_m"] > 0.0
    assert result.validation_metrics["residual_speed_error_mps"] >= 0.0


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
        (0.02, 0.0, 0.0),
        _ZERO3,
        _ZERO3,
        _ZERO3,
    )
    target_velocity = (
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.02, 0.0, 0.0),
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

    events = _braking_events(reference=reference, target=target)

    assert events[0][-1] == 0.1


def test_v3_metric_prioritizes_gate_feasibility_and_includes_stop_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = {
        "braking_distance_error_m": 0.004,
        "residual_speed_error_mps": 0.009,
        "stop_time_error_s": 0.20,
        "worst_axis_velocity_nrmse": 0.09,
        "yaw_rate_nrmse": 0.09,
        "heading_error_deg": 2.0,
        "combined_endpoint_error_m": 0.009,
    }
    base = MetricResult(
        metrics={},
        ranking_values=(0.004, 0.009, 0.09, 0.09, 2.0, 0.009, 0.0, 0.0),
        fit_metrics={},
        validation_metrics=validation,
    )
    monkeypatch.setattr(
        "robotics_experiments.alignment._metrics._metric_result",
        lambda _reference, _target: base,
    )

    result = _feasibility_metric_result((), ())

    assert result.ranking_values[:3] == pytest.approx((1.0, 1.0, 1.0))
    assert result.beam_values[2] == 2.0


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

    assert align(target, reference).selected_settings.kp_vel == (3.0, 1.0, 1.0)


def test_artifact_publication_tracks_latest_best_and_comparable_generation(
    tmp_path: Path,
) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
        parameters=(ParameterSpec("kp_vel", 1, ((1.0, 1.0, 1.0),)),),
    )
    root = tmp_path / "alignment" / "synthetic"
    first = replace(align(target, reference), run_id="first")
    first_dir = root / "first"
    write_alignment_artifact(artifact=first, run_dir=first_dir)
    publish_alignment(artifact=first, run_dir=first_dir)

    worse = replace(
        first,
        run_id="second",
        candidates=(replace(first.candidates[0], ranking_values=(1.0, 1.0, 1.0)),),
        selected_ranking_values=(1.0, 1.0, 1.0),
    )
    second_dir = root / "second"
    write_alignment_artifact(artifact=worse, run_dir=second_dir)
    publish_alignment(artifact=worse, run_dir=second_dir)
    assert (root / "latest").resolve() == second_dir
    assert (root / "best").resolve() == first_dir

    incompatible = replace(
        worse,
        run_id="third",
        comparison_fingerprint="new-generation",
        advisory_quality_passed=False,
        advisory_quality_reasons=("quality is advisory",),
    )
    third_dir = root / "third"
    write_alignment_artifact(artifact=incompatible, run_dir=third_dir)
    publish_alignment(artifact=incompatible, run_dir=third_dir)
    loaded = load_alignment_artifact(root / "best")
    assert loaded.run_id == "third"
    assert not loaded.advisory_quality_passed


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


def test_artifact_loader_rejects_nonfinite_settings(tmp_path: Path) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
        parameters=(ParameterSpec("kp_vel", 1, ((1.0, 1.0, 1.0),)),),
    )
    run_dir = tmp_path / "run"
    write_alignment_artifact(artifact=align(target, reference), run_dir=run_dir)
    path = run_dir / "alignment.json"
    payload = json.loads(path.read_text())
    payload["selected_settings"]["command_delay_s"] = float("nan")
    payload["candidates"][0]["settings"]["command_delay_s"] = float("nan")
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="finite"):
        load_alignment_artifact(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("horizontal_velocity_error_gain_scale", 0.9, "must be at least 1"),
        ("horizontal_velocity_error_gain_start_mps", 0.30, "start < full"),
    ],
)
def test_artifact_loader_rejects_invalid_horizontal_gain_schedule(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    artifact = align(
        _definition(backend="isaac-lab", evaluator=_Evaluator()),
        _definition(
            backend="px4-sitl",
            evaluator=_Evaluator(reference_settings=known),
        ),
    )
    run_dir = tmp_path / "run"
    write_alignment_artifact(artifact=artifact, run_dir=run_dir)
    path = run_dir / "alignment.json"
    payload = json.loads(path.read_text())
    payload["selected_settings"][field] = value
    payload["candidates"][artifact.selected_candidate_index]["settings"][field] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_alignment_artifact(path)


def test_artifact_loader_rejects_v1_without_upgrading_it(tmp_path: Path) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
    )
    run_dir = tmp_path / "run"
    write_alignment_artifact(artifact=align(target, reference), run_dir=run_dir)
    path = run_dir / "alignment.json"
    payload = json.loads(path.read_text())
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="unsupported schema_version=1"):
        load_alignment_artifact(path)


def test_artifact_schema_v6_and_legacy_controller_mappings(tmp_path: Path) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(backend="isaac-lab", evaluator=_Evaluator())
    run_dir = tmp_path / "run"
    artifact = align(target, reference)
    assert artifact.schema_version == 6
    write_alignment_artifact(artifact=artifact, run_dir=run_dir)
    path = run_dir / "alignment.json"
    payload = json.loads(path.read_text())
    loaded = load_alignment_artifact(path)

    assert loaded.schema_version == 6
    assert loaded.selected_settings.hover_thrust_normalized == pytest.approx(0.57)
    assert loaded.selected_settings.horizontal_velocity_error_gain_scale == 1.0

    payload["schema_version"] = 5
    for name in (
        "horizontal_velocity_error_gain_scale",
        "horizontal_velocity_error_gain_start_mps",
        "horizontal_velocity_error_gain_full_mps",
    ):
        payload["selected_settings"].pop(name)
        for candidate in payload["candidates"]:
            candidate["settings"].pop(name)
    path.write_text(json.dumps(payload))

    loaded = load_alignment_artifact(path)

    assert loaded.schema_version == 5
    assert loaded.selected_settings.horizontal_velocity_error_gain_scale == 1.0
    assert loaded.selected_settings.horizontal_velocity_error_gain_start_mps == 0.15
    assert loaded.selected_settings.horizontal_velocity_error_gain_full_mps == 0.30

    schema5_payload = json.loads(json.dumps(payload))
    payload["schema_version"] = 4
    payload["selected_settings"].pop("hover_thrust_normalized")
    for candidate in payload["candidates"]:
        candidate["settings"].pop("hover_thrust_normalized")
    path.write_text(json.dumps(payload))

    loaded = load_alignment_artifact(path)

    assert loaded.schema_version == 4
    assert loaded.selected_settings.hover_thrust_normalized == 0.0

    schema4_payload = json.loads(json.dumps(payload))
    payload["schema_version"] = 3
    payload["selected_settings"].pop("hover_thrust_fraction")
    for candidate in payload["candidates"]:
        candidate["settings"].pop("hover_thrust_fraction")
    path.write_text(json.dumps(payload))

    loaded = load_alignment_artifact(path)

    assert loaded.schema_version == 3
    assert loaded.selected_settings.hover_thrust_fraction == 0.0

    payload = schema4_payload
    payload["schema_version"] = 2
    payload["selected_settings"].pop("rate_output_scale_nm")
    payload["selected_settings"].pop("hover_thrust_fraction")
    for candidate in payload["candidates"]:
        candidate["settings"].pop("rate_output_scale_nm")
        candidate["settings"].pop("hover_thrust_fraction")
    path.write_text(json.dumps(payload))

    loaded = load_alignment_artifact(path)

    assert loaded.schema_version == 2
    assert loaded.selected_settings.rate_output_scale_nm == (1.0, 1.0, 1.0)
    assert loaded.selected_settings.hover_thrust_fraction == 0.0
    assert loaded.selected_settings.hover_thrust_normalized == 0.0

    payload = schema5_payload
    payload["selected_settings"]["hover_thrust_normalized"] = 0.0
    for candidate in payload["candidates"]:
        candidate["settings"]["hover_thrust_normalized"] = 0.0
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hover_thrust_normalized must be positive"):
        load_alignment_artifact(path)


def test_artifact_loader_rejects_unsupported_backend_and_ranking_mismatch(
    tmp_path: Path,
) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
    )
    run_dir = tmp_path / "run"
    write_alignment_artifact(artifact=align(target, reference), run_dir=run_dir)
    path = run_dir / "alignment.json"
    payload = json.loads(path.read_text())
    payload["target_backend"] = "real-flight"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unsupported target/reference backend"):
        load_alignment_artifact(path)

    payload["target_backend"] = "isaac-lab"
    payload["selected_ranking_values"][0] += 1.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="do not match selected candidate"):
        load_alignment_artifact(path)
