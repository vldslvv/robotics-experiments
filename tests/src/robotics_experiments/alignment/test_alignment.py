"""Host tests for controller definition, fitting, artifacts, and publication."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from robotics_experiments.alignment import (
    ControllerDefinition,
    ControllerSettings,
    align,
    load_alignment_artifact,
    load_controller_definition,
)
from robotics_experiments.alignment._align import _settings_candidates
from robotics_experiments.alignment._io import (
    publish_alignment,
    write_alignment_artifact,
)
from robotics_experiments.alignment._metrics import register_metric
from robotics_experiments.alignment._models import (
    CommandInterface,
    MetricResult,
    ParameterSpec,
    ProbeCommand,
    ProbeProfile,
    ProbeTrace,
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
        kd_vel=kd,
        kp_att=(6.0, 6.0, 2.0),
        kp_rate=(1.2, 1.2, 0.4),
        max_tilt_cmd_deg=30.0,
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
                        ProbeCommand(_ZERO3, 0.1),
                        ProbeCommand(tuple(positive), 0.15),
                        ProbeCommand(tuple(negative), 0.1),
                        ProbeCommand(_ZERO3, 0.1),
                    ),
                )
            )
    return tuple(profiles)


def _command_samples(profile: ProbeProfile, /) -> tuple[tuple[float, float, float], ...]:
    values = []
    for command in profile.commands:
        values.extend([command.velocity_body_flu_mps] * round(command.duration_s / 0.05))
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
        axis_index = {"x": 0, "y": 1, "z": 2}[profile.axis]
        velocities = []
        for index in range(len(commands)):
            delayed = _ZERO3 if index < delay_steps else commands[index - delay_steps]
            velocity = [0.0, 0.0, 0.0]
            velocity[axis_index] = (
                delayed[axis_index] * settings.kp_vel[axis_index] / settings.kd_vel[axis_index]
            )
            velocities.append(tuple(velocity))
        traces.append(
            ProbeTrace(
                profile=profile.name,
                split=profile.split,
                axis=profile.axis,
                time_s=tuple(index * 0.05 for index in range(len(commands))),
                command_velocity_body_flu_mps=commands,
                velocity_body_flu_mps=tuple(velocities),
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
    assert len(_settings_candidates(target)) == 36

    extended = replace(
        target,
        parameters=(
            *target.parameters,
            ParameterSpec("max_tilt_cmd_deg", (20.0, 30.0)),
        ),
    )
    candidates = _settings_candidates(extended)
    assert len(candidates) == 72
    assert {candidate.max_tilt_cmd_deg for candidate in candidates} == {20.0, 30.0}


def test_align_recovers_known_delay_and_per_axis_gain_on_validation(
    tmp_path: Path,
) -> None:
    known = _settings(delay=0.05, kp=(2.0, 3.0, 4.0))
    parameters = (
        ParameterSpec("command_delay_s", (0.0, 0.05, 0.1)),
        ParameterSpec(
            "kp_vel",
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
    parameters = (ParameterSpec("kp_vel", ((1.0, 1.0, 1.0), (3.0, 1.0, 1.0))),)
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
        parameters=(ParameterSpec("kp_vel", ((1.0, 1.0, 1.0),)),),
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


def test_artifact_loader_rejects_nonfinite_settings(tmp_path: Path) -> None:
    known = _settings(kp=(1.0, 1.0, 1.0))
    reference = _definition(
        backend="px4-sitl",
        evaluator=_Evaluator(reference_settings=known),
    )
    target = _definition(
        backend="isaac-lab",
        evaluator=_Evaluator(),
        parameters=(ParameterSpec("kp_vel", ((1.0, 1.0, 1.0),)),),
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
