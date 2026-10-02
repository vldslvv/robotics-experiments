from __future__ import annotations

import hashlib
import json
import math
import runpy
import subprocess
import sys
import types
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

import robotics_experiments.flight_parameters as flight_parameters
from robotics_experiments.flight_parameters import (
    FlightTrace,
    ResponseTraceEvidence,
    analyze_flight_parameters_stability,
    compress_flight_evidence,
    derive_flight_parameters,
    evaluate_hover_pose,
    finalize_flight_parameters_bundle,
    flight_parameters_from_payload,
    flight_trace_from_payload,
    flight_trials,
    load_flight_parameters,
    load_flight_parameters_config,
    publish_flight_parameters,
    validate_flight_parameters_bundle,
    write_flight_parameters,
)
from robotics_experiments.flight_parameters._artifact import raw_flight_trace_from_payload
from robotics_experiments.flight_parameters._models import RawFlightTrace
from robotics_experiments.flight_parameters.px4_evidence import (
    ParsedPx4HoverLog,
    verify_px4_hover_evidence,
)
from tests.support.paths import REPO_ROOT


def _config(tmp_path: Path, *, repetitions: int = 2):
    path = tmp_path / "flight.yaml"
    path.write_text(
        f"""
schema_version: 1
flight_parameters_id: test-hover-v1
source_backend: px4-sitl
state_source: simulator-ground-truth
experiment_config: configs/flight_parameters_x500_hover.yaml
vehicle: x500_01
starting_seed: 1
sets:
  fit:
    hover_count: {repetitions}
    response_count: 1
  validation:
    hover_count: 1
    response_count: 1
  acceptance:
    hover_count: 1
    response_count: 1
hover:
  kind: hover
  mode_transition_timeout_s: 3.0
  mode_report_max_age_s: 2.5
  settle_s: 2.0
  measure_s: 1.0
  sample_rate_hz: 2.0
  target_position_enu_m: [0.0, 0.0, 2.5]
  target_yaw_rad: 0.0
response:
  kind: controller-response
  experiment_config: configs/alignment_px4_x500_velocity.yaml
  definition: configs/alignment/controllers/px4_sitl_x500_velocity.yaml
policy_warmup_s: 0.5
position_margin_m: 0.005
attitude_margin_rad: 0.01
execution:
  worker_count: 1
""".lstrip(),
        encoding="utf-8",
    )
    return load_flight_parameters_config(path)


def _trace(*, config, index: int, x: float, yaw: float = 0.0) -> FlightTrace:
    attitude = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    trials = flight_trials(config=config, kind="hover")
    set_name, seed = trials[index - 1]
    return FlightTrace(
        artifact_kind="robotics-experiments-flight-parameters-trace",
        schema_version=1,
        flight_parameters_id=config.flight_parameters_id,
        campaign_run_id="run-1",
        trial_index=index,
        set_name=set_name,
        seed=seed,
        source_backend=config.source_backend,
        state_source=config.state_source,
        config_fingerprint=config.config_fingerprint,
        experiment_fingerprint="experiment-fingerprint",
        vehicle_fingerprint="vehicle-fingerprint",
        source_image="px4-image",
        verification_fingerprint="a" * 64,
        measurement_start_us=10_000_000,
        measurement_end_us=11_000_000,
        target_position_enu_m=config.hover.target_position_enu_m,
        target_attitude_flu_enu_wxyz=(1.0, 0.0, 0.0, 0.0),
        time_s=(0.0, 0.5),
        position_enu_m=((x, 0.0, 2.5), (x + 0.01, -0.02, 2.51)),
        attitude_flu_enu_wxyz=(attitude, attitude),
        linear_velocity_enu_mps=((0.3, 0.2, 0.1),) * 2,
        angular_velocity_enu_radps=((0.1, 0.2, 0.3),) * 2,
        completed=True,
        failure_reason="",
    )


def _response_trace(*, config, repetition: int = 1) -> ResponseTraceEvidence:
    set_name, seed = flight_trials(config=config, kind="response")[repetition - 1]
    return ResponseTraceEvidence(
        repetition=repetition,
        set_name=set_name,
        seed=seed,
        profile_count=1,
        reference_path=f"response/repetition-{repetition:03d}/flight_parameters_response.json",
        reference_sha256="b" * 64,
        ulog_path=f"response/repetition-{repetition:03d}/flight_parameters.ulg",
        ulog_sha256="c" * 64,
        bridge_log_path=f"response/repetition-{repetition:03d}/flight_parameters_bridge.log",
        bridge_log_sha256="d" * 64,
    )


def _array(values, dtype=float):
    return np.asarray(values, dtype=dtype)


def _parsed_log(*, measurement_mode: int = 4, commands=(400, 22)) -> ParsedPx4HoverLog:
    command_timestamps = [10_000_000 + index * 1_000_000 for index in range(len(commands))]
    return ParsedPx4HoverLog(
        px4_software_version="px4-test-version",
        vehicle_command={
            "timestamp": _array(command_timestamps, np.uint64),
            "command": _array(commands, np.uint32),
            "from_external": _array([1] * len(commands), np.uint8),
            "param5": _array([0.0, 47.3977][: len(commands)]),
            "param6": _array([0.0, 8.5456][: len(commands)]),
            "param7": _array([0.0, 490.5][: len(commands)]),
        },
        vehicle_command_ack={
            "timestamp": _array([value + 100_000 for value in command_timestamps], np.uint64),
            "command": _array(commands, np.uint32),
            "result": _array([0] * len(commands), np.uint8),
        },
        vehicle_status={
            "timestamp": _array([9_000_000, 11_100_000, 13_000_000, 15_500_000], np.uint64),
            "nav_state": _array([4, 17, 4, measurement_mode], np.uint8),
        },
        vehicle_control_mode={
            "timestamp": _array([14_500_000, 15_500_000], np.uint64),
            "flag_armed": _array([1, 1], np.uint8),
            "flag_control_auto_enabled": _array([1, 1], np.uint8),
            "flag_control_position_enabled": _array([1, 1], np.uint8),
            "flag_control_offboard_enabled": _array([0, 0], np.uint8),
            "flag_multicopter_position_control_enabled": _array([1, 1], np.uint8),
        },
        vehicle_land_detected={
            "timestamp": _array([14_500_000, 15_500_000], np.uint64),
            "landed": _array([0, 0], np.uint8),
            "maybe_landed": _array([0, 0], np.uint8),
            "ground_contact": _array([0, 0], np.uint8),
        },
        position_setpoint_triplet={
            "timestamp": _array([11_000_000, 13_000_000], np.uint64),
            "current.valid": _array([1, 1], np.uint8),
            "current.lat": _array([47.3977, 47.3977]),
            "current.lon": _array([8.5456, 8.5456]),
            "current.alt": _array([490.5, 490.5]),
            "current.yaw": _array([math.nan, math.nan]),
        },
        trajectory_setpoint={
            "timestamp": _array([15_000_000, 15_500_000], np.uint64),
            "yaw": _array([0.2, 0.2]),
        },
        vehicle_local_position={
            "timestamp": _array([15_000_000], np.uint64),
            "heading": _array([0.2]),
        },
    )


def _raw(config) -> RawFlightTrace:
    times = tuple(
        10_000_000 + index * 500_000 + (100_000 if index % 2 else 0) for index in range(13)
    )
    return RawFlightTrace(
        artifact_kind="robotics-experiments-flight-parameters-raw-trace",
        schema_version=1,
        flight_parameters_id=config.flight_parameters_id,
        campaign_run_id="run-1",
        trial_index=1,
        set_name="fit",
        seed=1,
        source_backend=config.source_backend,
        state_source=config.state_source,
        config_fingerprint=config.config_fingerprint,
        experiment_fingerprint="experiment-fingerprint",
        vehicle_fingerprint="vehicle-fingerprint",
        source_image="px4-image",
        requested_target_position_enu_m=config.hover.target_position_enu_m,
        requested_target_attitude_flu_enu_wxyz=(1.0, 0.0, 0.0, 0.0),
        px4_home_lat_lon_alt=(47.3977, 8.5456, 488.0),
        time_us=times,
        position_enu_m=tuple((0.0, 0.0, 2.5) for _ in times),
        attitude_flu_enu_wxyz=tuple((1.0, 0.0, 0.0, 0.0) for _ in times),
        linear_velocity_enu_mps=tuple((float(index), 0.1, 0.2) for index in range(len(times))),
        angular_velocity_enu_radps=tuple((0.3, float(index), 0.4) for index in range(len(times))),
        completed=True,
        failure_reason="",
    )


def _flight_parameters(tmp_path: Path):
    config = _config(tmp_path)
    traces = (
        _trace(config=config, index=1, x=-0.02),
        _trace(config=config, index=2, x=0.03),
        _trace(config=config, index=3, x=10.0),
        _trace(config=config, index=4, x=20.0),
    )
    return derive_flight_parameters(
        config=config,
        run_id="run-1",
        traces=traces,
        response_traces=tuple(
            _response_trace(config=config, repetition=index) for index in (1, 2, 3)
        ),
        source_commit="abc123",
        source_dirty=False,
        created_at_utc="2026-08-21T12:00:00Z",
    )


def test_config_is_strict_and_fingerprinted(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.hover.kind == "hover"
    assert len(config.config_fingerprint) == 64

    path = tmp_path / "unsupported.yaml"
    payload = (tmp_path / "flight.yaml").read_text().replace("kind: hover", "kind: cruise")
    path.write_text(payload)
    with pytest.raises(ValueError, match="only hover kind 'hover'"):
        load_flight_parameters_config(path)

    invalid_id = (
        (tmp_path / "flight.yaml")
        .read_text()
        .replace("flight_parameters_id: test-hover-v1", "flight_parameters_id: ../escape")
    )
    path.write_text(invalid_id)
    with pytest.raises(ValueError, match="flight_parameters_id must contain"):
        load_flight_parameters_config(path)


def test_production_hover_campaign_uses_independent_thirty_second_trials() -> None:
    config = load_flight_parameters_config(REPO_ROOT / "configs/flight/holybro_x500_v2_hover.yaml")

    assert tuple(group.name for group in config.sets) == ("fit", "validation", "acceptance")
    assert tuple(group.hover_count for group in config.sets) == (30, 10, 10)
    assert tuple(group.response_count for group in config.sets) == (1, 1, 1)
    assert config.starting_seed == 1001
    assert flight_trials(config=config, kind="hover") == tuple(
        (name, 1001 + index)
        for index, name in enumerate(["fit"] * 30 + ["validation"] * 10 + ["acceptance"] * 10)
    )
    assert flight_trials(config=config, kind="response") == (
        ("fit", 1051),
        ("validation", 1052),
        ("acceptance", 1053),
    )
    assert config.hover.mode_transition_timeout_s == 15.0
    assert config.hover.mode_report_max_age_s == 2.5
    assert config.hover.settle_s == 6.0
    assert config.hover.measure_s == 30.0
    assert config.hover.sample_rate_hz == 20.0
    assert config.execution.worker_count == 1


def test_px4_verification_selects_only_auto_loiter_samples(tmp_path: Path) -> None:
    config = _config(tmp_path, repetitions=1)
    verification, trace = verify_px4_hover_evidence(
        config=config,
        raw=_raw(config),
        log=_parsed_log(),
        raw_trace_sha256="1" * 64,
        ulog_sha256="2" * 64,
    )

    assert [event.mode for event in verification.mode_events] == [
        "AUTO_LOITER",
        "AUTO_TAKEOFF",
        "AUTO_LOITER",
    ]
    assert [event.command_id for event in verification.command_events] == [400, 22]
    assert verification.loiter_enter_us == 13_000_000
    assert trace.measurement_start_us == 15_000_000
    assert trace.measurement_end_us == 16_000_000
    assert trace.target_position_enu_m == pytest.approx((0.0, 0.0, 2.5))
    assert len(trace.time_s) == 2
    assert trace.linear_velocity_enu_mps == (
        _raw(config).linear_velocity_enu_mps[10],
        _raw(config).linear_velocity_enu_mps[11],
    )
    assert trace.angular_velocity_enu_radps == (
        _raw(config).angular_velocity_enu_radps[10],
        _raw(config).angular_velocity_enu_radps[11],
    )


def test_px4_verification_window_does_not_follow_loiter_report_time(tmp_path: Path) -> None:
    config = _config(tmp_path)
    traces = []
    for loiter_time in (12_500_000, 13_000_000):
        log = _parsed_log()
        log.vehicle_status["timestamp"][2] = loiter_time
        verification, trace = verify_px4_hover_evidence(
            config=config,
            raw=_raw(config),
            log=log,
            raw_trace_sha256="1" * 64,
            ulog_sha256="2" * 64,
        )
        assert verification.loiter_enter_us == loiter_time
        assert trace.measurement_start_us - loiter_time >= config.hover.settle_s * 1_000_000
        traces.append(trace)
    assert (
        replace(traces[0], verification_fingerprint=traces[1].verification_fingerprint) == traces[1]
    )


def test_px4_verification_rejects_late_loiter_without_moving_window(tmp_path: Path) -> None:
    config = _config(tmp_path)
    log = _parsed_log()
    log.vehicle_status["timestamp"][2] = 13_500_000
    with pytest.raises(ValueError, match="transition exceeded the configured timeout"):
        verify_px4_hover_evidence(
            config=config,
            raw=_raw(config),
            log=log,
            raw_trace_sha256="1" * 64,
            ulog_sha256="2" * 64,
        )


def test_px4_verification_rejects_takeoff_or_navigation_during_measurement(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, repetitions=1)
    with pytest.raises(ValueError, match="left AUTO_LOITER"):
        verify_px4_hover_evidence(
            config=config,
            raw=_raw(config),
            log=_parsed_log(measurement_mode=17),
            raw_trace_sha256="1" * 64,
            ulog_sha256="2" * 64,
        )
    returned_to_loiter = replace(
        _parsed_log(),
        vehicle_status={
            "timestamp": _array(
                [9_000_000, 11_100_000, 13_000_000, 14_000_000, 14_500_000], np.uint64
            ),
            "nav_state": _array([4, 17, 4, 14, 4], np.uint8),
        },
    )
    with pytest.raises(ValueError, match="left AUTO_LOITER"):
        verify_px4_hover_evidence(
            config=config,
            raw=_raw(config),
            log=returned_to_loiter,
            raw_trace_sha256="1" * 64,
            ulog_sha256="2" * 64,
        )
    with pytest.raises(ValueError, match="exactly arm then takeoff"):
        verify_px4_hover_evidence(
            config=config,
            raw=_raw(config),
            log=_parsed_log(commands=(400, 22, 192)),
            raw_trace_sha256="1" * 64,
            ulog_sha256="2" * 64,
        )


def test_trace_loader_rejects_nonmonotonic_time(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(json.dumps(asdict(_trace(config=config, index=1, x=0.0))))
    payload["time_s"] = [0.0, 0.0]
    with pytest.raises(ValueError, match="timestamps"):
        flight_trace_from_payload(payload)


def test_derivation_uses_all_extrema_and_authored_margins(tmp_path: Path) -> None:
    flight_parameters = _flight_parameters(tmp_path)
    position = flight_parameters.hover.position_error_enu_m
    assert position.observed_min == pytest.approx((-0.02, -0.02, 0.0))
    assert position.observed_max == pytest.approx((0.04, 0.0, 0.01))
    assert position.lower == pytest.approx((-0.025, -0.025, -0.005))
    assert position.upper == pytest.approx((0.045, 0.005, 0.015))
    assert flight_parameters.hover_trial_count == 4
    assert flight_parameters.hover_sample_count == 8
    assert len(flight_parameters.artifact_fingerprint) == 64
    assert len(flight_parameters.hover.trace_statistics) == 4
    assert flight_parameters.hover.trace_statistics[0].policy_window_sample_count == 1
    position_limit = next(
        item for item in flight_parameters.hover.acceptance_limits if item.name == "position_error"
    )
    assert position_limit.maximum_absolute > 0.0
    assert position_limit.rmse > 0.0
    assert position_limit.p95_absolute > 0.0
    assert position_limit.p99_absolute > 0.0


def test_derivation_rejects_incomplete_campaign(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="requires 4 traces"):
        derive_flight_parameters(
            config=config,
            run_id="run-1",
            traces=(_trace(config=config, index=1, x=0.0),),
            response_traces=tuple(
                _response_trace(config=config, repetition=index) for index in (1, 2, 3)
            ),
            source_commit="abc123",
            source_dirty=False,
            created_at_utc="2026-08-21T12:00:00Z",
        )


def test_orientation_is_quaternion_sign_invariant_and_wraps_yaw(tmp_path: Path) -> None:
    flight_parameters = _flight_parameters(tmp_path)
    positive = evaluate_hover_pose(
        flight_parameters=flight_parameters,
        target_position_enu_m=(0.0, 0.0, 2.5),
        target_attitude_flu_enu_wxyz=(1.0, 0.0, 0.0, 0.0),
        position_enu_m=(0.0, 0.0, 2.5),
        attitude_flu_enu_wxyz=(-1.0, 0.0, 0.0, 0.0),
    )
    assert positive.orientation_error_rad == pytest.approx(0.0)
    assert positive.inside


@pytest.mark.parametrize("compressed", [False, True])
def test_bundle_manifest_and_atomic_latest(tmp_path: Path, compressed: bool) -> None:
    config = _config(tmp_path, repetitions=1)
    root = tmp_path / "flight-parameters" / config.flight_parameters_id
    run_dir = root / "run-1"
    traces = []
    response_traces = []
    for index, (set_name, seed) in enumerate(flight_trials(config=config, kind="hover"), start=1):
        raw = replace(_raw(config), trial_index=index, set_name=set_name, seed=seed)
        evidence = run_dir / "evidence" / f"trial-{index:03d}"
        evidence.mkdir(parents=True)
        raw_path = evidence / "raw_trace.json"
        raw_path.write_text(json.dumps(asdict(raw), allow_nan=False), encoding="utf-8")
        ulog_path = evidence / "flight.ulg"
        ulog_path.write_text("synthetic ulog evidence")
        evidence.joinpath("bridge.log").write_text("bridge evidence\n", encoding="utf-8")
        verification, trace = verify_px4_hover_evidence(
            config=config,
            raw=raw,
            log=_parsed_log(),
            raw_trace_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            ulog_sha256=hashlib.sha256(ulog_path.read_bytes()).hexdigest(),
        )
        evidence.joinpath("hover_verification.json").write_text(
            json.dumps(asdict(verification), allow_nan=False), encoding="utf-8"
        )
        evidence.joinpath("trace.json").write_text(
            json.dumps(asdict(trace), allow_nan=False), encoding="utf-8"
        )
        traces.append(trace)
        response = _response_trace(config=config, repetition=index)
        response_reference = run_dir / response.reference_path
        response_reference.parent.mkdir(parents=True)
        response_ulog = run_dir / response.ulog_path
        response_bridge = run_dir / response.bridge_log_path
        response_reference.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "set_name": response.set_name,
                    "seed": response.seed,
                    "traces": [{}],
                }
            )
        )
        response_ulog.write_text("response ulog")
        response_bridge.write_text("response bridge\n", encoding="utf-8")
        response_traces.append(
            replace(
                response,
                reference_sha256=hashlib.sha256(response_reference.read_bytes()).hexdigest(),
                ulog_sha256=hashlib.sha256(response_ulog.read_bytes()).hexdigest(),
                bridge_log_sha256=hashlib.sha256(response_bridge.read_bytes()).hexdigest(),
            )
        )
    if compressed:
        for path in tuple(run_dir.rglob("*")):
            if path.is_file() and path.suffix in {".ulg", ".log"}:
                compress_flight_evidence(path=path, level=6)
                path.unlink()
        response_traces = [
            replace(
                record,
                ulog_path=record.ulog_path + ".gz",
                bridge_log_path=record.bridge_log_path + ".gz",
            )
            for record in response_traces
        ]
    flight_parameters = derive_flight_parameters(
        config=config,
        run_id="run-1",
        traces=tuple(traces),
        response_traces=tuple(response_traces),
        source_commit="abc123",
        source_dirty=False,
        created_at_utc="2026-08-21T12:00:00Z",
    )
    write_flight_parameters(flight_parameters=flight_parameters, run_dir=run_dir)
    finalize_flight_parameters_bundle(run_dir)
    validate_flight_parameters_bundle(run_dir)
    link = publish_flight_parameters(run_dir)
    assert link.is_symlink()
    assert link.readlink() == Path("run-1")
    validate_flight_parameters_bundle(link)
    assert load_flight_parameters(link) == flight_parameters

    evidence.joinpath("trace.json").write_text('{"changed":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest .* mismatch"):
        validate_flight_parameters_bundle(run_dir)


def test_flight_parameters_payload_fingerprint_is_strict(tmp_path: Path) -> None:
    payload = json.loads(json.dumps(asdict(_flight_parameters(tmp_path))))
    payload["measure_s"] = 2.0
    with pytest.raises(ValueError, match="artifact_fingerprint"):
        flight_parameters_from_payload(payload)


def test_failed_trace_requires_a_reason(tmp_path: Path) -> None:
    config = _config(tmp_path)
    payload = json.loads(
        json.dumps(asdict(replace(_trace(config=config, index=1, x=0.0), completed=False)))
    )
    with pytest.raises(ValueError, match="failed traces require one"):
        flight_trace_from_payload(payload)


@pytest.mark.parametrize("field", ["linear_velocity_enu_mps", "angular_velocity_enu_radps"])
@pytest.mark.parametrize(
    "invalid",
    [
        None,
        [],
        [[1.0, 2.0]],
        [[math.nan, 0.0, 0.0]] * 2,
        [[math.inf, 0.0, 0.0]] * 2,
        [[-math.inf, 0.0, 0.0]] * 2,
    ],
)
@pytest.mark.parametrize("raw", [False, True])
def test_motion_samples_reject_invalid_values(tmp_path: Path, field, invalid, raw) -> None:
    config = _config(tmp_path)
    trace = _raw(config) if raw else _trace(config=config, index=1, x=0.0)
    payload = json.loads(json.dumps(asdict(trace)))
    if invalid is None:
        del payload[field]
    else:
        payload[field] = invalid
        if raw and len(invalid) == 2:
            payload[field] = invalid[:1] * len(payload["time_us"])
    loader = raw_flight_trace_from_payload if raw else flight_trace_from_payload
    with pytest.raises(ValueError):
        loader(payload)


@pytest.mark.parametrize("field", ["hover_count", "response_count"])
@pytest.mark.parametrize("replacement", ["-1", "true", "1.5", "'1'"])
def test_config_rejects_invalid_flight_count(tmp_path: Path, field, replacement) -> None:
    _config(tmp_path)
    path = tmp_path / "flight.yaml"
    path.write_text(path.read_text().replace(f"{field}: 1", f"{field}: {replacement}", 1))
    with pytest.raises(ValueError, match=field):
        load_flight_parameters_config(path)


@pytest.mark.parametrize("field", ["hover_count", "response_count"])
def test_config_rejects_zero_flight_count(tmp_path: Path, field) -> None:
    _config(tmp_path)
    path = tmp_path / "flight.yaml"
    path.write_text(path.read_text().replace(f"{field}: 1", f"{field}: 0", 1))
    with pytest.raises(ValueError, match=field):
        load_flight_parameters_config(path)


@pytest.mark.parametrize("starting_seed", [0, 1001])
def test_flight_seeds_follow_global_order(tmp_path: Path, starting_seed) -> None:
    _config(tmp_path)
    path = tmp_path / "flight.yaml"
    path.write_text(path.read_text().replace("starting_seed: 1", f"starting_seed: {starting_seed}"))
    config = load_flight_parameters_config(path)
    trials = flight_trials(config=config, kind="hover") + flight_trials(
        config=config, kind="response"
    )
    assert tuple(name for name, _ in trials) == (
        "fit",
        "fit",
        "validation",
        "acceptance",
        "fit",
        "validation",
        "acceptance",
    )
    assert tuple(seed for _, seed in trials) == tuple(range(starting_seed, starting_seed + 7))


@pytest.mark.parametrize(
    ("before", "after"),
    [("starting_seed: 1", "starting_seed: 2"), ("response_count: 1", "response_count: 10")],
)
def test_seed_and_count_changes_are_fingerprinted(tmp_path: Path, before, after) -> None:
    config = _config(tmp_path)
    path = tmp_path / "flight.yaml"
    assert load_flight_parameters_config(path).config_fingerprint == config.config_fingerprint
    path.write_text(path.read_text().replace(before, after, 1))
    updated = load_flight_parameters_config(path)
    assert updated.config_fingerprint != config.config_fingerprint
    if before.startswith("response_count"):
        assert flight_trials(config=updated, kind="response") == (
            *(("fit", seed) for seed in range(5, 15)),
            ("validation", 15),
            ("acceptance", 16),
        )


def test_held_out_hover_flights_do_not_expand_reference_limits(tmp_path: Path) -> None:
    artifact = _flight_parameters(tmp_path)
    assert artifact.hover.position_error_enu_m.observed_max[0] == pytest.approx(0.04)
    held_out = artifact.hover.trace_statistics[2:]
    assert all(
        next(signal for signal in trace.policy_window if signal.name == "position_error").maximum
        > 9.0
        for trace in held_out
    )
    assert (
        next(
            signal for signal in artifact.hover.acceptance_limits if signal.name == "position_error"
        ).maximum
        < 0.1
    )


@pytest.mark.parametrize("kind", ["hover", "response"])
def test_derivation_rejects_wrong_set_assignment(tmp_path: Path, kind) -> None:
    config = _config(tmp_path)
    traces = tuple(_trace(config=config, index=index, x=0.0) for index in (1, 2, 3, 4))
    responses = tuple(_response_trace(config=config, repetition=index) for index in (1, 2, 3))
    if kind == "hover":
        traces = (replace(traces[0], set_name="acceptance"), *traces[1:])
    else:
        responses = (replace(responses[0], seed=999), *responses[1:])
    with pytest.raises(ValueError, match="configured flight sets and seeds"):
        derive_flight_parameters(
            config=config,
            run_id="run-1",
            traces=traces,
            response_traces=responses,
            source_commit="test",
            source_dirty=False,
            created_at_utc="2026-09-11T12:00:00Z",
        )


def _stability_run(
    *, root: Path, run_id: str, mutation: str = "", seed_offset: int = 0, compressed: bool = False
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    config = _config(root)
    config = replace(
        config,
        hover=replace(config.hover, measure_s=30.0, sample_rate_hz=20.0),
        starting_seed=config.starting_seed + seed_offset,
    )
    run_dir = root / run_id
    traces = []
    bridge_line = (
        "PX4 bridge dynamics: sim_time_us=8000 last_actuator_us=None armed=False "
        "commands=[0,0,0,0] position_enu=[0,0,0] quat_wxyz=[1,0,0,0] "
        "velocity_enu=[0,0,0] angular_velocity_body_flu=[0,0,0]\n"
    )
    for index in range(1, 5):
        position = np.tile([0.01, 0.0, 2.5], (600, 1))
        if index == 1:
            if mutation == "spike":
                position[301, 0] = 0.2
            elif mutation == "drift":
                position[:, 0] += np.arange(600) / 6000
            elif mutation == "bad-flight":
                position[:, 0] = 0.2
        trace = replace(
            _trace(config=config, index=index, x=0.01),
            campaign_run_id=run_id,
            measurement_end_us=40_000_000,
            time_s=tuple(np.arange(600) / 20),
            position_enu_m=tuple(tuple(row) for row in position),
            attitude_flu_enu_wxyz=((1.0, 0.0, 0.0, 0.0),) * 600,
            linear_velocity_enu_mps=((0.0, 0.0, 0.0),) * 600,
            angular_velocity_enu_radps=((0.0, 0.0, 0.0),) * 600,
        )
        evidence = run_dir / "evidence" / f"trial-{index:03d}"
        evidence.mkdir(parents=True)
        (evidence / "trace.json").write_text(json.dumps(asdict(trace)))
        (evidence / "bridge.log").write_text(bridge_line)
        traces.append(trace)
    responses = []
    for index in range(1, 4):
        record = _response_trace(config=config, repetition=index)
        reference = run_dir / record.reference_path
        reference.parent.mkdir(parents=True)
        reference.write_text(
            json.dumps(
                {
                    "set_name": record.set_name,
                    "seed": record.seed,
                    "controller_state_method": "px4-ulog-controller-boundary-v2",
                    "traces": [
                        {
                            "profile": "test",
                            "family": "velocity",
                            "control_mode": "velocity",
                            "axis": "x",
                            "split": "fit",
                            "completed": True,
                            "time_s": [0.0, 0.06 if mutation == "jitter" else 0.05],
                            "position_enu_m": [[0.0, 0.0, 2.5], [0.01, 0.0, 2.5]],
                            "initial_position_enu_m": [0.0, 0.0, 2.5],
                            "velocity_body_flu_mps": [[0.01, 0.0, 0.0]] * 2,
                            "angular_velocity_body_flu_radps": [[0.0, 0.0, 0.001]] * 2,
                            "command_velocity_body_flu_mps": [[0.0, 0.0, 0.0]] * 2,
                            "command_yaw_rate_radps": [0.0, 0.0],
                        }
                    ],
                }
            )
        )
        (run_dir / record.ulog_path).write_text("synthetic ulog")
        (run_dir / record.bridge_log_path).write_text(
            bridge_line.replace("sim_time_us=8000", "sim_time_us=None")
            if mutation == "missing-bridge-time"
            else bridge_line * (2 if mutation == "mixed-bridge" else 1)
        )
        responses.append(
            replace(
                record,
                reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                ulog_sha256=hashlib.sha256((run_dir / record.ulog_path).read_bytes()).hexdigest(),
                bridge_log_sha256=hashlib.sha256(
                    (run_dir / record.bridge_log_path).read_bytes()
                ).hexdigest(),
            )
        )
    if compressed:
        for path in tuple(run_dir.rglob("*")):
            if path.is_file() and path.suffix in {".ulg", ".log"}:
                compress_flight_evidence(path=path, level=6)
                path.unlink()
        responses = [
            replace(
                record,
                ulog_path=record.ulog_path + ".gz",
                bridge_log_path=record.bridge_log_path + ".gz",
            )
            for record in responses
        ]
    parameters = derive_flight_parameters(
        config=config,
        run_id=run_id,
        traces=tuple(traces),
        response_traces=tuple(responses),
        source_commit="abc123",
        source_dirty=False,
        created_at_utc="2026-09-26T00:00:00Z",
    )
    write_flight_parameters(flight_parameters=parameters, run_dir=run_dir)
    return run_dir


def test_stability_is_read_only_repeatable_and_requires_independent_evidence(
    tmp_path: Path,
) -> None:
    first = _stability_run(root=tmp_path, run_id="first")
    second = _stability_run(root=tmp_path, run_id="second")
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    kwargs = dict(
        run_dir=second,
        comparison_run_dirs=(first,),
        settings_path=REPO_ROOT / "configs/flight/stability/default.yaml",
    )
    report = analyze_flight_parameters_stability(**kwargs)
    assert report == analyze_flight_parameters_stability(**kwargs)
    assert report.status == "insufficient-evidence"
    assert report.comparisons[0].repeatability_status == "pass"
    assert report.runs[0].hover[0].sample_count == 2400
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ValueError, match="distinct producer"):
        analyze_flight_parameters_stability(**{**kwargs, "comparison_run_dirs": (second,)})


@pytest.mark.parametrize("mutation", ["spike", "drift", "bad-flight", "jitter"])
def test_stability_detects_extremes_drift_bad_flights_and_timing(
    tmp_path: Path, mutation: str
) -> None:
    first = _stability_run(root=tmp_path, run_id="first")
    second = _stability_run(root=tmp_path, run_id="second", mutation=mutation)
    report = analyze_flight_parameters_stability(
        run_dir=second,
        comparison_run_dirs=(first,),
        settings_path=REPO_ROOT / "configs/flight/stability/default.yaml",
    )
    assert report.status == ("insufficient-evidence" if mutation == "jitter" else "fail")
    summary = report.runs[0].hover[0]
    if mutation == "jitter":
        assert any(d.field == "time_s" and not d.equal for d in report.comparisons[0].differences)
    elif mutation == "drift":
        assert summary.maximum_endpoint_displacement_m > 0.08
        assert summary.maximum_position_trend_mps > 0.003
    else:
        assert summary.peak_flight == "fit/seed-1/trial-001"
        assert summary.excursion_flight_count == 1
        assert summary.longest_excursion_s == pytest.approx(0.05 if mutation == "spike" else 30)
        if mutation == "spike":
            horizontal = next(s for s in summary.signals if s.name == "horizontal_position_error")
            assert horizontal.maximum == pytest.approx(0.2)
            assert horizontal.p99_absolute == pytest.approx(0.01)
        else:
            assert summary.worst_flight_horizontal_rmse_m == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("mixed-bridge", "non-increasing bridge simulation timestamps"),
        ("missing-bridge-time", "missing bridge simulation timestamp"),
    ],
)
def test_stability_rejects_invalid_bridge_timelines(tmp_path: Path, mutation, message) -> None:
    first = _stability_run(root=tmp_path, run_id="first")
    second = _stability_run(root=tmp_path, run_id="second", mutation=mutation)
    with pytest.raises(ValueError, match=message):
        analyze_flight_parameters_stability(
            run_dir=second,
            comparison_run_dirs=(first,),
            settings_path=REPO_ROOT / "configs/flight/stability/default.yaml",
        )


def test_stability_population_does_not_treat_samples_as_independent_flights(tmp_path: Path) -> None:
    first = _stability_run(root=tmp_path, run_id="first")
    second = _stability_run(root=tmp_path, run_id="second", seed_offset=10000)
    report = analyze_flight_parameters_stability(
        run_dir=second,
        comparison_run_dirs=(first,),
        settings_path=REPO_ROOT / "configs/flight/stability/default.yaml",
    )
    assert report.status == "insufficient-evidence"
    assert report.comparisons[0].seed_relationship == "disjoint"
    rms = report.comparisons[0].population[0]
    assert rms.status == "insufficient-evidence"
    assert rms.confidence_interval is None
    response_rms = report.comparisons[0].response_population[0]
    assert response_rms.reference_flights == response_rms.current_flights == 3
    assert response_rms.status == "insufficient-evidence"


def test_campaign_analysis_retry_never_relaunches_capture(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "test-hover-v1"
    run_dir = _stability_run(root=root, run_id="retry")
    events = []

    def no_capture(*args, **kwargs):
        pytest.fail("analysis retry invoked capture or an external process")

    def analyze(**kwargs):
        assert (run_dir / "flight_parameters.json").is_file()
        assert len(list(run_dir.glob("response/*/*.json"))) == 3
        events.append("analysis")
        return analyze_flight_parameters_stability(**kwargs)

    def finalize(path, /):
        assert (path / "stability.json").is_file()
        events.append("finalize")

    def publish(path, /):
        events.append("publish")
        latest = path.parent / "latest"
        latest.symlink_to(path.name)
        return latest

    monkeypatch.setitem(
        sys.modules,
        "robotics_experiments.flight_parameters.px4_ulog",
        types.SimpleNamespace(verify_px4_hover_trial=no_capture),
    )
    monkeypatch.setattr(subprocess, "run", no_capture)
    monkeypatch.setattr(flight_parameters, "analyze_flight_parameters_stability", analyze)
    monkeypatch.setattr(flight_parameters, "finalize_flight_parameters_bundle", finalize)
    monkeypatch.setattr(flight_parameters, "publish_flight_parameters", publish)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--config",
            str(root / "flight.yaml"),
            "--output-root",
            str(tmp_path),
            "--run-id",
            "retry",
        ],
    )
    runpy.run_path(
        str(REPO_ROOT / "make/scripts/run_flight_parameters_campaign.py"), run_name="__main__"
    )
    assert events == ["analysis", "finalize", "publish"]
    assert json.loads((run_dir / "stability.json").read_text())["status"] == "insufficient-evidence"


def test_compression_preserves_stability_values_and_exact_trace_comparison(tmp_path: Path) -> None:
    plain = _stability_run(root=tmp_path, run_id="plain")
    packed = _stability_run(root=tmp_path, run_id="packed", compressed=True)
    report = analyze_flight_parameters_stability(
        run_dir=packed,
        comparison_run_dirs=(plain,),
        settings_path=REPO_ROOT / "configs/flight/stability/default.yaml",
    )
    assert report.runs[0].hover == report.runs[1].hover
    assert report.comparisons[0].repeatability_status == "pass"
    for item in (*report.comparisons[0].population, *report.comparisons[0].response_population):
        assert item.current_value == item.reference_value


@pytest.mark.parametrize(
    "interruption",
    [
        "processing",
        "compression",
        "capture",
        "low-space",
        "source-change",
        "seed-change",
        "corruption",
        "parallel",
    ],
)
def test_campaign_checkpoints_flights_and_preserves_failed_evidence(
    tmp_path: Path, monkeypatch, interruption: str
) -> None:
    config = _config(tmp_path)
    if interruption == "parallel":
        path = tmp_path / "flight.yaml"
        path.write_text(path.read_text().replace("worker_count: 1", "worker_count: 3"))
        config = load_flight_parameters_config(path)
    settings_path = tmp_path / "stability.yaml"
    settings_path.write_text(
        (REPO_ROOT / "configs/flight/stability/default.yaml")
        .read_text()
        .replace("drift_window_s: 5.0", "drift_window_s: 0.5")
    )
    captured = []
    inject = [True]
    bridge_line = (
        "PX4 bridge dynamics: sim_time_us=8000 last_actuator_us=None armed=False "
        "commands=[0,0,0,0] position_enu=[0,0,0] quat_wxyz=[1,0,0,0] "
        "velocity_enu=[0,0,0] angular_velocity_body_flu=[0,0,0]\n"
    )

    def verify(**kwargs):
        raw_path = kwargs["raw_trace_path"]
        raw = raw_flight_trace_from_payload(json.loads(raw_path.read_text()))
        verification, trace = verify_px4_hover_evidence(
            config=config,
            raw=raw,
            log=_parsed_log(),
            raw_trace_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            ulog_sha256=hashlib.sha256(kwargs["ulog_path"].read_bytes()).hexdigest(),
        )
        directory = kwargs["trial_dir"]
        (directory / "hover_verification.json").write_text(json.dumps(asdict(verification)))
        (directory / "trace.json").write_text(json.dumps(asdict(trace)))

    def execute(command, **kwargs):
        if command[0] == "make":
            values = dict(item.split("=", 1) for item in command if "=" in item)
            hover = "FLIGHT_PARAMETERS_TRIAL_INDEX" in values
            if interruption == "parallel" and not hover:
                assert len(tuple(run_dir.glob("evidence/*/complete.json"))) == 4
            kind = "hover" if hover else "response"
            index = int(
                values[
                    "FLIGHT_PARAMETERS_TRIAL_INDEX"
                    if hover
                    else "FLIGHT_PARAMETERS_RESPONSE_REPETITION"
                ]
            )
            group, seed = flight_trials(config=config, kind=kind)[index - 1]
            directory = Path(
                values["FLIGHT_PARAMETERS_TRIAL_DIR" if hover else "FLIGHT_PARAMETERS_RESPONSE_DIR"]
            )
            captured.append((kind, index))
            ulog_name, bridge_name = (
                ("flight.ulg", "bridge.log")
                if hover
                else ("flight_parameters.ulg", "flight_parameters_bridge.log")
            )
            (directory / ulog_name).write_text(f"{kind} ulog {index}")
            (directory / bridge_name).write_text(bridge_line)
            harness = Path(values["INTEGRATION_ROOT"]) / f"attempt-{len(captured)}"
            harness.mkdir(parents=True)
            (harness / "flight.ulg").write_bytes((directory / ulog_name).read_bytes())
            (harness / "flight_v0_bridge.log").write_text(bridge_line + "closed\n")
            if hover:
                raw = replace(_raw(config), trial_index=index, set_name=group, seed=seed)
                (directory / "raw_trace.json").write_text(json.dumps(asdict(raw)))
            else:
                (directory / "flight_parameters_response.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "set_name": group,
                            "seed": seed,
                            "traces": [
                                {
                                    "profile": "test",
                                    "family": "velocity",
                                    "control_mode": "velocity",
                                    "axis": "x",
                                    "split": "fit",
                                    "completed": True,
                                    "failure_reason": None,
                                    "initial_controller_state_valid": True,
                                    "time_s": [0.0, 0.05],
                                    "position_enu_m": [[0.0, 0.0, 2.5], [0.01, 0.0, 2.5]],
                                    "initial_position_enu_m": [0.0, 0.0, 2.5],
                                    "velocity_body_flu_mps": [[0.01, 0.0, 0.0]] * 2,
                                    "angular_velocity_body_flu_radps": [[0.0, 0.0, 0.001]] * 2,
                                    "command_velocity_body_flu_mps": [[0.0, 0.0, 0.0]] * 2,
                                    "command_yaw_rate_radps": [0.0, 0.0],
                                }
                            ],
                        }
                    )
                )
                if inject[0] and interruption == "capture":
                    inject[0] = False
                    raise OSError("injected capture failure")
        else:
            assert command[0] == "uv" and "--augment-reference" in command
            if inject[0] and interruption in {
                "processing",
                "source-change",
                "seed-change",
                "corruption",
            }:
                inject[0] = False
                raise OSError("injected processing failure")
            path = Path(command[command.index("--augment-reference") + 1])
            payload = json.loads(path.read_text())
            payload["controller_state_method"] = "px4-ulog-controller-boundary-v2"
            path.write_text(json.dumps(payload))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setitem(
        sys.modules,
        "robotics_experiments.flight_parameters.px4_ulog",
        types.SimpleNamespace(verify_px4_hover_trial=verify),
    )
    monkeypatch.setattr(subprocess, "run", execute)
    namespace = runpy.run_path(str(REPO_ROOT / "make/scripts/run_flight_parameters_campaign.py"))
    main = namespace["main"]
    provenance = {
        "source_commit": "test",
        "source_dirty": True,
        "source_sha256": {"capture.py": "same"},
        "images": {"px4": "image-id"},
    }
    monkeypatch.setitem(main.__globals__, "_source_provenance", lambda root: provenance)
    original_compress = main.__globals__["compress_flight_evidence"]

    def compress(**kwargs):
        result = original_compress(**kwargs)
        if inject[0] and interruption == "compression":
            inject[0] = False
            raise OSError("injected compression failure")
        return result

    monkeypatch.setitem(main.__globals__, "compress_flight_evidence", compress)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--config",
            str(tmp_path / "flight.yaml"),
            "--processing-config",
            str(REPO_ROOT / "configs/flight/processing/default.yaml"),
            "--stability-config",
            str(settings_path),
            "--output-root",
            str(tmp_path / "outputs"),
            "--run-id",
            "run-1",
        ],
    )
    run_dir = tmp_path / "outputs/test-hover-v1/run-1"
    if interruption == "low-space":
        with monkeypatch.context() as patch:
            patch.setattr(
                namespace["shutil"], "disk_usage", lambda path: types.SimpleNamespace(free=0)
            )
            with pytest.raises(OSError, match="insufficient space"):
                main()
        assert not captured
    elif interruption != "parallel":
        with pytest.raises(OSError, match="injected"):
            main()
    assert not (run_dir / "manifest.json").exists()
    if interruption == "source-change":
        with monkeypatch.context() as patch:
            patch.setitem(
                main.__globals__,
                "_source_provenance",
                lambda root: {**provenance, "source_commit": "changed"},
            )
            before = list(captured)
            with pytest.raises(ValueError, match="existing campaign record differs"):
                main()
            assert captured == before
    if interruption == "seed-change":
        path = tmp_path / "flight.yaml"
        original_config = path.read_text()
        path.write_text(original_config.replace("starting_seed: 1", "starting_seed: 2"))
        before = list(captured)
        with pytest.raises(ValueError, match="existing campaign record differs"):
            main()
        assert captured == before
        path.write_text(original_config)
    if interruption == "corruption":
        archive = run_dir / "evidence/trial-001/flight.ulg.gz"
        original = archive.read_bytes()
        archive.write_bytes(b"corrupt")
        before = list(captured)
        with pytest.raises(ValueError, match="checkpoint evidence differs"):
            main()
        assert captured == before
        archive.write_bytes(original)
    main()
    validate_flight_parameters_bundle(run_dir)
    assert (run_dir.parent / "latest").resolve() == run_dir
    assert len(tuple(run_dir.glob("evidence/*/complete.json"))) == 4
    assert len(tuple(run_dir.glob("response/*/complete.json"))) == 3
    assert len(captured) == (8 if interruption == "capture" else 7)
    assert len(set(captured)) == 7
    inputs = json.loads((run_dir / "capture_inputs.json").read_text())
    assert tuple(flight["seed"] for flight in inputs["flights"]) == tuple(range(1, 8))
    assert not tuple(run_dir.glob("evidence/*/*.ulg"))
    assert not tuple(run_dir.glob("response/*/*.ulg"))
    campaign = json.loads((run_dir / "campaign.json").read_text())
    assert campaign["failed_capture_attempts"] == (1 if interruption == "capture" else 0)
    if interruption == "capture":
        assert tuple((run_dir / "failed").rglob("*.ulg"))
    assert json.loads((run_dir / "stability.json").read_text())["status"] == "insufficient-evidence"
