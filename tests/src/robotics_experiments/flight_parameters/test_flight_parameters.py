from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from robotics_experiments.flight_parameters import (
    FlightTrace,
    ResponseTraceEvidence,
    derive_flight_parameters,
    evaluate_hover_pose,
    finalize_flight_parameters_bundle,
    flight_parameters_from_payload,
    flight_trace_from_payload,
    load_flight_parameters,
    load_flight_parameters_config,
    publish_flight_parameters,
    validate_flight_parameters_bundle,
    write_flight_parameters,
)
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
hover:
  kind: hover
  repetitions: {repetitions}
  mode_transition_timeout_s: 15.0
  mode_report_max_age_s: 2.5
  settle_s: 2.0
  measure_s: 1.0
  sample_rate_hz: 2.0
  target_position_enu_m: [0.0, 0.0, 2.5]
  target_yaw_rad: 0.0
response:
  kind: controller-response
  repetitions: 1
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
    return FlightTrace(
        artifact_kind="robotics-experiments-flight-parameters-trace",
        schema_version=1,
        flight_parameters_id=config.flight_parameters_id,
        campaign_run_id="run-1",
        trial_index=index,
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
        completed=True,
        failure_reason="",
    )


def _response_trace() -> ResponseTraceEvidence:
    return ResponseTraceEvidence(
        repetition=1,
        profile_count=1,
        reference_path="response/repetition-001/flight_parameters_response.json",
        reference_sha256="b" * 64,
        ulog_path="response/repetition-001/flight_parameters.ulg",
        ulog_sha256="c" * 64,
        bridge_log_path="response/repetition-001/flight_parameters_bridge.log",
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
        completed=True,
        failure_reason="",
    )


def _flight_parameters(tmp_path: Path):
    config = _config(tmp_path)
    traces = (_trace(config=config, index=1, x=-0.02), _trace(config=config, index=2, x=0.03))
    return derive_flight_parameters(
        config=config,
        run_id="run-1",
        traces=traces,
        response_traces=(_response_trace(),),
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

    assert config.hover.repetitions == 30
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
    assert flight_parameters.hover_trial_count == 2
    assert flight_parameters.hover_sample_count == 4
    assert len(flight_parameters.artifact_fingerprint) == 64
    assert len(flight_parameters.hover.trace_statistics) == 2
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
    with pytest.raises(ValueError, match="requires 2 traces"):
        derive_flight_parameters(
            config=config,
            run_id="run-1",
            traces=(_trace(config=config, index=1, x=0.0),),
            response_traces=(_response_trace(),),
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


def test_bundle_manifest_and_atomic_latest(tmp_path: Path) -> None:
    config = _config(tmp_path, repetitions=1)
    raw = _raw(config)
    root = tmp_path / "flight-parameters" / config.flight_parameters_id
    run_dir = root / "run-1"
    evidence = run_dir / "evidence" / "trial-001"
    evidence.mkdir(parents=True)
    raw_path = evidence / "raw_trace.json"
    raw_path.write_text(json.dumps(asdict(raw), allow_nan=False), encoding="utf-8")
    ulog_path = evidence / "flight.ulg"
    ulog_path.write_bytes(b"synthetic ulog evidence")
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
    response_dir = run_dir / "response/repetition-001"
    response_dir.mkdir(parents=True)
    response_reference = response_dir / "flight_parameters_response.json"
    response_ulog = response_dir / "flight_parameters.ulg"
    response_bridge = response_dir / "flight_parameters_bridge.log"
    response_reference.write_text('{"schema_version":1,"traces":[{}]}\n', encoding="utf-8")
    response_ulog.write_bytes(b"response ulog")
    response_bridge.write_text("response bridge\n", encoding="utf-8")
    response_trace = replace(
        _response_trace(),
        reference_sha256=hashlib.sha256(response_reference.read_bytes()).hexdigest(),
        ulog_sha256=hashlib.sha256(response_ulog.read_bytes()).hexdigest(),
        bridge_log_sha256=hashlib.sha256(response_bridge.read_bytes()).hexdigest(),
    )
    flight_parameters = derive_flight_parameters(
        config=config,
        run_id="run-1",
        traces=(trace,),
        response_traces=(response_trace,),
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
