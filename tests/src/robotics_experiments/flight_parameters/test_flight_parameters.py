from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from omegaconf import OmegaConf

from robotics_experiments.flight_parameters import (
    analyze_flight_parameters_stability,
    compress_flight_evidence,
    flight_seeds,
    load_flight_dataset,
    load_flight_parameters_config,
)
from robotics_experiments.flight_parameters._dataset import (
    finalize_flight_dataset,
    write_flight_dataset,
)
from robotics_experiments.flight_parameters._models import RawFlightTrace
from robotics_experiments.flight_parameters.px4_evidence import (
    ParsedPx4HoverLog,
    verify_px4_hover_evidence,
)
from tests.support.paths import REPO_ROOT


def _config(tmp_path: Path, repetitions: int = 2, /):
    payload = cast(
        dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.load(REPO_ROOT / "configs/flight/holybro_x500_v2_suite.yaml")
        ),
    )
    payload["flight_count"] = repetitions
    payload["start_seed"] = 1
    payload["stages"][0]["hover"].update(mode_transition_timeout_s=3.0, settle_s=2.0, measure_s=1.0)
    path = tmp_path / "flight.yaml"
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload)))
    return replace(load_flight_parameters_config(path=path), sample_rate_hz=2.0)


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
    period_us = round(1e6 / config.sample_rate_hz)
    times = tuple(
        10_000_000 + index * period_us + (period_us // 5 if index % 2 else 0)
        for index in range(round(6 * config.sample_rate_hz) + 1)
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


@pytest.mark.parametrize("sample_rate", [2.0, 4.0])
def test_px4_verification_selects_only_auto_loiter_samples(tmp_path: Path, sample_rate) -> None:
    config = replace(_config(tmp_path, 1), sample_rate_hz=sample_rate)
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
    assert config.command_rate_hz == 20.0
    assert len(trace.time_s) == round(config.hover.measure_s * sample_rate)
    measured = slice(round(5 * sample_rate), round(6 * sample_rate))
    assert trace.linear_velocity_enu_mps == _raw(config).linear_velocity_enu_mps[measured]
    assert trace.angular_velocity_enu_radps == _raw(config).angular_velocity_enu_radps[measured]


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
    config = _config(tmp_path, 1)
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


def test_external_offboard_commands_after_hover_are_allowed(tmp_path: Path) -> None:
    config = _config(tmp_path)
    log = _parsed_log(commands=(400, 22, 192))
    log.vehicle_command["timestamp"][-1] = 16_100_000
    verification, _ = verify_px4_hover_evidence(
        config=config, raw=_raw(config), log=log, raw_trace_sha256="1" * 64, ulog_sha256="2" * 64
    )
    assert [event.command_id for event in verification.command_events] == [400, 22]


def test_canonical_suite_preserves_physical_commands_and_seed_order() -> None:
    config = load_flight_parameters_config(
        path=REPO_ROOT / "configs/flight/holybro_x500_v2_suite.yaml"
    )
    assert config.flight_count == 10
    assert len(config.stages) == 24
    assert config.stages[0].kind == "hover"
    assert flight_seeds(config=config) == tuple(range(1001, 1011))
    # Protect the physical suite while alignment still owns its duplicated definitions.
    source = cast(
        dict[str, Any],
        OmegaConf.to_container(
            OmegaConf.load(REPO_ROOT / "configs/alignment/controllers/px4_sitl_x500_velocity.yaml")
        ),
    )
    authored = [
        (family["control_mode"], profile)
        for family in source["probe_families"]
        for profile in family["profiles"]
    ]
    for stage, (mode, profile) in zip(config.stages[1:], authored, strict=True):
        assert stage.name == profile["name"] and stage.axis == profile["axis"]
        assert stage.control_mode == mode
        for command, original in zip(stage.commands, profile["commands"], strict=True):
            captured = asdict(command)
            assert all(
                captured[key] == tuple(value) if isinstance(value, list) else captured[key] == value
                for key, value in original.items()
            )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_campaign_size_fails(count) -> None:
    with pytest.raises(ValueError, match="flight_count"):
        load_flight_parameters_config(
            path=REPO_ROOT / "configs/flight/holybro_x500_v2_suite.yaml", flight_count=count
        )


def test_count_override_changes_identity_without_changing_stages() -> None:
    path = REPO_ROOT / "configs/flight/holybro_x500_v2_suite.yaml"
    a = load_flight_parameters_config(path=path)
    b = load_flight_parameters_config(path=path, flight_count=23)
    assert a.stages == b.stages
    assert a.config_fingerprint != b.config_fingerprint
    assert flight_seeds(config=b) == tuple(range(1001, 1024))


def _digest(path: Path, /) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path, /) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _dataset(
    *, root: Path, run_id: str, seed_offset: int = 0, flights: int = 10, shift: float = 0.0
) -> Path:
    run = root / run_id
    run.mkdir()
    (run / "definition.json").write_text(
        json.dumps(
            {
                "flight_parameters_id": "test",
                "start_seed": 1001 + seed_offset,
                "flight_count": flights,
                "stages": [{"name": "hover"}, {"name": "velocity"}],
            }
        )
    )
    records = []
    for index in range(flights):
        seed = 1001 + index + seed_offset
        directory = run / "flights" / f"flight-{index:03d}"
        directory.mkdir(parents=True)
        stages = [
            {
                "name": "hover",
                "kind": "hover",
                "control_mode": "auto-loiter",
                "axis": "pose",
                "commands": [],
                "hover": {},
                "stage_index": 0,
                "start_sim_time_us": 1_000_000,
                "end_sim_time_us": 11_000_000,
                "sample_start": 0,
                "sample_stop": 20,
                "effective_target_position_enu_m": [0.0, 0.0, 2.5],
                "effective_target_attitude_flu_enu_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            {
                "name": "velocity",
                "kind": "controller-response",
                "control_mode": "velocity",
                "axis": "x",
                "commands": [],
                "hover": None,
                "stage_index": 1,
                "start_sim_time_us": 11_000_000,
                "end_sim_time_us": 12_000_000,
                "sample_start": 20,
                "sample_stop": 22,
                "initial_position_enu_m": [0.0, 0.0, 2.5],
                "initial_controller_state_valid": True,
            },
        ]
        metadata = {
            "completed": True,
            "sample_count": 22,
            "flight_index": index,
            "seed": seed,
            "run_id": run_id,
            "sample_rate_hz": 2.0,
            "controller_state_method": "px4-ulog-controller-boundary-v2",
            "stages": stages,
        }
        (directory / "flight.json").write_text(json.dumps(metadata))
        rows = [
            {
                "sim_time_us": 1_000_000 + i * 500_000,
                "stage_index": 0 if i < 20 else 1,
                "phase": "measurement",
                "position_enu_m": [0.01 + shift, 0.0, 2.5],
                "attitude_flu_enu_wxyz": [1.0, 0.0, 0.0, 0.0],
                "linear_velocity_enu_mps": [0.01, 0.0, 0.0],
                "velocity_body_flu_mps": [0.01, 0.0, 0.0],
                "angular_velocity_body_flu_radps": [0.0, 0.0, 0.01],
                "angular_velocity_enu_radps": [0.0, 0.0, 0.01],
                "requested_command": {
                    "forward_mps": 0.02,
                    "right_mps": 0.0,
                    "down_mps": 0.0,
                    "yaw_rate_radps": 0.0,
                },
            }
            for i in range(22)
        ]
        (directory / "motion.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        bridge = directory / "bridge.log"
        bridge.write_text(
            "PX4 bridge dynamics: sim_time_us=1000000 last_actuator_us=1000000 "
            "armed=True commands=[0.1] position_enu=[0,0,2.5] quat_wxyz=[1,0,0,0] "
            "velocity_enu=[0,0,0] angular_velocity_body_flu=[0,0,0]\n"
        )
        compress_flight_evidence(path=bridge, level=1)
        bridge.unlink()
        completed = {
            "flight": {
                "index": index,
                "seed": seed,
                "directory": directory.relative_to(run).as_posix(),
            },
            "completed": True,
            "files": [_record(path, directory) for path in sorted(directory.iterdir())],
        }
        (directory / "complete.json").write_text(json.dumps(completed))
        records.append(_record(directory / "complete.json", run))
    write_flight_dataset(
        run_dir=run,
        payload={
            "flight_parameters_id": "test",
            "run_id": run_id,
            "completed": True,
            "definition": _record(run / "definition.json", run),
            "flights": records,
        },
    )
    return run


def test_dataset_publication_freezes_metadata_and_checks_all_hashes(tmp_path: Path) -> None:
    run = _dataset(root=tmp_path, run_id="run-a")
    latest = finalize_flight_dataset(run)
    dataset = load_flight_dataset(latest)
    assert dataset.run_dir == run and dataset.flights[0].seed == 1001
    with pytest.raises(TypeError):
        dataset.flights[0].metadata["seed"] = 7
    (run / "flights/flight-000/motion.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        load_flight_dataset(latest)


def test_stability_resamples_whole_flights_and_requires_comparison_batches(tmp_path: Path) -> None:
    a = _dataset(root=tmp_path, run_id="run-a")
    b = _dataset(root=tmp_path, run_id="run-b")
    c = _dataset(root=tmp_path, run_id="run-c", seed_offset=100)
    settings = REPO_ROOT / "configs/flight/stability/default.yaml"
    report = analyze_flight_parameters_stability(
        run_dir=a, comparison_run_dirs=(), settings_path=settings
    )
    assert report.status == "insufficient-evidence"
    assert report.runs[0].hover[0].sample_count == 200
    assert report.runs[0].response[0].flight_count == 10
    report = analyze_flight_parameters_stability(
        run_dir=a, comparison_run_dirs=(b, c), settings_path=settings
    )
    assert report.status == "pass"
    assert report.comparisons[0].population[0].current_flights == 10
    bad = _dataset(root=tmp_path, run_id="bad", shift=0.1)
    report = analyze_flight_parameters_stability(
        run_dir=bad, comparison_run_dirs=(a,), settings_path=settings
    )
    assert report.status == "fail"
    assert report.runs[0].hover[0].excursion_flight_count == 10


@pytest.mark.parametrize(("flight_count", "workers"), ((1, 1), (4, 2)))
def test_campaign_identity_and_resume_preserve_evidence(
    tmp_path, monkeypatch, flight_count, workers
):
    import runpy
    import sys
    import types
    from threading import Barrier, Lock

    pyulog = types.ModuleType("pyulog")
    pyulog.ULog = object  # Unit test supplies processing rather than a ULog decoder.
    monkeypatch.setitem(sys.modules, "pyulog", pyulog)
    namespace = runpy.run_path(str(REPO_ROOT / "make/scripts/run_flight_parameters_campaign.py"))
    main = namespace["main"]
    globals_ = main.__globals__
    captures = []
    capture_barrier = Barrier(workers, timeout=10)
    capture_lock = Lock()
    active_captures = 0
    peak_captures = 0
    monkeypatch.setitem(globals_, "_check_space", lambda **_kwargs: None)

    def capture(command, *, cwd, check):
        nonlocal active_captures, peak_captures
        assert check
        values = dict(item.split("=", 1) for item in command if "=" in item)
        directory = Path(values["FLIGHT_PARAMETERS_TRIAL_DIR"])
        index = int(values["FLIGHT_PARAMETERS_FLIGHT_INDEX"])
        for previous in range(index // workers * workers):
            completed = directory.parent / f"flight-{previous:03d}"
            assert (completed / "complete.json").is_file()
            assert (completed / "flight.ulg.gz").is_file()
            assert not (completed / "flight.ulg").exists()
        with capture_lock:
            active_captures += 1
            peak_captures = max(peak_captures, active_captures)
        capture_barrier.wait()
        integration = Path(values["INTEGRATION_ROOT"]) / "capture"
        integration.mkdir(parents=True)
        for name, content in (
            ("raw_flight.json", "{}"),
            ("motion.jsonl", "{}\n"),
            ("flight.ulg", "ulog evidence"),
            ("bridge.log", "bridge evidence\n"),
        ):
            (directory / name).write_text(content)
        (integration / "flight.ulg").write_bytes((directory / "flight.ulg").read_bytes())
        (integration / "capture_bridge.log").write_bytes((directory / "bridge.log").read_bytes())
        captures.append(directory)
        capture_barrier.wait()
        with capture_lock:
            active_captures -= 1

    monkeypatch.setitem(globals_, "subprocess", types.SimpleNamespace(run=capture))

    def process(*, repo_root, config, run_dir, flight):
        (run_dir / flight.directory / "flight.json").write_text(
            json.dumps(
                {
                    "flight_index": flight.index,
                    "seed": flight.seed,
                    "run_id": run_dir.name,
                    "completed": True,
                    "stages": [asdict(stage) for stage in config.stages],
                }
            )
        )

    monkeypatch.setitem(globals_, "_process_flight", process)

    def interrupted_analysis(**_kwargs):
        raise RuntimeError("analysis interrupted")

    monkeypatch.setitem(globals_, "_publish_campaign", interrupted_analysis)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "flight.yaml"
    config_path.write_text(
        (REPO_ROOT / "configs/flight/holybro_x500_v2_suite.yaml")
        .read_text()
        .replace("worker_count: 1", f"worker_count: {workers}")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "campaign",
            "--config",
            str(config_path),
            "--flight-count",
            str(flight_count),
            "--run-id",
            "checkpoint",
            "--output-root",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(RuntimeError, match="analysis interrupted"):
        main()
    assert len(captures) == flight_count and peak_captures == workers
    directory = captures[0]
    assert (directory / "flight.ulg.gz").is_file()
    assert not (directory / "flight.ulg").exists()
    complete = (directory / "complete.json").read_bytes()
    header = json.loads((directory.parents[1] / "flight_parameters.json").read_text())
    assert header["completed"]
    assert set(header) == {
        "artifact_kind",
        "schema_version",
        "flight_parameters_id",
        "run_id",
        "completed",
        "definition",
        "flights",
    }
    definition = json.loads((directory.parents[1] / "definition.json").read_text())
    assert definition["flight_count"] == flight_count and len(definition["stages"]) == 24
    dataset = load_flight_dataset(directory.parents[1])
    assert [(flight.flight_index, flight.seed) for flight in dataset.flights] == [
        (index, definition["start_seed"] + index) for index in range(flight_count)
    ]
    calls = []
    monkeypatch.setitem(globals_, "_publish_campaign", lambda **kwargs: calls.append(kwargs))
    main()
    assert len(captures) == flight_count and len(calls) == 1
    assert (directory / "complete.json").read_bytes() == complete
    (directory / "bridge.log.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash or size mismatch"):
        main()
    assert len(captures) == flight_count
