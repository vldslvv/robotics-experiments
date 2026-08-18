"""Capture raw task-neutral PX4 SITL pose evidence for flight parameters."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from _px4_takeoff_scenario import (
    Px4TakeoffScenarioOptions,
    run_px4_takeoff_scenario,
)
from _runtime import SceneTestContext, run_scene_test

from robotics_experiments.flight_parameters import load_flight_parameters_config
from robotics_experiments.px4_bridge import VehicleState

_TEST_NAME = "flight_parameters_capture"
_CONFIG_ENV = "FLIGHT_PARAMETERS_CONFIG"
_RUN_ID_ENV = "FLIGHT_PARAMETERS_RUN_ID"
_TRIAL_INDEX_ENV = "FLIGHT_PARAMETERS_TRIAL_INDEX"
_TRIAL_DIR_ENV = "FLIGHT_PARAMETERS_TRIAL_DIR"
_DEFAULT_CONFIG = "configs/flight/holybro_x500_v2_hover.yaml"


def _plain(value: Any, /) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _fingerprint(value: object, /) -> str:
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _float3(values, /) -> tuple[float, float, float]:
    return float(values[0]), float(values[1]), float(values[2])


def _float4(values, /) -> tuple[float, float, float, float]:
    return float(values[0]), float(values[1]), float(values[2]), float(values[3])


def _yaw_quaternion(yaw_rad: float, /) -> tuple[float, float, float, float]:
    return math.cos(0.5 * yaw_rad), 0.0, 0.0, math.sin(0.5 * yaw_rad)


def _run(*, context: SceneTestContext) -> None:
    config = load_flight_parameters_config(os.environ.get(_CONFIG_ENV, _DEFAULT_CONFIG))
    run_id = os.environ[_RUN_ID_ENV]
    trial_index = int(os.environ[_TRIAL_INDEX_ENV])
    trial_dir = Path(os.environ[_TRIAL_DIR_ENV])
    if tuple(context.config.vehicles) != (config.vehicle,):
        raise ValueError(
            f"flight parameter experiment must contain only vehicle {config.vehicle!r}"
        )
    routine_steps = context.config.vehicles[config.vehicle].routine.steps
    expected_kinds = ("arm", "takeoff", "hover")
    if tuple(step["kind"] for step in routine_steps) != expected_kinds:
        raise ValueError(f"flight parameter routine must be {expected_kinds}")
    hover_hold_s = float(routine_steps[2]["hold_s"])
    expected_hold_s = (
        config.hover.mode_transition_timeout_s + config.hover.settle_s + config.hover.measure_s
    )
    if not math.isclose(hover_hold_s, expected_hold_s, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(
            f"flight parameter hover hold {hover_hold_s} does not equal {expected_hold_s}"
        )

    expected_samples = round(expected_hold_s * config.hover.sample_rate_hz)
    positions: list[tuple[float, float, float]] = []
    attitudes: list[tuple[float, float, float, float]] = []
    times_us: list[int] = []
    capture_start_us: int | None = None

    def routine_tick(fleet) -> None:
        nonlocal capture_start_us
        flying = fleet.flying[0]
        progress = flying.driver.routine_state
        if progress.cursor != 2:
            return
        from robotics_experiments.runtime import read_bridge_host_status

        host = flying.spawned.host
        observer = flying.spawned.observer
        if host is None or observer is None:
            raise AssertionError("flight parameter capture requires a live bridge observer")
        sim_time_us = read_bridge_host_status(host=host).sim_time_us
        if capture_start_us is None:
            capture_start_us = sim_time_us
        if len(times_us) >= expected_samples:
            return
        sample_target_us = capture_start_us + round(
            len(times_us) * 1_000_000 / config.hover.sample_rate_hz
        )
        if sim_time_us < sample_target_us:
            return
        state: VehicleState = observer.read_state(sim_time_us)
        times_us.append(sim_time_us)
        positions.append(_float3(state.position_enu))
        attitudes.append(_float4(state.orientation_flu_enu_wxyz))

    def post_routine(_context: SceneTestContext, fleet) -> None:
        if len(times_us) != expected_samples:
            raise AssertionError(
                f"flight parameter trial captured {len(times_us)} raw samples, "
                f"expected {expected_samples}"
            )
        from robotics_experiments.runtime import read_bridge_host_status

        flying = fleet.flying[0]
        host = flying.spawned.host
        if host is None:
            raise AssertionError("flight parameter capture retained no bridge host")
        debug_log_path = read_bridge_host_status(host=host).debug_log_path
        if debug_log_path is None:
            raise AssertionError("flight parameter capture retained no bridge log")
        if not trial_dir.is_dir() or any(trial_dir.iterdir()):
            raise ValueError("flight parameter trial directory must exist and be empty")
        shutil.copy2(debug_log_path, trial_dir / "bridge.log")
        experiment = _plain(context.config)
        vehicle_config = context.config.vehicles[config.vehicle]
        vehicle = _plain(vehicle_config)
        payload = {
            "artifact_kind": "robotics-experiments-flight-parameters-raw-trace",
            "schema_version": 1,
            "flight_parameters_id": config.flight_parameters_id,
            "campaign_run_id": run_id,
            "trial_index": trial_index,
            "source_backend": config.source_backend,
            "state_source": config.state_source,
            "config_fingerprint": config.config_fingerprint,
            "experiment_fingerprint": _fingerprint(experiment),
            "vehicle_fingerprint": _fingerprint(vehicle),
            "source_image": (
                f"isaac={os.environ.get('ISAAC_BENCHMARK_IMAGE', 'unknown-isaac-image')};"
                f"px4={os.environ.get('PX4_BENCHMARK_IMAGE', 'unknown-px4-image')}"
            ),
            "requested_target_position_enu_m": list(config.hover.target_position_enu_m),
            "requested_target_attitude_flu_enu_wxyz": list(
                _yaw_quaternion(config.hover.target_yaw_rad)
            ),
            "px4_home_lat_lon_alt": [
                vehicle_config.bridge.home_lat_deg,
                vehicle_config.bridge.home_lon_deg,
                vehicle_config.bridge.home_alt_m,
            ],
            "time_us": times_us,
            "position_enu_m": positions,
            "attitude_flu_enu_wxyz": attitudes,
            "completed": True,
            "failure_reason": "",
        }
        (trial_dir / "raw_trace.json").write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    run_px4_takeoff_scenario(
        context,
        Px4TakeoffScenarioOptions(
            test_name=_TEST_NAME,
            px4_container_log_name="flight_parameters_capture_px4.log",
            routine_tick_hook=routine_tick,
            post_routine_hook=post_routine,
            success_label="raw task-neutral hover evidence",
        ),
    )


def main() -> None:
    run_scene_test(
        test_name=_TEST_NAME,
        default_scene=None,
        manage_stage=False,
        extra_python_paths=["isaac/runtime/extensions/px4_sitl_brigde"],
        test_body=lambda context: _run(context=context),
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
