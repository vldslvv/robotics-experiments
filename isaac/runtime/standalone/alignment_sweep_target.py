"""Replay captured PX4 probes against every declarative Isaac Lab candidate."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import queue
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robotics_experiments.alignment import load_controller_definition
from robotics_experiments.alignment._align import _staged_beam_search
from robotics_experiments.alignment._io import _trace
from robotics_experiments.alignment._models import (
    ControllerDefinition,
    ControllerSettings,
    ProbeTrace,
)
from robotics_experiments.config import ExperimentConfig, load_config

_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"
_TARGET_ENV = "ALIGNMENT_TARGET"
_RUN_DIR_ENV = "ROBOTICS_ALIGNMENT_RUN_DIR"
_DEFAULT_TARGET = "configs/alignment/controllers/isaac_lab_x500_velocity.yaml"


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _settings_fingerprint(settings: ControllerSettings, /) -> str:
    encoded = json.dumps(
        asdict(settings),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _partition_settings(
    settings: tuple[ControllerSettings, ...],
    worker_count: int,
    /,
) -> tuple[tuple[tuple[int, ControllerSettings], ...], ...]:
    active_workers = min(worker_count, len(settings))
    base, extra = divmod(len(settings), active_workers)
    shards = []
    offset = 0
    for worker_index in range(active_workers):
        size = base + int(worker_index < extra)
        shards.append(tuple(enumerate(settings[offset : offset + size], start=offset)))
        offset += size
    return tuple(shards)


def _merge_indexed_results(
    *,
    result_count: int,
    shards: list[tuple[tuple[int, list[dict[str, Any]] | str], ...]],
) -> tuple[list[dict[str, Any]] | str, ...]:
    merged: list[list[dict[str, Any]] | str | None] = [None] * result_count
    for shard in shards:
        for index, result in shard:
            if merged[index] is not None:
                raise RuntimeError(f"alignment worker returned duplicate candidate index {index}")
            merged[index] = result
    missing = [index for index, result in enumerate(merged) if result is None]
    if missing:
        raise RuntimeError(f"alignment workers omitted candidate indices {missing}")
    return tuple(result for result in merged if result is not None)


class _GpuSampler:
    def __init__(self, *, run_dir: Path, target_percent: float, interval_s: float) -> None:
        self._path = run_dir / "gpu_utilization.json"
        self._target_percent = target_percent
        self._interval_s = interval_s
        self._active = threading.Event()
        self._stop = threading.Event()
        self._samples: list[tuple[float, float, float]] = []
        self._warning = ""
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_active(self, active: bool) -> None:
        if active:
            self._active.set()
        else:
            self._active.clear()

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            if not self._active.wait(timeout=self._interval_s):
                continue
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                rows = [
                    tuple(float(value.strip()) for value in line.split(","))
                    for line in result.stdout.splitlines()
                    if line.strip()
                ]
                if rows:
                    self._samples.append(
                        (
                            statistics.fmean(row[0] for row in rows),
                            sum(row[1] for row in rows),
                            sum(row[2] for row in rows),
                        )
                    )
            except (OSError, subprocess.CalledProcessError, ValueError) as exc:
                self._warning = f"GPU sampling unavailable: {exc}"
                self._stop.set()
                break
            self._stop.wait(self._interval_s)

    def finish(
        self,
        *,
        worker_count: int,
        evaluations: int,
        batches: int,
        elapsed_s: float,
    ) -> None:
        self._active.clear()
        self._stop.set()
        self._thread.join()
        utilization = [sample[0] for sample in self._samples]
        ordered = sorted(utilization)
        percentile_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        average = statistics.fmean(utilization) if utilization else None
        payload = {
            "schema_version": 1,
            "worker_count": worker_count,
            "target_utilization_percent": self._target_percent,
            "sample_interval_s": self._interval_s,
            "sample_count": len(self._samples),
            "average_utilization_percent": average,
            "p95_utilization_percent": ordered[percentile_index] if ordered else None,
            "peak_utilization_percent": max(utilization) if utilization else None,
            "peak_memory_used_mib": (
                max(sample[1] for sample in self._samples) if self._samples else None
            ),
            "total_memory_mib": (
                max(sample[2] for sample in self._samples) if self._samples else None
            ),
            "candidate_evaluations": evaluations,
            "gpu_batches": batches,
            "elapsed_s": elapsed_s,
            "candidates_per_second": evaluations / elapsed_s if elapsed_s > 0.0 else 0.0,
            "target_reached": bool(
                utilization and average is not None and average >= self._target_percent
            ),
            "warning": self._warning,
        }
        self._path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        if self._warning:
            _report(self._warning)
        elif average is None:
            _report("GPU sampling returned no active-evaluation samples")
        elif not payload["target_reached"]:
            _report(
                f"GPU utilization averaged {average:.1f}% below the "
                f"{self._target_percent:.1f}% target; preserve deterministic work "
                "and inspect the recorded bottleneck metrics"
            )


def _make_env(
    *,
    config: ExperimentConfig,
    num_envs: int,
    settings: ControllerSettings,
):
    from isaaclab.envs import DirectRLEnv

    from robotics_experiments.workflows.isaac_lab import (
        DroneHoverEnv,
        make_drone_hover_env_cfg,
    )

    class _AlignmentReplayEnv(DroneHoverEnv):
        def _get_dones(self):
            no_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return no_done, no_done

        def _get_rewards(self):
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        def step_with_physics_steps(self, *, action: torch.Tensor, physics_steps: int):
            self.cfg.decimation = physics_steps
            return DirectRLEnv.step(self, action)

        def apply_controller_settings_batch(
            self,
            *,
            controller_settings: tuple[ControllerSettings, ...],
            traces_per_candidate: int,
        ) -> None:
            self._set_controller_settings_batch(
                controller_settings=controller_settings,
                environments_per_setting=traces_per_candidate,
            )

        def set_initial_states(self, *, traces: list[dict[str, Any]]) -> None:
            from isaaclab.utils import math as math_utils

            from robotics_experiments.workflows.isaac_lab.tasks.flight import (
                seed_flight_controller_state,
            )

            if len(traces) != self.num_envs:
                raise ValueError(
                    f"alignment replay has {len(traces)} traces for {self.num_envs} environments"
                )
            root_state = self.drone.data.default_root_state.clone()
            root_state[:, :3] = (
                torch.as_tensor(
                    [trace["initial_position_enu_m"] for trace in traces],
                    dtype=torch.float32,
                    device=self.device,
                )
                + self.scene.env_origins
            )
            quaternion = torch.as_tensor(
                [trace["initial_attitude_flu_enu_wxyz"] for trace in traces],
                dtype=torch.float32,
                device=self.device,
            )
            root_state[:, 3:7] = quaternion
            root_state[:, 7:10] = torch.as_tensor(
                [trace["initial_linear_velocity_enu_mps"] for trace in traces],
                dtype=torch.float32,
                device=self.device,
            )
            angular_body = torch.as_tensor(
                [trace["initial_angular_velocity_body_flu_radps"] for trace in traces],
                dtype=torch.float32,
                device=self.device,
            )
            root_state[:, 10:13] = math_utils.quat_apply(quaternion, angular_body)
            self._reset_drone(
                env_ids=torch.arange(self.num_envs, device=self.device),
                root_state=root_state,
            )
            self._controller_state = seed_flight_controller_state(
                state=self._controller_state,  # type: ignore[has-type]  # Isaac base owns state.
                params=self.actuation_params,
                linear_velocity_world=root_state[:, 7:10],
                angular_velocity_body=angular_body,
                acceleration_setpoint_world=torch.as_tensor(
                    [
                        trace.get(
                            "initial_acceleration_setpoint_enu_mps2",
                            (0.0, 0.0, 0.0),
                        )
                        for trace in traces
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
                linear_acceleration_world=torch.as_tensor(
                    [
                        trace.get(
                            "initial_linear_acceleration_enu_mps2",
                            (0.0, 0.0, 0.0),
                        )
                        for trace in traces
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
                angular_acceleration_body=torch.as_tensor(
                    [
                        trace.get(
                            "initial_angular_acceleration_body_flu_radps2",
                            (0.0, 0.0, 0.0),
                        )
                        for trace in traces
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
                rate_integral_body=torch.as_tensor(
                    [
                        trace.get(
                            "initial_rate_integral_body_flu",
                            (0.0, 0.0, 0.0),
                        )
                        for trace in traces
                    ],
                    dtype=torch.float32,
                    device=self.device,
                ),
                valid=torch.as_tensor(
                    [trace.get("initial_controller_state_valid", False) for trace in traces],
                    dtype=torch.bool,
                    device=self.device,
                ),
            )

        def body_velocities(self) -> list[tuple[float, float, float]]:
            from isaaclab.utils import math as math_utils

            velocities = math_utils.quat_apply_inverse(
                self.drone.data.root_quat_w,
                self.drone.data.root_lin_vel_w,
            )
            return [
                (float(value[0]), float(value[1]), float(value[2]))
                for value in velocities.cpu().tolist()
            ]

        def flight_states(
            self,
        ) -> list[
            tuple[
                tuple[float, float, float],
                tuple[float, float, float, float],
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ]:
            positions = (self.drone.data.root_pos_w - self.scene.env_origins).cpu().tolist()
            attitudes = self.drone.data.root_quat_w.cpu().tolist()
            linear_velocities = self.drone.data.root_lin_vel_w.cpu().tolist()
            angular_velocities = self.drone.data.root_ang_vel_b.cpu().tolist()
            return [
                (
                    (float(position[0]), float(position[1]), float(position[2])),
                    (
                        float(attitude[0]),
                        float(attitude[1]),
                        float(attitude[2]),
                        float(attitude[3]),
                    ),
                    (
                        float(linear_velocity[0]),
                        float(linear_velocity[1]),
                        float(linear_velocity[2]),
                    ),
                    (
                        float(angular_velocity[0]),
                        float(angular_velocity[1]),
                        float(angular_velocity[2]),
                    ),
                )
                for position, attitude, linear_velocity, angular_velocity in zip(
                    positions,
                    attitudes,
                    linear_velocities,
                    angular_velocities,
                    strict=True,
                )
            ]

    cfg = make_drone_hover_env_cfg(experiment_config=config)
    cfg.seed = config.rl.training.seed
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = 120.0
    return _AlignmentReplayEnv(
        cfg=cfg,
        experiment_config=config,
        controller_settings=settings,
    )


def _assert_flight_envelope(
    *,
    state: tuple[
        tuple[float, float, float],
        tuple[float, float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    initial_position: tuple[float, float, float],
    routine,
) -> None:
    position, attitude, linear_velocity, angular_velocity = state
    _w, x, y, _z = attitude
    body_z_alignment = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    tilt_deg = math.degrees(math.acos(body_z_alignment))
    lateral_drift = math.hypot(
        position[0] - initial_position[0],
        position[1] - initial_position[1],
    )
    angular_speed = math.sqrt(sum(value * value for value in angular_velocity))
    failures = []
    if tilt_deg > routine.max_tilt_deg:
        failures.append(f"tilt {tilt_deg:.1f}deg exceeds {routine.max_tilt_deg:.1f}deg")
    if lateral_drift > routine.max_lateral_drift_m:
        failures.append(
            f"lateral drift {lateral_drift:.2f}m exceeds " f"{routine.max_lateral_drift_m:.2f}m"
        )
    if linear_velocity[2] > routine.max_vertical_speed_m_s:
        failures.append(
            f"vertical speed {linear_velocity[2]:.2f}m/s exceeds up limit "
            f"{routine.max_vertical_speed_m_s:.2f}m/s"
        )
    if linear_velocity[2] < routine.max_downward_speed_m_s:
        failures.append(
            f"vertical speed {linear_velocity[2]:.2f}m/s exceeds down limit "
            f"{routine.max_downward_speed_m_s:.2f}m/s"
        )
    if angular_speed > routine.max_angular_speed_rad_s:
        failures.append(
            f"angular speed {angular_speed:.2f}rad/s exceeds "
            f"{routine.max_angular_speed_rad_s:.2f}rad/s"
        )
    if failures:
        raise RuntimeError("target flight-envelope failure: " + "; ".join(failures))


def _candidate_trace_batch(
    *,
    env,
    config: ExperimentConfig,
    settings: tuple[ControllerSettings, ...],
    candidate_capacity: int,
    reference_traces: list[dict[str, Any]],
    vehicle_label: str,
) -> tuple[list[dict[str, Any]] | str, ...]:
    if not settings or len(settings) > candidate_capacity:
        raise ValueError(
            f"candidate batch contains {len(settings)} settings for capacity "
            f"{candidate_capacity}"
        )
    traces_per_candidate = len(reference_traces)
    padded_settings = settings + (settings[-1],) * (candidate_capacity - len(settings))
    physics_hz = config.sim.physics_hz
    action_scale = config.rl.action.max_body_velocity_mps
    routine = config.vehicles[vehicle_label].routine
    env.apply_controller_settings_batch(
        controller_settings=padded_settings,
        traces_per_candidate=traces_per_candidate,
    )
    env.set_initial_states(traces=reference_traces * candidate_capacity)
    times = [[float(value) for value in source["time_s"]] for source in reference_traces]
    sample_indices = [1 for _source in reference_traces]
    trace_initial_positions = [
        (
            float(source["initial_position_enu_m"][0]),
            float(source["initial_position_enu_m"][1]),
            float(source["initial_position_enu_m"][2]),
        )
        for source in reference_traces
    ]
    active_env_count = len(settings) * traces_per_candidate
    initial_states = env.flight_states()[:active_env_count]
    velocities = [[value] for value in env.body_velocities()[:active_env_count]]
    positions = [[state[0]] for state in initial_states]
    attitudes = [[state[1]] for state in initial_states]
    world_velocities = [[state[2]] for state in initial_states]
    angular_velocities = [[state[3]] for state in initial_states]
    failures: list[str | None] = [None for _settings in settings]
    elapsed_s = 0.0
    while any(
        sample_index < len(trace_times)
        for sample_index, trace_times in zip(sample_indices, times, strict=True)
    ):
        next_sample_s = min(
            trace_times[sample_index]
            for sample_index, trace_times in zip(sample_indices, times, strict=True)
            if sample_index < len(trace_times)
        )
        interval_s = next_sample_s - elapsed_s
        physics_steps = round(interval_s * physics_hz)
        if not math.isclose(
            physics_steps / physics_hz,
            interval_s,
            abs_tol=1e-9,
        ):
            raise ValueError(
                f"reference cadence interval {interval_s} is not representable "
                f"at {physics_hz} Hz"
            )
        action = torch.zeros(
            (env.num_envs, config.rl.action.action_dim),
            device=env.device,
        )
        for candidate_index in range(candidate_capacity):
            for trace_index, source in enumerate(reference_traces):
                sample_index = sample_indices[trace_index]
                if sample_index >= len(times[trace_index]):
                    continue
                command = source["command_velocity_body_flu_mps"][sample_index - 1]
                env_index = candidate_index * traces_per_candidate + trace_index
                action[env_index, :3] = (
                    torch.as_tensor(
                        command,
                        dtype=torch.float32,
                        device=env.device,
                    )
                    / action_scale
                )
                action[env_index, 3] = float(
                    source["command_yaw_rate_radps"][sample_index - 1]
                ) / float(config.rl.action.max_yaw_rate_radps)
        env.step_with_physics_steps(
            action=action,
            physics_steps=physics_steps,
        )
        elapsed_s = next_sample_s
        body_velocities = env.body_velocities()
        flight_states = env.flight_states()
        for trace_index, source in enumerate(reference_traces):
            sample_index = sample_indices[trace_index]
            if sample_index >= len(times[trace_index]) or not math.isclose(
                times[trace_index][sample_index],
                elapsed_s,
                abs_tol=1e-9,
            ):
                continue
            for candidate_index in range(len(settings)):
                if failures[candidate_index] is not None:
                    continue
                env_index = candidate_index * traces_per_candidate + trace_index
                velocity = body_velocities[env_index]
                flight_state = flight_states[env_index]
                try:
                    if not np.all(np.isfinite(velocity)):
                        raise RuntimeError(
                            f"target probe {source['profile']!r} produced " "non-finite velocity"
                        )
                    _assert_flight_envelope(
                        state=flight_state,
                        initial_position=trace_initial_positions[trace_index],
                        routine=routine,
                    )
                except RuntimeError as exc:
                    failures[candidate_index] = str(exc)
                    continue
                velocities[env_index].append(velocity)
                positions[env_index].append(flight_state[0])
                attitudes[env_index].append(flight_state[1])
                world_velocities[env_index].append(flight_state[2])
                angular_velocities[env_index].append(flight_state[3])
            sample_indices[trace_index] += 1

    results: list[list[dict[str, Any]] | str] = []
    for candidate_index, failure in enumerate(failures):
        if failure is not None:
            results.append(failure)
            continue
        raw_traces = []
        for trace_index, source in enumerate(reference_traces):
            env_index = candidate_index * traces_per_candidate + trace_index
            raw_traces.append(
                {
                    **source,
                    "position_enu_m": [list(value) for value in positions[env_index]],
                    "attitude_flu_enu_wxyz": [list(value) for value in attitudes[env_index]],
                    "velocity_body_flu_mps": [list(value) for value in velocities[env_index]],
                    "velocity_enu_mps": [list(value) for value in world_velocities[env_index]],
                    "angular_velocity_body_flu_radps": [
                        list(value) for value in angular_velocities[env_index]
                    ],
                    "completed": True,
                    "failure_reason": None,
                }
            )
        results.append(raw_traces)
    return tuple(results)


def _worker_main(
    *,
    worker_index: int,
    config_path: str,
    target_path: str,
    reference_traces: list[dict[str, Any]],
    candidate_capacity: int,
    task_queue,
    result_queue,
) -> None:
    simulation_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(headless=True)
        simulation_app = app_launcher.app
        config = load_config(config_path)
        target = load_controller_definition(target_path)
        env = _make_env(
            config=config,
            num_envs=candidate_capacity * len(reference_traces),
            settings=target.base_settings,
        )
        result_queue.put(("ready", worker_index, None))
        while True:
            task = task_queue.get()
            if task is None:
                break
            batch_id, indexed_settings = task
            settings = tuple(setting for _index, setting in indexed_settings)
            raw = _candidate_trace_batch(
                env=env,
                config=config,
                settings=settings,
                candidate_capacity=candidate_capacity,
                reference_traces=reference_traces,
                vehicle_label=target.vehicle,
            )
            result_queue.put(
                (
                    "result",
                    worker_index,
                    (
                        batch_id,
                        tuple(
                            (index, result)
                            for (index, _setting), result in zip(indexed_settings, raw, strict=True)
                        ),
                    ),
                )
            )
    except BaseException as exc:
        result_queue.put(("error", worker_index, f"{type(exc).__name__}: {exc}"))
    finally:
        if env is not None:
            env.close()
        if simulation_app is not None:
            simulation_app.close()


class _WorkerPool:
    def __init__(
        self,
        *,
        config_path: str,
        target_path: Path,
        reference_traces: list[dict[str, Any]],
        worker_count: int,
        candidate_capacity: int,
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.worker_count = worker_count
        self._result_queue = context.Queue()
        self._task_queues = [context.Queue() for _index in range(worker_count)]
        self._processes = [
            context.Process(
                target=_worker_main,
                kwargs={
                    "worker_index": index,
                    "config_path": config_path,
                    "target_path": str(target_path),
                    "reference_traces": reference_traces,
                    "candidate_capacity": candidate_capacity,
                    "task_queue": self._task_queues[index],
                    "result_queue": self._result_queue,
                },
                name=f"alignment-worker-{index}",
            )
            for index in range(worker_count)
        ]
        for process in self._processes:
            process.start()
        try:
            self._wait_for_ready()
        except BaseException:
            self.terminate()
            raise
        self._batch_id = 0

    def _receive(self):
        while True:
            try:
                return self._result_queue.get(timeout=1.0)
            except queue.Empty:
                failed = [
                    f"{process.name} exited with code {process.exitcode}"
                    for process in self._processes
                    if process.exitcode is not None and process.exitcode != 0
                ]
                if failed:
                    raise RuntimeError("alignment worker failure: " + "; ".join(failed)) from None

    def _wait_for_ready(self) -> None:
        ready: set[int] = set()
        while len(ready) < self.worker_count:
            kind, worker_index, payload = self._receive()
            if kind == "error":
                raise RuntimeError(f"alignment worker {worker_index} failed to start: {payload}")
            if kind != "ready":
                raise RuntimeError(f"unexpected alignment worker message {kind!r}")
            ready.add(worker_index)

    def evaluate(
        self,
        settings: tuple[ControllerSettings, ...],
        /,
    ) -> tuple[list[dict[str, Any]] | str, ...]:
        indexed_shards = _partition_settings(settings, self.worker_count)
        batch_id = self._batch_id
        self._batch_id += 1
        for worker_index, shard in enumerate(indexed_shards):
            self._task_queues[worker_index].put((batch_id, shard))
        results: list[tuple[tuple[int, list[dict[str, Any]] | str], ...]] = []
        while len(results) < len(indexed_shards):
            kind, worker_index, payload = self._receive()
            if kind == "error":
                raise RuntimeError(f"alignment worker {worker_index} failed: {payload}")
            if kind != "result":
                raise RuntimeError(f"unexpected alignment worker message {kind!r}")
            result_batch_id, shard = payload
            if result_batch_id != batch_id:
                raise RuntimeError(
                    f"alignment worker returned batch {result_batch_id}, expected {batch_id}"
                )
            results.append(shard)
        return _merge_indexed_results(result_count=len(settings), shards=results)

    def close(self) -> None:
        for task_queue in self._task_queues:
            task_queue.put(None)
        for process in self._processes:
            process.join()
        failures = [
            f"{process.name} exited with code {process.exitcode}"
            for process in self._processes
            if process.exitcode != 0
        ]
        if failures:
            raise RuntimeError("alignment worker shutdown failure: " + "; ".join(failures))

    def terminate(self) -> None:
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join()


class _LiveEvaluator:
    def __init__(
        self,
        *,
        pool: _WorkerPool,
        sampler: _GpuSampler,
    ) -> None:
        self._pool = pool
        self._sampler = sampler
        self._evaluations = 0
        self._batches = 0
        self._started_s = time.monotonic()

    def finish_sampling(self) -> None:
        self._sampler.finish(
            worker_count=self._pool.worker_count,
            evaluations=self._evaluations,
            batches=self._batches,
            elapsed_s=time.monotonic() - self._started_s,
        )

    def evaluate(
        self,
        *,
        definition: ControllerDefinition,
        settings: ControllerSettings,
        reference_traces: tuple[ProbeTrace, ...] | None,
    ) -> tuple[ProbeTrace, ...]:
        result = self.evaluate_batch(
            definition=definition,
            settings=(settings,),
            reference_traces=reference_traces,
        )[0]
        if isinstance(result, str):
            raise RuntimeError(result)
        return result

    def evaluate_batch(
        self,
        *,
        definition: ControllerDefinition,
        settings: tuple[ControllerSettings, ...],
        reference_traces: tuple[ProbeTrace, ...] | None,
    ) -> tuple[tuple[ProbeTrace, ...] | str, ...]:
        del definition, reference_traces
        self._evaluations += len(settings)
        self._batches += 1
        self._sampler.set_active(True)
        try:
            raw_results = self._pool.evaluate(settings)
        finally:
            self._sampler.set_active(False)
        elapsed_s = time.monotonic() - self._started_s
        _report(
            f"alignment target evaluated {self._evaluations} candidates in "
            f"{self._batches} GPU batches ({elapsed_s:.1f}s)"
        )
        parsed: list[tuple[ProbeTrace, ...] | str] = []
        for candidate_index, raw_result in enumerate(raw_results):
            if isinstance(raw_result, str):
                parsed.append(raw_result)
                continue
            parsed.append(
                tuple(
                    _trace(
                        value=raw_trace,
                        name=f"live_target[{candidate_index}][{trace_index}]",
                    )
                    for trace_index, raw_trace in enumerate(raw_result)
                )
            )
        return tuple(parsed)


def _run(
    *,
    config: ExperimentConfig,
    target_path: Path,
    run_dir: Path,
) -> None:
    target = load_controller_definition(target_path)
    reference_path = run_dir / "reference.json"
    payload = json.loads(reference_path.read_text())
    reference_traces = payload.get("traces")
    if not isinstance(reference_traces, list):
        raise ValueError(f"{reference_path} has no traces list")
    maximum_expansion = max(
        target.search.beam_width * len(parameter.candidates) for parameter in target.parameters
    )
    worker_count = min(config.alignment.execution.worker_count, maximum_expansion)
    candidate_capacity = math.ceil(maximum_expansion / worker_count)
    sampler = _GpuSampler(
        run_dir=run_dir,
        target_percent=config.alignment.execution.gpu_target_percent,
        interval_s=config.alignment.execution.gpu_sample_interval_s,
    )
    pool = _WorkerPool(
        config_path=os.environ[_CONFIG_PATH_ENV],
        target_path=target_path,
        reference_traces=reference_traces,
        worker_count=worker_count,
        candidate_capacity=candidate_capacity,
    )
    sampler.start()
    evaluator = _LiveEvaluator(pool=pool, sampler=sampler)
    try:
        live_target = replace(
            target,
            evaluator=evaluator,
        )
        parsed_reference = tuple(
            _trace(value=trace, name=f"reference[{index}]")
            for index, trace in enumerate(reference_traces)
        )
        outcome = _staged_beam_search(
            target=live_target,
            reference_traces=parsed_reference,
        )
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        evaluator.finish_sampling()
    candidates = {
        _settings_fingerprint(settings): [asdict(trace) for trace in traces]
        for settings, traces in outcome.target_traces.items()
    }
    failures = {
        _settings_fingerprint(score.settings): score.failure_reason
        for score in outcome.candidate_scores
        if not score.usable
    }
    _report(
        f"alignment target search evaluated {len(outcome.candidate_scores)} unique "
        f"candidates; selected {outcome.selected.index}"
    )
    (run_dir / "target_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "controller_id": target.controller_id,
                "definition_fingerprint": target.definition_fingerprint,
                "image": os.environ.get("ISAAC_BENCHMARK_IMAGE", target.image),
                "candidates": candidates,
                "failures": failures,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def main() -> None:
    try:
        config = load_config(os.environ[_CONFIG_PATH_ENV])
        _run(
            config=config,
            target_path=Path(os.environ.get(_TARGET_ENV, _DEFAULT_TARGET)),
            run_dir=Path(os.environ[_RUN_DIR_ENV]),
        )
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise


if __name__ == "__main__":
    main()
