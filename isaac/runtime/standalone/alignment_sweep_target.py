"""Replay captured PX4 probes against every declarative Isaac Lab candidate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robotics_experiments.alignment import load_controller_definition
from robotics_experiments.alignment._align import _settings_candidates
from robotics_experiments.alignment._models import ControllerSettings
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


def _make_env(
    *,
    config: ExperimentConfig,
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

        def apply_controller_settings(
            self,
            *,
            controller_settings: ControllerSettings,
        ) -> None:
            self._controller_settings = controller_settings
            self._command_delay_steps = round(
                controller_settings.command_delay_s * self._experiment_config.sim.physics_hz
            )
            self._command_delay_index = 0
            self._command_delay_buffer = torch.zeros(
                (
                    self._command_delay_steps,
                    self.num_envs,
                    self.cfg.action_space,
                ),
                dtype=self.actions.dtype,
                device=self.device,
            )
            self._applied_actions.zero_()
            self._setup_actuation()

        def set_initial_state(self, *, trace: dict[str, Any]) -> None:
            from isaaclab.utils import math as math_utils

            root_state = self.drone.data.default_root_state[:1].clone()
            root_state[:, :3] = torch.as_tensor(
                trace["initial_position_enu_m"],
                dtype=torch.float32,
                device=self.device,
            )
            quaternion = torch.as_tensor(
                trace["initial_attitude_flu_enu_wxyz"],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 4)
            root_state[:, 3:7] = quaternion
            root_state[:, 7:10] = torch.as_tensor(
                trace["initial_linear_velocity_enu_mps"],
                dtype=torch.float32,
                device=self.device,
            )
            angular_body = torch.as_tensor(
                trace["initial_angular_velocity_body_flu_radps"],
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, 3)
            root_state[:, 10:13] = math_utils.quat_apply(quaternion, angular_body)
            self._reset_drone(
                env_ids=torch.tensor([0], device=self.device),
                root_state=root_state,
            )

        def body_velocity(self) -> tuple[float, float, float]:
            from isaaclab.utils import math as math_utils

            velocity = math_utils.quat_apply_inverse(
                self.drone.data.root_quat_w[:1],
                self.drone.data.root_lin_vel_w[:1],
            )[0]
            values = tuple(float(value) for value in velocity.cpu().tolist())
            return values[0], values[1], values[2]

        def flight_state(
            self,
        ) -> tuple[
            tuple[float, float, float],
            tuple[float, float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]:
            def values(tensor, /) -> tuple[float, ...]:
                return tuple(float(value) for value in tensor[0].cpu().tolist())

            position = values(self.drone.data.root_pos_w)
            attitude = values(self.drone.data.root_quat_w)
            linear_velocity = values(self.drone.data.root_lin_vel_w)
            angular_velocity = values(self.drone.data.root_ang_vel_b)
            return (
                (position[0], position[1], position[2]),
                (attitude[0], attitude[1], attitude[2], attitude[3]),
                (
                    linear_velocity[0],
                    linear_velocity[1],
                    linear_velocity[2],
                ),
                (
                    angular_velocity[0],
                    angular_velocity[1],
                    angular_velocity[2],
                ),
            )

    cfg = make_drone_hover_env_cfg(experiment_config=config)
    cfg.scene.num_envs = 1
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


def _candidate_traces(
    *,
    env,
    config: ExperimentConfig,
    settings: ControllerSettings,
    reference_traces: list[dict[str, Any]],
    vehicle_label: str,
) -> list[dict[str, Any]]:
    env.apply_controller_settings(controller_settings=settings)
    physics_hz = config.sim.physics_hz
    action_scale = config.rl.action.max_body_velocity_mps
    routine = config.vehicles[vehicle_label].routine
    output = []
    for source in reference_traces:
        env.set_initial_state(trace=source)
        times = [float(value) for value in source["time_s"]]
        commands = source["command_velocity_body_flu_mps"]
        raw_initial_position = source["initial_position_enu_m"]
        initial_position = (
            float(raw_initial_position[0]),
            float(raw_initial_position[1]),
            float(raw_initial_position[2]),
        )
        velocities = [env.body_velocity()]
        for index in range(1, len(times)):
            interval_s = times[index] - times[index - 1]
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
                (1, config.rl.action.action_dim),
                device=env.device,
            )
            action[0, :3] = (
                torch.as_tensor(
                    commands[index - 1],
                    dtype=torch.float32,
                    device=env.device,
                )
                / action_scale
            )
            env.step_with_physics_steps(
                action=action,
                physics_steps=physics_steps,
            )
            velocity = env.body_velocity()
            if not np.all(np.isfinite(velocity)):
                raise RuntimeError(
                    f"target probe {source['profile']!r} produced non-finite velocity"
                )
            _assert_flight_envelope(
                state=env.flight_state(),
                initial_position=initial_position,
                routine=routine,
            )
            velocities.append(velocity)
        output.append(
            {
                **source,
                "velocity_body_flu_mps": [list(value) for value in velocities],
                "completed": True,
                "failure_reason": None,
            }
        )
    return output


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
    candidates: dict[str, list[dict[str, Any]]] = {}
    failures: dict[str, str] = {}
    env = _make_env(config=config, settings=target.base_settings)
    try:
        for index, settings in enumerate(_settings_candidates(target)):
            fingerprint = _settings_fingerprint(settings)
            try:
                candidates[fingerprint] = _candidate_traces(
                    env=env,
                    config=config,
                    settings=settings,
                    reference_traces=reference_traces,
                    vehicle_label=target.vehicle,
                )
                _report(f"alignment target candidate {index} COMPLETE")
            except Exception as exc:  # noqa: BLE001 - retain an unusable candidate diagnostic
                failures[fingerprint] = str(exc)
                _report(f"alignment target candidate {index} UNUSABLE: {exc}")
    finally:
        env.close()
    (run_dir / "target_candidates.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
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
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
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
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
