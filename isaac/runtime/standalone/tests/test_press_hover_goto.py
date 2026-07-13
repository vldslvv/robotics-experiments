"""Standalone in-Lab hover + goto validation gate for the F1 actuation seam.

Drives the flyable ``DroneHoverEnv`` with an analytic (non-RL) outer position
controller feeding the env's ``velocity`` action, to confirm the in-Lab inner
loop is trustworthy before any RL: command a hover, then a 2 m lateral goto, and
assert every parallel env holds hover and tracks the goto. This is the F1 Step-7
gate and the precondition for F3.

Unlike the ``_runtime.py`` standalone tests (which drive the single-env numpy
``IsaacArticulationIO``), this exercises the Isaac Lab ``DirectRLEnv`` surface, so
it launches its own ``AppLauncher`` and builds the env directly. Failures print
the ``[isaac-standalone-test] ERROR:`` sentinel the make harness greps for.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh \
        /isaac-sim/proj/isaac/runtime/standalone/tests/test_press_hover_goto.py
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from robotics_experiments.config import ExperimentConfig, load_config

if TYPE_CHECKING:
    from robotics_experiments.config import PressActionConfig
    from robotics_experiments.workflows.isaac_lab import (
        DroneHoverEnv,
        DroneHoverEnvCfg,
    )

# Analytic outer-loop gain + tolerances for the gate (validation-only, not a
# config surface): a position P controller produces the velocity command the
# env's velocity actuation tracks.
_POSITION_KP = 1.0
_HOVER_SETTLE_S = 4.0
_HOVER_TOLERANCE_M = 0.1
_GOTO_DISTANCE_M = 2.0
_GOTO_SETTLE_S = 6.0
_GOTO_TOLERANCE_M = 0.2
_SETTLED_WINDOW_S = 1.0
_EPISODE_MARGIN_S = 1.0
_DEFAULT_SEED = 7
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"

# F2 side-channel validation: one switch prim placed this far beyond the goto
# target along +X, outward plate normal facing the approaching drone, so the
# settled drone sits inside the handoff gate's switch-range threshold.
_SWITCH_BEYOND_GOTO_M = 0.25
_SWITCH_PRIM_PATH = "/World/PressValidationSwitch"


@dataclass(frozen=True)
class _DriveStats:
    final_error_m: float
    last_window_max_error_m: float
    peak_error_m: float


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _position_action(
    env: DroneHoverEnv, target_w: torch.Tensor, max_velocity: float
) -> torch.Tensor:
    position = env.drone.data.root_pos_w
    velocity_command = torch.clamp(
        _POSITION_KP * (target_w - position), -max_velocity, max_velocity
    )
    action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    # Spawn yaw is 0, so the world velocity command is the body-heading command;
    # the env's velocity actuation tracks it. action[3] (yaw rate) stays 0.
    action[:, :3] = velocity_command / max_velocity
    return action


def _yaw_tilt_from_quat(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    quat = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    body_z_world_z = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
    tilt = torch.acos(body_z_world_z)
    return yaw, tilt


def _diagnostic_sample(
    env: DroneHoverEnv, target_w: torch.Tensor, action: torch.Tensor, max_velocity: float
) -> str:
    # Isaac Lab workflow code is imported after AppLauncher.
    from robotics_experiments.workflows.isaac_lab import compute_lab_rotor_omega

    position_error = env.drone.data.root_pos_w - target_w
    error_norm = torch.linalg.norm(position_error, dim=-1)
    env_index = int(torch.argmax(error_norm).item())
    yaw, tilt = _yaw_tilt_from_quat(env.drone.data.root_quat_w)

    omega = compute_lab_rotor_omega(
        env.actuation_params,
        action,
        env.drone.data.root_quat_w,
        env.drone.data.root_lin_vel_w,
        env.drone.data.root_ang_vel_b,
    )
    velocity_cmd = action[:, :3] * max_velocity
    sim_time = float(env.common_step_counter * env.step_dt)

    pos = position_error[env_index].detach().cpu().tolist()
    vel_w = env.drone.data.root_lin_vel_w[env_index].detach().cpu().tolist()
    vel_b = env.drone.data.root_lin_vel_b[env_index].detach().cpu().tolist()
    ang_b = env.drone.data.root_ang_vel_b[env_index].detach().cpu().tolist()
    cmd = velocity_cmd[env_index].detach().cpu().tolist()

    return (
        f"diag t={sim_time:.2f}s env={env_index} "
        f"err={float(error_norm[env_index].item()):.3f} "
        f"pos_err=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
        f"vel_w=({vel_w[0]:+.3f},{vel_w[1]:+.3f},{vel_w[2]:+.3f}) "
        f"vel_b=({vel_b[0]:+.3f},{vel_b[1]:+.3f},{vel_b[2]:+.3f}) "
        f"yaw={float(yaw[env_index].item()):+.3f} "
        f"tilt={float(tilt[env_index].item()):.3f} "
        f"ang_b=({ang_b[0]:+.3f},{ang_b[1]:+.3f},{ang_b[2]:+.3f}) "
        f"cmd_vel=({cmd[0]:+.3f},{cmd[1]:+.3f},{cmd[2]:+.3f}) "
        f"omega=[{float(omega.min().item()):.1f},{float(omega.max().item()):.1f}]"
    )


def _drive_to(
    env: DroneHoverEnv,
    target_w: torch.Tensor,
    settle_s: float,
    max_velocity: float,
    diag: bool,
    log_every: int,
) -> _DriveStats:
    steps = max(1, round(settle_s / env.step_dt))
    window_steps = max(1, round(_SETTLED_WINDOW_S / env.step_dt))
    max_errors: list[float] = []
    for step in range(steps):
        action = _position_action(env, target_w, max_velocity)
        if diag and step % log_every == 0:
            _report(_diagnostic_sample(env, target_w, action, max_velocity))
        _, _, terminated, truncated, _ = env.step(action)
        if bool(torch.any(terminated)):
            raise AssertionError("a drone terminated (crash / out-of-bounds) before settling")
        if bool(torch.any(truncated)):
            raise AssertionError("a drone episode timed out before settling")
        max_errors.append(_max_position_error(env, target_w))

    last_window = max_errors[-window_steps:]
    return _DriveStats(
        final_error_m=max_errors[-1],
        last_window_max_error_m=max(last_window),
        peak_error_m=max(max_errors),
    )


def _max_position_error(env: DroneHoverEnv, target_w: torch.Tensor) -> float:
    return float(torch.linalg.norm(env.drone.data.root_pos_w - target_w, dim=-1).max().item())


def _reset_env(env: DroneHoverEnv, seed: int) -> None:
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()


def _make_env(
    *,
    experiment_config: ExperimentConfig,
    seed: int,
    num_envs: int | None,
) -> DroneHoverEnvCfg:
    from robotics_experiments.workflows.isaac_lab import make_drone_hover_env_cfg

    cfg = make_drone_hover_env_cfg(experiment_config=experiment_config)
    cfg.episode_length_s = _HOVER_SETTLE_S + _GOTO_SETTLE_S + _EPISODE_MARGIN_S
    if hasattr(cfg, "seed"):
        cfg.seed = seed
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg


def _ctbr_hover_action(
    env: DroneHoverEnv, action_cfg: PressActionConfig
) -> tuple[torch.Tensor, float]:
    mass = float(env.drone.root_physx_view.get_masses()[0].sum().item())
    max_total_thrust = float(env.actuation_params.max_total_thrust)
    hover_fraction = mass * 9.81 / max_total_thrust
    action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    action[:, 0] = hover_fraction / float(action_cfg.hover_thrust_fraction) - 1.0
    return action, hover_fraction


def _run_ctbr_sanity(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    action_cfg: PressActionConfig,
    env_cls: type[DroneHoverEnv],
) -> None:
    # Isaac Lab workflow code is imported after AppLauncher.
    from robotics_experiments.workflows.isaac_lab import compute_lab_rotor_omega

    _seed_everything(args.seed)
    num_envs = args.num_envs if args.num_envs is not None else 1
    env = env_cls(
        cfg=_make_env(
            experiment_config=config,
            seed=args.seed,
            num_envs=num_envs,
        ),
        experiment_config=config,
    )
    try:
        _reset_env(env, args.seed)
        hover_action, hover_fraction = _ctbr_hover_action(env, action_cfg)
        start_z = env.drone.data.root_pos_w[:, 2].clone()
        steps = max(1, round(3.0 / env.step_dt))
        pulse_start = round(0.5 / env.step_dt)
        pulse_end = round(1.0 / env.step_dt)
        max_tilt = 0.0
        max_yaw_rate = 0.0

        for step in range(steps):
            action = hover_action.clone()
            if pulse_start <= step < pulse_end:
                action[:, 3] = 0.08
            _, _, terminated, truncated, _ = env.step(action)
            if bool(torch.any(terminated)) or bool(torch.any(truncated)):
                raise AssertionError("ctbr sanity terminated or timed out")

            _yaw, tilt = _yaw_tilt_from_quat(env.drone.data.root_quat_w)
            max_tilt = max(max_tilt, float(tilt.max().item()))
            yaw_rate = torch.abs(env.drone.data.root_ang_vel_b[:, 2]).max()
            max_yaw_rate = max(max_yaw_rate, float(yaw_rate.item()))

        z_drift = torch.max(torch.abs(env.drone.data.root_pos_w[:, 2] - start_z))
        final_z_drift = float(z_drift.item())
        omega = compute_lab_rotor_omega(
            env.actuation_params,
            hover_action,
            env.drone.data.root_quat_w,
            env.drone.data.root_lin_vel_w,
            env.drone.data.root_ang_vel_b,
        )
        _report(
            "ctbr sanity: "
            f"hover_fraction={hover_fraction:.3f} "
            f"max_tilt={max_tilt:.3f} rad "
            f"max_yaw_rate={max_yaw_rate:.3f} rad/s "
            f"z_drift={final_z_drift:.3f} m "
            f"omega=[{float(omega.min().item()):.1f},{float(omega.max().item()):.1f}]"
        )
        if max_tilt > 0.15:
            raise AssertionError(f"ctbr yaw-rate pulse tilted too far: {max_tilt:.3f} rad")
        if final_z_drift > 0.05:
            raise AssertionError(f"ctbr hover altitude drift too high: {final_z_drift:.3f} m")
    finally:
        env.close()


def _setup_press_side_channel(*, goto_target_env0: np.ndarray, switch_usd_path: str):
    """Place the F2 validation switch prim and return its ground-truth provider.

    The prim's world pose is what the provider reads; its authored orientation
    (pi about Z) makes the local +X outward plate normal face the approaching
    drone, per the RL contact-frame convention.
    """
    # Isaac-app imports (omni/pxr and the press Isaac leaf) stay deferred until
    # after AppLauncher, per the standalone import-ordering rule.
    import omni.usd
    from pxr import Gf, UsdGeom

    from robotics_experiments.rl.press import GroundTruthPoseProvider

    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(_SWITCH_PRIM_PATH, "Xform")
    prim.GetReferences().AddReference(switch_usd_path)
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*(float(v) for v in goto_target_env0))
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.0, Gf.Vec3d(0.0, 0.0, 1.0)))
    return GroundTruthPoseProvider(stage, _SWITCH_PRIM_PATH)


def _press_observation_sample(env: DroneHoverEnv, provider, action_dim: int):
    """Assemble a PressObservation for env 0 from the provider + sim state.

    This is the F2 deploy-shape assembly running as a side channel: it never
    touches the env's training observation.
    """
    from robotics_experiments.px4_bridge import quat_wxyz_to_xyzw, world_enu_to_body_flu
    from robotics_experiments.rl.press import (
        PressObservation,
        to_vector,
        world_switch_to_contact_frame,
    )

    position = env.drone.data.root_pos_w[0].cpu().numpy().astype(np.float64)
    quat_wxyz = env.drone.data.root_quat_w[0].cpu().numpy().astype(np.float64)
    lin_vel_w = env.drone.data.root_lin_vel_w[0].cpu().numpy().astype(np.float64)
    ang_vel_b = env.drone.data.root_ang_vel_b[0].cpu().numpy().astype(np.float64)
    sample = provider.sample(0.0)
    sq = sample.world_switch_quat_xyzw
    switch_quat_wxyz = np.array([sq[3], sq[0], sq[1], sq[2]], dtype=np.float64)
    rel_pos, rel_quat = world_switch_to_contact_frame(
        (position, quat_wxyz), (sample.world_switch_pos_m, switch_quat_wxyz)
    )
    observation = PressObservation(
        relative_switch_pos_m=rel_pos,
        relative_switch_quat_xyzw=rel_quat,
        pose_confidence=sample.confidence,
        pose_age_s=sample.age_s,
        detection_valid=sample.valid,
        body_lin_vel_mps=world_enu_to_body_flu(quat_wxyz, lin_vel_w),
        body_ang_vel_radps=ang_vel_b,
        attitude_quat_xyzw=quat_wxyz_to_xyzw(quat_wxyz),
        pusher_compression_m=0.0,
        pusher_compression_rate_mps=0.0,
        contact_valid=False,
        contact_flag=False,
        lamp_state=-1,
        lamp_confidence=0.0,
        lamp_age_s=0.0,
        previous_action=np.zeros(action_dim),
        handoff_mode=0,
        estimator_health=1.0,
        obs_mask=0,
    )
    assert to_vector(observation).shape == (30 + action_dim,)
    return observation


def _validate_press_side_channel(*, start_obs, end_obs, handoff_config) -> None:
    """F2 exit-gate checks over the start/end side-channel observations."""
    from robotics_experiments.rl.press import HandoffGate, handoff_ready

    start_range = float(start_obs.relative_switch_pos_m[0])
    end_range = float(end_obs.relative_switch_pos_m[0])
    _report(f"press observation wall-normal range: start={start_range:.3f} m end={end_range:.3f} m")
    if not end_range < start_range - 1.5:
        raise AssertionError(
            f"relative switch range did not shrink across the goto: "
            f"{start_range:.3f} m -> {end_range:.3f} m"
        )
    gate = HandoffGate.from_config(handoff_config)
    start_result = handoff_ready(start_obs, gate)
    if start_result.ready:
        raise AssertionError("handoff gate was already ready 2 m from the switch")
    end_result = handoff_ready(end_obs, gate)
    _report(
        f"handoff gate: start_reason={start_result.reason} "
        f"end_ready={end_result.ready} end_reason={end_result.reason}"
    )
    if not end_result.ready:
        raise AssertionError(
            f"handoff gate not ready at the settled goto endpoint: {end_result.reason}"
        )


def _run(*, args: argparse.Namespace, config: ExperimentConfig) -> None:
    action_cfg = config.rl.action
    if args.ctbr_sanity:
        action_cfg = replace(action_cfg, kind="ctbr")

    # DroneHoverEnv is Isaac Lab task code, so it is imported after AppLauncher.
    from robotics_experiments.workflows.isaac_lab import (
        DroneHoverEnv,
    )

    if args.ctbr_sanity:
        _run_ctbr_sanity(
            args=args,
            config=config,
            action_cfg=action_cfg,
            env_cls=DroneHoverEnv,
        )
        return

    if action_cfg.kind != "velocity":
        raise AssertionError(
            f"hover+goto gate drives the analytic velocity controller; "
            f"rl.action.kind={action_cfg.kind!r} is not supported here"
        )
    max_velocity = float(action_cfg.max_body_velocity_mps)

    _seed_everything(args.seed)
    num_envs = args.num_envs
    if num_envs is None and args.diag:
        num_envs = 1

    env = DroneHoverEnv(
        cfg=_make_env(
            experiment_config=config,
            seed=args.seed,
            num_envs=num_envs,
        ),
        experiment_config=config,
    )
    try:
        _reset_env(env, args.seed)
        # The hover target is the (un-randomized) spawn pose per env.
        target = env.scene.env_origins + env.drone.data.default_root_state[:, :3]

        hover = _drive_to(
            env,
            target,
            _HOVER_SETTLE_S,
            max_velocity,
            diag=args.diag,
            log_every=args.log_every,
        )
        _report(
            "hover position error: "
            f"final={hover.final_error_m:.3f} m "
            f"last{_SETTLED_WINDOW_S:.1f}s_max={hover.last_window_max_error_m:.3f} m "
            f"peak={hover.peak_error_m:.3f} m"
        )
        if hover.last_window_max_error_m > _HOVER_TOLERANCE_M:
            raise AssertionError(
                "hover not held: "
                f"last-window max error {hover.last_window_max_error_m:.3f} m "
                f"> {_HOVER_TOLERANCE_M} m"
            )

        goto_target = target + torch.tensor([_GOTO_DISTANCE_M, 0.0, 0.0], device=env.device)
        # F2 side channel: a switch just beyond env 0's goto target; the
        # ground-truth provider + PressObservation assembly run alongside the
        # gate without touching the env's training observation.
        switch_pos_env0 = goto_target[0].cpu().numpy().astype(np.float64) + np.array(
            [_SWITCH_BEYOND_GOTO_M, 0.0, 0.0]
        )
        provider = _setup_press_side_channel(
            goto_target_env0=switch_pos_env0,
            switch_usd_path=config.switches.switch_usd_path,
        )
        action_dim = int(env.cfg.action_space)
        press_start_obs = _press_observation_sample(env, provider, action_dim)
        goto = _drive_to(
            env,
            goto_target,
            _GOTO_SETTLE_S,
            max_velocity,
            diag=args.diag,
            log_every=args.log_every,
        )
        _report(
            "goto position error: "
            f"final={goto.final_error_m:.3f} m "
            f"last{_SETTLED_WINDOW_S:.1f}s_max={goto.last_window_max_error_m:.3f} m "
            f"peak={goto.peak_error_m:.3f} m"
        )
        if goto.last_window_max_error_m > _GOTO_TOLERANCE_M:
            raise AssertionError(
                "goto not tracked: "
                f"last-window max error {goto.last_window_max_error_m:.3f} m "
                f"> {_GOTO_TOLERANCE_M} m"
            )

        press_end_obs = _press_observation_sample(env, provider, action_dim)
        _validate_press_side_channel(
            start_obs=press_start_obs,
            end_obs=press_end_obs,
            handoff_config=config.rl.press.handoff,
        )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # --scene / --strict are accepted for make-harness parity; the Lab env builds
    # its own scene and the gate always enforces its tolerances.
    parser.add_argument(
        "--scene",
        default=None,
        help="accepted for integration-runner compatibility; ignored by this Isaac Lab gate",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--diag", action="store_true", help="log per-step flight diagnostics")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ctbr-sanity", action="store_true", help=argparse.SUPPRESS)
    args, _unknown = parser.parse_known_args()
    if args.log_every < 1:
        raise SystemExit("--log-every must be >= 1")
    if args.num_envs is not None and args.num_envs < 1:
        raise SystemExit("--num-envs must be >= 1")
    if args.scene:
        _report(
            f"ignoring --scene {args.scene!r}; "
            "the Isaac Lab hover+goto gate builds DroneHoverEnv directly"
        )

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        try:
            config_path = os.environ[_CONFIG_PATH_ENV]
        except KeyError as exc:
            raise RuntimeError(f"missing required environment variable {_CONFIG_PATH_ENV}") from exc
        _run(args=args, config=load_config(config_path))
        if args.ctbr_sanity:
            _report("ctbr sanity validation PASSED")
        else:
            _report("press hover+goto validation PASSED")
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
