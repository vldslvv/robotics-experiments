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
import sys
from dataclasses import dataclass

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


@dataclass(frozen=True)
class _DriveStats:
    final_error_m: float
    last_window_max_error_m: float
    peak_error_m: float


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


def _seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _position_action(env, target_w, max_velocity: float):
    import torch

    position = env.drone.data.root_pos_w
    velocity_command = torch.clamp(
        _POSITION_KP * (target_w - position), -max_velocity, max_velocity
    )
    action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    # Spawn yaw is 0, so the world velocity command is the body-heading command;
    # the env's velocity actuation tracks it. action[3] (yaw rate) stays 0.
    action[:, :3] = velocity_command / max_velocity
    return action


def _yaw_tilt_from_quat(quat_wxyz):
    import torch

    quat = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(1e-9)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    body_z_world_z = (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
    tilt = torch.acos(body_z_world_z)
    return yaw, tilt


def _diagnostic_sample(env, target_w, action, max_velocity: float) -> str:
    import torch

    position_error = env.drone.data.root_pos_w - target_w
    error_norm = torch.linalg.norm(position_error, dim=-1)
    env_index = int(torch.argmax(error_norm).item())
    yaw, tilt = _yaw_tilt_from_quat(env.drone.data.root_quat_w)

    actuation = vars(env)["_actuation"]
    omega = actuation.action_to_rotor_omega(
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
    env,
    target_w,
    settle_s: float,
    max_velocity: float,
    *,
    diag: bool,
    log_every: int,
) -> _DriveStats:
    import torch

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


def _max_position_error(env, target_w) -> float:
    import torch

    return float(torch.linalg.norm(env.drone.data.root_pos_w - target_w, dim=-1).max().item())


def _reset_env(env, seed: int) -> None:
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()


def _make_env(env_cfg_cls, *, seed: int, num_envs: int | None):
    cfg = env_cfg_cls()
    cfg.episode_length_s = _HOVER_SETTLE_S + _GOTO_SETTLE_S + _EPISODE_MARGIN_S
    if hasattr(cfg, "seed"):
        cfg.seed = seed
    if num_envs is not None:
        cfg.scene.num_envs = num_envs
    return cfg


def _ctbr_hover_action(env, action_cfg):
    import torch

    actuation = vars(env)["_actuation"]
    mass = float(env.drone.root_physx_view.get_masses()[0].sum().item())
    max_total_thrust = float(vars(actuation)["_max_total_thrust"])
    hover_fraction = mass * 9.81 / max_total_thrust
    action = torch.zeros((env.num_envs, env.cfg.action_space), device=env.device)
    action[:, 0] = hover_fraction / float(action_cfg.hover_thrust_fraction) - 1.0
    return action, hover_fraction


def _run_ctbr_sanity(args: argparse.Namespace, action_cfg, env_cls, env_cfg_cls) -> None:
    import torch

    _seed_everything(args.seed)
    num_envs = args.num_envs if args.num_envs is not None else 1
    env = env_cls(_make_env(env_cfg_cls, seed=args.seed, num_envs=num_envs))
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
        omega = vars(env)["_actuation"].action_to_rotor_omega(
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


def _run(args: argparse.Namespace) -> None:
    import torch

    from robotics_experiments.config import get_config

    action_cfg = get_config().press.action
    if args.ctbr_sanity:
        action_cfg.kind = "ctbr"

    from robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env import (
        DroneHoverEnv,
        DroneHoverEnvCfg,
    )

    if args.ctbr_sanity:
        _run_ctbr_sanity(args, action_cfg, DroneHoverEnv, DroneHoverEnvCfg)
        return

    if action_cfg.kind != "velocity":
        raise AssertionError(
            f"hover+goto gate drives the analytic velocity controller; "
            f"press.action.kind={action_cfg.kind!r} is not supported here"
        )
    max_velocity = float(action_cfg.max_body_velocity_mps)

    _seed_everything(args.seed)
    num_envs = args.num_envs
    if num_envs is None and args.diag:
        num_envs = 1

    env = DroneHoverEnv(_make_env(DroneHoverEnvCfg, seed=args.seed, num_envs=num_envs))
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
        _run(args)
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
