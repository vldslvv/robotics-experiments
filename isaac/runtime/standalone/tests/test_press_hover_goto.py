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

# Analytic outer-loop gain + tolerances for the gate (validation-only, not a
# config surface): a position P controller produces the velocity command the
# env's velocity actuation tracks.
_POSITION_KP = 1.5
_HOVER_SETTLE_S = 4.0
_HOVER_TOLERANCE_M = 0.1
_GOTO_DISTANCE_M = 2.0
_GOTO_SETTLE_S = 4.0
_GOTO_TOLERANCE_M = 0.2


def _report(message: str) -> None:
    print(f"[isaac-standalone-test] {message}", file=sys.stderr, flush=True)


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


def _drive_to(env, target_w, settle_s: float, max_velocity: float) -> None:
    import torch

    steps = max(1, round(settle_s / env.step_dt))
    for _ in range(steps):
        _, _, terminated, _, _ = env.step(_position_action(env, target_w, max_velocity))
        if bool(torch.any(terminated)):
            raise AssertionError("a drone terminated (crash / out-of-bounds) before settling")


def _max_position_error(env, target_w) -> float:
    import torch

    return float(torch.linalg.norm(env.drone.data.root_pos_w - target_w, dim=-1).max().item())


def _run() -> None:
    import torch

    from robotics_experiments.config import get_config
    from robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env import (
        DroneHoverEnv,
        DroneHoverEnvCfg,
    )

    action_cfg = get_config().press.action
    if action_cfg.kind != "velocity":
        raise AssertionError(
            f"hover+goto gate drives the analytic velocity controller; "
            f"press.action.kind={action_cfg.kind!r} is not supported here"
        )
    max_velocity = float(action_cfg.max_body_velocity_mps)

    env = DroneHoverEnv(DroneHoverEnvCfg())
    try:
        env.reset()
        # The hover target is the (un-randomized) spawn pose per env.
        target = env.scene.env_origins + env.drone.data.default_root_state[:, :3]

        _drive_to(env, target, _HOVER_SETTLE_S, max_velocity)
        hover_error = _max_position_error(env, target)
        _report(f"hover max position error: {hover_error:.3f} m")
        if hover_error > _HOVER_TOLERANCE_M:
            raise AssertionError(
                f"hover not held: max error {hover_error:.3f} m > {_HOVER_TOLERANCE_M} m"
            )

        goto_target = target + torch.tensor([_GOTO_DISTANCE_M, 0.0, 0.0], device=env.device)
        _drive_to(env, goto_target, _GOTO_SETTLE_S, max_velocity)
        goto_error = _max_position_error(env, goto_target)
        _report(f"goto max position error: {goto_error:.3f} m")
        if goto_error > _GOTO_TOLERANCE_M:
            raise AssertionError(
                f"goto not tracked: max error {goto_error:.3f} m > {_GOTO_TOLERANCE_M} m"
            )
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # --scene / --strict are accepted for make-harness parity; the Lab env builds
    # its own scene and the gate always enforces its tolerances.
    parser.add_argument("--scene", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.parse_known_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app
    try:
        _run()
        _report("press hover+goto validation PASSED")
    except Exception as exc:  # noqa: BLE001 - report any failure as the make sentinel
        _report(f"ERROR: {exc}")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
