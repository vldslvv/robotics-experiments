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
import math
import os
import random
import sys
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import torch

from robotics_experiments.config import ExperimentConfig, PressActionConfig, load_config

if TYPE_CHECKING:
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
_GOTO_SETTLE_S = 8.0
_GOTO_TOLERANCE_M = 0.2
_SETTLED_WINDOW_S = 1.0
_EPISODE_MARGIN_S = 1.0
_DEFAULT_SEED = 7
_CONFIG_PATH_ENV = "ROBOTICS_EXPERIMENT_CONFIG_PATH"

# F2 side-channel validation: one switch prim placed this far beyond the goto
# target along +X, outward plate normal facing the approaching drone. This keeps
# the fully deployed pusher clear during the non-contact goto while leaving the
# settled drone inside the configured handoff range.
_SWITCH_BEYOND_GOTO_M = 0.48
_SWITCH_PRIM_PATH = "/World/PressValidationSwitch"
_CONTACT_LAMP_PRIM_PATH = "/World/envs/env_0/Lamp"


@dataclass(frozen=True)
class _DriveStats:
    final_error_m: float
    last_window_max_error_m: float
    peak_error_m: float


class _SwitchJointAdapter:
    """Expose the Lab switch's real joint through production step_switches."""

    def __init__(self, articulation, joint_id: int):
        self._articulation = articulation
        self._joint_id = joint_id

    def get_joint_positions(self):
        return self._articulation.data.joint_pos[0, self._joint_id].reshape(1).cpu().numpy()


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


def _spawn_contact_chain_lamp(*, lamp_usd_path: str):
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.DefinePrim(_CONTACT_LAMP_PRIM_PATH, "Xform")
    prim.GetReferences().AddReference(lamp_usd_path)
    UsdGeom.Xformable(prim).AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(0.4, 0.0, 2.0)
    )
    if not prim.IsValid():
        raise AssertionError("paired lamp prim did not resolve")
    return stage


def _collision_paths(stage, root_path: str) -> list[str]:
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(root_path)
    return [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(root.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def _collision_contract_summary(stage, drone_root_path: str, wall_root_path: str) -> str:
    from pxr import UsdPhysics

    drone_root = stage.GetPrimAtPath(drone_root_path)
    drone_colliders = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(drone_root.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    arm_booms = [
        prim
        for prim in drone_colliders
        if "/body/arms/" in str(prim.GetPath()) and prim.GetName() == "boom"
    ]
    propeller_discs = [prim for prim in drone_colliders if prim.GetName() == "prop_collision"]
    if not arm_booms or len(propeller_discs) != len(arm_booms):
        raise AssertionError(
            "drone requires one arm and swept-propeller collision per rotor: "
            f"arms={len(arm_booms)} props={len(propeller_discs)}"
        )
    disabled = [
        str(prim.GetPath())
        for prim in (*arm_booms, *propeller_discs)
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
    ]
    if disabled:
        raise AssertionError(f"arm/propeller collision geometry is disabled: {disabled}")

    wall_root = stage.GetPrimAtPath(wall_root_path)
    wall_colliders = [
        prim
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(wall_root.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False
    ]
    if not wall_colliders:
        raise AssertionError("press task wall has no enabled collision geometry")
    return (
        f"arm_colliders={len(arm_booms)} "
        f"propeller_colliders={len(propeller_discs)} "
        f"wall_colliders={[str(prim.GetPath()) for prim in wall_colliders]}"
    )


def _pusher_tool_geometry_summary(stage, root_path: str) -> str:
    from pxr import UsdGeom, UsdPhysics

    paths = {
        "mount": f"{root_path}/body/pusher/mount",
        "boom": f"{root_path}/body/pusher/fixed_boom",
        "plunger": f"{root_path}/pusher/pad/plunger",
        "pad": f"{root_path}/pusher/pad/geom",
    }
    mount = stage.GetPrimAtPath(paths["mount"])
    boom = UsdGeom.Cylinder(stage.GetPrimAtPath(paths["boom"]))
    plunger = UsdGeom.Cylinder(stage.GetPrimAtPath(paths["plunger"]))
    pad = UsdGeom.Cylinder(stage.GetPrimAtPath(paths["pad"]))
    if not mount.IsValid() or not mount.HasAPI(UsdPhysics.CollisionAPI):
        raise AssertionError("pusher mount is missing or non-collidable")
    if not boom or not boom.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise AssertionError("fixed pusher boom is missing or non-collidable")
    if not plunger or plunger.GetAxisAttr().Get() != "X":
        raise AssertionError("pusher plunger is missing or not aligned with body +X")
    if not pad or pad.GetAxisAttr().Get() != "X":
        raise AssertionError("pusher pad is missing or not forward-facing along body +X")
    if not pad.GetPrim().HasAPI(UsdPhysics.CollisionAPI):
        raise AssertionError("forward-facing pusher pad has no collision geometry")
    visible_gprims = (
        UsdGeom.Cube(stage.GetPrimAtPath(f"{paths['mount']}/geom")),
        boom,
        plunger,
        pad,
    )
    if any(not gprim or not gprim.GetDisplayColorAttr().Get() for gprim in visible_gprims):
        raise AssertionError("pusher mount/boom/plunger/pad must all have visible display colors")

    xform_cache = UsdGeom.XformCache()
    boom_center = xform_cache.GetLocalToWorldTransform(boom.GetPrim()).ExtractTranslation()
    plunger_center = xform_cache.GetLocalToWorldTransform(plunger.GetPrim()).ExtractTranslation()
    pad_center = xform_cache.GetLocalToWorldTransform(pad.GetPrim()).ExtractTranslation()
    if not boom_center[0] < plunger_center[0] < pad_center[0]:
        raise AssertionError(
            "pusher geometry is not ordered mount -> boom -> plunger -> pad along body +X"
        )
    return (
        f"paths={paths} "
        f"boom_center={tuple(round(float(value), 4) for value in boom_center)} "
        f"plunger_center={tuple(round(float(value), 4) for value in plunger_center)} "
        f"pad_center={tuple(round(float(value), 4) for value in pad_center)}"
    )


def _run_contact_chain(*, args: argparse.Namespace, config: ExperimentConfig) -> None:
    from pxr import UsdLux

    from robotics_experiments.environment import SwitchState, step_switches
    from robotics_experiments.workflows.isaac_lab import SwitchPressEnv, make_switch_press_env_cfg

    class _ContactChainEnv(SwitchPressEnv):
        """Test-local physical path independent of Stage 1 policy behavior."""

        def _pre_physics_step(self, actions, /):
            super()._pre_physics_step(actions)
            self._applied_actions = self.actions

        def _get_dones(self):
            no_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            return no_done, no_done

        def _get_rewards(self):
            return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    if config.rl.action.kind != "velocity":
        raise AssertionError("contact-chain gate requires rl.action.kind=velocity")
    if args.num_envs not in (None, 1):
        raise AssertionError("--contact-chain requires exactly one environment")
    _seed_everything(args.seed)
    cfg = make_switch_press_env_cfg(experiment_config=config)
    cfg.scene.num_envs = 1
    cfg.episode_length_s = config.rl.episode_length_s + _EPISODE_MARGIN_S
    if hasattr(cfg, "seed"):
        cfg.seed = args.seed
    env = _ContactChainEnv(cfg=cfg, experiment_config=config)
    try:
        _reset_env(env, args.seed)
        stage = _spawn_contact_chain_lamp(lamp_usd_path=config.switches.lamp_usd_path)
        switch_joint_ids, _ = env.switch.find_joints(".*")
        joint_id = int(switch_joint_ids[0])
        pusher_joint_ids, _ = env.drone.find_joints("pusher_joint")
        pusher_joint_id = int(pusher_joint_ids[0])
        initial_pusher_position = float(env.drone.data.joint_pos[0, pusher_joint_id].item())
        pusher_upper_limit = float(env.drone.data.joint_pos_limits[0, pusher_joint_id, 1].item())
        if not np.isclose(initial_pusher_position, pusher_upper_limit, atol=1e-5):
            raise AssertionError(
                "pusher did not start fully deployed: "
                f"position={initial_pusher_position:.6f} upper={pusher_upper_limit:.6f}"
            )
        adapter = _SwitchJointAdapter(env.switch, joint_id)
        switch_state = (SwitchState("/World/envs/env_0/Switch", _CONTACT_LAMP_PRIM_PATH),)
        toggles: list[bool] = []
        max_pusher = 0.0
        max_button = 0.0
        contact_seen = False
        contact_steps = 0
        max_contact_force = 0.0
        retreat_started = False
        stable_retreat = False
        stable_retreat_steps = 0
        max_tilt = 0.0
        steps = round(config.rl.episode_length_s / env.step_dt)
        max_velocity = float(config.rl.action.max_body_velocity_mps)
        # This deterministic physical-actuation gate is independent of the
        # approach reward's safe closing-speed schedule.
        approach_speed = 2.0 * config.rl.press.goal.linear_speed_max_mps
        retreat_speed = approach_speed * 0.75
        initial_range = float((env.switch.data.root_pos_w - env.drone.data.root_pos_w)[0, 0].item())
        _report(
            "contact geometry: "
            f"switch_bodies={env.switch.body_names} drone_bodies={env.drone.body_names} "
            f"switch_joint={env.switch.joint_names[joint_id]} "
            f"switch_limits={env.switch.data.joint_pos_limits[0, joint_id].cpu().tolist()} "
            f"pusher_deployed={initial_pusher_position:.5f} m"
        )
        _report(
            "pusher tool geometry: "
            f"{_pusher_tool_geometry_summary(stage, '/World/envs/env_0/Drone')}"
        )
        collision_summary = _collision_contract_summary(
            stage,
            "/World/envs/env_0/Drone",
            "/World/envs/env_0/Wall",
        )
        _report(f"collision contract: {collision_summary}")
        _report(f"drone collision prims={_collision_paths(stage, '/World/envs/env_0/Drone')}")
        for _step in range(steps):
            switch_offset = env.switch.data.root_pos_w - env.drone.data.root_pos_w
            action = torch.zeros((1, env.cfg.action_space), device=env.device)
            if retreat_started:
                speed = -retreat_speed
            elif contact_seen:
                contact_steps += 1
                speed = min(
                    max_velocity,
                    approach_speed + contact_steps * approach_speed * 0.5,
                )
            else:
                speed = approach_speed
            action[:, 0] = speed / max_velocity
            action[:, 1] = torch.clamp(2.0 * switch_offset[:, 1] / max_velocity, -0.2, 0.2)
            action[:, 2] = torch.clamp(2.0 * switch_offset[:, 2] / max_velocity, -0.2, 0.2)
            _observation, _reward, _terminated, _truncated, _info = env.step(action)
            pusher = float(
                env.drone.data.joint_pos_limits[0, pusher_joint_id, 1].item()
                - env.drone.data.joint_pos[0, pusher_joint_id].item()
            )
            button = float(env.switch.data.joint_pos[0, joint_id].item())
            max_pusher = max(max_pusher, pusher)
            max_button = max(max_button, button)
            contact_force = float(
                torch.linalg.norm(env.contact_sensor.data.net_forces_w, dim=-1).max().item()
            )
            max_contact_force = max(max_contact_force, contact_force)
            contact_now = contact_force >= config.rl.press.termination.contact_force_threshold_n
            contact_seen = contact_seen or contact_now
            switch_state = step_switches(
                stage=stage,
                switch_state=switch_state,
                articulations=(adapter,),
                config=config.switches,
                on_toggle=lambda _index, state: toggles.append(state),
            )
            if toggles:
                retreat_started = True
            _yaw, tilt = _yaw_tilt_from_quat(env.drone.data.root_quat_w)
            max_tilt = max(max_tilt, float(tilt[0].item()))
            relative = env.switch.data.root_pos_w - env.drone.data.root_pos_w
            linear_speed = float(torch.linalg.norm(env.drone.data.root_lin_vel_b[0]).item())
            angular_speed = float(torch.linalg.norm(env.drone.data.root_ang_vel_b[0]).item())
            retreat_stable_now = (
                retreat_started
                and not contact_now
                and float(relative[0, 0].item()) >= initial_range
                and float(torch.linalg.norm(relative[0, 1:3]).item())
                <= config.rl.press.goal.alignment_error_max_m
                and linear_speed <= config.rl.press.goal.closing_speed_far_mps
                and angular_speed <= config.rl.press.goal.angular_speed_max_radps
            )
            stable_retreat_steps = stable_retreat_steps + 1 if retreat_stable_now else 0
            if stable_retreat_steps >= config.rl.press.goal.stable_steps:
                stable_retreat = True
                break
            if float(tilt[0].item()) > math.acos(config.rl.press.termination.crash_tilt_cos):
                raise AssertionError("contact-chain drone breached its tilt envelope")

        light_prim = stage.GetPrimAtPath(f"{_CONTACT_LAMP_PRIM_PATH}/Lamp/SphereLight")
        intensity = float(UsdLux.SphereLight(light_prim).GetIntensityAttr().Get())
        _report(
            "contact chain: "
            f"contact={contact_seen} max_pusher={max_pusher:.5f} m "
            f"max_button={max_button:.5f} m toggles={len(toggles)} "
            f"max_contact_force={max_contact_force:.3f} N "
            f"lamp_intensity={intensity:.1f} stable_retreat={stable_retreat} "
            f"max_tilt={max_tilt:.3f} rad"
        )
        _report(
            "contact final: "
            f"drone_pos={env.drone.data.root_pos_w[0].cpu().tolist()} "
            f"drone_body_pos={env.drone.data.body_pos_w[0].cpu().tolist()} "
            f"switch_pos={env.switch.data.root_pos_w[0].cpu().tolist()} "
            f"switch_body_pos={env.switch.data.body_pos_w[0].cpu().tolist()}"
        )
        if not contact_seen or max_pusher <= 0.0:
            raise AssertionError("pusher contact/compression was not observed")
        if max_button < config.switches.press_threshold:
            raise AssertionError("real button did not cross the configured press threshold")
        if toggles != [True]:
            raise AssertionError(
                f"production latch toggled {len(toggles)} times, expected exactly one"
            )
        if intensity != config.switches.lamp_on_intensity:
            raise AssertionError("paired lamp did not make its configured off-to-on transition")
        if not stable_retreat:
            raise AssertionError("drone did not complete a stable post-toggle retreat")
    finally:
        env.close()


def _setup_press_side_channel(*, goto_target_env0: np.ndarray, switch_usd_path: str):
    """Place the F2 validation switch prim and return its ground-truth provider.

    The prim's world pose is what the provider reads. The source asset's local
    -Y outward normal faces the approaching drone after a -pi/2 rotation about
    Z; the provider then converts it to canonical local +X.
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
    sqrt_half = math.sqrt(0.5)
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(sqrt_half, Gf.Vec3d(0.0, 0.0, -sqrt_half))
    )
    return GroundTruthPoseProvider(stage, _SWITCH_PRIM_PATH)


def _press_observation_sample(env: DroneHoverEnv, provider, action_dim: int):
    """Assemble a PressObservation for env 0 from the provider + sim state.

    This is the F2 deploy-shape assembly running as a side channel: it never
    touches the env's training observation.
    """
    from robotics_experiments.px4_bridge import quat_wxyz_to_xyzw, world_enu_to_body_flu
    from robotics_experiments.rl.press import (
        PressObservation,
        PressObservationMask,
        to_vector,
        world_switch_to_contact_frame,
    )

    position = env.drone.data.root_pos_w[0].cpu().numpy().astype(np.float64)
    quat_wxyz = env.drone.data.root_quat_w[0].cpu().numpy().astype(np.float64)
    lin_vel_w = env.drone.data.root_lin_vel_w[0].cpu().numpy().astype(np.float64)
    ang_vel_b = env.drone.data.root_ang_vel_b[0].cpu().numpy().astype(np.float64)
    sample = provider.sample(0.0)
    _report(
        "press pose sample: "
        f"drone={position.tolist()} switch={sample.world_switch_pos_m.tolist()}"
    )
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
        contact_valid=True,
        contact_flag=False,
        lamp_state=0,
        lamp_confidence=1.0,
        lamp_age_s=0.0,
        previous_action=np.zeros(action_dim),
        handoff_mode=1,
        estimator_health=1.0,
        obs_mask=int(
            PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
        ),
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
    if args.contact_chain:
        _run_contact_chain(args=args, config=config)
        return
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
    parser.add_argument(
        "--contact-chain",
        action="store_true",
        help="run the one-env real pusher/button/latch/lamp/retreat gate",
    )
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
        if args.contact_chain:
            _report("press contact-chain validation PASSED")
        elif args.ctbr_sanity:
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
