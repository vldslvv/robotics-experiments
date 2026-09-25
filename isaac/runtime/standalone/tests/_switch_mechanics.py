"""Controlled physical pushers for switch force, stroke and collision verification."""

import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from robotics_experiments.config import ExperimentConfig, load_switch_model
from robotics_experiments.environment import SwitchState, apply_button_sample
from robotics_experiments.rl.press.contact import classify_press_contacts, contact_force_totals
from robotics_experiments.switches import load_switch_asset

__all__: list[str] = []


def _run_switch_mechanics(config: ExperimentConfig, /) -> None:
    # AppLauncher is running before these simulator imports.
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from pxr import Gf, UsdGeom, UsdPhysics

    verification = config.switches.verification
    dt = 1 / config.sim.physics_hz
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt, device="cuda:0"))
    rigs: list[dict[str, Any]] = []
    try:
        stage = sim_utils.get_current_stage()
        cases = ("activation", "near_stop", "full_travel", "housing", "wall", "simultaneous")
        root = Path(__file__).resolve().parents[4]
        for profile in ("easy", "nominal"):
            model = load_switch_model(root / f"configs/experiment/switches/model/{profile}.yaml")
            asset = load_switch_asset(model=model)
            for case in cases:
                index = len(rigs)
                offset = index * config.rl.env_spacing
                namespace = f"/World/rig_{index}"
                switch = Articulation(
                    ArticulationCfg(
                        prim_path=f"{namespace}/Switch",
                        spawn=sim_utils.UsdFileCfg(usd_path=str(asset.path)),
                        init_state=ArticulationCfg.InitialStateCfg(pos=(offset, 0, 1)),
                        actuators={
                            "button": ImplicitActuatorCfg(
                                joint_names_expr=[".*"], stiffness=None, damping=None
                            )
                        },
                    )
                )
                pad_width = (
                    (model.housing_width_m - model.button_width_m - 2 * model.clearance_m) / 4
                    if case == "housing"
                    else model.button_width_m / 3
                )
                lateral = {
                    "housing": (
                        model.housing_width_m + model.button_width_m + 2 * model.clearance_m
                    )
                    / 4,
                    "wall": model.housing_width_m / 2 + pad_width,
                    "simultaneous": model.button_width_m / 2,
                }.get(case, 0)
                pad_depth = model.button_thickness_m
                center = (
                    offset + lateral,
                    model.button_rest_y_m
                    - model.button_thickness_m / 2
                    - pad_depth / 2
                    - model.clearance_m,
                    1,
                )
                wrapper = UsdGeom.Xform.Define(stage, f"{namespace}/Pusher")
                UsdPhysics.ArticulationRootAPI.Apply(wrapper.GetPrim())
                for name in ("Base", "Pad"):
                    body = UsdGeom.Xform.Define(stage, f"{wrapper.GetPath()}/{name}")
                    body.AddTranslateOp().Set(Gf.Vec3d(*center))
                    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
                    UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(
                        verification.pusher_mass_kg
                    )
                pad_path = f"{wrapper.GetPath()}/Pad"
                pad_cfg = sim_utils.CuboidCfg(
                    size=(pad_width, pad_depth, model.button_height_m / 3),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=model.contact_offset_m, rest_offset=0
                    ),
                )
                pad_cfg.func(f"{pad_path}/Shape", pad_cfg)
                sim_utils.activate_contact_sensors(pad_path)
                fixed = UsdPhysics.FixedJoint.Define(stage, f"{wrapper.GetPath()}/Fixed")
                fixed.CreateBody1Rel().SetTargets([f"{wrapper.GetPath()}/Base"])
                fixed.CreateLocalPos0Attr(Gf.Vec3f(*center))
                fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
                joint = UsdPhysics.PrismaticJoint.Define(stage, f"{wrapper.GetPath()}/Push")
                joint.CreateBody0Rel().SetTargets([f"{wrapper.GetPath()}/Base"])
                joint.CreateBody1Rel().SetTargets([pad_path])
                joint.CreateAxisAttr("Y")
                joint.CreateLocalPos0Attr(Gf.Vec3f(0))
                joint.CreateLocalPos1Attr(Gf.Vec3f(0))
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
                drive.CreateStiffnessAttr(verification.pusher_stiffness_n_m)
                drive.CreateDampingAttr(verification.pusher_damping_n_s_m)
                pusher = Articulation(
                    ArticulationCfg(
                        prim_path=str(wrapper.GetPath()),
                        spawn=None,
                        actuators={
                            "push": ImplicitActuatorCfg(
                                joint_names_expr=[".*"], stiffness=None, damping=None
                            )
                        },
                    )
                )
                wall_cfg = sim_utils.CuboidCfg(
                    size=(
                        config.rl.press.reset.wall_half_extents_m[1] * 2,
                        config.rl.press.reset.wall_half_extents_m[0] * 2,
                        config.rl.press.reset.wall_half_extents_m[2] * 2,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        contact_offset=model.contact_offset_m, rest_offset=0
                    ),
                )
                wall_cfg.func(
                    f"{namespace}/Wall",
                    wall_cfg,
                    translation=(
                        offset,
                        config.rl.press.reset.switch_wall_offset_m
                        + config.rl.press.reset.wall_half_extents_m[0],
                        1,
                    ),
                )
                sensor = ContactSensor(
                    ContactSensorCfg(
                        prim_path=pad_path,
                        update_period=0,
                        max_contact_data_count_per_prim=config.rl.press.termination.contact_data_capacity_per_body,
                        filter_prim_paths_expr=[
                            f"{namespace}/Switch/LightSwitch/Button",
                            f"{namespace}/Switch/LightSwitch/BaseMesh",
                            f"{namespace}/Wall",
                            f"{namespace}/Pusher/Base",
                        ],
                    )
                )
                goal = {
                    "activation": model.activation_displacement_m,
                    "near_stop": model.travel_m - verification.near_stop_margin_m,
                }.get(case, model.travel_m + verification.overtravel_m)
                if case == "wall":
                    goal = (
                        config.rl.press.reset.switch_wall_offset_m
                        - model.button_rest_y_m
                        + model.button_thickness_m / 2
                        + verification.overtravel_m
                    )
                # Compensate pusher drive deflection without altering the switch drive.
                target = (
                    goal
                    + model.clearance_m
                    + goal * model.stiffness_n_m / verification.pusher_stiffness_n_m
                )
                rigs.append(
                    dict(
                        profile=profile,
                        case=case,
                        model=model,
                        asset=asset,
                        switch=switch,
                        pusher=pusher,
                        sensor=sensor,
                        target=target,
                        samples=[],
                        state=SwitchState(prim_path=namespace),
                        toggles=0,
                    )
                )
        sim.reset()
        duration = (
            max(rig["target"] for rig in rigs) / verification.pusher_speed_mps + verification.hold_s
        )
        for step in range(math.ceil(duration / dt)):
            time_s = (step + 1) * dt
            for rig in rigs:
                target = min(time_s * verification.pusher_speed_mps, rig["target"])
                rig["pusher"].set_joint_position_target(
                    torch.full((1, 1), target, device=sim.device)
                )
                rig["pusher"].write_data_to_sim()
            sim.step(render=False)
            for rig in rigs:
                rig["switch"].update(dt)
                rig["pusher"].update(dt)
                forces, _points, _normals, _distances, counts, starts = rig[
                    "sensor"
                ].contact_physx_view.get_contact_data(dt=dt)
                totals = contact_force_totals(forces=forces, counts=counts, starts=starts).reshape(
                    1, 4
                )
                categories = classify_press_contacts(
                    pad_forces_n=totals,
                    body_force_n=totals.new_zeros(1),
                    threshold_n=config.rl.press.termination.contact_force_threshold_n,
                )
                displacement = float(rig["switch"].data.joint_pos[0, 0])
                velocity = float(rig["switch"].data.joint_vel[0, 0])
                rig["state"], event = apply_button_sample(
                    switch=rig["state"], button_position=displacement, dt_s=dt, model=rig["model"]
                )
                rig["toggles"] += event
                rig["samples"].append(
                    [time_s, displacement, velocity, *totals[0].tolist(), *categories[0].tolist()]
                )
        records: list[dict[str, Any]] = []
        for rig in rigs:
            samples = torch.tensor(rig["samples"], dtype=torch.float64)
            held = samples[-round(verification.hold_s / dt / 2) :]
            displacement = float(held[:, 1].mean())
            force = float(held[:, 3].mean())
            model = rig["model"]
            categories = samples[:, 7:].bool().any(dim=0).tolist()
            record = dict(
                profile=rig["profile"],
                case=rig["case"],
                asset=rig["asset"].record(),
                displacement_m=displacement,
                force_n=force,
                spring_force_n=model.stiffness_n_m * displacement,
                damping_force_n=model.damping_n_s_m * float(held[:, 2].mean()),
                stop_reaction_n=(
                    force
                    - model.stiffness_n_m * displacement
                    - model.damping_n_s_m * float(held[:, 2].mean())
                ),
                categories=categories,
                toggles=rig["toggles"],
                samples=rig["samples"],
            )
            records.append(record)
        output = root / ".build/assets/switch-mechanics"
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{time.time_ns()}.json"
        path.write_text(json.dumps(records) + "\n")
        print(f"Switch mechanics evidence: {path}", flush=True)
        for rig, record in zip(rigs, records, strict=True):
            model, case = rig["model"], rig["case"]
            print(
                json.dumps({key: value for key, value in record.items() if key != "samples"}),
                flush=True,
            )
            flags = record["categories"]
            if case in {"activation", "near_stop"}:
                expected = (
                    model.activation_displacement_m
                    if case == "activation"
                    else model.travel_m - verification.near_stop_margin_m
                )
                assert (
                    abs(record["displacement_m"] - expected) < verification.displacement_tolerance_m
                )
                assert (
                    abs(record["force_n"] - model.stiffness_n_m * record["displacement_m"])
                    < verification.force_tolerance_n
                )
                assert abs(record["damping_force_n"]) < verification.force_tolerance_n
                assert not any(flags[1:])
            elif case == "full_travel":
                assert (
                    abs(record["displacement_m"] - model.travel_m)
                    < verification.displacement_tolerance_m
                )
                assert record["toggles"] == 1 and not any(flags[1:])
            elif case == "housing":
                assert flags[1]
            elif case == "wall":
                assert flags[2]
            else:
                assert flags[0] and flags[1]

    finally:
        rigs.clear()
        sim.clear_all_callbacks()
        sim.clear_instance()
