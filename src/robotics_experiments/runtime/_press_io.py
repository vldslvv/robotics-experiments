"""Isaac Sim 5.1 resource adapter for live press-policy observations.

Every operation reads or configures a live USD/PhysX resource. This private
runtime leaf stays off the host import path until a press artifact is selected.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.physics import ContactSensor
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics

from robotics_experiments.px4_bridge import (
    body_velocity_flu_from_state,
    quat_wxyz_to_xyzw,
)
from robotics_experiments.rl.press import (
    GroundTruthPoseProvider,
    PressObservation,
    PressObservationMask,
    world_switch_to_contact_frame,
)

__all__ = ["PressObservationIO"]


class PressObservationIO:
    """Mutable handles for one drone/pusher, paired switch, lamp, and pad sensor."""

    def __init__(
        self,
        *,
        stage: Any,
        articulation_path: str,
        switch_index: int,
        action_dim: int,
        contact_force_threshold_n: float,
        button_geometry_scale: float,
        button_stiffness_scale: float,
        switch_wall_offset_m: float,
        wall_half_extents_m: tuple[float, float, float],
        pusher_effort_limit_n: float,
        pusher_velocity_limit_mps: float,
    ) -> None:
        self._stage = stage
        self._drone_path = articulation_path
        self._switch_path = f"/Root/LightSwitch_{switch_index}"
        self._lamp_path = f"/Root/Lamp_{switch_index}"
        self._action_dim = int(action_dim)
        self._contact_force_threshold_n = float(contact_force_threshold_n)
        self._drone: SingleArticulation | None = None
        self._pusher_joint_index = -1
        self._pusher_extension_m = 0.0
        self._pose_provider = GroundTruthPoseProvider(stage, self._switch_path)

        self._configure_switch(
            geometry_scale=float(button_geometry_scale),
            stiffness_scale=float(button_stiffness_scale),
            wall_offset_m=float(switch_wall_offset_m),
            wall_half_extents_m=(
                float(wall_half_extents_m[0]),
                float(wall_half_extents_m[1]),
                float(wall_half_extents_m[2]),
            ),
        )
        self._configure_drone(
            effort_limit_n=float(pusher_effort_limit_n),
            velocity_limit_mps=float(pusher_velocity_limit_mps),
        )
        sensor_path = f"{articulation_path}/pusher/pad/geom/press_contact_sensor"
        self._contact_sensor = ContactSensor(
            prim_path=sensor_path,
            min_threshold=self._contact_force_threshold_n,
            max_threshold=100_000.0,
            radius=-1.0,
        )

    def _configure_switch(
        self,
        *,
        geometry_scale: float,
        stiffness_scale: float,
        wall_offset_m: float,
        wall_half_extents_m: tuple[float, float, float],
    ) -> None:
        root = self._stage.GetPrimAtPath(self._switch_path)
        if not root.IsValid():
            raise ValueError(f"switch prim not found at {self._switch_path!r}")
        buttons = [
            prim
            for prim in self._stage.Traverse()
            if prim.GetPath().HasPrefix(root.GetPath()) and prim.GetName() == "ButtonMesh"
        ]
        if len(buttons) != 1:
            raise RuntimeError(f"switch requires exactly one ButtonMesh, found {len(buttons)}")
        button = buttons[0]
        if button.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(button)
            points = mesh.GetPointsAttr().Get()
            mesh.GetPointsAttr().Set(
                [
                    Gf.Vec3f(
                        point[0] * geometry_scale,
                        point[1] * geometry_scale,
                        point[2] * geometry_scale,
                    )
                    for point in points
                ]
            )
        elif button.IsA(UsdGeom.Cube):
            cube = UsdGeom.Cube(button)
            cube.GetSizeAttr().Set(float(cube.GetSizeAttr().Get()) * geometry_scale)
        else:
            raise RuntimeError(f"unsupported ButtonMesh type {button.GetTypeName()!r}")

        joints = [
            prim
            for prim in self._stage.Traverse()
            if prim.GetPath().HasPrefix(root.GetPath()) and prim.IsA(UsdPhysics.PrismaticJoint)
        ]
        if len(joints) != 1:
            raise RuntimeError(f"switch requires exactly one prismatic joint, found {len(joints)}")
        drive = UsdPhysics.DriveAPI.Get(joints[0], "linear")
        stiffness = drive.GetStiffnessAttr().Get()
        if stiffness is None:
            raise RuntimeError("switch button drive has no authored stiffness")
        drive.GetStiffnessAttr().Set(float(stiffness) * stiffness_scale)

        # The source switch's local -Y is the outward normal, so its wall lies
        # behind it along local +Y. Author a collision-only proxy as a child of
        # the switch root; switch teardown then removes the proxy atomically.
        normal_half, lateral_half, vertical_half = wall_half_extents_m
        wall_path = f"{self._switch_path}/PressWallCollider"
        wall = UsdGeom.Xform.Define(self._stage, wall_path)
        UsdGeom.XformCommonAPI(wall).SetTranslate(Gf.Vec3d(0.0, wall_offset_m + normal_half, 0.0))
        wall_geom = UsdGeom.Cube.Define(self._stage, f"{wall_path}/geom")
        wall_geom.CreateSizeAttr(1.0)
        wall_geom.AddScaleOp().Set(
            Gf.Vec3f(2.0 * lateral_half, 2.0 * normal_half, 2.0 * vertical_half)
        )
        UsdPhysics.CollisionAPI.Apply(wall_geom.GetPrim())
        wall_geom.MakeInvisible()

    def _configure_drone(self, *, effort_limit_n: float, velocity_limit_mps: float) -> None:
        root = self._stage.GetPrimAtPath(self._drone_path)
        if not root.IsValid():
            raise ValueError(f"drone prim not found at {self._drone_path!r}")
        booms = [
            prim
            for prim in self._stage.Traverse()
            if prim.GetPath().HasPrefix(root.GetPath())
            and prim.GetName() == "boom"
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not booms:
            raise RuntimeError("press-policy drone has no boom collision prims")
        disabled_booms = [
            str(boom.GetPath())
            for boom in booms
            if UsdPhysics.CollisionAPI(boom).GetCollisionEnabledAttr().Get() is False
        ]
        if disabled_booms:
            raise RuntimeError(f"press-policy drone has disabled boom collisions: {disabled_booms}")
        prop_collisions = [
            prim
            for prim in self._stage.Traverse()
            if prim.GetPath().HasPrefix(root.GetPath())
            and prim.GetName() == "prop_collision"
            and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(prop_collisions) != len(booms):
            raise RuntimeError(
                "press-policy drone requires one propeller collision disc per arm; "
                f"found {len(prop_collisions)} discs and {len(booms)} arms"
            )

        joint = self._stage.GetPrimAtPath(f"{self._drone_path}/pusher_joint")
        if not joint.IsValid():
            raise RuntimeError("press-policy drone has no pusher_joint")
        drive = UsdPhysics.DriveAPI.Get(joint, "linear")
        upper = UsdPhysics.PrismaticJoint(joint).GetUpperLimitAttr().Get()
        drive.CreateTargetPositionAttr(float(upper))
        drive.CreateMaxForceAttr(effort_limit_n)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(joint)
        physx_joint.CreateMaxJointVelocityAttr().Set(velocity_limit_mps)

    def _initialize(self) -> None:
        if self._drone is not None:
            return
        drone = SingleArticulation(prim_path=self._drone_path)
        drone.initialize()
        if drone.dof_names.count("pusher_joint") != 1:
            raise RuntimeError(
                "press-policy drone requires exactly one pusher_joint; "
                f"found {drone.dof_names!r}"
            )
        index = drone.get_dof_index("pusher_joint")
        self._pusher_extension_m = float(drone.dof_properties["upper"][index])
        self._pusher_joint_index = index
        self._contact_sensor.initialize()
        self._drone = drone

    def read(
        self,
        *,
        vehicle_state: Any,
        sim_time_s: float,
    ) -> PressObservation:
        """Assemble the exact 34-value policy input from live resources."""
        self._initialize()
        assert self._drone is not None
        self._drone.apply_action(
            ArticulationAction(
                joint_positions=np.array([self._pusher_extension_m], dtype=np.float64),
                joint_indices=np.array([self._pusher_joint_index], dtype=np.int64),
            )
        )
        joint_index = np.array([self._pusher_joint_index], dtype=np.int64)
        pusher_position = float(self._drone.get_joint_positions(joint_indices=joint_index)[0])
        pusher_velocity = float(self._drone.get_joint_velocities(joint_indices=joint_index)[0])
        frame = self._contact_sensor.get_current_frame()
        contact_force = float(frame.get("force", 0.0))
        contact_flag = bool(frame.get("in_contact", False)) and (
            contact_force >= self._contact_force_threshold_n
        )

        pose = self._pose_provider.sample(sim_time_s)
        switch_quat_wxyz = np.array(
            [
                pose.world_switch_quat_xyzw[3],
                pose.world_switch_quat_xyzw[0],
                pose.world_switch_quat_xyzw[1],
                pose.world_switch_quat_xyzw[2],
            ],
            dtype=np.float64,
        )
        relative_position, relative_quat_xyzw = world_switch_to_contact_frame(
            (vehicle_state.position_enu, vehicle_state.orientation_flu_enu_wxyz),
            (pose.world_switch_pos_m, switch_quat_wxyz),
        )
        light = self._stage.GetPrimAtPath(f"{self._lamp_path}/Lamp/SphereLight")
        if not light.IsValid():
            raise RuntimeError(f"paired lamp light prim not found for {self._lamp_path!r}")
        lamp_state = int(float(UsdLux.SphereLight(light).GetIntensityAttr().Get()) > 0.0)
        mask = int(
            PressObservationMask.POSE | PressObservationMask.CONTACT | PressObservationMask.LAMP
        )
        return PressObservation(
            relative_switch_pos_m=relative_position,
            relative_switch_quat_xyzw=relative_quat_xyzw,
            pose_confidence=pose.confidence,
            pose_age_s=pose.age_s,
            detection_valid=pose.valid,
            body_lin_vel_mps=body_velocity_flu_from_state(vehicle_state),
            body_ang_vel_radps=vehicle_state.angular_velocity_flu,
            attitude_quat_xyzw=quat_wxyz_to_xyzw(vehicle_state.orientation_flu_enu_wxyz),
            pusher_compression_m=self._pusher_extension_m - pusher_position,
            pusher_compression_rate_mps=-pusher_velocity,
            contact_valid=True,
            contact_flag=contact_flag,
            lamp_state=lamp_state,
            lamp_confidence=1.0,
            lamp_age_s=0.0,
            previous_action=np.zeros(self._action_dim, dtype=np.float64),
            handoff_mode=2 if contact_flag else 1,
            estimator_health=1.0,
            obs_mask=mask,
        )
