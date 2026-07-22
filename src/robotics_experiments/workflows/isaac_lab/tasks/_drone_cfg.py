"""Config-driven Isaac Lab ``ArticulationCfg`` builder for the drone USD.

Torch/isaaclab leaf (off the core import path). Moved here from
``isaac/runtime/lab/drone_cfg.py`` so importable task code lives in the package,
and made config-driven: the USD asset comes from ``drone.usd_path`` (Hydra) rather
than a hardcoded assets path. The USD carries static geometry/topology, mass
properties, and the authoritative pusher stiffness/damping. This cfg applies
only Lab-specific solver caps, effort and velocity limits, and initial state
from ``rl.lab_vehicle``.

Assumes the pusher-enabled variant (the default ``holybro_x500_v2``): a
pusher-less variant built with ``drone.pusher.enabled=false`` has no
``pusher_joint`` and would need the pusher actuator + init entries dropped.
"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import RigidBodyPropertiesCfg, UsdFileCfg

__all__: list[str] = []


def _make_drone_articulation_cfg(*, usd_path: str, prim_path: str, lab_vehicle) -> ArticulationCfg:
    """Build the drone ``ArticulationCfg`` for the Lab env from config.

    Args:
        usd_path: Absolute path to the drone USD asset (``drone.usd_path``).
        prim_path: Per-env articulation prim path (an Isaac Lab regex such as
            ``/World/envs/env_.*/Drone``).
    """
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=UsdFileCfg(
            usd_path=usd_path,
            # Rigid-body solver caps protect against numerical blowups when a
            # policy outputs extreme thrusts early in training.
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=lab_vehicle.max_linear_velocity_mps,
                max_angular_velocity=lab_vehicle.max_angular_velocity_radps,
                max_depenetration_velocity=lab_vehicle.max_depenetration_velocity_mps,
                enable_gyroscopic_forces=True,
            ),
            activate_contact_sensors=True,
            # No mass override: the USD carries the computed body mass/inertia/CoM
            # (robotics_experiments.drone._airframe.compute_body_mass_properties).
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, lab_vehicle.spawn_height_m),
            joint_pos={"pusher_joint": 0.0},
            joint_vel={"pusher_joint": 0.0},
        ),
        actuators={
            "pusher": ImplicitActuatorCfg(
                joint_names_expr=["pusher_joint"],
                stiffness=None,
                damping=None,
                effort_limit=lab_vehicle.pusher_effort_limit_n,
                velocity_limit=lab_vehicle.pusher_velocity_limit_mps,
            ),
        },
    )
