"""Config-driven Isaac Lab ``ArticulationCfg`` builder for the drone USD.

Torch/isaaclab leaf (off the core import path). Moved here from
``isaac/runtime/lab/drone_cfg.py`` so importable task code lives in the package,
and made config-driven: the USD asset comes from ``drone.usd_path`` (Hydra) rather
than a hardcoded assets path. The USD carries the static geometry/topology and the
computed body mass/inertia/CoM; this cfg applies the spawn-time overrides Isaac Lab
randomizes per env (rigid-body caps, pusher actuator drive, initial joint state).

Assumes the pusher-enabled variant (the default ``holybro_x500_v2``): a
pusher-less variant built with ``drone.pusher.enabled=false`` has no
``pusher_joint`` and would need the pusher actuator + init entries dropped.
"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

__all__: list[str] = []

# Pusher prismatic-joint drive. cfg values override the USD defaults at spawn, so
# these are authoritative for RL training; EventTerm randomization (in the env's
# EventCfg) scales them per env without re-authoring the USD.
_PUSHER_STIFFNESS = 800.0
_PUSHER_DAMPING = 40.0
_PUSHER_EFFORT_LIMIT = 50.0  # N -- rough cap for the prismatic pusher motor


def _make_drone_articulation_cfg(usd_path: str, prim_path: str) -> ArticulationCfg:
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
            rigid_props=PhysxRigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=20.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            # No mass override: the USD carries the computed body mass/inertia/CoM
            # (robotics_experiments.drone._airframe.compute_body_mass_properties).
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={"pusher_joint": 0.0},
            joint_vel={"pusher_joint": 0.0},
        ),
        actuators={
            "pusher": ImplicitActuatorCfg(
                joint_names_expr=["pusher_joint"],
                stiffness=_PUSHER_STIFFNESS,
                damping=_PUSHER_DAMPING,
                effort_limit=_PUSHER_EFFORT_LIMIT,
                velocity_limit=1.0,  # m/s -- pad can't slam faster than this
            ),
        },
    )
