# drone_cfg.py — Isaac Lab ArticulationCfg for the generated drone USD.
#
# Paired with build_copter_usd_without_pxr.py. That script emits the static
# geometry/topology once; this file wraps the resulting .usda so Isaac Lab can
# spawn N parallel copies with per-env randomization for sim-to-real training.
#
# Boundary (who owns what):
#   USD (authored once, static):
#     - Prim tree, meshes, collision shapes
#     - Joint topology (which body pairs are connected, axis, limits)
#     - Custom attributes like drone:spinDir on props
#     - ArticulationRootAPI on the wrapper prim
#   Cfg (this file, applied at spawn, randomizable per env):
#     - Mass / inertia / center of mass
#     - Joint drive stiffness / damping / effort limits
#     - Initial pose and joint positions
#     - Rigid body solver flags (max velocities, gravity toggle, CCD)
#
# Usage in an Isaac Lab env (not part of this repo yet):
#
#   from drone_cfg import QUADCOPTER_500MM_CFG
#
#   @configclass
#   class MySceneCfg(InteractiveSceneCfg):
#       drone = QUADCOPTER_500MM_CFG.replace(prim_path="{ENV_REGEX_NS}/Drone")
#
# Example randomization (lives in the env's EventCfg, NOT here, because
# EventTerm/SceneEntityCfg are env-scoped):
#
#   @configclass
#   class EventCfg:
#       reset_mass = EventTerm(
#           func=mdp.randomize_rigid_body_mass,
#           mode="reset",
#           params={"asset_cfg": SceneEntityCfg("drone", body_names=["body"]),
#                   "mass_distribution_params": (0.9, 1.1),  # +/-10%
#                   "operation": "scale"},
#       )
#       reset_pusher_gains = EventTerm(
#           func=mdp.randomize_actuator_gains,
#           mode="reset",
#           params={"asset_cfg": SceneEntityCfg("drone",
#                                               joint_names=["pusher_joint"]),
#                   "stiffness_distribution_params": (600.0, 1000.0),
#                   "damping_distribution_params": (30.0, 50.0)},
#       )

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import (
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
    UsdFileCfg,
)

# Mirrored from build_copter_usd_without_pxr.py rather than imported because
# importing that module triggers SimulationApp startup at Python import time.
ASSETS_DIR = "/isaac-sim/proj/assets/LightSwitchWarehouse"

# Mirrored from the same file's pusher constants. Keep in sync if you retune
# the USD defaults; cfg values here override the USD at spawn time, so this is
# the authoritative source for RL training.
_PUSHER_STIFFNESS = 800.0
_PUSHER_DAMPING = 40.0
_PUSHER_EFFORT_LIMIT = 50.0  # N — rough cap for the prismatic pusher motor


def _make_drone_cfg(
    usd_filename: str,
    total_mass: float,
    prim_name: str,
) -> ArticulationCfg:
    """Build an ArticulationCfg for one drone variant.

    The prim_path uses {ENV_REGEX_NS} so Isaac Lab swaps it with the per-env
    namespace at spawn time (one articulation per parallel env).
    """
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        spawn=UsdFileCfg(
            usd_path=f"{ASSETS_DIR}/{usd_filename}",
            # Rigid body solver props apply to every rigid body under the
            # articulation root. Caps protect against numerical blowups when a
            # policy outputs extreme thrusts early in training.
            rigid_props=RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=20.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            # Overrides the USD's MassAPI. Kept here so EventTerm can scale it
            # per-env without re-authoring the USD.
            mass_props=MassPropertiesCfg(mass=total_mass),
        ),
        # Initial state applies after spawn. pos is world-relative to the env
        # origin; pusher_joint=0.0 means the pad sits fully retracted.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            joint_pos={"pusher_joint": 0.0},
            joint_vel={"pusher_joint": 0.0},
        ),
        # Actuators are keyed by a logical name ("pusher"), with
        # joint_names_expr matching the joint prim name in the USD.
        # ImplicitActuatorCfg uses PhysX's built-in PD controller — fast and
        # differentiable under standard PPO/SAC workflows.
        actuators={
            "pusher": ImplicitActuatorCfg(
                joint_names_expr=["pusher_joint"],
                stiffness=_PUSHER_STIFFNESS,
                damping=_PUSHER_DAMPING,
                effort_limit=_PUSHER_EFFORT_LIMIT,
                velocity_limit=1.0,  # m/s — pad can't slam faster than this
            ),
        },
    )


# Masses mirror the `total_mass` field in build_copter_usd_without_pxr.py.
QUADCOPTER_500MM_CFG = _make_drone_cfg(
    usd_filename="quadcopter_500mm.usda",
    total_mass=1.5,
    prim_name="Quadcopter",
)

HEXACOPTER_S550_CFG = _make_drone_cfg(
    usd_filename="hexacopter_s550.usda",
    total_mass=1.929,
    prim_name="Hexacopter",
)
