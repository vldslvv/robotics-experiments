"""Task-neutral, versioned policy observation layouts.

PARITY INVARIANT: the same packing function is fed from Lab tensors in
training and from deploy-time state. Any field a deployment caller cannot
assemble is forbidden in the layout.
"""

from __future__ import annotations

import numpy as np

from robotics_experiments.px4_bridge import VehicleState
from robotics_experiments.rl._interfaces import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
)

__all__ = [
    "HOVER_OBSERVATION_DIM",
    "HOVER_OBSERVATION_SCHEMA_VERSION",
    "hover_observation_from_state",
    "pack_hover_observation",
]


# Hover layout v1: [pos_err_enu(3), quat_flu_enu_wxyz(4), lin_vel_enu(3),
# ang_vel_body_flu(3)].
def pack_hover_observation(
    position_err_enu: np.ndarray,
    quat_flu_enu_wxyz: np.ndarray,
    linear_velocity_enu: np.ndarray,
    angular_velocity_body_flu: np.ndarray,
) -> np.ndarray:
    """Pack the hover schema-v1 vector in its stable field order."""
    return np.concatenate(
        (
            np.asarray(position_err_enu, dtype=np.float64).reshape(3),
            np.asarray(quat_flu_enu_wxyz, dtype=np.float64).reshape(4),
            np.asarray(linear_velocity_enu, dtype=np.float64).reshape(3),
            np.asarray(angular_velocity_body_flu, dtype=np.float64).reshape(3),
        )
    )


def hover_observation_from_state(state: VehicleState, target_enu: np.ndarray) -> np.ndarray:
    """Assemble the hover layout from deploy-time vehicle state."""
    return pack_hover_observation(
        state.position_enu - np.asarray(target_enu, dtype=np.float64).reshape(3),
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
        state.angular_velocity_flu,
    )
