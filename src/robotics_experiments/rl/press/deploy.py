"""Pure RL-action -> PX4 OFFBOARD setpoint mapping (deploy-time seam).

The deploy-time analogue of the RL actuation contract: in Lab the action drives the
in-sim actuation model; at deploy the *same* normalized action maps to a PX4
OFFBOARD setpoint streamed over the GCS link. This module owns only the pure
conversion (body-FLU RL action -> the PX4 body-FRD wire payloads exported by
``commanding``); the side-effecting stream is the command port's
``send_velocity_setpoint`` / ``send_attitude_target``. Keeping it here (not in
the env leaf) keeps it torch/Isaac-free and unit-testable. The public entrypoint
is exported through ``robotics_experiments.rl``.

Frame note: the RL action is body-FLU (forward / left / up, yaw +CCW); PX4's
setpoint frames are body-FRD (forward / right / down, yaw +CW), so ``y`` and
``z`` (and yaw) flip sign. The ``velocity`` action is interpreted in the
body-yaw (heading) frame, matching the ``velocity`` actuation cascade; PX4
tracks it
via ``MAV_FRAME_BODY_NED``.
"""

from __future__ import annotations

import numpy as np

from robotics_experiments.commanding import AttitudeRateSetpoint, VelocitySetpoint

__all__ = ["press_action_to_offboard"]


def _velocity_setpoint(action: np.ndarray, action_cfg) -> VelocitySetpoint:
    max_velocity = float(action_cfg.max_body_velocity_mps)
    max_yaw_rate = float(action_cfg.max_yaw_rate_radps)
    return VelocitySetpoint(
        forward_mps=float(action[0]) * max_velocity,
        right_mps=-float(action[1]) * max_velocity,
        down_mps=-float(action[2]) * max_velocity,
        yaw_rate_radps=-float(action[3]) * max_yaw_rate,
    )


def _attitude_target(action: np.ndarray, action_cfg) -> AttitudeRateSetpoint:
    hover_fraction = float(action_cfg.hover_thrust_fraction)
    max_body_rate = float(action_cfg.max_body_rate_radps)
    thrust = float(np.clip(hover_fraction * (1.0 + float(action[0])), 0.0, 1.0))
    return AttitudeRateSetpoint(
        thrust=thrust,
        roll_rate_radps=float(action[1]) * max_body_rate,
        pitch_rate_radps=-float(action[2]) * max_body_rate,
        yaw_rate_radps=-float(action[3]) * max_body_rate,
    )


def press_action_to_offboard(action: np.ndarray, action_cfg):
    """Map a normalized RL action to a PX4 OFFBOARD setpoint wire payload.

    Args:
        action: The normalized policy action (``[vx, vy, vz, yaw_rate]`` for
            ``velocity`` / ``[thrust, wx, wy, wz]`` for ``ctbr``), body-FLU.
        action_cfg: The ``rl.action`` config (``kind`` + the command scales
            ``max_body_velocity_mps`` / ``max_yaw_rate_radps`` /
            ``max_body_rate_radps`` / ``hover_thrust_fraction``).

    Returns:
        A ``VelocitySetpoint`` (``velocity``) or ``AttitudeRateSetpoint``
        (``ctbr``) for the command port to stream.

    Raises:
        ValueError: for an unknown or not-yet-implemented action level.
    """
    action = np.asarray(action, dtype=np.float64)
    kind = action_cfg.kind
    if kind == "velocity":
        return _velocity_setpoint(action, action_cfg)
    if kind == "ctbr":
        return _attitude_target(action, action_cfg)
    raise ValueError(
        f"unsupported rl.action.kind={kind!r}; implemented levels are 'velocity' and 'ctbr'"
    )
