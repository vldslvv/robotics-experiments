"""Pure unit tests for the RL-action -> PX4 OFFBOARD setpoint deploy mapping.

Assert the body-FLU RL action converts to the body-FRD wire payloads with the
correct sign flips (y / z / yaw) and thrust normalization -- the deploy-time
analogue of the in-Lab actuation, tested without pymavlink or Isaac.
"""

import numpy as np
import pytest

from robotics_experiments.rl.press import PolicyContract, press_action_to_offboard


def _contract(*, action_kind: str) -> PolicyContract:
    return PolicyContract(
        policy_interface="hover-v1",
        obs_schema_version=1,
        obs_dim=13,
        policy_rate_hz=20.0,
        action_kind=action_kind,
        action_dim=4,
        max_body_velocity_mps=3.0,
        max_yaw_rate_radps=1.5,
        max_body_rate_radps=6.0,
        hover_thrust_fraction=0.4,
    )


def test_velocity_action_flips_flu_to_frd():
    setpoint = press_action_to_offboard(
        np.array([1.0, 0.5, 0.25, 0.4]),
        _contract(action_kind="velocity"),
    )

    assert setpoint.forward_mps == pytest.approx(3.0)
    assert setpoint.right_mps == pytest.approx(-1.5)  # +left FLU -> -right FRD
    assert setpoint.down_mps == pytest.approx(-0.75)  # +up FLU -> -down FRD
    assert setpoint.yaw_rate_radps == pytest.approx(-0.6)  # +CCW FLU -> -CW NED


def test_ctbr_action_maps_thrust_and_body_rates():
    setpoint = press_action_to_offboard(
        np.array([0.2, 0.1, -0.1, 0.3]),
        _contract(action_kind="ctbr"),
    )

    assert setpoint.thrust == pytest.approx(0.4 * 1.2)
    assert setpoint.roll_rate_radps == pytest.approx(0.6)  # +x FLU == +x FRD
    assert setpoint.pitch_rate_radps == pytest.approx(0.6)  # +y FLU -> -y FRD
    assert setpoint.yaw_rate_radps == pytest.approx(-1.8)  # +z FLU -> -z FRD


def test_ctbr_thrust_is_clipped_to_unit_interval():
    contract = _contract(action_kind="ctbr")
    high = press_action_to_offboard(np.array([5.0, 0.0, 0.0, 0.0]), contract)
    low = press_action_to_offboard(np.array([-5.0, 0.0, 0.0, 0.0]), contract)

    assert high.thrust == pytest.approx(1.0)
    assert low.thrust == pytest.approx(0.0)


def test_deploy_rejects_unimplemented_kind():
    with pytest.raises(ValueError, match="attitude"):
        press_action_to_offboard(np.zeros(4), _contract(action_kind="attitude"))
