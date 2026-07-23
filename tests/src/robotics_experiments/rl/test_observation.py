"""Pure tests for the versioned hover observation layout."""

import numpy as np

from robotics_experiments.px4_bridge import VehicleState
from robotics_experiments.rl import (
    HOVER_OBSERVATION_DIM,
    HOVER_OBSERVATION_SCHEMA_VERSION,
    hover_observation_from_state,
    pack_hover_observation,
)


def test_hover_layout_length_and_order() -> None:
    vector = pack_hover_observation(
        position_err_enu=np.array([1.0, 2.0, 3.0]),
        quat_flu_enu_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        linear_velocity_enu=np.array([4.0, 5.0, 6.0]),
        angular_velocity_body_flu=np.array([7.0, 8.0, 9.0]),
    )
    assert vector.shape == (HOVER_OBSERVATION_DIM,)
    assert vector.dtype == np.float64
    np.testing.assert_allclose(vector[0:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(vector[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(vector[7:10], [4.0, 5.0, 6.0])
    np.testing.assert_allclose(vector[10:13], [7.0, 8.0, 9.0])


def test_hover_layout_field_changes_move_expected_slots() -> None:
    base = pack_hover_observation(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
    )
    changed = pack_hover_observation(
        np.zeros(3),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, 0.5]),
    )
    assert np.nonzero(changed - base)[0].tolist() == [12]


def test_hover_layout_schema_version_pinned() -> None:
    assert HOVER_OBSERVATION_SCHEMA_VERSION == 1
    assert HOVER_OBSERVATION_DIM == 13


def test_hover_observation_from_state_matches_pack() -> None:
    state = VehicleState(
        position_enu=np.array([2.0, 1.0, 3.0]),
        orientation_flu_enu_wxyz=np.array([0.9238795, 0.0, 0.0, 0.3826834]),
        linear_velocity_enu=np.array([0.1, -0.2, 0.3]),
        angular_velocity_flu=np.array([0.01, 0.02, -0.03]),
        sim_time_us=1_000_000,
    )
    target = np.array([2.0, 1.0, 2.0])
    from_state = hover_observation_from_state(state, target)
    packed = pack_hover_observation(
        state.position_enu - target,
        state.orientation_flu_enu_wxyz,
        state.linear_velocity_enu,
        state.angular_velocity_flu,
    )
    np.testing.assert_allclose(from_state, packed)
    np.testing.assert_allclose(from_state[0:3], [0.0, 0.0, 1.0])
