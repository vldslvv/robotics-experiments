"""The pose-provider seam: the single swap point for the switch pose source.

Pure module (numpy + stdlib). The provider owns only the switch pose estimate;
drone state comes from the env/bridge directly. Phase F ships the ground-truth
provider (Isaac leaf) and the synthetic-noise provider (F5); Phase G plugs real
sensor/fusion providers behind the identical ``sample()`` signature.

INVARIANT: swapping the provider must not change ``PressObservation``'s shape
or the ``to_vector`` output length ("policy API change: none") -- only the
values (and validity/confidence/age) change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = ["AbstractPoseProvider", "PoseSample"]


@dataclass(frozen=True)
class PoseSample:
    """One switch-pose estimate.

    Attributes:
        world_switch_pos_m: Switch plate position, world-ENU meters (3,).
        world_switch_quat_xyzw: Switch plate orientation, xyzw; local ``+X`` is
            the outward plate normal (the contact-frame convention exported by
            :mod:`robotics_experiments.rl`).
        confidence: Estimate confidence in [0, 1].
        age_s: Age of the estimate, seconds.
        valid: Whether the estimate is usable at all.
    """

    world_switch_pos_m: np.ndarray
    world_switch_quat_xyzw: np.ndarray
    confidence: float
    age_s: float
    valid: bool


class AbstractPoseProvider(Protocol):
    """Protocol every switch-pose provider implements."""

    def sample(self, sim_time_s: float) -> PoseSample:
        """Return the current switch-pose estimate at ``sim_time_s``."""
        ...
