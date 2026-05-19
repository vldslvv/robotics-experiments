"""Conversion from PX4-normalized rotor commands to per-motor forces in N.

The thrust model is intentionally a separate object from the MAVLink port:
domain randomization, voltage sag, and response-lag effects belong here, and
must not be entangled with the protocol layer. Swap a different model in
without touching the bridge or the port.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

__all__ = ["RotorThrustModel", "LinearThrustModel"]


class RotorThrustModel(ABC):
    """Abstract per-motor force model.

    Implementations take an ``np.ndarray`` of ``num_rotors`` normalized
    commands (each in ``[0, 1]`` after PX4's mixer) and return an
    ``np.ndarray`` of forces in Newtons to apply along each rotor's local
    thrust axis.
    """

    @property
    @abstractmethod
    def num_rotors(self) -> int: ...

    @abstractmethod
    def commands_to_forces(self, controls: np.ndarray) -> np.ndarray: ...


class LinearThrustModel(RotorThrustModel):
    """Trivial uniform linear model: ``force_i = clip(c_i, 0, 1) * max_force``.

    Picks no opinion about voltage, response time, or per-motor variation.
    Intended as the seed model - replace with a curve-fit/airframe-specific
    model once domain randomization is wired up.
    """

    def __init__(self, *, num_rotors: int, max_force_n: float = 8.0):
        """Initialize the model.

        Args:
            num_rotors: Number of rotors driven by this model.
            max_force_n: Force in Newtons produced at command 1.0.
        """
        self._num_rotors = int(num_rotors)
        self._max_force_n = float(max_force_n)

    @property
    def num_rotors(self) -> int:
        return self._num_rotors

    @property
    def max_force_n(self) -> float:
        return self._max_force_n

    def commands_to_forces(self, controls: np.ndarray) -> np.ndarray:
        c = np.clip(np.asarray(controls, dtype=np.float64), 0.0, 1.0)
        if c.shape != (self._num_rotors,):
            raise ValueError(
                f"LinearThrustModel expects shape ({self._num_rotors},), got {c.shape}"
            )
        return c * self._max_force_n
