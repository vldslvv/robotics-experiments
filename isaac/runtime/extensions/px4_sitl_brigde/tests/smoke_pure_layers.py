"""Stage-1 smoke test: pure-Python layers (no Isaac, no PX4, no MAVLink).

Run with the Isaac Kit Python interpreter or any Python with ``numpy`` and
``scipy`` installed:

    python tests/smoke_pure_layers.py

Exercises:
  - ``VehicleState.hover_at`` constructs without error
  - ``frames.magnetometer_body_flu`` returns the expected FLU vector
  - ``frames.synth_gps_from_enu`` shifts lat/lon by the right sign
  - ``LinearThrustModel`` clips and scales as advertised

Prints PASS/FAIL per assertion. Exit status is 0 only when every check passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make the ``Px4SitlBridge_python`` package importable when run from anywhere.
EXT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT_ROOT))

from Px4SitlBridge_python import frames  # noqa: E402
from Px4SitlBridge_python.thrust_model import LinearThrustModel  # noqa: E402
from Px4SitlBridge_python.vehicle_state import VehicleState  # noqa: E402

_failed = 0


def _check(name: str, ok: bool, detail: str = "") -> None:
    global _failed
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        _failed += 1


def main() -> int:
    s = VehicleState.hover_at()
    _check("hover state position", np.allclose(s.position_enu, [0, 0, 1]))
    _check("hover state quat wxyz", np.allclose(s.orientation_flu_enu_wxyz, [1, 0, 0, 0]))

    mag = frames.magnetometer_body_flu(np.array([1.0, 0, 0, 0]))
    # Default NED ref [0.215, 0, 0.43] → ENU [0, 0.215, -0.43] → body FLU same (level).
    _check(
        "magnetometer FLU level",
        np.allclose(mag, [0, 0.215, -0.43], atol=1e-6),
        f"got {mag}",
    )

    gps = frames.synth_gps_from_enu(np.array([10.0, 20.0, 1.0]), np.zeros(3))
    _check(
        "gps lat shift sign (north→larger lat)",
        gps["latitude"] > frames.DEFAULT_HOME_LAT_DEG,
    )
    _check(
        "gps lon shift sign (east→larger lon)",
        gps["longitude"] > frames.DEFAULT_HOME_LON_DEG,
    )
    _check(
        "gps alt shift",
        abs(gps["altitude"] - (frames.DEFAULT_HOME_ALT_M + 1.0)) < 1e-6,
    )

    model = LinearThrustModel(num_rotors=4, max_force_n=8.0)
    forces = model.commands_to_forces(np.array([0.0, 0.5, 1.0, 1.5]))
    _check(
        "linear thrust model clips above 1.0",
        np.allclose(forces, [0.0, 4.0, 8.0, 8.0]),
        f"got {forces}",
    )

    print()
    if _failed:
        print(f"{_failed} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
