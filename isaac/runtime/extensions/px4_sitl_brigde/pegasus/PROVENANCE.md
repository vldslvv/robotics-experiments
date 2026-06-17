# Pegasus vendoring provenance and divergence index

This directory is a vendored, stripped snapshot of the Pegasus Simulator. It is
the single source of truth for what the snapshot is, which modules are
load-bearing, and which deviations from upstream are intentional and must be
preserved across any future re-sync.

## Upstream

- Project: Pegasus Simulator -- `https://github.com/PegasusSimulator/PegasusSimulator`
- License: BSD-3-Clause, (c) 2023 Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
- Commit/version: `bef3c57` ("fix: bug where the linear velocity in the body
  frame would return the acceleration"). Synced 2026-06-17. The prior snapshot
  predated provenance tracking (commit `UNKNOWN`); the diff against upstream at
  re-sync time showed the only substantive change in the reachable set was the
  `State.get_linear_body_velocity_ned_frd` fix in `bef3c57`, which was applied.
- Scope: per `README.MD`, the tree is "stripped from everything but the logic that
  bridges Isaac Sim with PX4 SITL, and its dependencies."

## Reachable module set (the surface to maintain)

The PX4 bridge imports the following lazily (so a lightweight
`import robotics_experiments` does not pull in Pegasus/scipy). Entry imports:

- `src/robotics_experiments/px4_bridge/_pegasus_sensors.py:42-45,84` ->
  `simulator/logic/sensors/{barometer,gps,imu,magnetometer}.py`,
  `simulator/logic/state.py`
- `src/robotics_experiments/px4_bridge/mavlink.py:190-193` ->
  `simulator/logic/backends/px4_mavlink_backend.py`
  (`PX4MavlinkBackend`, `PX4MavlinkBackendConfig`)

Transitively reachable from those entries:

- `simulator/logic/sensors/sensor.py` (base class)
- `simulator/logic/sensors/geo_mag_utils.py` (WMM2018 tables + reprojection)
- `simulator/logic/rotations.py` (frame constants)
- `simulator/logic/backends/backend.py` (abstract base)
- `simulator/logic/backends/tools/px4_launch_tool.py` (see dead-at-runtime note)

Every remaining `.py` in this tree is reachable; nothing is dead-import. Earlier
dead modules were removed in commit `Cleanup unused pegasus modules -- Step 9`.

## Dead-at-runtime (kept intentionally)

`simulator/logic/backends/tools/px4_launch_tool.py` and the `px4_autolaunch`
branches in `px4_mavlink_backend.py` (around lines 509-531) are retained verbatim
from upstream but are never exercised: the bridge constructs the backend with
`"px4_autolaunch": False` (`src/robotics_experiments/px4_bridge/mavlink.py:207`),
and PX4 SITL is launched separately (`make run-px4`). They are kept to minimize
divergence from upstream so a future sync stays mechanical; do not treat them as
live code paths.

## Intentional divergences (load-bearing fixes, NOT drift)

Each row is a deliberate deviation from upstream Pegasus. Do not "sync it away"
during a re-sync; re-apply it and confirm its guarding test still passes. The
three vendored-tree divergences carry inline `# Intentional divergence from
upstream Pegasus` comments at the cited lines.

| Divergence | Location | Upstream behavior | Why preserved | Guarding test |
|---|---|---|---|---|
| GPS ground-truth lat/lon in degrees | `simulator/logic/sensors/gps.py:88-94` and `:189-193` | radians | The backend consumes `latitude_gt`/`longitude_gt` via `* 10000000` expecting degrees, matching the primary `latitude`/`longitude` fields | `tests/test_frames_pegasus_equivalence.py::test_pegasus_gps_ground_truth_is_degrees_for_any_home` |
| Barometer applies accumulated drift, not drift rate | `simulator/logic/sensors/barometer.py:131-134` | adds the per-second drift rate directly each tick | Pressure and pressure-altitude stay mutually consistent | `tests/test_px4_bridge_sensors.py::test_vendored_pegasus_barometer_applies_accumulated_drift` |
| Barometer exposes `noise_std_pa` (0 disables noise) | `simulator/logic/sensors/barometer.py:60,106-126` | hard-coded 1 Pa RMS noise with no disable knob | Lets bridge precision tests make the sensor exact via Hydra; default 1.0 preserves upstream behavior | `tests/test_px4_bridge_sensors.py::test_mavlink_port_pegasus_sensors_can_disable_noise_for_precision` |
| HIL_SENSOR 14th field carries temperature | `simulator/logic/backends/px4_mavlink_backend.py:710-717` | sends GPS altitude in this field | The MAVLink `HIL_SENSOR` 14th field is temperature; sending altitude is wrong | `tests/test_px4_bridge_sensors.py::test_vendored_pegasus_backend_sends_barometer_temperature` |
| Sim-time lockstep (`dt` from `sim_time_us`, not wallclock) -- bridge-side, NOT in this tree | `src/robotics_experiments/px4_bridge/_pegasus_sensors.py:68-74`; documented at `src/robotics_experiments/px4_bridge/mavlink.py:94-96` | wallclock-derived timing / free-run | Makes `HIL_SENSOR.time_usec` sim-speed-agnostic and lockstep deterministic | `tests/test_px4_bridge_sensors.py::test_mavlink_port_uses_sim_time_and_gates_gps`; `tests/test_px4_bridge_lockstep.py::test_mavlink_port_skips_duplicate_sim_time_in_lockstep` |

## Comment-only annotations (no behavior change)

These deviate from upstream only in comment text -- the executable code is
byte-identical to upstream. They carry no guarding test because there is nothing
behavioral to guard, but they will show up in a re-sync diff. Keep the
annotation; do not "sync away" the explanatory text.

- **GPS velocity noise is sampled but not applied** --
  `simulator/logic/sensors/gps.py:153-156`. Upstream samples `self._noise_gps_vel`
  but the velocity line (`velocity = state.linear_velocity  # + self._noise_gps_vel`)
  leaves it commented out, so `HIL_GPS` velocity stays exact ENU->NED ground
  truth. The vendored comment just documents this upstream behavior so future
  readers do not mistake it for a bug; the code line itself matches upstream.

## Re-sync procedure

1. Record the upstream commit you are syncing to (replace the `UNKNOWN` above).
2. Diff upstream into the reachable module set only; do not reintroduce modules
   outside that set.
3. Re-apply each divergence in the table above (the three vendored-tree ones are
   inside this tree; the sim-time lockstep one is bridge-side and usually
   untouched by an upstream diff).
4. Run the unit suite (`uv run pytest tests/`) -- the four guarding tests above are
   the drift guard. Then run the smoke tests under
   `isaac/runtime/extensions/px4_sitl_brigde/tests/`
   (`smoke_pure_layers.py`, `smoke_port_with_px4.py`).
