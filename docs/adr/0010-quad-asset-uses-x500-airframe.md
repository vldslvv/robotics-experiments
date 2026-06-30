# ADR 0010: Bind the quad asset to a square-X airframe (X500/4019), not Iris

- Status: Accepted (Implemented)
- Date: 2026-06-30
- Related: ADR 0008 (airframe mechanism), ADR 0009 (hexa geometry), ADR 0011 (config generator)
- Full design: `docs/QUAD_GENERIC_AIRFRAME_PLAN.md`

## Context

The default `quadcopter_500mm.usda` is a 500mm **square-X** quad, but ADR 0008
bound it to `10016_none_iris` -- PX4's *Iris*, a wide/rectangular geometry
(`CA_ROTOR` at 0.1515 x 0.245/0.1875). It passed the integration test only because
the controller tolerated the mismatch; it was not correct.

Findings (verified against `vv-px4-sitl:v1.14.3`):

- **`4001` is unusable in SITL**: posix already ships `4001_gz_x500` (Gazebo); a
  second `4001_*` collides and `SYS_AUTOSTART=4001` launches gz, not
  `simulator_mavlink`. The generic `4001_quad_x` is NuttX-only.
- **`4019` (Holybro X500 V2) matches the asset exactly** by motor index (FRD
  +/-0.25, +/-0.25, CW->+KM / CCW->-KM) and is free in the posix set.

## Decision

Bind the quad asset to **`4019` / `none_x500_v2`**, shipped as a generated SITL
airframe with `CA_ROTOR` derived from the asset geometry.

- `configs/airframes/quadcopter_500mm.yaml` sets `sim_model: none_x500_v2`,
  `sys_autostart: 4019`; the Dockerfile px4 stage generates `4019_none_x500_v2`.
- Guard the `4001`/`gz_x500` collision at build time
  (`generate_px4_airframes.py::_conflicting_airframe_files`, with a
  `--prune-conflicts` seam).
- The separate unbound `4018_none_quad_x` generic airframe was NOT shipped
  (YAGNI -- no asset binds to it). `10016_none_iris` stays untouched, reserved for
  a future Iris asset. Octo (8001) / octo-coax (12001) are documented reservations
  only.

## Consequences

- Bridge-side 4-element thrust arrays are unchanged (geometry-independent).
- Switching Iris (wide) -> X500 (square) changes control allocation; because
  `4019` matches the asset exactly, the quad regression
  (`test-integration-px4-takeoff` / `-navigation`) was the key check and passed.
