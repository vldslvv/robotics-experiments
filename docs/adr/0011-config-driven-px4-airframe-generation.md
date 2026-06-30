# ADR 0011: Single-source PX4 airframes from `configs/airframes` (generated at build)

- Status: Accepted
- Date: 2026-06-30
- Related: ADR 0008/0009/0010 (the hand-written airframes this supersedes)
- Source: no standalone design doc; the contract lives in `AGENTS.md` and the
  `configs/airframes/*.yaml` headers (memory: `px4-airframe-generation`)

## Context

ADR 0008-0010 shipped hand-written `px4/docker/airframes/*` files plus a Python
`_PX4_AIRFRAME_BY_ASSET` map. That meant three independent sources for one drone's
geometry -- the USD asset, the bridge rotor arrays, and the PX4 `CA_ROTOR`
allocation -- kept consistent only by hand, exactly the class of mismatch that
caused the hexa tip-over (ADR 0009).

## Decision

Single-source every PX4 SITL airframe (and the asset->airframe mapping) from
`configs/airframes/*.yaml`, the same source the USD asset reads:

- The Dockerfile px4 stage runs `px4/scripts/generate_px4_airframes.py` to
  generate the `none_*` posix airframe scripts (`CA_ROTOR` **derived**, not
  hand-authored), register them in PX4's airframe CMake list, write an image
  manifest, then `make px4_sitl_default`. A host freshness check compares the
  manifest against the configs.
- `resolve_px4_sitl_launch_config(vehicle)` maps the drone USD asset basename to
  the generated airframe (`quadcopter_500mm.usda` -> `none_x500_v2`/`4019`;
  `hexacopter_s550.usda` -> `none_hexa_x`/`6001`).
- **`SYS_AUTOSTART` ids must be FREE in the posix set:** `rcS` globs
  `airframes/<id>_*` and sources the last match, so a stock `<id>_*` would shadow
  a generated one. The quad therefore uses the free `4019` (X500 V2), not `4001`
  (`4001_gz_x500` collides). The generator fails the build loudly on such a
  collision (`_conflicting_airframe_files`), with a `--prune-conflicts` seam.

## Consequences

- USD geometry, bridge rotor arrays, and PX4 control allocation share one source;
  a geometry change can no longer drift between sim and flight controller.
- The generated airframe scripts are transient build artifacts (not committed).
- Supersedes the hand-written airframe mechanism of ADR 0008-0010; those ADRs
  remain accurate for the geometry decisions, only the delivery mechanism changed.
