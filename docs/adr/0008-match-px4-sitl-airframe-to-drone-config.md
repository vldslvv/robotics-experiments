# ADR 0008: Match PX4 SITL airframe to the drone config

- Status: Accepted (hand-written mechanism superseded by ADR 0011 / config generator)
- Date: 2026-06-29
- Related: ADR 0009 (hexa tuning), ADR 0010 (quad X500), ADR 0011 (config generator)
- Full design: `docs/PX4_SITL_AIRFRAME_MATCH_PLAN.md`

## Context

Every PX4 SITL vehicle launched with the quad `none_iris` airframe regardless of
its USD asset. Hexacopters therefore ran with a quad control allocation. The
launch airframe was not derived from the resolved per-vehicle config.

Verified against `vv-px4-sitl:v1.14.3`: `make px4_sitl_default` embeds only the
*posix* airframe set, whose sole external-MAVLink (none-family) airframe is
`10016_none_iris` (a quad). `6001_hexa_x` exists only in the NuttX set, so
`SYS_AUTOSTART=6001` aborted with "no autostart file found".

## Decision

Derive the SITL airframe from each resolved drone/PX4 config via a pure seam:

- Frozen `Px4SitlLaunchConfig(instance_index, sim_model, sys_autostart)` produced
  by the config loader (no FS/Docker/subprocess reads per vehicle); unknown
  PX4-enabled asset raises `ValueError`.
- `get_enabled_px4_launch_configs()` returns launch specs in fleet order.
- Persistent and integration launch flows export `PX4_INSTANCE`, `PX4_SIM_MODEL`,
  `PX4_SYS_AUTOSTART`; the common launch script defaults
  `PX4_SIM_MODEL=${PX4_SIM_MODEL:-none_iris}` only for manual/unset launches.
- Ship a custom SITL `6001_none_hexa_x` airframe (none-family template + canonical
  `hexa_x` geometry), registered into the posix CMakeLists and baked by the
  Dockerfile px4 stage.

## Consequences

- Hexacopters boot/arm/take off on a 6-rotor airframe; quad behavior unchanged.
- Asset-vs-airframe rotor ordering/spin was NOT yet validated: the hexa tipped
  over at ~0.17 m due to a geometry/index mismatch -> ADR 0009.
- The hand-written `px4/docker/airframes/*` + `_PX4_AIRFRAME_BY_ASSET` map was
  later replaced by the config-driven generator (`configs/airframes/*.yaml`).
