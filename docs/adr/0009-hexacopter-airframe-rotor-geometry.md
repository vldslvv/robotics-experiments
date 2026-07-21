# ADR 0009: Make the hexa airframe geometry match the USD asset

- Status: Accepted (Resolved)
- Date: 2026-06-29
- Related: ADR 0008 (airframe mechanism), ADR 0010 (quad X500)

## Context

With ADR 0008 the hexacopter booted and armed but tipped over at ~0.17 m on
takeoff, breaching the tilt gate (`tilt 35.8deg > 35.0deg`). ULog showed
asymmetric motor saturation (`[0.46, 0.46, 0.92, 0, 0.92, 0]`) -- the signature of
a control allocator fighting an attitude error it cannot resolve.

Root cause: rotor-geometry / motor-index mismatch. PX4's stock `6001_hexa_x` uses
canonical clocking (rotors at +/-30/+/-90/+/-150 deg, none on the forward axis),
but `hexacopter_s550.usda` authors a rotor on the forward axis (index 0 at 0 deg,
then 60/120/...). No PX4 `CA_ROTOR` index matched the asset's physical rotor.
`rotor_discovery.py` validated order only for the quad case, so the mismatch was
silent.

## Decision

Keep the USD asset authoritative; rewrite the airframe to match it.

- Rewrite `CA_ROTOR0..5` in `6001_none_hexa_x` to the asset's per-index layout
  (ordered by `px4_motor_index`): position `PX = R*cos(az)`, `PY = -R*sin(az)`
  (FLU->FRD, R = 0.275 m); KM sign from `spin_dir`, calibrated against the working
  iris quad mapping: **CW -> +0.05, CCW -> -0.05**.
- Add `_validate_generic_hexa_x_order` to `rotor_discovery.py` (each marker's
  body-FLU heading must equal `60*index` deg) so a future mismatch fails loudly at
  spawn. Unit-tested.
- Fix `ulog_timeline.py` to count motor columns (was hardcoded to 4).

## Consequences

- `make test-integration-px4-hexacopter-navigation` passes the full
  arm/takeoff/hover/goto/goto/hover/land routine.
- Thrust was checked and NOT changed (T/W ~3.3); the non-climb resolved once the
  allocator stopped fighting a wrong-geometry model.
- The FLU->FRD + spin->KM convention validated here is reused by ADR 0010.
