# 0012 - Hardware-faithful X500 V2 quad with computed, vendor-cited airframe properties

Date: 2026-06-30

## Context

The default quad asset was a placeholder (`quadcopter_500mm`): X2212-class
motors, 9-inch props, 1.5 kg, and a 0.3535 m boom length that actually modelled a
707 mm-diagonal frame, with the verbatim PX4/Gazebo iris thrust curve
(`rotor_constant 8.54858e-6`, `max_rotor_velocity 1100`). The roadmap
(`LIGHT_SWITCH_MULTIROTOR_ROADMAP.md`, Phase B/E) flagged it as a placeholder for
the Holybro X500 V2 Stage 1 X4 in `archive/HOLYBRO_X8_STAGED_BOM_v2.md`.

Beyond making it faithful, the requirement was that the config be **verifiable
and single-source**: every physical number should trace to a vendor datasheet /
BOM section and be fixable in one edit, with no hand-authored value duplicating
information that lives elsewhere (the prior config hand-authored `total_mass`,
`diagonal_inertia`, and `rotor_constant`, all of which silently go stale when a
component value changes).

## Decision

Rebuild the quad as `holybro_x500_v2` (renamed from `quadcopter_500mm`), faithful
to the BOM, and split every airframe value into authored **cited leaves** vs
**computed derivations**:

- **Cited leaves** in `configs/airframes/holybro_x500_v2.yaml`: per-component
  mass, geometry, and the per-rotor `max_thrust_per_rotor_g`, each carrying an
  inline `-- BOM x.y` / vendor citation. Faithful values: 0.25 m motor radius
  (500 mm diagonal wheelbase, 354 mm adjacent), MN4006 EVO KV380 (68 g),
  P134.4 13x4.4 (0.16510 m), Tattu 4S 5000 mAh (460 g), ~2.251 kg AUW.
- **Computed body mass properties**: the USD builder
  (`build_copter_usd_standalone.compute_body_mass_properties`) computes
  `total_mass` / `center_of_mass` / `diagonal_inertia` from the component
  masses + geometry (box/cylinder principal inertia + parallel-axis to the CoM,
  excluding the separate pusher pad). These fields are removed from the YAML and
  the builder's `AirframeConfig`.
- **Computed thrust coefficient**: `config.loading._compute_rotor_constants` derives
  `rotor_constant` from `max_thrust_per_rotor_g` + `max_rotor_velocity` (standard
  gravity gram-force -> Newton), so `T = rotor_constant * omega^2` reaches the
  cited static max at `max_rotor_velocity`. `rotor_constant` is removed from every
  airframe/px4 YAML; `Px4BridgeConfig.rotor_constant` becomes a computed field.
- **Physical RPM thrust scaling** for the MN4006 EVO KV380 @ 4S: per-motor max
  1565 gf at `max_rotor_velocity` 525 rad/s, `input_scaling` 475 +
  `zero_position_armed` 50 (controls 0..1 -> 50..525 rad/s). Yields T:W 2.78 and
  hover ~56%, matching the BOM.

The `hexacopter_s550` placeholder adopts the same computed-properties schema
(its `max_thrust_per_rotor_g` back-computed from the prior iris curve) so it
flies identically.

## Consequences

- Fixing a vendor mismatch is a one-line leaf edit, then
  `make generate-drone-usd` + `make build-px4` (the airframe manifest staleness
  check stops a stale image shipping). No second place to update.
- The PX4 `CA_ROTOR` positions move to +/-0.1768 (radius 0.25); the SITL slot
  (`none_x500_v2`/`4019`, ADR 0010) is unchanged. The Generic-Quad-X sign pattern
  is preserved (only the radius scales).
- Isaac Lab's `drone_cfg.py` drops its hand-authored `mass_props` override and
  inherits the USD's computed mass (single source); EventTerm randomization still
  scales it per-env.
- Guard rings and GPS are folded into existing component masses (no new USD
  geometry); Stage 2 X8 stays deferred (ADR 0001). Renames cascade through the
  drone config, USD builder default, and tests; the old
  `quadcopter_500mm.usda` artifact is removed.
- Guards: `tests/test_usd_builder_config.py` checks the computed mass/CoM/inertia;
  `tests/test_config.py` checks the derived `rotor_constant`;
  `tests/test_px4_airframe_generator.py` checks the +/-0.1768 `CA_ROTOR` table.
