# Architecture Decision Records

One decision per file, numbered in the chronological order the decisions were
made. Each ADR is short (Context / Decision / Consequences) and links to the
fuller design doc under `docs/` where one exists.

## Platform

- [0001](0001-staged-x500-platform.md) (2026-05-17) -- Staged Holybro X500 V2
  platform: ship a quadcopter first, upgrade to X8 coaxial only on demonstrated need.

## Runtime Design

- [0002](0002-three-layer-feature-design.md) (2026-05-20) -- Split each feature
  into core / Kit extension / standalone workflow, each owning one concern.

## Simulation Stability

- [0003](0003-px4-takeoff-instability-mitigation.md) (2026-06-16) -- Separate
  real takeoff instability from PX4 infra flakes and record the mitigations.

## Architecture

- [0004](0004-enforce-package-layering.md) (2026-06-16) -- Enforce the dependency
  direction; the core package must import without Isaac, test-guarded.
- [0005](0005-pegasus-vendoring-policy.md) (2026-06-16) -- Govern the vendored
  Pegasus snapshot via a single PROVENANCE doc; divergences are load-bearing fixes.

## Commanding

- [0006](0006-command-routine-contract-layer.md) (2026-06-22) -- Add the
  `commanding` layer as a pure-function contract mirroring the PX4 bridge contract.

## Testing

- [0007](0007-tests-assert-public-behavior.md) (2026-06-28) -- Assert public
  behavior over internals; enable Ruff `SLF001` repo-wide to lock it in.

## PX4 airframes

- [0008](0008-match-px4-sitl-airframe-to-drone-config.md) (2026-06-29) -- Derive
  the SITL airframe per-vehicle from config (replaces the global `none_iris`).
- [0009](0009-hexacopter-airframe-rotor-geometry.md) (2026-06-29) -- Rewrite the
  hexa airframe `CA_ROTOR` to match the USD asset (fixes the takeoff tip-over).
- [0010](0010-quad-asset-uses-x500-airframe.md) (2026-06-30) -- Bind the square-X
  quad asset to `4019`/`none_x500_v2` instead of the wide Iris geometry (numeric
  ID later superseded by 0015).
- [0011](0011-config-driven-px4-airframe-generation.md) (2026-06-30) -- Generate
  airframes from `configs/airframes` at build time; single source for sim + PX4.
  Supersedes the hand-written mechanism of 0008-0010.
- [0012](0012-hardware-faithful-x500-and-computed-airframe-properties.md)
  (2026-06-30) -- Rebuild the quad as hardware-faithful `holybro_x500_v2`; author
  only cited per-component leaves and compute body mass/inertia + `rotor_constant`
  from them.
- [0015](0015-px4-1.17-and-custom-airframe-ids.md) (2026-07-17) -- Upgrade PX4
  SITL to 1.17.0 and move generated airframes to PX4's reserved custom model ID
  range, superseding the earlier numeric bindings.

## RL press policy

- [0013](0013-in-lab-rl-action-level.md) (2026-07-01) -- In-Lab RL action level:
  pluggable `rl.action.kind` seam shipping `velocity` (default) + `ctbr`, plus
  the PX4 OFFBOARD deploy mapping; retires the `isaac_lab` group for `rl`.
- [0014](0014-press-observation-provider-contract.md) (2026-07-05) -- Press
  observation contract: versioned pure layouts with a train/deploy parity
  invariant, the `AbstractPoseProvider` swap seam (+ contact frame convention),
  and the config-built handoff gate.

## Not captured as ADRs (intentionally)

These `docs/` files stay as-is -- they are living references, plans, or records,
not decision snapshots:

- Living references (AGENTS.md points at them): `COMMAND_ROUTINE_CONTRACT.md`,
  `INTEGRATION_TEST_ARCHITECTURE.md`.
- Canonical roadmap and supporting hardware detail for ADR 0001:
  `LIGHT_SWITCH_MULTIROTOR_ROADMAP.md`; historical source docs
  `archive/QUADCOPTER_PLAN_REVISED.md` and
  `archive/HOLYBRO_X8_STAGED_BOM_v2.md`; calculators `thrust_to_weight.py` and
  `evaluate_motors_revised.py`.
