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
  quad asset to `4019`/`none_x500_v2` instead of the wide Iris geometry.
- [0011](0011-config-driven-px4-airframe-generation.md) (2026-06-30) -- Generate
  airframes from `configs/airframes` at build time; single source for sim + PX4.
  Supersedes the hand-written mechanism of 0008-0010.

## Not captured as ADRs (intentionally)

These `docs/` files stay as-is -- they are living references, plans, or records,
not decision snapshots:

- Living references (AGENTS.md points at them): `COMMAND_ROUTINE_CONTRACT.md`,
  `INTEGRATION_TEST_ARCHITECTURE.md`.
- Plans / roadmaps: `INTEGRATION_TEST_SPEEDUP_PLAN.md`.
- Investigations / logs / scratch: `PX4_TAKEOFF_TEST_INSTABILITY.md` and
  `flight_analysis_00_26_50.md` are distilled into ADR 0003; `notes.txt` and
  `todos.txt` are scratch.
- Supporting hardware detail for ADR 0001: `QUADCOPTER_HARDWARE_BOM.md`,
  `thrust_to_weight.py`, `evaluate_motors_revised.py`.
