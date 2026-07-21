# ADR 0004: Enforce package layering and an Isaac-free core import path

- Status: Accepted
- Date: 2026-06-16
- Related: ADR 0002 (feature layering), ADR 0005 (Pegasus vendoring)

## Context

The intended dependency direction is
`config -> domain (drone / environment / simulation / px4_bridge) -> Isaac
extensions`, with simulator imports lazy. But `drone/spawn.py`,
`environment/stage.py`, and `environment/step.py` imported `carb`/`pxr`/`isaacsim`
at module top level, coupling import time to a live simulator and breaking
`import robotics_experiments` outside Isaac. Separately, `frames.py` and the
vendored Pegasus `rotations.py` held parallel quaternion implementations with no
declared owner.

## Decision

Make the core package import cleanly without Isaac, and enforce it:

- A wholly-Isaac leaf module (every function operates on a live USD stage) may
  keep top-level `carb`/`pxr`/`isaacsim` imports, but must (a) carry a docstring
  declaring the exception and (b) stay off the core import path, shielded by a
  lazy package `__init__` (re-export-only or `__getattr__` deferral). The three
  leaf modules above follow the `simulation` package's private Isaac backend precedent.
- `tests/src/robotics_experiments/test_core_imports.py` blocks
  `carb`/`pxr`/`isaacsim` via a
  `_BlockedDependencyFinder` and asserts the leaf modules are not eager-imported,
  so the boundary cannot silently regress.
- `px4_bridge/frames.py` is documented as the single canonical source for
  bridge-side frame math;
  `tests/src/robotics_experiments/px4_bridge/test_frames_pegasus_equivalence.py`
  is the drift
  guard against the vendored `rotations.py`.
- Importable Isaac Lab task code lives under
  `src/robotics_experiments/workflows/isaac_lab/` (lazy-import clean), with the
  Gymnasium `Isaac-...-Direct-v0` naming and a minimal `DirectRLEnv` skeleton --
  so future RL work has a defined home off the core path.

## Consequences

- `python -c "import robotics_experiments"` works outside Isaac; the layering is
  test-enforced, not just conventional.
- One canonical frame library removes the risk of the two implementations drifting.
