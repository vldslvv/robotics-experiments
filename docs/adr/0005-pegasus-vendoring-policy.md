# ADR 0005: Pegasus vendoring / de-vendoring policy

- Status: Accepted
- Date: 2026-06-16
- Related: ADR 0004 (layering / frame math)
- Policy lives in `isaac/runtime/extensions/px4_sitl_brigde/pegasus/PROVENANCE.md`

## Context

~2,100 LOC of the Pegasus Simulator is vendored under
`isaac/runtime/extensions/px4_sitl_brigde/pegasus/` and lint-excluded. The bridge
depends on it, but the version was unpinned and the project's intentional
divergences (HIL `HIL_SENSOR` temperature fix, accumulated barometer drift, GPS
degrees, bridge-side sim-time lockstep) were spread across the tree with no
single record -- so a future re-sync could silently revert a load-bearing fix.

## Decision

Treat the vendored tree as a stripped, unpinned source snapshot governed by a
single provenance document, not a PyPI dependency:

- `pegasus/PROVENANCE.md` is the single source of truth: upstream repo / BSD-3
  license, the synced commit (recorded honestly as `UNKNOWN` rather than guessed),
  the bridge-reachable module set, the dead-at-runtime modules, the full
  divergence index (each with `file:line` and its guarding test), and the re-sync
  procedure.
- Keep only the modules the bridge transitively imports; flag every intentional
  divergence inline pointing at PROVENANCE.md, so a future sync is mechanical.
- The divergences are **load-bearing fixes with guarding unit tests** -- never
  "sync them away."
- `AGENTS.md` carries a short "Pegasus vendoring policy" section pointing here.

Dead-at-runtime code (`px4_launch_tool.py`, autolaunch branches under
`px4_autolaunch=False`) was deliberately kept and documented rather than removed,
to minimize divergence from upstream.

## Consequences

- A re-sync is a documented, auditable procedure instead of a risky bulk merge.
- The bridge's intentional behavior is protected by tests, not by memory.
