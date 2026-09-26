# ADR 0002: Three-layer feature design (core / Kit extension / standalone)

- Status: Accepted (living convention)
- Date: 2026-05-20
- Related: ADR 0004 (layering enforcement), ADR 0006 (commanding layer)
- Current convention: [development workflow](../DEVELOPMENT_WORKFLOW.md#isaac-facing-feature-design)

## Context

Isaac-facing features risked piling business logic, Kit lifecycle wiring, and
test scaffolding into the same place (typically a Kit extension or a standalone
script), making behavior hard to reuse and easy to drift between the live and
test paths.

## Decision

Split every Isaac-facing feature into three layers, each owning one concern:

- **Core** (`src/robotics_experiments/`) -- business logic, validation, reusable
  APIs. Accepts explicit config objects; keeps Isaac-only imports lazy or behind
  private backend modules; avoids loading heavy Hydra machinery at import time.
- **Kit extension** (`isaac/runtime/extensions/`) -- runtime wiring only: event
  subscriptions, active-stage lookup, lifecycle state, subscription cleanup, and
  calls into core. A small extension-local config adapter maps the shared Hydra
  config to the feature's config object.
- **Standalone workflow** (`isaac/runtime/standalone/`) -- executable integration
  coverage: starts `SimulationApp`, loads real scenes, asserts simulator state.

Extraction rule: move logic into core when it changes simulator behavior,
validates config, mutates USD/PhysX/carb, or will be reused; keep it in the
extension when it is Kit lifecycle / events / UI; do not add a shared abstraction
just to remove a small import adapter. Error policy: invalid config raises in
core; recoverable fallbacks log a warning at the runtime boundary; extensions
must not silently swallow core errors.

## Consequences

- Cross-component composition later moved *above* the extensions into the
  host-agnostic frozen `FleetState` lifecycle transitions, with extensions kept
  single-responsibility -- a direct extension of this layering (see AGENTS.md).
- The dependency direction this implies is enforced by a regression guard in
  ADR 0004.
