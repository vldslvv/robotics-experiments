# Feature Design

> The layering decision is recorded as ADR 0002 (`docs/adr/`); this doc is the
> living convention behind it.

This repository splits each Isaac-facing feature into shared core behavior,
Kit extension wiring, and standalone workflow coverage. Keep changes scoped to
the layer that owns the behavior being changed.

## Layer Responsibilities

- Core packages under `src/robotics_experiments/` own business logic,
  validation, and reusable APIs. Public APIs should accept explicit config
  objects when practical and keep Isaac-only imports lazy or inside private
  backend modules.
- Core public entrypoints should also avoid loading heavy runtime config
  machinery during lightweight imports. Type-only config references should
  come from lightweight schema/type paths; Hydra composition belongs behind
  explicit config-loading calls.
- Isaac Kit extensions under `isaac/runtime/extensions/` own runtime wiring:
  event subscriptions, active-stage lookup, lifecycle state, and calls into the
  core package. Extension-local config modules should adapt the shared
  `robotics_experiments.config` Hydra config to the feature-specific config
  object.
- Standalone workflows under `isaac/runtime/standalone/` own executable
  integration coverage. They may start `SimulationApp`, load real scenes,
  reset/recompose Hydra config for tests, and assert externally visible
  simulator state.

## Simulation Settings Example

- Core behavior lives in `src/robotics_experiments/simulation`. The public
  `apply_settings(stage, settings=None)` entrypoint is safe to import from
  normal Python tooling and lazily loads the Isaac backend only when settings
  are applied.
- Default setting resolution should remain separate from import-time module
  setup: importing the entrypoint should not require Hydra, and only applying
  settings without an explicit config should resolve the shared experiment
  config.
- The private Isaac backend owns USD/carb/PhysX mutation: finding or creating
  the physics scene, validating gravity and positive rates, applying PhysX
  attributes, setting the Kit render-rate limit, and logging the applied values.
- The fleet runtime owns settings application: the live `experiment_runtime`
  shell rebinds `apply_settings(state=...)` on stage-open (and the
  standalone runner calls it after opening the scene), which delegates to the
  core `apply_settings(stage, settings)` API. (This replaced the former
  standalone `sim_settings` extension.) It does not duplicate USD mutation logic.
- The standalone test starts a real headless Isaac workflow, opens a real USD
  scene, calls the production core API, and verifies physics scene attributes
  plus Kit run-loop settings.

## Extraction Rules

- Move logic from an extension into core when it changes simulator behavior,
  validates configuration, mutates USD/PhysX/carb state, or will be reused by
  standalone, Lab, or another extension workflow.
- Keep logic in an extension when it is specific to Kit lifecycle, timeline or
  stage events, ROS node ownership, UI, or subscription cleanup.
- Keep logic in standalone workflows when it exists only to launch a real
  runtime, choose test scenes, report test progress, or assert integration
  outcomes.
- Do not add a shared abstraction just to remove a small import adapter. Extract
  only when it removes meaningful duplication or prevents behavior drift.

## Error Handling

- Invalid configuration must raise an error in the core layer.
- Recoverable runtime fallbacks should log a warning near the runtime boundary.
- Extensions should avoid swallowing core errors silently; a failed feature
  setup should be visible in the Isaac logs or test output.
