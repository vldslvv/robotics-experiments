# Phase F records and current workflow

The F1/F2/F3 plans and decision logs in this directory retain the inputs,
results, and design decisions of their dated experiments. Their actor layouts,
artifact compatibility rules, and training commands are historical records.
Use [the RL workflow](../RL_WORKFLOW.md) and
[the flight-to-policy report](../FLIGHT_TO_POLICY_IMPROVEMENTS.md) for the
current executable workflow and qualification status.

Current contracts:

- Standalone hover uses observation schema 2 (12 body-relative values).
- Press uses observation schema 3 (35 values), sharing flight and goal-pose
  calculations with the task-neutral RL package. Actions retain four values.
- The outer policy artifact schema is version 1. Older observation schemas
  require retraining; published evidence is never migrated.
- The ordered press curriculum starts with hover, then approach. Later stages
  restore the complete accepted learner from the immediately preceding stage.
- Controller and flight-bundle provenance is pinned before training. Validation
  selects a checkpoint; a fresh disjoint test alone determines acceptance.
- Runtime adapters own task observation packing and completion. The generic
  commanding driver stores opaque progress and handles OFFBOARD transmission.

The previous schema-2 press-hover actor passed Lab acceptance but failed PX4
transfer at a different world heading. Body-relative observation verification
is reported separately from learned-policy qualification; training on the new
schema has not started.
