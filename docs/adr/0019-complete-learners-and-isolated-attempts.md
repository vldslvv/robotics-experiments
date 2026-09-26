# ADR 0019: Restore complete learners and isolate training attempts

- Status: Accepted
- Recorded: 2026-09-25, consolidating learner and attempt-lifecycle decisions
- Related: [0017](0017-policy-selection-acceptance-and-publication.md)

## Context

Actor-only fine-tuning discarded learning state and did not reliably preserve
useful behavior. Continuing from a phase's final weights could lose its earlier
best checkpoint. Separately, the first multi-seed campaign stalled when building
the next Isaac environment even after Python/CUDA cleanup. Simulator lifetime
therefore could not be treated as interchangeable with learner lifetime.

## Decision

Use one strict complete-learner checkpoint owner. Restore actor, critic, learned
exploration, both observation normalizers, optimizer moments and steps, and
selected-branch progress together. Keep RNG snapshots as evidence, but reseed new
attempts from their configured seeds. There is no partial-learner fallback or
artifact migration. An initializer may come from a failed or interrupted run if
its selected checkpoint, exported actor, contracts, and pinned inputs validate.

Run each attempt in a distinct Isaac process with fresh environments, rollout
buffers, and runner. The coordinator owns sequencing and reports, not a simulator.
Within an attempt, later phases restore the complete selected learner, apply
their learning-rate/schedule/entropy and normalizer-update settings, then reset
environments before collecting new rollouts. Simulator state never crosses
attempt or stage boundaries.

Apply this boundary to ordered curricula, including single-stage invocations.
Stages own complete experiments; labels do not select behavior or compatibility.
Publish an accepted stage before passing its exact pinned learner to the next
stage. Rejection, publication error, runtime failure, or interruption stops the
curriculum. No process resume, implicit curriculum retry, or checkpoint merging
is supported. Each report has one owner and remains immutable after completion.

## Consequences

Restoration preserves learning continuity without pretending to resume a physical
trajectory. Selected learner ancestry may rewind; completed curriculum work never
rewinds and starts at zero for a new invocation. Process isolation costs startup
time but releases simulator resources reliably. Procedural smoke exercises every
phase and attempt boundary without consuming final-test seeds or publishing a
policy. The [RL workflow](../RL_WORKFLOW.md) specifies current checkpoint fields,
compatibility, report ownership, and curriculum behavior.

Historical sources at Git revision `a97389e0325dd54b3190752db3faf2f189509c33`:
`docs/phase_f/F3_TRAINING_DECISION_LOG.md` (actor-only continuation and entry 181)
and `docs/phase_f/PRESS_POLICY_TRAINING_DECISION_LOG.md` (entries 45-47).
Run `20260811T-isolated-attempt-smoke` exercised three distinct worker processes
after `20260811T195121Z.1232485` exposed same-process simulator retention.
