# ADR 0017: Separate policy selection, acceptance, and deployment

- Status: Accepted
- Recorded: 2026-09-25, consolidating the policy qualification decisions

## Context

Longer training could destroy an earlier useful actor, and high success rates
could conceal a small number of unsafe episodes. Repeatedly testing selected
checkpoints on the same purported final workload turned qualification into
another tuning loop. A later manual deployment of a validation-failed actor also
showed that the active deployment label cannot serve as proof of acceptance.

## Decision

Evaluate every checkpoint on one fixed validation workload. The evaluator returns
raw named measurements; experiment-owned plateau and selection rules consume
them. Validation-passing candidates take priority, with exact ties retaining the
earlier candidate. Preserve the selected diagnostic candidate even when no actor
qualifies, rather than presenting the latest optimizer state as the best result.

After a completed attempt has an eligible candidate, freeze its exact exported
actor bytes and evaluate them once in a fresh, uncached worker on disjoint test
seeds. Only that raw evaluation determines acceptance. Every required check is
binding, including physical failures, finite trajectories, and measured hover
statistics. Unavailable successful-hold statistics cannot become passing results.
Test failure ends the campaign; it cannot select another actor, trigger another
attempt, or steer refinement.

Automatic publication preserves the tested actor bytes and advances labels only
after acceptance. `best` compares accepted actors within one fingerprint.
`latest` identifies the active operator-selected deployment: an explicit override
may select a strict diagnostic artifact, but preserves its failed or unavailable
qualification evidence and never advances `best`. Producer reports remain
immutable. Routine invocation, deterministic feasibility, Lab acceptance, and PX4
transfer are separate claims with separate evidence.

## Consequences

Training progress, low mean error, or a successful demonstration cannot substitute
for the final test. Reports retain exact selected/tested/published identities and
failure reasons. Lab acceptance qualifies the frozen actor on its recorded
workload, not training repeatability or transfer at every heading. Operators can
inspect unqualified behavior without silently converting deployment into approval.

The [RL workflow](../RL_WORKFLOW.md) owns current selection, metric-population,
acceptance, and publication contracts.

Historical sources at Git revision `a97389e0325dd54b3190752db3faf2f189509c33`:
`docs/phase_f/PRESS_POLICY_TRAINING_DECISION_LOG.md` (entries 43-47),
`docs/phase_f/F3_TRAINING_DECISION_LOG.md` (entries 395-397), and
`docs/FLIGHT_TO_POLICY_IMPROVEMENTS.md` (acceptance and transfer outcomes).
Campaign `20260812T054123Z.124480` retained the explicitly deployed unqualified
candidate; the later Lab pass `20260913T095657Z` did not establish PX4 transfer.
