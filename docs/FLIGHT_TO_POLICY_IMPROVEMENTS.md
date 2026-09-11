# Flight-to-policy improvements

Stage 1 improves experiment traceability and evaluation correctness for
comparing a small number of consecutive policy runs. The
implemented behavior is described below; impacts are expectations, not measured
improvements in policy performance.

| Change | Implemented behavior | Expected impact |
| --- | --- | --- |
| Record experiment inputs | Existing campaign and attempt reports retain the full resolved configuration, pinned input paths and identities, and runtime versions alongside seeds, checkpoint references and results. Attempt initialization is recorded. | Two policy runs can be compared using the recorded training and evaluation inputs, including failed or interrupted runs. |
| Document pooled acceptance metrics | The workflow documents sample pooling, successful-hold windows, each metric's population, and conservative per-flight reference limits. | Acceptance results are easier to interpret without changing the statistical calculation or thresholds. |
| Reject NaN and infinity | Evaluation checks active observations, raw actions before clipping, transition metric inputs and aggregate results; invalid values raise an error with context. | Invalid values cannot silently become clipped actions or valid-looking acceptance evidence. Failed evaluation stops the workflow before publication. |

Report schema 26 adds provenance; the policy schema and comparison fingerprint
remain unchanged. Tests cover provenance serialization and retention without
launching subprocesses, pooled statistics, NaN and both infinities, completed episode
masking, and unavailable successful-hold statistics.

Finite checks add evaluation work and device synchronization. GPU overhead and
policy performance have not been measured for this change. Training does not
collect Git commits, working-tree state or container identity. Keep the checkout
and configuration fixed during a campaign.

Reset mixtures, training settings, acceptance thresholds, comparison mechanics
and sensor noise retain their existing behavior. Later stages are deferred.
The detailed current contract is in [RL workflow](RL_WORKFLOW.md).
