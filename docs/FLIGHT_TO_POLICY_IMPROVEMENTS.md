# Flight-to-policy improvements

Stages 1 and 3 improve experiment traceability, evaluation correctness, and
configuration for a small number of consecutive policy runs. The implemented
behavior is described below; impacts are expectations, not measured improvements
in policy performance.

| Change | Implemented behavior | Expected impact |
| --- | --- | --- |
| Record experiment inputs | Existing campaign and attempt reports retain the full resolved configuration, pinned input paths and identities, and runtime versions alongside seeds, checkpoint references and results. Attempt initialization is recorded. | Two policy runs can be compared using the recorded training and evaluation inputs, including failed or interrupted runs. |
| Document pooled acceptance metrics | The workflow documents sample pooling, successful-hold windows, each metric's population, and conservative per-flight reference limits. | Acceptance results are easier to interpret without changing the statistical calculation or thresholds. |
| Reject NaN and infinity | Evaluation checks active observations, raw actions before clipping, transition metric inputs and aggregate results; invalid values raise an error with context. | Invalid values cannot silently become clipped actions or valid-looking acceptance evidence. Failed evaluation stops the workflow before publication. |
| Author press duration once (stage 3) | The selected Hydra curriculum stage owns episode duration. Press experiment overrides are removed, and both the environment and comparison record read the stage value. The standalone drone-hover task retains its generic duration. | A duration change requires one edit. Current hover/approach durations remain 40/45 seconds, with unchanged comparison identities. |
| Train one candidate policy (stage 3) | Shipped Hydra configs use one training attempt, seed 42. Checkpoints are selected on validation seeds 101/202/303, then the frozen actor is tested once on acceptance seeds 404/505/606. Each workload has 100 episodes per seed. | Campaigns have one training-attempt budget while retaining evaluation across multiple seeds. Results qualify the selected actor; they do not measure training repeatability across initializations. |

Report schema 26 adds provenance; the policy schema and comparison fingerprint
remain unchanged. Tests cover provenance serialization and retention without
launching subprocesses, pooled statistics, NaN and both infinities, completed episode
masking, and unavailable successful-hold statistics.

Finite checks add evaluation work and device synchronization. GPU overhead and
policy performance have not been measured for this change. Training does not
collect Git commits, working-tree state or container identity. Keep the checkout
and configuration fixed during a campaign.

Stage 3 checks cover changing a stage's duration through Hydra without a second
edit, comparison identity following that duration, single-attempt configuration,
and the existing validation-selection and final-test publication rules.
The change does not alter the campaign driver or its stopping rules. PPO phases,
reset mixtures, acceptance thresholds, comparison mechanics, and sensor noise
retain their existing behavior. Controller fitting and policy training remain
separate sequential activities. Later stages are deferred.
The detailed current contract is in [RL workflow](RL_WORKFLOW.md).
