# ADR 0016: Measured flight evidence and certified controllers

- Status: Accepted
- Recorded: 2026-09-25, consolidating the alignment and flight-to-policy decisions
- Related: [0013](0013-in-lab-rl-action-level.md)

## Context

Policy failures initially prompted controller-delay and gain sweeps inside the
learning workflow. A motion fit could pass on one capture yet fail on a fresh
reference, controller replay, or live hover. Training against that fit mixed
policy quality with an unqualified plant. Repeated hover also showed that pose
alone was insufficient reset evidence: measured velocity belongs to the same
physical sample. An accepted controller is useful only with the exact evidence
and control modes that qualified it.

## Decision

Separate three owners. `flight_parameters` publishes task-neutral hover and
lossless controller-response evidence. `alignment` searches declarative
controller candidates and qualifies them. `controller` owns the shared settings,
strict artifact, and normalized NumPy reference consumed by both alignment and
RL; RL never imports alignment or reopens its candidate population.

Assign complete independent recordings to fitting, validation, and acceptance.
Derive hover limits and paired pose/motion resets from fitting flights only.
Search on fitting responses, select on validation, and freeze the controller
before acceptance. Require motion quality, actuator audit, controller-boundary
replay, and settings-bound live checks rather than treating a good aggregate
score as certification. Deterministic feedback feasibility checks distinguish
controller/action/reset problems from PPO learning problems.

Publish immutable schema-1 bundles. Keep candidate search evidence separate from
the compact controller contract. Consumers resolve selectors once, pin canonical
producer identities and digests, and reference producer files directly. Training
embeds the certified settings, control modes, and provenance before PPO. Changes
to measured semantics or comparison inputs require a new fingerprint.

## Consequences

Poor-quality but valid results remain diagnostic evidence; they cannot publish
as certified controllers or enter training. A failed fresh qualification is not
repaired by migrating evidence, relaxing acceptance, or tuning the policy around
it. Capture and qualification cost more than a single convenient reference but
make downstream failures interpretable. Certification remains limited to the
recorded workload and does not establish policy or hardware qualification.

Current contracts: [alignment](../SIM_TO_SIM_ALIGNMENT.md) and
[flight parameters](../FLIGHT_PARAMETERS.md).

Historical sources at Git revision `a97389e0325dd54b3190752db3faf2f189509c33`:
`docs/phase_f/F3_TRAINING_DECISION_LOG.md` (alignment and reproduction entries)
and `docs/FLIGHT_TO_POLICY_IMPROVEMENTS.md` (stages 4-5).
Decisive producer runs: flight bundle `20260912T102640Z` and controller
`20260912T211745Z`; their recorded qualification is historical evidence.
