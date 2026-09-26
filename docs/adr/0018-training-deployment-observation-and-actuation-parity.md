# ADR 0018: Match training and deployment observations and actuation

- Status: Accepted
- Recorded: 2026-09-25, consolidating the observation and command-timing decisions
- Supersedes: the controller ownership/frame description in
  [0013](0013-in-lab-rl-action-level.md) and actor-layout details in
  [0014](0014-press-observation-provider-contract.md)

## Context

A policy accepted in Lab failed after deployment at a different world heading.
World-oriented features and quaternion representation exposed unnecessary
coordinate dependence. Other comparisons showed that matching normalized actions
alone was insufficient: changing when body commands become world commands alters
the plant response. A route could also reach its waypoint without entering the
actor's training distribution, so handoff failure needed to remain distinguishable
from policy failure.

## Decision

Share NumPy/Torch body-relative observation calculations. Express goal-minus-drone
position, motion, and gravity in body FLU; represent desired orientation using
goal-to-body rotation columns. Equivalent global translation/yaw and quaternion
sign changes must not alter the corresponding flight features. Keep physical
reset and acceptance frames independent of actor representation. Export the actor
normalizer with the network and never update it during evaluation or deployment.
Version layouts strictly; incompatible actors require retraining, not mutation.

Retain normalized action interpretation in RL and controller dynamics in the
certified controller. Hold policy output between inference ticks. Convert the
retained body velocity using the attitude at each OFFBOARD transmission, then
hold the world command through controller/physics substeps. Match transport,
controller state, allocation, and actuator behavior through replay and sequential
NumPy/Torch checks; a scalar delay alone is not a substitute for that contract.

Keep `{kind: policy}` generic. Runtime adapters own task observations, handoff,
completion, and opaque progress; commanding owns cadence and transmission.
Actor entry must use the applicable training reset envelope. Preparatory feedback
does not become previous-policy-action history.

## Consequences

Schema and timing changes require explicit comparison identities and new
qualification. Observation invariance and deterministic feedback success establish
representation/feasibility properties, not learned-policy transfer. Contact
physics, task acceptance, and PX4 deployment support remain separate contracts.
See the [RL workflow](../RL_WORKFLOW.md) for exact layouts and cadence.

Historical sources at Git revision `a97389e0325dd54b3190752db3faf2f189509c33`:
`docs/phase_f/F3_TRAINING_DECISION_LOG.md` (generic adapters, command latching,
replay, and handoff) and `docs/FLIGHT_TO_POLICY_IMPROVEMENTS.md` (2026-09-13
body-relative observations). Evidence remains in
`.build/body-relative-observations/recorded-reset-verification.json` and
`.build/hover-feasibility/20260913T220033604090Z/feasibility.json`.
