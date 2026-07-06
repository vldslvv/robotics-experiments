# ADR 0014: Press observation contract + pose-provider seam

- Status: Accepted
- Date: 2026-07-05
- Related: ADR 0013 (in-Lab RL action level), ADR 0006 (command/routine contract)
- Source: `docs/phase_f/F2_observation_and_handoff.md`,
  `docs/phase_f/F2b_task_to_policy_pipeline.md`

## Context

The press RL policy needs an input that survives three planned data-source
swaps (ground-truth pose -> synthetic noise in F5 -> real perception in Phase
G) and a train->deploy hop (Isaac Lab tensors in training,
`VehicleState`/bridge state at deploy). Feeding raw sensors or simulator
internals to the policy would couple it to whichever source existed first and
make every swap a retrain-plus-rewire.

## Decision

Three load-bearing seams in `robotics_experiments.rl.press` (pure, core import
path):

1. **Versioned observation layouts** (`rl/press/observation.py`). Each layout is a
   documented field order packed by a pure function and stamped with a schema
   version (`HOVER_OBSERVATION_SCHEMA_VERSION`, `OBSERVATION_SCHEMA_VERSION`).
   The version is recorded in the exported policy artifact's metadata and
   validated at `load_policy` time. PARITY INVARIANT: the same packing function
   is fed from Lab tensors in training and from `VehicleState` (+ provider
   state) at deploy; a field the deploy side cannot assemble is forbidden.
   The press layout's task-relative pose lives in the **contact frame**
   (x = wall-normal press axis, y/z = plate tangents; the switch prim's local
   +X is the outward plate normal), computed by the shared
   `world_switch_to_contact_frame` so every provider agrees on the frame, with
   the relative quaternion canonicalized to non-negative w.

2. **Pose-provider seam** (`rl.press` facade; implementation in
   `rl/press/_provider.py`). `AbstractPoseProvider.sample
   (sim_time_s) -> PoseSample` (pos, quat, confidence, age, valid) is the single
   swap point for the switch pose source. The provider owns only the switch
   pose estimate; drone state comes from the env/bridge. Swapping providers
   must not change the observation shape ("policy API change: none").
   Perception uncertainty is explicit policy input
   (`pose_confidence`/`pose_age_s`/`detection_valid`), never hidden failure.

3. **Handoff gate** (`rl.press` facade; implementation in
   `rl/press/_handoff.py`). `handoff_ready(obs, gate)` is a pure
   predicate over the observation with config-built thresholds
   (`rl.press.handoff`), returning the first violated check as a typed reason. It
   mirrors the command layer's pure-invariant style but is a separate predicate
   (it gates *entering* contact; the envelope gates a flight in progress).

`GroundTruthPoseProvider` (`rl/press/_isaac_provider.py`) is an Isaac-only leaf
shielded by the lazy `rl/press/__init__.__getattr__`, mirroring
`environment/stage.py`; `tests/test_core_imports.py` guards the boundary.

## Consequences

- F5's synthetic estimator and Phase G's sensor providers implement `sample()`
  and change nothing else; the F3 press env consumes `PressObservation` as its
  policy input via `to_vector`.
- Layout changes are conscious acts: tests pin the field order and the schema
  version, and a stale policy artifact is refused at load rather than
  misinterpreted.
- The hover pipeline (F2b) uses the same mechanism with its own smaller layout,
  so the artifact/loader contract is identical for every task.
