# Phase F implementation status

Phase F now uses explicit, separately named curriculum stages. The first stage
is contact-free approach. There is no automatic stage promotion and no fixed
three-stage enum.

Status as of 2026-07-26:

| Plan | Status | Executable evidence or blocker |
|------|--------|--------------------------------|
| F1 in-Lab flight | Validated | `make test-integration-press-hover-goto` flies the hover/goto gate. |
| F1b controller tuning | Validated | The recorded gains are live and the seeded gate passed repeatedly. |
| F2 observation and handoff | Validated | Pure observation/provider/handoff tests pass; the hover/goto side channel reaches handoff. |
| F2b task-to-policy pipeline | Validated | Hover training, artifact publication, and strict PX4 deployment passed. |
| F3 Stage 1 approach | Lab accepted | Collision-aware run `approach-20260727a`, checkpoint 49 passed selection and independent evaluation at 100% success with zero combined/body/pusher collisions, flight failures, or timeouts. |
| F3 physical contact chain | Implemented | The real pusher/button/latch/lamp/retreat gate uses a test-local no-dones environment and is independent of Stage 1 termination. |
| F3b approach deployment | Dynamics mismatch recorded; failed | The accepted Lab policy contacted the pusher at 0.3807 m and 0.0782 m/s in the live PX4 routine. Keep collision privileged and investigate controller/simulator dynamics instead of changing `press-v1`. |
| Later contact/toggle/retreat stage | Planned | It must be a new named Hydra option and policy namespace initialized actor-only from a compatible checkpoint. |
| F4 randomization | Planned | Add stage-owned randomization without changing the shared 34-by-4 actor contract. |
| F5 synthetic pose | Planned | Only the ground-truth provider exists. |

## Current contracts

- Press task config is composed from
  `configs/experiment/rl/press/approach.yaml`.
- `robotics_experiments.rl.press` publicly exposes approach-specific task and
  deployment records. The removed phase/latch task API has no compatibility
  aliases.
- The Stage 1 observation remains 34 values and the action remains four values.
  Compression, contact, and lamp fields remain packed for checkpoint reuse by
  later stages.
- PhysX body/pusher collision flags are training-only inputs to reward and
  termination. They are absent from policy state, observations, critic inputs,
  and deployment sensing.
- Press artifacts require both `policy_interface: press-v1` and
  `curriculum_stage: approach`. Stage-less and unknown-stage artifacts fail.
- Actor-only initialization may read any validated
  `.build/training/<source-policy>/<run>/logs/model_N.pt`; manual export remains
  restricted to the checkpoint's owning configured namespace.
- Project-owned progress is a frozen dataclass advanced by a stateless
  transition. Mutable classes remain external Isaac, PX4, and torch adapters.
- The generic commanding driver interprets only policy completion/failure and
  retains the opaque final policy state.

Training, selection, evaluation, and mission evidence is recorded in
`F3_TRAINING_DECISION_LOG.md`.
