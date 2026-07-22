# F3 -- Ground-truth switch-press task and first policy

Status: IMPLEMENTED; POLICY TRAINED; ACCEPTANCE TUNING IN PROGRESS (2026-07-22).

F3 provides the easy, ground-truth contact task that precedes curriculum,
randomization, and synthetic perception. It does not add production
`{kind: press}` behavior, live observation injection, mission retry/abort
logic, or a committed policy binary; those remain F3b/F4/F5 work.

## Implemented task contract

`Isaac-Switch-Press-Direct-v0` is registered independently of
`Isaac-Drone-Hover-Direct-v0`. Both tasks share the private DirectRLEnv flight
base for exact-average cadence, action buffering, rotor actuation, pusher hold,
and drone reset. `make_isaac_lab_env(...)` resolves either task through the
registration-owned config factory, so training and evaluation contain no task
ID branches.

Each press environment contains the real pusher-enabled `holybro_x500_v2`, one
cloned real switch articulation, and a static wall. The source switch's local
outward normal is converted to canonical contact-frame `+X`, and only the
button geometry is enlarged for the easy task. The source plate and joint
physics remain authoritative. The pad contact sensor, pusher joint, and button
joint provide the ground-truth contact, compression, and actuation inputs.

The 34-value observation uses the existing `PressObservation` field order and
the named `PressObservationMask` bits. Pose, contact, and lamp values are marked
valid and fresh with confidence 1. `handoff_mode` is 1 before first contact and
2 afterward.

## Functional transition

`PressTaskState` is a frozen backend-neutral batched state. One
`step_press_task(...)` call advances its latch, reward, termination, and metric
logic. The vector latch has the same threshold/confirmation/release semantics
as `environment.apply_button_sample`; generic contact never earns toggle
reward. DirectRLEnv advances the transition once in `_get_dones`, then caches
the result for rewards and observations.

Success requires a confirmed lamp toggle followed by a stable retreat. Crash,
altitude, relative-envelope, and timeout outcomes remain distinct. Reward
terms cover stability/alive, approach, gentle first contact, compression and
button progress, one-shot toggle, retreat, success, action energy/delta,
contact oscillation, scraping, sustained misalignment, terminal failure, and
timeout. Approach and retreat progress are signed, so moving in the wrong
direction cannot farm reward.

All task geometry, thresholds, weights, limits, seeds, episode counts, and
acceptance targets live under `rl.lab_vehicle` and `rl.press` Hydra config.
The Lab actuator inherits pusher stiffness/damping from the generated USD;
Lab-only spawn height, rigid-body caps, effort, and velocity limits are config
values.

## Contact-chain prerequisite

`make test-integration-press-contact-chain` runs one real switch/lamp pair,
rotates it into the contact frame, performs a slow scripted press, advances the
production N-step latch, verifies exactly one lamp transition, retreats, and
checks stability. The current 2026-07-22 run passed with:

- maximum pusher displacement: 0.01219 m
- maximum button displacement: 0.01047 m
- maximum contact force: 10.432 N
- lamp transitions: 1; final intensity: 5000
- stable retreat beyond 0.35 m: true; maximum tilt: 0.448 rad

The passing evidence is stored under
`.build/docker/integration/test-integration-press-contact-chain.20260722T160119Z.248449`.

Switch/lamp scene spawning is atomic and fail-closed: both assets are
preflighted, every configured pair is validated, partial canonical prims are
removed on failure, and a wrong pair count raises. Config loading rejects
non-positive counts/confirmation/retry values and malformed or unordered
bounds.

## Training, artifacts, and evaluation

Use the shared training entrypoint with the press selection:

```console
make train-rl HYDRA_CONFIG=configs/train_press_quad.yaml
make eval-press
```

Each task config owns `observation_schema_version` and `observation_space`, and
artifact metadata is exported from those fields. `load_vector_policy(...)` is
the generic validated observation-vector loader; `build_policy_step_fn(...)`
remains the hover-specific `VehicleState` to OFFBOARD composer.

The evaluator loads `.build/training/press/latest` and runs 100 deterministic
episodes for each of three configured non-training seeds. It reports reward
term means plus press success at latch confirmation, retreat rate among toggled
episodes, first-contact stability among contacted episodes, median
time-to-toggle, and peak compression overshoot. It exits nonzero unless press
success is at least 80%, first-contact stability at least 85%, and retreat is at
least 95%.

The five-iteration/two-environment smoke completed on 2026-07-21. The full
training run then completed 300 PPO updates with 49,152 environments in
3,242.884 seconds. It normally held 88-95% GPU compute and 22.4-23.1 of
24.6 GB VRAM. The final printed training sample reported mean reward 2.15 and
mean episode length 199/200; those are optimization diagnostics, not acceptance
metrics. That initial artifact is
`.build/training/press/20260721T231700Z.19`.

The repaired deterministic metric gate completed all 300 episodes on
2026-07-22. The baseline policy confirmed a toggle in 100% of episodes, but
first-contact stability was 8% and stable retreat was 0%. It therefore failed
the 85% contact and 95% retreat gates. A first tuned run then completed at
`.build/training/press/20260722T150005Z.19`; it reached 100% stable first
contact but 0% toggles and consequently 0% retreat. This exchanged aggressive
pressing for a gentle-touch-and-wait local optimum and also failed acceptance.
`.build/training/press/latest` currently points to that first tuned artifact.

The second-cycle settings are staged with 2.0 m/s action authority but
deliberately untrained. Its attempted
2026-07-22 launch was stopped after about 19 seconds at the user's request,
before it produced any checkpoint. F3 is therefore not yet validated. Resume
the exact staged command and then rerun the complete three-seed evaluation as
documented in the decision log. Evaluate every new artifact with:

```console
make -o generate-drone-usd eval-press
```

The evaluator lifecycle diagnosis, baseline reward-term means, and current
evidence-based tuning decisions are recorded in the separate decision log.

The chronological parameter-tuning decisions, failed trials, GPU measurements,
and RL justifications are recorded separately in
[F3_TRAINING_DECISION_LOG.md](F3_TRAINING_DECISION_LOG.md).

Implementation base revision: `c6ef12cadcedaf28c50bebdf3535c0f927b34ab9`.
