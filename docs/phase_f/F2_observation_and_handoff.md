# F2 -- Observation Interface + pose-provider seam + handoff gate

Goal served: goal 1 (a stable policy input for the easy task) and the substrate for
goals 3/4. Depends on: nothing (parallel with F1). Blocks: F3.
No deferred decisions.

Why this exists: the policy must observe a normalized, task-relative observation --
never raw sensors and never simulator internals -- so the data source can change
(ground truth -> synthetic noise in F5 -> sensors in Phase G) without changing the
policy input. This is the same pattern as the bridge/command seam (pure `decide`
over `AbstractCommandPort`); see `docs/COMMAND_ROUTINE_CONTRACT.md`.

---

## Step 1 -- Define the pure observation dataclass

- Action: create `src/robotics_experiments/press/_observation.py` (pure; numpy +
  stdlib only). Define `@dataclass(frozen=True) class PressObservation` with the
  task-relative fields (units/frames in docstrings):
  - `relative_switch_pos_m: np.ndarray` (3,), in the contact frame (Step 2)
  - `relative_switch_quat_xyzw: np.ndarray` (4,)
  - `pose_confidence: float`, `pose_age_s: float`, `detection_valid: bool`
  - `body_lin_vel_mps: np.ndarray` (3,), `body_ang_vel_radps: np.ndarray` (3,)
  - `attitude_quat_xyzw: np.ndarray` (4,)
  - `pusher_compression_m: float`, `pusher_compression_rate_mps: float`
  - `contact_valid: bool`, `contact_flag: bool`
  - `lamp_state: int` (-1 unknown / 0 off / 1 on), `lamp_confidence: float`,
    `lamp_age_s: float`
  - `previous_action: np.ndarray` (action_dim,)
  - `handoff_mode: int` (0 nav / 1 precontact / 2 contact)
  - `estimator_health: float`, `obs_mask: int`
- Add a pure `to_vector(obs, layout) -> np.ndarray` that flattens to a fixed-length
  float vector in a documented, versioned order (`OBSERVATION_SCHEMA_VERSION`
  constant). The env leaf packs this into a torch tensor; keep the ordering here so
  it is testable without torch.
- `__all__`: `PressObservation`, `to_vector`, `OBSERVATION_SCHEMA_VERSION`.
- Rationale for `pose_confidence`/`pose_age_s`/`detection_valid` being mandatory:
  they turn perception uncertainty into explicit control state, which is what lets
  F5 and Phase G swap providers without touching the policy. Note: times are
  seconds (repo convention: sim-time in seconds/`us`, not ms) rather than ROADMAP1's
  ms.
- Test: `tests/test_press_observation.py` (pure): `to_vector` length equals the
  declared layout length; field order is stable; changing a field changes the
  expected slot.

## Step 2 -- Define the contact frame + transform helpers

- Action: in `press/_observation.py`, document and implement the contact frame:
  x = wall-normal press axis (drone toward switch surface), y = plate horizontal
  tangent, z = plate vertical tangent. Add a pure
  `world_switch_to_contact_frame(drone_pose, switch_pose) -> (rel_pos, rel_quat)`
  used by every provider so all providers agree on the frame.
- Reuse existing frame utilities where possible
  (`px4_bridge/frames.py`) rather than re-deriving quaternion math.
- Test: pure transform round-trips a known drone/switch geometry to the expected
  relative pose (switch straight ahead at 0.2 m -> `rel_pos = (0.2, 0, 0)`).

## Step 3 -- Define the pose-provider seam

- Action: create `src/robotics_experiments/press/_provider.py` (pure). Define
  `class AbstractPoseProvider(Protocol)` with:
  - `sample(sim_time_s) -> PoseSample` where
    `@dataclass(frozen=True) class PoseSample` carries
    `world_switch_pos_m`, `world_switch_quat_xyzw`, `confidence`, `age_s`,
    `valid`.
  - The provider owns only the switch pose estimate; drone state comes from the
    env/bridge directly (it is not "estimated" in Phase F).
- This is the single swap point. Phase F ships `GroundTruthPoseProvider` (Step 4)
  and `SyntheticPoseProvider` (F5). Phase G adds sensor/fusion providers behind the
  identical `sample()` signature. INVARIANT to assert in tests and state in the
  docstring: swapping the provider must not change `PressObservation`'s shape or the
  `to_vector` output length ("policy API change: none").
- `__all__`: `AbstractPoseProvider`, `PoseSample`.

## Step 4 -- Ground-truth provider (Isaac leaf)

- Action: create `src/robotics_experiments/press/_isaac_provider.py` -- a guarded
  Isaac leaf (imports isaacsim/pxr; off the core path; re-exported lazily only when
  Isaac is present, mirroring `environment/__init__.py __getattr__`). Implement
  `GroundTruthPoseProvider` that reads the switch prim world pose from the live
  stage/articulation and returns a `PoseSample` with `confidence=1.0`, `age_s=0.0`,
  `valid=True`.
- Keep `press/__init__.py` re-export-only and lazy for this symbol so
  `import robotics_experiments.press` stays Isaac-free.
- Test: `tests/test_core_imports.py` asserts `press` imports with Isaac absent and
  that `_isaac_provider` is not eager-imported.

## Step 5 -- Handoff gate (pure predicate + config)

- Action: create `src/robotics_experiments/press/_handoff.py` (pure). Implement
  `handoff_ready(obs: PressObservation, gate: HandoffGate) -> HandoffResult` where
  the result is either ready or a reason enum (`NOT_VALID`, `LOW_CONFIDENCE`,
  `STALE`, `TOO_FAR`, `LATERAL`, `WALL_NORMAL_SPEED`, `LATERAL_SPEED`,
  `ANGULAR_SPEED`, `ESTIMATOR`). Checks, in order: `detection_valid`, confidence >=
  threshold, age <= threshold, wall-normal range <= threshold, lateral alignment
  error <= threshold, wall-normal speed <= threshold, lateral speed <= threshold,
  angular-rate magnitude <= threshold, estimator_health >= threshold.
- `HandoffGate` is a frozen dataclass built from config (no magic numbers in code).
- Action: append a `PressHandoffConfig` to `config/_types.py` and a
  `handoff:` block to `configs/experiment/press/default.yaml` with the initial
  thresholds from ROADMAP1's handoff table (`pose_confidence_min`, `pose_age_max_s`,
  `switch_range_max_m`, `lateral_align_max_m`, `wall_normal_speed_max_mps`,
  `lateral_speed_max_mps`, `angular_speed_max_radps`, `estimator_health_min`). Add
  `handoff: PressHandoffConfig` to `PressConfig`.
- Reuse note: this mirrors the command layer's continuous envelope invariants
  (`commanding/_routine.py`: `tilt_below`, `lateral_drift_below`, ...). Keep the
  same pure-predicate style; do not duplicate the envelope-gate logic, but a
  separate handoff predicate is warranted (different thresholds, different frame).
- Test: `tests/test_press_handoff.py` (pure): a calm, close, confident, fresh obs
  is ready; each individual violation returns its specific reason; boundary values
  behave.

## Step 6 -- Wire the observation into the env

- Action: in the env leaf (`drone_hover_env.py` for now; F3's press env after), add
  `_assemble_observation()` that builds a `PressObservation` from the provider's
  `PoseSample` + the drone's sim state (vel, ang vel, attitude), packs it with
  `to_vector`, and returns `{"policy": tensor}`. Set the env's
  `observation_space` from the layout length.
- Contact/pusher/lamp fields are placeholders here (F3 fills them from the pusher
  joint + contact reporter + latch); set `contact_flag=False`,
  `pusher_compression_m` from the pusher joint position if present, else 0.
- Done when: the env emits a fixed-length observation whose length equals
  `to_vector` layout, and a ground-truth run shows `relative_switch_pos_m` shrinking
  as the drone approaches a placed switch.

## Files touched by F2

- add `src/robotics_experiments/press/_observation.py`, `.../_provider.py`,
  `.../_handoff.py`, `.../_isaac_provider.py`
- edit `src/robotics_experiments/press/__init__.py` (lazy re-exports)
- edit `src/robotics_experiments/config/_types.py` (`PressHandoffConfig`,
  `PressObservationConfig` if needed), `configs/experiment/press/default.yaml`
- edit the env leaf to assemble/pack the observation
- add `tests/test_press_observation.py`, `tests/test_press_handoff.py`; extend
  `tests/test_core_imports.py`
- record an ADR for the observation/provider contract (it is a load-bearing seam)

## Exit gate

Pure observation + handoff tests pass; `press` imports Isaac-free; a ground-truth
run produces a shrinking relative-switch-pose observation and the handoff predicate
flips to ready inside the gate. This plus F1's gate are the preconditions for F3.
