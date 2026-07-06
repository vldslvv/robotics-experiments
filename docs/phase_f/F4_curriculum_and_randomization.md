# F4 -- Contact curriculum + domain randomization (tighten to real size/forces)

Goal served: goal 2 (tighten the task to real light-switch size and forces).
Depends on: F3 (a policy that solves the easy task). Blocks: F5.
No deferred decisions.

Why this exists: the F3 policy solves an oversized/soft switch at close range with
no randomization. This plan tightens the task through config-driven curriculum
stages down to the real `holybro_x500_v2` pusher against real switch geometry and
force, and adds domain randomization so the policy generalizes. All stage/DR values
are config (single source of truth), never hardcoded.

---

## Step 1 -- Curriculum config + pure advance logic

- Action: create `src/robotics_experiments/rl/press/_curriculum.py` (pure). Model a
  stage as parameters + a promotion predicate:
  - `@dataclass(frozen=True) class CurriculumStage` with the tightening knobs from
    ROADMAP1's tables: `switch_size_scale`, `press_force_scale`, `travel_scale`,
    `button_stiffness`, `press_threshold`, `handoff_distance_m`, `wall_tilt_deg`,
    `wall_friction`, `pose_noise` (0 until F5), plus a `promote_when` threshold set
    (min success rate over a window, min first-contact stability).
  - `advance(stage_index, recent_metrics, stages) -> new_stage_index` (pure):
    promote when the window metrics clear the stage's `promote_when`.
- Action: add `PressCurriculumConfig` (ordered `stages` list + window size) to
  `config/types.py` and a `curriculum:` block to
  `configs/experiment/rl/default.yaml`. Author the ordered stages:
  1. stabilization: free hover -> wall approach (no switch) -> stable contact ->
     randomized wall tilt/contact point (the ROADMAP1 stabilization curriculum);
  2. contact: easy (F3) -> real geometry/easy actuation -> nominal geometry/rising
     load -> randomized contact (the ROADMAP1 contact curriculum), ending at the
     real X500 pusher + real switch size/force.
- Test: `tests/test_press_curriculum.py` (pure): promotion fires only when metrics
  clear thresholds; stages advance in order; the final stage is the real geometry.
- Done when: the pure curriculum advances a synthetic metric stream through the
  authored stages.

## Step 2 -- Apply stage parameters at reset (env side)

- Action: in `SwitchPressEnv._reset_idx`, apply the active stage's parameters:
  scale the switch USD size / button stiffness / press threshold, set the wall tilt
  and contact point, set the drone start distance (handoff distance), set wall
  friction. Use Isaac Lab reset-mode events where possible; scale geometry via the
  spawn cfg / prim attributes.
- Action: add a curriculum-manager hook that reads recent per-env success metrics
  and calls the pure `advance()` between iterations (Isaac Lab
  `CurriculumTermCfg`-style, or a manual counter in the env). Keep the decision in
  the pure function; the env only applies the result.
- Done when: training visibly moves the switch from oversized/soft toward real
  size/force as success accrues, and the metrics keep clearing each stage.

## Step 3 -- Domain randomization via EventCfg

- Action: add an `EventCfg` to `SwitchPressEnvCfg` (the template already exists in
  `isaac/runtime/lab/drone_cfg.py`: `randomize_rigid_body_mass` on the body,
  `randomize_actuator_gains` on `pusher_joint`). Extend with:
  - vehicle dynamics: body mass/inertia, motor lag, thrust scale, hover-thrust bias
    (contact stage onward);
  - contact mechanics: pusher stiffness/damping, contact restitution, wall friction
    (stabilization stage onward);
  - task geometry: switch pose, plate size, switch depth, wall tilt, stand-off
    (early curriculum onward);
  - safety harness tether: configured anchor + slack length, zero force while
    slack, spring-damper pull when taut (contact stage onward; model and
    rationale in `docs/SAFETY_HARNESS_PLAN.md` Step 4).
- Action: add `PressRandomizationConfig` to `config/types.py` and a
  `randomization:` block to `configs/experiment/rl/default.yaml`. Each randomized
  parameter carries a record: `name`, `units`, `nominal`, `train_range`,
  `acceptance_range`, `sampling_law`, `stage_introduced` (so DR is
  curriculum-coupled: narrow near nominal first, widen as stages advance, and an
  acceptance range fixed for eval-only).
- Rule: randomization ranges are config, seedable, and reproducible. No range
  literal in Python.
- Test: `tests/test_press_randomization.py` (pure): range resolution per stage;
  acceptance ranges are held fixed; a seed reproduces a sample sequence.
- Done when: DR is active from its configured stage and the ranges widen with the
  curriculum.

## Step 4 -- Retrain / continue training to the real geometry

- Action: run the shared F2b training entrypoint (`make train-rl`, pointed at
  the press task per F3 Step 7) with the curriculum + DR config; continue from
  the F3 checkpoint. Log per-stage metrics + per-term rewards.
- Metrics targets (ROADMAP1 mature/randomized columns): press success >= 85%
  randomized, first-contact stability >= 95%, retreat >= 99%, bounded compression
  overshoot.
- Done when: the policy holds through the final (real X500 pusher + real switch
  size/force) stage under DR at the target metrics.

## Files touched by F4

- add `src/robotics_experiments/rl/press/_curriculum.py`
- edit `.../tasks/switch_press_env.py` (EventCfg, stage apply, curriculum hook)
- edit `config/types.py` (`PressCurriculumConfig`, `PressRandomizationConfig`),
  `configs/experiment/rl/default.yaml`
- add `tests/test_press_curriculum.py`, `tests/test_press_randomization.py`

## Exit gate

The policy trained through the curriculum + DR presses the real-geometry, real-force
switch (nominal `holybro_x500_v2` pusher, real switch size/stiffness) under domain
randomization at the mature/randomized metric targets. This is the "tightened to
real size and forces" milestone and the precondition for F5.
