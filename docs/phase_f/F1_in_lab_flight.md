# F1 -- In-Lab flight actuation (make the DirectRLEnv fly)

Status: COMPLETE. The hover env flies N parallel drones through the actuation
seam and the Step-7 gate (`make test-integration-press-hover-goto`, on the
Isaac Lab image) passes; the config-only gain retune is recorded in
`F1b_velocity_controller_tuning.md`.

Goal served: enabler for goal 1 (an easy task at all needs a flyable Lab env).
Depends on: nothing. Blocks: F2b, F3.
Action-level decision: resolved in ADR 0013. `rl.action.kind=velocity` is the
default; `ctbr` is implemented as the switchable alternative.
Post-F1 refactor: the Protocol/class seam described in Steps 4-5 was replaced by
frozen params dataclasses + pure functions (`build_actuation_params` /
`action_to_rotor_omega` exported by `robotics_experiments.rl.press` and implemented in
`rl/press/actuation.py`, mirrored in `flight.py`); same
math and gates.
Post-F1 follow-up (F2b): the Step-5 placeholder reward/termination/reset module
constants move into config (`rl.press.hover_task`, task-as-config) and the
observation's angular velocity switches to body frame in F2b Steps 1-2, so the
hover task becomes the first YAML-defined task of the task->policy pipeline.
Post-F1 cadence/config alignment: shared Lab scale, action, training, and
`policy_rate_hz` live under `rl`; hover/handoff task values live under
`rl.press`, composed from `configs/experiment/rl/press/default.yaml`. The
default policy rate is 20 Hz. The env schedules exact-average policy intervals
from `sim.physics_hz` (250/20 alternates 13 and 12 physics steps), so there is no
authored fixed physics decimation.

Why this was needed: `workflows/isaac_lab/tasks/drone_hover_env.py` was a no-op
scaffold (`_apply_action` returned `None`, rewards were zero). PX4 SITL cannot run per-env at
RL scale (one lockstep process per vehicle), so training needs an in-Lab inner
loop. This plan makes the env actually fly N parallel drones and puts the action
behind a pluggable seam so the inner-loop modeling choice is a config switch.

---

## Step 1 -- Create the pure `press` package skeleton

- Action: create `src/robotics_experiments/rl/press/__init__.py` (re-export only) and
  `src/robotics_experiments/rl/press/actuation.py`.
- `rl/press/__init__.py`: `__all__` empty for now; add exports as later steps land.
  Import nothing from torch/isaaclab/carb/pxr at module scope.
- Rule: this package is on the core import path. It must import only stdlib +
  numpy. The torch/isaaclab env code lives in `workflows/isaac_lab/tasks/`, not
  here.
- Done when: `python -c "import robotics_experiments.rl.press"` works in a plain
  `.venv` (no Isaac/torch).

## Step 2 -- Add the `press` Hydra config group (scaffold only)

- Action: edit the owning private RL config module.
  - Add `@dataclass class PressActionConfig` with bare annotations:
    `kind: str` (the deferred action level, see Step 6),
    `action_dim: int`,
    `max_body_velocity_mps: float`,
    `max_yaw_rate_radps: float`,
    `max_body_rate_radps: float`,
    `max_tilt_cmd_deg: float`,
    `hover_thrust_fraction: float`,
    `controller_kp_vel: List[float]`, `controller_kd_vel: List[float]`,
    `controller_kp_att: List[float]`, `controller_kp_rate: List[float]`.
    (Superset covering the candidate action levels; the selected `kind` reads the
    subset it needs. Keep unused ones -- they are cheap and keep the switch a
    config change.)
  - Add `@dataclass class RlConfig` with `action: PressActionConfig` and the
    shared Lab-scale fields moved from `IsaacLabConfig` (`num_envs: int`,
    `env_spacing: float`, `episode_length_s: float`, `policy_rate_hz: float`).
    Decimation is derived from the simulation period and policy rate rather
    than authored.
  - Add `rl: RlConfig` to `ExperimentConfig`. Later config refactoring nested
    task-specific values under `rl.press: PressTaskConfig`; config types are
    imported by consumers from `robotics_experiments.config`.
- Action: edit `src/robotics_experiments/config/_schemas.py::_register_schemas`:
  add `cs.store(group="experiment/rl", name="base_press", node=RlConfig)`.
- Action: create `configs/experiment/rl/default.yaml` with concrete values (16
  envs, 4 m spacing, 5 s episode, `policy_rate_hz: 20`, action gains) and compose
  the nested press-task config from
  `configs/experiment/rl/press/default.yaml`.
- Action: edit `configs/base.yaml` defaults to add
  `experiment/rl@press: default`.
- Test: extend `tests/` with a config test asserting
  `load_config("configs/takeoff_quad.yaml").rl.action`
  resolves and `rl.num_envs > 0`.
- Done when: `uv run python -c "from robotics_experiments.config import load_config;
  print(load_config('configs/takeoff_quad.yaml').rl)"` prints the resolved config.

## Step 3 -- Add a vectorized force sink for Isaac Lab

- Action: create `src/robotics_experiments/workflows/isaac_lab/tasks/flight.py`
  (torch/isaaclab leaf).
- Provide pure batched `rotor_omega_to_body_wrench(...)` plus the sole
  side-effecting `apply_body_wrench(...)` sink. The former reduces per-rotor
  body-FLU thrust at the configured offsets, yaw reaction, and drag to an
  equivalent body wrench; the latter writes it to every cloned articulation at
  once. This is the vectorized analogue of the bridge articulation's
  `apply_dynamics` sink.
- Reuse the physical constants already resolved per-airframe: rotor count + spin
  dirs from USD rotor metadata (`px4_bridge/rotor_discovery.py`), `rotor_constant`,
  `max_rotor_velocity`, `linear_drag` from `cfg.px4` (already computed in
  the config package). Do not re-derive thrust constants here.
- Done when: a single-env manual test applies uniform hover thrust and the drone
  holds altitude (not sink/climb) for 2 s in a scripted Lab session.

## Step 4 -- Implement the pluggable actuation seam

- Action: in `rl/press/actuation.py` (pure), define the seam that maps a
  normalized action to per-rotor angular-velocity references, independent of
  torch. The landed contract is frozen `ActuationParams` / `ActuationState`
  plus `build_actuation_params(...)` / `action_to_rotor_omega(...)`, with pure
  branches keyed by action `kind`. It implements the velocity cascade and CTBR
  rate loop using the airframe mixer geometry for allocation.
  - The rotor mix inverts the same allocation the thrust model realizes, so
    feeding its output through `PegasusQuadraticThrustModel.commands_to_dynamics`
    reproduces the commanded wrench.
- Action: in `flight.py` (torch leaf), add a vectorized implementation of the
  selected `kind` that matches the pure reference (batched over envs). The torch
  version is the one the env runs; the numpy version is the test oracle.
- Test: `tests/src/robotics_experiments/rl/press/test_press_actuation.py`
  (pure): for a hover action the model
  returns rotor omegas whose `commands_to_dynamics` net force approximately cancels
  weight; for a +x velocity command the resulting body wrench tilts toward +x. Assert
  on the realized wrench (public thrust-model output), not internals.
- Done when: pure actuation tests pass and match hand-computed hover/ tilt cases.

## Step 5 -- Turn `DroneHoverEnv` into a real flyable env

- Action: edit `workflows/isaac_lab/tasks/drone_hover_env.py`.
  - `_setup_scene`: spawn the training vehicle's configured USD through the
    task-local `_make_drone_articulation_cfg` builder (moved from the original
    runtime-local Lab builder) so the pusher joint and
    rigid caps come along. Keep the ground plane.
  - `action_space` / `observation_space`: set from `cfg.rl.action.action_dim`
    and the F2 observation length (until F2 lands, keep a plain flight-state obs:
    pos err, quat, lin vel, ang vel).
  - `_pre_physics_step`: store the clipped normalized action.
  - `_apply_action`: convert action -> rotor omega through the frozen batched
    params and pure torch function -> body wrench -> Step-3 sink. Run every
    physics substep. The env selects 12/13-step intervals as needed to average
    the configured 20 Hz policy rate over 250 Hz physics.
  - `_get_rewards`: a temporary hover-tracking reward (negative pos/vel error) so
    the env is trainable end-to-end for the Step-7 validation only; F3 replaces it.
  - `_reset_idx`: randomize start pose within a small box around the hover target.
- Rename note: keep task id `Isaac-Drone-Hover-Direct-v0` for now (F3 adds
  `Isaac-Switch-Press-Direct-v0`). Update the README in `isaac/runtime/lab/` to say
  the hover task now flies.
- Done when: the env steps N envs with actions applied and no NaNs over a 5 s
  episode headless.

## Step 6 -- Decision: RL action level

ADR 0013 (`docs/adr/0013-in-lab-rl-action-level.md`) accepts a pluggable
`rl.action.kind` seam with `velocity` as the default and `ctbr` as the
implemented alternative. Selection criteria were: (a) sim-to-real transfer quality
on the flight controller, (b) clean mapping to a PX4 OFFBOARD setpoint at deploy
time, (c) training stability/throughput, and (d) enough authority for contact.

Options (industry approaches to actuating an aerial RL policy without a
flight-controller-in-the-loop):

- `velocity` -- implemented and accepted as the default. RL commands body-frame
  linear velocity + yaw rate; an in-sim cascaded PD/geometric controller tracks it.
  Deploys as PX4 OFFBOARD body-velocity setpoints. Prior art: OmniDrones and
  Aerial Gym expose this as a controller layer. Pros: simplest, most stable,
  cleanest PX4 deploy, and directly supports the hover+goto validation gate. Cons:
  least authority for fast contact; controller must be tuned to the airframe.
- `ctbr` -- implemented switchable alternative. RL commands collective thrust +
  body-rate setpoint; an in-sim body-rate controller closes the loop. Prior art:
  this is the action space that champion-level RL drone work converged on
  (Kaufmann et al., "Swift", Nature 2023; Song et al.) for its transfer quality.
  Deploys as PX4 OFFBOARD attitude/rate setpoints. Pros: proven best sim-to-real,
  good authority. Cons: slightly more deploy integration than velocity.
- `attitude` -- attitude quaternion + collective thrust. Middle ground; deploys as
  PX4 OFFBOARD attitude setpoints. Reserved; selecting it raises until implemented.
- `rotor` -- direct per-rotor thrust/RPM. Max authority; worst flight-controller
  sim-to-real gap; does not use PX4 as the inner loop. Reserved ablation; selecting
  it raises until implemented.
- `residual` -- learn a residual on top of one of the above analytic controllers.
  Robustness win; needs a base controller anyway. Reserved; selecting it raises
  until implemented.
- `hierarchical` -- pretrain + freeze a low-level velocity/rate policy, train the
  contact policy on top. Most moving parts; reserved unless the analytic
  controllers underperform.

Accepted default: `velocity`, because it is the simplest stable training surface,
is the direct target of the hover+goto validation gate, and maps cleanly to PX4
body-velocity OFFBOARD setpoints. `ctbr` remains implemented for later
transfer/contact-authority experiments. Switching between the two is a one-line
`rl.action.kind` config edit.

## Step 7 -- Validation gate: hover + goto in Lab

- Action: add a scripted (non-RL) check that drives the actuation seam directly
  with an analytic controller (no policy): command a hover then a 2 m lateral
  goto; assert the vectorized env holds hover (pos error < 0.1 m, settled) and
  tracks the goto (final pos error < 0.2 m) across all envs. Put it under
  `isaac/runtime/standalone/tests/` (guarded, container-run) with a thin make
  target, mirroring the existing standalone test pattern.
- Done when: hover holds and goto tracks for the chosen `rl.action.kind`,
  confirming the in-Lab inner loop is trustworthy before any RL. This gate is the
  precondition for starting F3.
- Status: complete. `make test-integration-press-hover-goto` runs on the
  `isaac-lab` Docker image, seeds the env for reproducibility, and requires the
  final 1.0 s hover/goto windows to stay within tolerance (`0.1 m` hover,
  `0.2 m` goto). The config-only F1b retune and measured results are recorded in
  `F1b_velocity_controller_tuning.md`.

## Files touched by F1

- add `src/robotics_experiments/rl/press/__init__.py`, `.../rl/press/actuation.py`
- add `src/robotics_experiments/workflows/isaac_lab/tasks/flight.py`
- edit `src/robotics_experiments/workflows/isaac_lab/tasks/drone_hover_env.py`
- edit the private RL config/schema modules
- add `configs/experiment/rl/default.yaml` and the nested
  `configs/experiment/rl/press/default.yaml`; edit `configs/base.yaml`
- move/adopt the old runtime-local Lab config as
  `workflows/isaac_lab/tasks/_drone_cfg.py`
- add the actuation test under
  `tests/src/robotics_experiments/rl/press/`; extend the config and
  `tests/src/robotics_experiments/test_core_imports.py` guards
- add a standalone hover+goto validation script + make target
- record ADR after Step 6 decision
