# ADR 0013: In-Lab RL action level (velocity default + ctbr) and OFFBOARD deploy

- Status: Accepted
- Date: 2026-07-01
- Related: ADR 0006 (command/routine contract layer), ADR 0002 (feature layering)
- Source: `docs/phase_f/F1_in_lab_flight.md` (the F1 plan; Step 6 deferred decision)

## Context

Isaac Lab RL cannot run one PX4 SITL lockstep process per env at training scale,
so the DirectRLEnv needs an in-Lab inner loop that maps a normalized policy action
to per-rotor forces. F1 Step 6 deferred the action level (the layer the RL policy
commands): `velocity`, `ctbr`, `attitude`, `rotor`, `residual`, or `hierarchical`.
The choice trades sim-to-real transfer, clean mapping to a PX4 OFFBOARD setpoint at
deploy, training stability, and contact authority. The repo's commanding layer
streamed only position setpoints (`SET_POSITION_TARGET_GLOBAL_INT`), so the deploy
mapping was net-new either way.

## Decision

Build a pluggable actuation seam (`rl.action.kind`) and ship **both** reference
levels, defaulting to `velocity`:

- `velocity` (**default**): cascaded geometric controller tracking a yaw-frame
  body velocity + yaw rate. Simplest/most training-stable, directly powers the
  Step-7 hover+goto validation gate (it *is* a position/velocity cascade), and
  deploys onto the existing position-setpoint stream extended with velocity fields.
- `ctbr`: collective thrust + body-rate setpoint (the inner loop of `velocity`).
  The best-documented sim-to-real level (Swift/Kaufmann); a one-line `kind` switch
  once training needs its transfer quality / contact authority.
- `attitude` / `rotor` / `residual` / `hierarchical` remain seam options that raise
  if selected until implemented.

The pure single-env reference models are exported through `robotics_experiments.rl.press`
and live in `rl/press/actuation.py` (numpy, the
test oracle); the batched torch versions the env runs live in
`workflows/isaac_lab/tasks/flight.py`. Both invert the PX4 bridge's
`PegasusQuadraticThrustModel` allocation, so an action reproduces the commanded
wrench. F1 also added the OFFBOARD deploy plumbing (both levels): `velocity` ->
`SET_POSITION_TARGET_LOCAL_NED` (body velocity), `ctbr` -> `SET_ATTITUDE_TARGET`
(thrust + body rates), with a pure `robotics_experiments.rl.press.press_action_to_offboard`
mapper. The mission-level press routine step / `PressDriver` stays in F3 (only the
port-level plumbing + mapper land in F1).

## Consequences

- Switching the RL action level is a config edit, not a code change; the Step-7
  gate validates the chosen level in Lab before any RL.
- The numpy reference is the host-tested oracle; the torch env path is validated in
  the container by the hover+goto gate (`make test-integration-press-hover-goto`).
- The `isaac_lab` Hydra group + `IsaacLabConfig` are retired; the Lab-scale
  tunables and the action level live in the `rl` group (single source).
- The deploy setpoint types are available for F3's press behavior, which layers a
  `PressDriver` over the same command port without new wire plumbing.
