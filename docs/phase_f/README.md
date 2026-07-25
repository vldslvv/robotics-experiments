# Phase F implementation status

Phase F builds the local switch-contact policy from a flyable Isaac Lab task,
then hardens it from ground-truth geometry to randomized contact physics and a
degraded pose provider. The detailed files in this directory are executable
plans: completed plans record the implementation that landed; planned files
must follow the current repository contracts summarized below.

Status as of 2026-07-22:

| Plan | Status | Executable evidence or blocker |
|------|--------|--------------------------------|
| F1 in-Lab flight | Validated | `make test-integration-press-hover-goto` flies the hover/goto gate. |
| F1b controller tuning | Validated | The recorded gains are live and the seeded gate passed repeatedly. |
| F2 observation and handoff | Validated | Pure observation/provider/handoff tests pass; the hover/goto side channel reaches the handoff. |
| F2b task-to-policy pipeline | Validated | `make train-rl` produced the hover artifact and `make test-integration-press-policy` passed strict deployment through PX4. |
| F3 ground-truth press task | Implemented; acceptance tuning paused | Baseline: 100% toggles, 8% stable contact, 0% retreat. First tune/current `latest`: 0% toggles, 100% stable contact, 0% retreat. The staged second tune is untrained. |
| F3b generic contact-policy deployment | Partially implemented; mission blocked by F3 acceptance | The `press-v1` adapter, live 34-value observation, pure completion transition, and full `{kind: policy}` flight routine exist. The strict mission awaits an accepted newly exported artifact; scripted comparison remains planned. |
| F4 curriculum/randomization | Planned; blocked by F3 | No press curriculum or contact-domain randomization exists. |
| F5 synthetic pose | Planned; blocked by F4 | Only the ground-truth provider exists; no synthetic estimator/provider exists. |

## Current contracts for remaining work

- The config boundary is immutable `ExperimentConfig`. Shared Lab scale,
  action, and training config live under `rl`; press-task config lives under
  `rl.press`, composed from `configs/experiment/rl/press/default.yaml`.
- The default policy rate is 20 Hz. Isaac Lab schedules it from
  `sim.physics_hz` with the shared exact-average cadence while training.
  Exported artifacts own evaluation and deployment cadence; deployment
  transmits at
  `driver.offboard_setpoint_rate_hz`, and ticks drivers at
  `driver.control_rate_hz`. Do not collapse these into one fixed decimation or
  assume they are equal.
- `robotics_experiments.rl` owns task-neutral actuation, hover layout, artifact
  metadata/loading, and normalized-action-to-OFFBOARD conversion.
  `robotics_experiments.rl.press` owns only the press observation, handoff,
  task/provider, and immutable deployment progress. Policy metadata includes
  observation schema/dimension, action contract, and `policy_rate_hz`; these
  values are artifact-owned after export.
- Project-owned evolving state in `runtime`, `commanding`, and `px4_bridge` is a
  frozen dataclass advanced by stateless transitions. The remaining Phase F
  pure state machines, curriculum state, estimator state, and retry progress
  must follow the same model; mutable classes are reserved for external
  Isaac/PX4/torch adapters.
- Consumers import config values from `robotics_experiments.config`,
  task-neutral RL APIs from `robotics_experiments.rl`, press-only APIs from
  `robotics_experiments.rl.press`, and Isaac Lab workflow APIs from
  `robotics_experiments.workflows.isaac_lab`. New public exports are added only
  for production consumers.
- Host tests mirror source paths under `tests/src/robotics_experiments/`; Isaac
  standalone tests live under `isaac/runtime/standalone/tests/`.

The learned-policy mission gate and F4 must wait until F3 produces a policy
that meets the recorded metrics. The generic F3b deployment plumbing is
implemented, but its artifact-dependent integration target must not be treated
as validated until that gate passes. The scripted baseline can then proceed
without adding a task-specific commanding kind.
