# F4 -- Contact curriculum and domain randomization

Current switch geometry, mechanics and asset preparation are defined in
[Generated simulation assets](../ASSET_GENERATION.md). Full curriculum training
and domain randomization remain subsequent work. The validation record below
predates the shared switch models and is retained as historical evidence.

## Validation before unified asset generation

Status: Task implementation and curriculum configuration are available, but
nominal pressing is physically blocked; PPO has not started. Domain randomization
remains planned and depends on an accepted nominal contact policy. See
`configs/train/press_curriculum_quad.yaml` and the RL workflow for the provisional
80/85/95 Lab requirements; these must tighten after the curriculum is trainable.

The 2026-09-25 validation-seed feedback checks passed hover, approach, gentle
contact and easy press-and-retreat. Easy pressing completed all 300 episodes
with one toggle, full retreat, zero unsafe contacts and 99.7% stable first
contacts. Nominal pressing produced 300 unintended pad collisions and no toggles.
The unscaled button protrudes 8.399 mm from the solid base collider; at the
configured 10 mm threshold its collider is fully enclosed by that base.
The existing deployment experiments instead use a 5 mm latch threshold.
Resolve that task-definition mismatch before launching the full curriculum;
do not lower collision acceptance or modify assets to hide it. Exact geometry,
asset digest and runtime evidence are in
`.build/press-curriculum-20260925/nominal-blocker.json` and the adjacent reports.

The shared stage runner and ordered curriculum `stages` provide execution and
checkpoint forwarding. Curriculum configurations live in `configs/train/` and
their complete stage experiments in `configs/train/stage/`.
See [RL workflow](../RL_WORKFLOW.md) for the current
contract. F4 must define complete task experiments rather than add a second
curriculum manager, promotion loop, or workflow framework.

## Task experiments

Define complete Hydra experiments that progressively tighten switch geometry,
button stiffness, press threshold, wall tilt, contact point, stand-off distance,
and friction toward the real X500 pusher and switch. Each experiment owns its
reward, reset distribution, training phases, plateau rule, validation workload,
and acceptance criteria. Curriculum entries contain only names and config paths.
Task code implements the configured behavior; labels never select behavior.

Each accepted stage supplies its complete selected learner to the next stage.
The curriculum reads only the acceptance boolean. It must not inspect contact
measurements or promote a task from rolling training metrics. Fresh environments
apply the next experiment's task parameters; simulator state is not inherited.
The contact and press tasks share the schema-3 observation contract and have
separate physical completion rules and acceptance requirements.

## Domain randomization

Use seedable Isaac Lab reset events and ordinary task code for configured
randomization. Candidate areas include:

- Vehicle mass/inertia, motor response, thrust scale, and hover-thrust bias.
- Pusher stiffness/damping, contact restitution, and wall friction.
- Switch pose, plate dimensions, switch depth, wall tilt, and stand-off distance.
- The configured safety harness described in `docs/SAFETY_HARNESS_PLAN.md`:
  anchor, slack length, and spring-damper pull while taut.

Each experiment authors its own ranges, units, nominal values, sampling law, and
fixed evaluation distribution. There is no `stage_introduced` field or implicit
lookup by curriculum label. Training and evaluation must reproduce their respective
sample sequences from recorded seeds. Controller randomization must respect the
controller-artifact and flight-evidence ownership rules in `AGENTS.md`.

## Verification and training

Test the task's reset parameter application, reproducible sampling, reward
calculations, and independent acceptance measurements before training. Reuse the
shared curriculum sequencing tests; contact behavior does not need another
advancement implementation.

A future F4 campaign may start from a compatible F3 learner through the
curriculum's `initial_learner` producer-report reference. Each stage must pass its
own fresh acceptance test before publication or advancement.
Run it with `make train-rl`, retain per-stage
reports and monitoring, and stop on failed acceptance. Recorded targets for the
mature randomized contact task are press success at least 85%, first-contact
stability at least 95%, retreat at least 99%, and bounded compression overshoot.
These are future contact requirements, not changes to current hover or approach
acceptance.

Completion requires an accepted policy for the real switch geometry and forces
under the authored randomization workload. It is the precondition for F5.
