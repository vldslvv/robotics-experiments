# Isaac Lab Workflow

This directory contains Isaac Lab-specific configs, task definitions, and future
environment code.

Keep direct Isaac Sim scripts in `isaac/runtime/standalone` and Kit extensions
in `isaac/runtime/extensions`.

## Task layout

Importable Isaac Lab task code lives under
`robotics_experiments.workflows.isaac_lab`, not directly in this runtime
directory. Keep this directory for workflow notes, local Isaac Lab runtime
assets/config examples, and files that are meant to be copied or mounted into an
Isaac Lab session.

The first registered task is `Isaac-Drone-Hover-Direct-v0`, implemented as a
minimal DirectRLEnv scaffold in
`robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env`. It establishes
the environment lifecycle, action/observation surfaces, timeout termination, and
Gymnasium registration without implementing thrust dynamics. Future RL dynamics,
rewards, sensors, and contact behavior should extend that task or add sibling
task modules under the same package.

Task-level tunables for these skeletons live in the Hydra `isaac_lab` group
(`configs/experiment/isaac_lab/default.yaml`) with matching typed config fields.

Use guarded registration through
`robotics_experiments.workflows.isaac_lab.register_isaac_lab_tasks()`. The helper
returns `False` on host environments without Isaac Lab/Gymnasium, so core package
imports and normal unit tests stay simulator-free.
