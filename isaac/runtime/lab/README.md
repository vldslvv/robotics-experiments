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

The first registered task is `Isaac-Drone-Hover-Direct-v0`, a flyable DirectRLEnv
in `robotics_experiments.workflows.isaac_lab.tasks.drone_hover_env`. It flies N
parallel drones through the pluggable in-Lab actuation seam
(`robotics_experiments.press`), which maps a normalized RL action to per-rotor
forces via the batched torch model in
`robotics_experiments.workflows.isaac_lab.tasks._flight` (no PX4 lockstep per
env). Future RL rewards, sensors, and contact behavior should extend that task or
add sibling task modules under the same package.

Task-level tunables (Lab-scale + the RL action level) live in the Hydra `press`
group (`configs/experiment/press/default.yaml`) with matching typed config
fields. The env spawns from the config-driven `ArticulationCfg` builder in
`robotics_experiments.workflows.isaac_lab.tasks._drone_cfg` (reads
`drone.usd_path`), not a hardcoded asset path.

## Docker image

Isaac Lab is not in the default `isaac-sim` image. It is baked into the
`isaac-lab` Dockerfile stage (layered `FROM isaac-sim`; same kit python, USD, and
PhysX) built by `make build-isaac-lab` as `ISAAC_LAB_IMAGE`
(`vv-isaac-lab-custom:<tag>`), from `isaac-sim/IsaacLab` pinned by the
`ISAACLAB_REF` build ARG (default `v2.3.2`, built on Isaac Sim 5.1). The
`test-integration-press-hover-goto` gate (and future F2/F3 training) runs on this
image; every other integration target uses the leaner `isaac-sim` image.

Use guarded registration through
`robotics_experiments.workflows.isaac_lab.register_isaac_lab_tasks()`. The helper
returns `False` on host environments without Isaac Lab/Gymnasium, so core package
imports and normal unit tests stay simulator-free.
