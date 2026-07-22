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
Isaac Lab session. Code outside that package should import only from the package
root; the task submodules are internal implementation details except for
Gymnasium's registered entry-point strings.

The registered tasks are exported as `DRONE_HOVER_TASK_ID`
(`Isaac-Drone-Hover-Direct-v0`) and `SWITCH_PRESS_TASK_ID`
(`Isaac-Switch-Press-Direct-v0`), with both env/config/factory triples available
lazily from `robotics_experiments.workflows.isaac_lab`. They fly N
parallel drones through the pluggable in-Lab actuation seam
(`robotics_experiments.rl`), which maps a normalized RL action to per-rotor
forces via the internal batched torch model (no PX4 lockstep per env). The root
export `compute_lab_rotor_omega` is the supported external diagnostic/composition
entry point for that model. `make_isaac_lab_env(...)` is the generic task
factory used by training and evaluation. The press task adds the cloned real
switch, wall, pad contact sensor, and frozen functional task transition; task
modules remain internal outside the package.

Shared RL tunables (Lab-scale + the RL action level) live in the Hydra `rl`
group (`configs/experiment/rl/default.yaml`), including Lab-only vehicle limits
under `rl.lab_vehicle`, with press task tunables under the `rl.press` config
group (`configs/experiment/rl/press/default.yaml`). The env spawns from an
internal config-driven `ArticulationCfg` builder that reads `drone.usd_path`,
not a hardcoded asset path.

## Docker image

Isaac Lab is not in the default `isaac-sim` image. It is baked into the
`isaac-lab` Dockerfile stage (layered `FROM isaac-sim`; same kit python, USD, and
PhysX) built by `make build-isaac-lab` as `ISAAC_LAB_IMAGE`
(`vv-isaac-lab-custom:<tag>`), from `isaac-sim/IsaacLab` pinned by the
`ISAACLAB_REF` build ARG (default `v2.3.2`, built on Isaac Sim 5.1). The
`test-integration-press-hover-goto`, `test-integration-press-contact-chain`,
training, and press evaluation targets run on this image; every other
integration target uses the leaner `isaac-sim` image.

Use guarded registration through
`robotics_experiments.workflows.isaac_lab.register_isaac_lab_tasks()`. The helper
returns `False` on host environments without Isaac Lab/Gymnasium, so core package
imports and normal unit tests stay simulator-free.
