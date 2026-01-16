# AGENTS.md — Robotics stack (Isaac Sim + ROS 2)

## Project goal
This repository integrates:
- Isaac Sim (simulation)
- ROS 2 (control, perception)
- Docker-based development

Prefer minimal, scoped changes.
Do not introduce new tooling unless explicitly requested.

## Repository layout
- isaac_sim/        → Isaac Sim scripts, extensions, graphs
- ros2_ws/          → ROS 2 workspace (colcon)
- docs/             → Design and usage docs

## Common commands

### ROS 2
Avoid calling ROS2, it's meant to be used in dockerized environment

### Isaac Sim
Avoid calling ROS2, it's meant to be used in dockerized environment

## Hard rules
- Do NOT commit USD assets or binary files.
- Do NOT regenerate assets unless explicitly asked.
- Do NOT hardcode absolute paths.
- Prefer modifying existing files over adding new entrypoints.

## Docker assumptions
- Containers often run with --network=host
- GPU support is required (NVIDIA runtime)
- Headless must work by default

## When uncertain
Don't do anything, ask for instructions

## Recent agent operations
- Added a top-level `pyproject.toml` to centralize Python build metadata and tooling configuration (Black/isort/Ruff/mypy/pytest).
- Synced dependency pins with `requirements.txt` / `requirements_dev.txt` and `.pre-commit-config.yaml`.
