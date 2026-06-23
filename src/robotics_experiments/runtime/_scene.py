"""Pure scene-path resolution shared by the orchestrator and the standalone harness.

No Isaac imports: it only turns the experiment ``scene`` string (``@`` =
repository-relative) into an absolute path, so it stays usable before a
``SimulationApp`` exists and on the core import path.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["resolve_scene_path"]


def resolve_scene_path(scene: str, *, repo_root: Path) -> Path:
    """Resolve an experiment scene string to an absolute path.

    A leading ``@`` is repository-relative (mapped under ``repo_root``); any
    other value is treated as a filesystem path.
    """
    if scene.startswith("@"):
        return (repo_root / scene[1:]).resolve()
    return Path(scene).expanduser().resolve()
