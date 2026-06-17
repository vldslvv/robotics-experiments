"""Isaac Lab workflow integration.

This package stays lightweight at import time. Isaac Lab task implementations
live under ``robotics_experiments.workflows.isaac_lab.tasks`` and are imported
only by Isaac Lab / Gymnasium entry points.
"""

from ._registration import DRONE_HOVER_TASK_ID, register_isaac_lab_tasks

__all__ = ["DRONE_HOVER_TASK_ID", "register_isaac_lab_tasks"]
