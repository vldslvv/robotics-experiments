"""Isaac Lab workflow integration.

This package stays lightweight at import time. Isaac Lab task implementations
live under ``robotics_experiments.workflows.isaac_lab.tasks`` and are imported
only by Isaac Lab / Gymnasium entry points.
"""

from robotics_experiments.workflows.isaac_lab.registration import register_isaac_lab_tasks

__all__ = ["register_isaac_lab_tasks"]
