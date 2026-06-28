"""Simulation package public API.

This package exposes lightweight entrypoints that are safe to import from
normal Python tooling, tests, and non-Isaac processes. Isaac-specific modules
are imported only when their functionality is called.
"""

from ._settings import apply_settings

__all__ = ["apply_settings"]
