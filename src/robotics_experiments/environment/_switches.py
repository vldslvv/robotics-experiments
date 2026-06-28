from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["SwitchState"]


@dataclass
class SwitchState:
    """Runtime state for a single spawned light switch."""

    prim_path: str
    lamp_prim_path: str = ""
    articulation: Any | None = field(default=None, repr=False)
    state: bool = False
    last_button_state: bool = False


def _apply_button_sample(
    switch: SwitchState,
    *,
    button_position: float,
    press_threshold: float,
) -> bool:
    """Update one switch from a button position sample.

    Returns ``True`` only when the sample is a rising-edge press that toggled
    the switch state.
    """
    button_pressed = button_position >= press_threshold
    toggled = button_pressed and not switch.last_button_state
    if toggled:
        switch.state = not switch.state
    switch.last_button_state = button_pressed
    return toggled
