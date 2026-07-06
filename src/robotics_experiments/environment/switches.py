from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["SwitchState", "apply_button_sample"]


@dataclass
class SwitchState:
    """Runtime state for a single spawned light switch.

    ``press_steps`` counts how many consecutive samples the button has stayed at
    or above the press threshold, and ``toggle_latched`` records that the current
    sustained press has already produced its single toggle. Together they drive
    the N-step confirmation window; both reset
    when the button releases.
    """

    prim_path: str
    lamp_prim_path: str = ""
    articulation: Any | None = field(default=None, repr=False)
    state: bool = False
    press_steps: int = 0
    toggle_latched: bool = False


def apply_button_sample(
    switch: SwitchState,
    *,
    button_position: float,
    press_threshold: float,
    confirm_steps: int,
) -> bool:
    """Update one switch from a button position sample.

    Implements the light-state latch: a press toggles the switch only once it
    has stayed at or above ``press_threshold`` for
    ``confirm_steps`` consecutive samples. A rapid tap that releases before the
    window completes does not count; a sustained press toggles exactly once and
    will not re-toggle until the button is released and pressed again.

    Returns ``True`` only on the sample that confirms (and applies) the toggle.
    """
    if button_position < press_threshold:
        switch.press_steps = 0
        switch.toggle_latched = False
        return False

    switch.press_steps += 1
    if switch.press_steps >= confirm_steps and not switch.toggle_latched:
        switch.state = not switch.state
        switch.toggle_latched = True
        return True
    return False
