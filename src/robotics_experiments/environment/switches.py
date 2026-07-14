from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["SwitchState", "apply_button_sample"]


@dataclass(frozen=True)
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
    state: bool = False
    press_steps: int = 0
    toggle_latched: bool = False


def apply_button_sample(
    *,
    switch: SwitchState,
    button_position: float,
    press_threshold: float,
    confirm_steps: int,
) -> tuple[SwitchState, bool]:
    """Return the successor state for one button position sample.

    Implements the light-state latch: a press toggles the switch only once it
    has stayed at or above ``press_threshold`` for
    ``confirm_steps`` consecutive samples. A rapid tap that releases before the
    window completes does not count; a sustained press toggles exactly once and
    will not re-toggle until the button is released and pressed again.

    The input state is never mutated. The boolean result is ``True`` only on
    the sample that confirms the toggle.
    """
    if button_position < press_threshold:
        return replace(switch, press_steps=0, toggle_latched=False), False

    press_steps = switch.press_steps + 1
    toggled = press_steps >= confirm_steps and not switch.toggle_latched
    return (
        replace(
            switch,
            state=not switch.state if toggled else switch.state,
            press_steps=press_steps,
            toggle_latched=switch.toggle_latched or toggled,
        ),
        toggled,
    )
