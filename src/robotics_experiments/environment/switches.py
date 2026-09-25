"""One simulation-time button latch for scalar runtime and batched Lab consumers."""

import math
from dataclasses import dataclass, replace

import numpy as np

from robotics_experiments.config import SwitchModelConfig
from robotics_experiments.switches import ButtonLatchState, step_button_latch

__all__ = ["SwitchState", "apply_button_sample"]


@dataclass(frozen=True)
class SwitchState:
    """One lamp and its button's immutable progress."""

    prim_path: str
    lamp_prim_path: str = ""
    state: bool = False
    press_duration_s: float = 0.0
    toggle_latched: bool = False


def apply_button_sample(
    *,
    switch: SwitchState,
    button_position: float,
    dt_s: float,
    model: SwitchModelConfig,
) -> tuple[SwitchState, bool]:
    """Apply the shared vector transition to one runtime switch and lamp."""
    if not math.isfinite(button_position):
        raise ValueError("button displacement must be finite")
    latch, event = step_button_latch(
        state=ButtonLatchState(
            np.asarray(switch.press_duration_s), np.asarray(switch.toggle_latched)
        ),
        displacement_m=np.asarray(button_position),
        dt_s=dt_s,
        model=model,
    )
    toggled = bool(event)
    return replace(
        switch,
        press_duration_s=float(latch.duration_s),
        toggle_latched=bool(latch.latched),
        state=not switch.state if toggled else switch.state,
    ), toggled
