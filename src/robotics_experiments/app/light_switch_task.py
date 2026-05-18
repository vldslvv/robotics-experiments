from __future__ import annotations

from typing import Callable

from robotics_experiments.config import LightSwitchConfig
from robotics_experiments.domain.switches import SwitchState


def apply_switch_sample(
    switch: SwitchState,
    *,
    button_position: float,
    press_threshold: float,
) -> bool:
    from robotics_experiments.domain.switches import apply_button_sample

    return apply_button_sample(
        switch,
        button_position=button_position,
        press_threshold=press_threshold,
    )


def spawn_switches(stage, config: LightSwitchConfig | None = None) -> list[SwitchState]:
    from robotics_experiments.adapters.isaac.stage_switches import (
        spawn_switches as _spawn,
    )

    return _spawn(stage, config)


def despawn_switches(stage, switch_state: list[SwitchState]) -> None:
    from robotics_experiments.adapters.isaac.stage_switches import (
        despawn_switches as _despawn,
    )

    _despawn(stage, switch_state)


def step_switches(
    stage,
    switch_state: list[SwitchState],
    on_toggle: Callable[[int, bool], None] | None = None,
    config: LightSwitchConfig | None = None,
) -> None:
    from robotics_experiments.adapters.isaac.light_switch_step import (
        step_switches as _step,
    )

    _step(stage, switch_state, on_toggle=on_toggle, config=config)
