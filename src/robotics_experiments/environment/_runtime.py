"""Functional light-switch lifecycle over live Isaac resource adapters.

This module is intentionally Isaac-only: every transition owns or operates on
a live USD stage or PhysX articulation. The lazy ``environment`` package root
keeps it off the core import path when Isaac is unavailable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import TYPE_CHECKING

import carb

from robotics_experiments.config import LightSwitchConfig
from robotics_experiments.environment.stage import (
    despawn_switches,
    spawn_switches,
    validate_switches,
)
from robotics_experiments.environment.step import init_articulations, step_switches
from robotics_experiments.environment.switches import SwitchState

if TYPE_CHECKING:
    from isaacsim.core.prims import SingleArticulation
    from pxr import Usd

__all__ = [
    "SwitchesRuntimePhase",
    "SwitchesRuntimeState",
    "adopt_switches_stage",
    "create_switches_runtime",
    "pause_switches_runtime",
    "play_switches_runtime",
    "step_switches_runtime",
    "teardown_switches_runtime",
]

_LOG_PREFIX = "[LightSwitch]"


class SwitchesRuntimePhase(Enum):
    """Lifecycle phase for the light-switch runtime."""

    IDLE = auto()
    ARMING = auto()
    RUNNING = auto()
    PAUSED = auto()


@dataclass(frozen=True)
class SwitchesRuntimeState:
    """Immutable switch progress plus live Isaac resource references."""

    config: LightSwitchConfig
    stage: Usd.Stage | None
    switches: tuple[SwitchState, ...]
    articulations: tuple[SingleArticulation, ...]
    phase: SwitchesRuntimePhase


def create_switches_runtime(
    *, config: LightSwitchConfig, stage: Usd.Stage | None
) -> SwitchesRuntimeState:
    """Create an idle runtime bound to the current stage."""
    return SwitchesRuntimeState(
        config=config,
        stage=stage,
        switches=(),
        articulations=(),
        phase=SwitchesRuntimePhase.IDLE,
    )


def teardown_switches_runtime(*, state: SwitchesRuntimeState) -> SwitchesRuntimeState:
    """Despawn tracked prims and return the runtime to its idle phase."""
    despawn_switches(state.stage, state.switches)
    return replace(
        state,
        switches=(),
        articulations=(),
        phase=SwitchesRuntimePhase.IDLE,
    )


def adopt_switches_stage(*, state: SwitchesRuntimeState, stage: Usd.Stage) -> SwitchesRuntimeState:
    """Adopt a newly opened stage and clear runtime progress on that stage."""
    return teardown_switches_runtime(state=replace(state, stage=stage))


def play_switches_runtime(*, state: SwitchesRuntimeState) -> SwitchesRuntimeState:
    """Spawn and validate switches, or resume a paused runtime."""
    if state.stage is None:
        carb.log_error(f"{_LOG_PREFIX} No stage loaded.")
        return state

    if state.phase is SwitchesRuntimePhase.PAUSED:
        phase = SwitchesRuntimePhase.RUNNING if state.articulations else SwitchesRuntimePhase.ARMING
        carb.log_info(f"{_LOG_PREFIX} Simulation started.")
        return replace(state, phase=phase)
    if state.phase is not SwitchesRuntimePhase.IDLE:
        return state

    state = teardown_switches_runtime(state=state)
    switches = spawn_switches(stage=state.stage, config=state.config)
    spawned = replace(state, switches=switches)
    if not validate_switches(state.stage, switches):
        carb.log_error(f"{_LOG_PREFIX} Scene validation failed; aborting simulation start.")
        return teardown_switches_runtime(state=spawned)

    carb.log_info(f"{_LOG_PREFIX} Simulation started.")
    return replace(spawned, phase=SwitchesRuntimePhase.ARMING)


def pause_switches_runtime(*, state: SwitchesRuntimeState) -> SwitchesRuntimeState:
    """Pause per-frame work while preserving spawned prims and handles."""
    if state.phase in {SwitchesRuntimePhase.ARMING, SwitchesRuntimePhase.RUNNING}:
        return replace(state, phase=SwitchesRuntimePhase.PAUSED)
    return state


def step_switches_runtime(
    *,
    state: SwitchesRuntimeState,
    on_toggle: Callable[[int, bool], None] | None = None,
) -> SwitchesRuntimeState:
    """Retry articulation setup or advance switch interaction by one tick."""
    if state.phase is SwitchesRuntimePhase.ARMING:
        articulations = init_articulations(
            prim_paths=tuple(switch.prim_path for switch in state.switches)
        )
        if articulations is None:
            return state
        return replace(
            state,
            articulations=tuple(articulations),
            phase=SwitchesRuntimePhase.RUNNING,
        )
    if state.phase is not SwitchesRuntimePhase.RUNNING:
        return state

    switches = step_switches(
        stage=state.stage,
        switch_state=state.switches,
        articulations=state.articulations,
        config=state.config,
        on_toggle=on_toggle,
    )
    return replace(state, switches=switches)
