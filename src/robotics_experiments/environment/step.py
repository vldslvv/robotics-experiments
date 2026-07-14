"""Switch articulation init and per-step toggle handling on a USD stage.

This module is intentionally Isaac-only: it imports ``carb``, ``pxr``, and
``isaacsim`` at import time because every function operates on a live USD stage
and PhysX articulations. The core import path never loads it directly; Isaac
runtime code imports this leaf module explicitly, so ``import
robotics_experiments`` works outside the Isaac runtime. See
``robotics_experiments.simulation`` for the same pattern and
``tests/test_core_imports.py`` for the boundary guard.
"""

from collections.abc import Callable, Sequence

import carb
from isaacsim.core.prims import SingleArticulation
from pxr import Sdf, Usd, UsdLux, UsdShade

from robotics_experiments.config import LightSwitchConfig
from robotics_experiments.environment.switches import SwitchState, apply_button_sample

__all__ = ["init_articulations", "step_switches"]


def _set_lamp(
    stage: Usd.Stage,
    lamp_prim_path: str,
    on: bool,
    intensity: float,
) -> None:
    """Drive both the ``SphereLight`` intensity and the shader emission of a lamp.

    The lamp is rendered via an emissive shader as well as a light source;
    keeping the two in sync is what makes the lamp visibly switch on and off.

    Args:
        stage: USD stage that owns the lamp.
        lamp_prim_path: Absolute path to the lamp root prim.
        on: ``True`` to enable the lamp, ``False`` to turn it off.
        intensity: Light intensity used when ``on``; ignored when off.
    """
    light_prim = stage.GetPrimAtPath(f"{lamp_prim_path}/Lamp/SphereLight")
    if light_prim.IsValid():
        UsdLux.SphereLight(light_prim).GetIntensityAttr().Set(intensity if on else 0.0)

    shader_prim = stage.GetPrimAtPath(f"{lamp_prim_path}/Lamp/Mesh/Looks/OmniPBR/Shader")
    if shader_prim.IsValid():
        shader = UsdShade.Shader(shader_prim)
        shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(
            intensity if on else 0.0
        )
        shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(on)


def init_articulations(*, prim_paths: Sequence[str]) -> list[SingleArticulation] | None:
    """Initialize PhysX articulations for each tracked switch.

    Must be called only after the simulation has begun playing; PhysX does
    not expose articulations until the first physics step. On failure the
    caller can retry on a later tick.

    Args:
        prim_paths: Switch prim paths whose articulations should be initialized.

    Returns:
        All initialized articulation handles, or ``None`` on the first
        exception.
    """
    try:
        articulations = []
        for prim_path in prim_paths:
            art = SingleArticulation(prim_path=prim_path)
            art.initialize()
            articulations.append(art)

        carb.log_info(f"[LightSwitch] Initialized {len(articulations)} articulations.")
        if articulations:
            carb.log_info(f"[LightSwitch] DOFs: {articulations[0].dof_names}")
        return articulations

    except Exception as e:
        carb.log_error(f"[LightSwitch] Articulation init failed: ({e})")
        return None


def step_switches(
    *,
    stage: Usd.Stage,
    switch_state: tuple[SwitchState, ...],
    articulations: Sequence[SingleArticulation],
    config: LightSwitchConfig,
    on_toggle: Callable[[int, bool], None] | None = None,
) -> tuple[SwitchState, ...]:
    """Detect confirmed button presses and toggle the paired lamps.

    Reads the first DOF of each articulation; once it has stayed at or above the
    configured press threshold for ``press_confirm_steps`` consecutive steps (the
    light-state latch), the switch's ``state`` flips, the lamp is
    updated on stage, and ``on_toggle`` is invoked with the switch index and new
    state.

    Args:
        stage: USD stage to mutate for lamp visuals.
        switch_state: Immutable per-switch progress snapshots.
        articulations: PhysX articulation handles parallel to ``switch_state``.
        on_toggle: Optional callback receiving ``(index, new_state)`` on every
            successful toggle.
        config: Spawner configuration.
    """
    successor = []
    for i, (sw, articulation) in enumerate(zip(switch_state, articulations, strict=True)):
        joint_positions = articulation.get_joint_positions()
        button_pos = joint_positions[0]

        next_switch, toggled = apply_button_sample(
            switch=sw,
            button_position=float(button_pos),
            press_threshold=config.press_threshold,
            confirm_steps=config.press_confirm_steps,
        )
        successor.append(next_switch)
        if toggled:
            _set_lamp(
                stage,
                next_switch.lamp_prim_path,
                next_switch.state,
                config.lamp_on_intensity,
            )
            if on_toggle is not None:
                on_toggle(i, next_switch.state)
    return tuple(successor)
