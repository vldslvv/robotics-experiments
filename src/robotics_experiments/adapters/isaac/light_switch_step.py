from typing import Callable, List

import carb
from isaacsim.core.prims import SingleArticulation
from pxr import Sdf, Usd, UsdLux, UsdShade

from robotics_experiments.config import LightSwitchConfig
from robotics_experiments.config import get_config as _get_experiment_config
from robotics_experiments.domain.switches import SwitchState, apply_button_sample


def init_articulations(switch_state: List[SwitchState]) -> bool:
    """Initialize PhysX articulations for each tracked switch.

    Must be called only after the simulation has begun playing; PhysX does
    not expose articulations until the first physics step. On failure all
    articulation handles are cleared so the caller can retry on a later tick.

    Args:
        switch_state: Per-switch state entries whose articulations should be
            initialized in-place.

    Returns:
        ``True`` if every articulation initialized, ``False`` on the first
        exception (with articulations reset to ``None``).
    """
    try:
        for sw in switch_state:
            art = SingleArticulation(prim_path=sw.prim_path)
            art.initialize()
            sw.articulation = art

        carb.log_info(f"[LightSwitch] Initialized {len(switch_state)} articulations.")
        carb.log_info(f"[LightSwitch] DOFs: {switch_state[0].articulation.dof_names}")
        return True

    except Exception as e:
        carb.log_error(f"[LightSwitch] Articulation init failed: ({e})")
        for sw in switch_state:
            sw.articulation = None
        return False


def step_switches(
    stage: Usd.Stage,
    switch_state: List[SwitchState],
    on_toggle: Callable[[int, bool], None] | None = None,
    config: LightSwitchConfig | None = None,
) -> None:
    """Detect rising-edge button presses and toggle the paired lamps.

    Reads the first DOF of each articulation; when it crosses the configured
    press threshold upward, the switch's ``state`` flips, the lamp is updated
    on stage, and ``on_toggle`` is invoked with the switch index and new
    state.

    Args:
        stage: USD stage to mutate for lamp visuals.
        switch_state: Per-switch state entries; ``state`` and
            ``last_button_state`` are updated in place.
        on_toggle: Optional callback receiving ``(index, new_state)`` on every
            successful toggle.
        config: Spawner configuration; when ``None``, resolved via
            :func:`.config.get_config` at call time.
    """
    if config is None:
        config = _get_experiment_config().switches
    for i, sw in enumerate(switch_state):
        joint_positions = sw.articulation.get_joint_positions()
        button_pos = joint_positions[0]

        if apply_button_sample(
            sw,
            button_position=float(button_pos),
            press_threshold=config.press_threshold,
        ):
            _set_lamp(stage, sw.lamp_prim_path, sw.state, config.lamp_on_intensity)
            if on_toggle is not None:
                on_toggle(i, sw.state)


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
