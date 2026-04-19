from typing import Callable, List

import carb
from isaacsim.core.prims import SingleArticulation
from pxr import Sdf, Usd, UsdLux, UsdShade

from .config import LIGHTSWITCH_CONFIG, LightSwitchConfig
from .state import SwitchState


def init_articulations(switch_state: List[SwitchState]) -> bool:
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
    config: LightSwitchConfig = LIGHTSWITCH_CONFIG,
) -> None:
    for i, sw in enumerate(switch_state):
        joint_positions = sw.articulation.get_joint_positions()
        button_pos = joint_positions[0]

        button_pressed = button_pos >= config.press_threshold
        if button_pressed and not sw.last_button_state:
            sw.state = not sw.state
            _set_lamp(stage, sw.lamp_prim_path, sw.state, config.lamp_on_intensity)
            if on_toggle is not None:
                on_toggle(i, sw.state)

        sw.last_button_state = button_pressed


def _set_lamp(stage: Usd.Stage, lamp_prim_path: str, on: bool, intensity: float) -> None:
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
