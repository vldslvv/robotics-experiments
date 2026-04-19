import os
import random
from typing import List

import carb
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from .config import LIGHTSWITCH_CONFIG, LightSwitchConfig
from .state import SwitchState


def spawn_switches(
    stage: Usd.Stage, config: LightSwitchConfig = LIGHTSWITCH_CONFIG
) -> List[SwitchState]:
    positions = _sample_positions(
        config.num_switches,
        config.spawn_bounds,
        config.min_spawn_dist,
        config.max_spawn_retries,
    )
    switch_paths = _spawn_switches(stage, config.switch_usd_path, positions)
    lamp_paths = _spawn_lamps(stage, config.lamp_usd_path, positions)
    return [
        SwitchState(prim_path=sp, lamp_prim_path=lp)
        for sp, lp in zip(switch_paths, lamp_paths)
    ]


def despawn_switches(stage: Usd.Stage, switch_state: List[SwitchState]) -> None:
    if stage is None:
        return
    for sw in switch_state:
        for path in (sw.prim_path, sw.lamp_prim_path):
            if not path:
                continue
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(Sdf.Path(path))


def validate_switches(stage: Usd.Stage, switch_state: List[SwitchState]) -> bool:
    for sw in switch_state:
        prim = stage.GetPrimAtPath(sw.prim_path)
        if not prim.IsValid():
            carb.log_error(f"[LightSwitch] Prim not found at '{sw.prim_path}'.")
            return False
    return True


def _sample_positions(
    count: int,
    bounds: List[Gf.Range3d],
    min_distance: float,
    max_retries: int,
) -> List[Gf.Vec3d]:
    positions: List[Gf.Vec3d] = []

    def _dist(a: Gf.Vec3d, b: Gf.Vec3d) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5

    for _ in range(count):
        fallback = None
        placed = False
        for _ in range(max_retries):
            b = random.choice(bounds)
            mn, mx = b.min, b.max
            c = Gf.Vec3d(
                random.uniform(mn[0], mx[0]),
                random.uniform(mn[1], mx[1]),
                random.uniform(mn[2], mx[2]),
            )
            if fallback is None:
                fallback = c
            if all(_dist(c, p) >= min_distance for p in positions):
                positions.append(c)
                placed = True
                break

        if not placed:
            carb.log_warn(
                f"[LightSwitch] Could not place switch {len(positions)} "
                f"with min separation {min_distance} m after {max_retries} retries; "
                "accepting overlapping position."
            )
            positions.append(fallback)

    return positions


def _spawn_switches(stage: Usd.Stage, usd_path: str, positions: List[Gf.Vec3d]) -> List[str]:
    if not os.path.isfile(usd_path):
        carb.log_error(f"[LightSwitch] USD asset not found: '{usd_path}'")
        return []

    prim_paths: List[str] = []
    for i, pos in enumerate(positions):
        prim_path = f"/Root/LightSwitch_{i}"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(usd_path)

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(pos)

        prim_paths.append(prim_path)

    carb.log_info(f"[LightSwitch] Spawned {len(prim_paths)} switches.")
    return prim_paths


def _spawn_lamps(stage: Usd.Stage, usd_path: str, positions: List[Gf.Vec3d]) -> List[str]:
    if not os.path.isfile(usd_path):
        carb.log_error(f"[LightSwitch] Lamp USD asset not found: '{usd_path}'")
        return []

    prim_paths: List[str] = []
    for i, pos in enumerate(positions):
        lamp_pos = Gf.Vec3d(pos[0], pos[1], random.uniform(3.4, 5.0))
        prim_path = f"/Root/Lamp_{i}"

        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(usd_path)

        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(lamp_pos)

        light_prim = stage.GetPrimAtPath(f"{prim_path}/Lamp/SphereLight")
        if light_prim.IsValid():
            UsdLux.SphereLight(light_prim).GetIntensityAttr().Set(0.0)

        shader_prim = stage.GetPrimAtPath(f"{prim_path}/Lamp/Mesh/Looks/OmniPBR/Shader")
        if shader_prim.IsValid():
            shader = UsdShade.Shader(shader_prim)
            shader.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(1.0, 0.9, 0.7)
            )
            shader.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(False)

        prim_paths.append(prim_path)

    carb.log_info(f"[LightSwitch] Spawned {len(prim_paths)} lamps.")
    return prim_paths
