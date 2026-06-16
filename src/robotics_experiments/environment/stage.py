import os
import random
from typing import List

import carb
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from robotics_experiments.config._resolve import resolve_section
from robotics_experiments.config.types import BoundingBox, LightSwitchConfig
from robotics_experiments.environment.switches import SwitchState

__all__ = ["spawn_switches", "despawn_switches", "validate_switches"]


def _to_range(box: BoundingBox) -> Gf.Range3d:
    """Convert a serializable :class:`BoundingBox` to a ``pxr.Gf.Range3d``.

    Args:
        box: Corner-tuple bounding box authored in YAML.

    Returns:
        Equivalent ``Gf.Range3d`` usable by the sampling helpers.
    """
    return Gf.Range3d(Gf.Vec3d(*box.min), Gf.Vec3d(*box.max))


def _sample_positions(
    count: int,
    bounds: List[Gf.Range3d],
    min_distance: float,
    max_retries: int,
) -> List[Gf.Vec3d]:
    """Sample ``count`` positions inside ``bounds`` with a minimum separation.

    For each position, a random bound is picked and a point uniformly sampled
    inside it. If ``max_retries`` attempts fail to satisfy the minimum
    separation, the first sampled fallback is accepted and a warning is
    logged rather than aborting the spawn.

    Args:
        count: Number of positions to sample.
        bounds: Candidate axis-aligned regions to sample from.
        min_distance: Minimum pairwise distance between accepted positions.
        max_retries: Retry budget per position before accepting overlap.

    Returns:
        List of sampled positions with length ``count``.
    """
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


def _spawn_switches(
    stage: Usd.Stage,
    usd_path: str,
    positions: List[Gf.Vec3d],
) -> List[str]:
    """Create ``Xform`` prims that reference the switch USD at each position.

    Args:
        stage: USD stage to mutate.
        usd_path: Absolute path to the switch USD asset.
        positions: World-space positions for each spawned switch.

    Returns:
        Absolute prim paths of the created switches, or an empty list when
        the USD asset is missing.
    """
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


def _spawn_lamps(
    stage: Usd.Stage,
    usd_path: str,
    positions: List[Gf.Vec3d],
) -> List[str]:
    """Create lamp prims above each switch with lights and emission set to off.

    The lamp Z is randomized between 3.4 and 5.0 m above the ground so lamps
    sit plausibly on a ceiling; the ``SphereLight`` intensity and the
    ``OmniPBR`` shader emission are both pre-initialized to the off state.

    Args:
        stage: USD stage to mutate.
        usd_path: Absolute path to the lamp USD asset.
        positions: Switch positions used for lamp X/Y placement.

    Returns:
        Absolute prim paths of the created lamps, or an empty list when the
        USD asset is missing.
    """
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


def spawn_switches(stage: Usd.Stage, config: LightSwitchConfig | None = None) -> List[SwitchState]:
    """Spawn switches and their paired lamps at randomly sampled positions.

    Args:
        stage: USD stage to mutate.
        config: Spawner configuration; when ``None``, resolved via
            :func:`.config.get_config` at call time.

    Returns:
        One :class:`SwitchState` per successfully spawned switch, pairing the
        switch prim path with its lamp prim path.
    """
    config = resolve_section("switches", config)
    positions = _sample_positions(
        config.num_switches,
        [_to_range(b) for b in config.spawn_bounds],
        config.min_spawn_dist,
        config.max_spawn_retries,
    )
    switch_paths = _spawn_switches(stage, config.switch_usd_path, positions)
    lamp_paths = _spawn_lamps(stage, config.lamp_usd_path, positions)
    return [
        SwitchState(prim_path=sp, lamp_prim_path=lp)
        for sp, lp in zip(switch_paths, lamp_paths, strict=False)
    ]


def despawn_switches(stage: Usd.Stage, switch_state: List[SwitchState]) -> None:
    """Remove all switch and lamp prims tracked by ``switch_state``.

    Safe to call when ``stage`` is ``None`` or prims have already been
    removed; missing prims are skipped silently.

    Args:
        stage: USD stage to mutate, or ``None`` for a no-op.
        switch_state: Per-switch state entries whose prims should be removed.
    """
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
    """Verify that every tracked switch prim resolves on the stage.

    Args:
        stage: USD stage to inspect.
        switch_state: Per-switch state entries to validate.

    Returns:
        ``True`` if all switch prims are valid, ``False`` if any is missing.
    """
    for sw in switch_state:
        prim = stage.GetPrimAtPath(sw.prim_path)
        if not prim.IsValid():
            carb.log_error(f"[LightSwitch] Prim not found at '{sw.prim_path}'.")
            return False
    return True
