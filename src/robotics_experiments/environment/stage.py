"""Switch and lamp asset spawning on a USD stage.

This module is intentionally Isaac-only: it imports ``carb`` and ``pxr`` at
import time because every function operates on a live USD stage. The core import
path never loads it directly; Isaac runtime code imports this leaf module
explicitly, so ``import robotics_experiments`` works outside the Isaac runtime.
See ``robotics_experiments.simulation`` for the same pattern and
``tests/test_core_imports.py`` for the boundary guard.
"""

import os
import random
from typing import List

import carb
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

from robotics_experiments.config import BoundingBox, LightSwitchConfig
from robotics_experiments.environment.switches import SwitchState
from robotics_experiments.switches import load_switch_asset

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
    *,
    rng: random.Random,
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
        rng: Seeded random source owned by the spawn operation.
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
            b = rng.choice(bounds)
            mn, mx = b.min, b.max
            c = Gf.Vec3d(
                rng.uniform(mn[0], mx[0]),
                rng.uniform(mn[1], mx[1]),
                rng.uniform(mn[2], mx[2]),
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


def _reference_asset_at(
    stage: Usd.Stage,
    prim_path: str,
    usd_path: str,
    pos: Gf.Vec3d,
) -> Usd.Prim:
    """Define an ``Xform`` prim that references ``usd_path`` and translate it.

    Args:
        stage: USD stage to mutate.
        prim_path: Absolute prim path to create.
        usd_path: Absolute path to the USD asset to reference.
        pos: World-space translation for the created prim.

    Returns:
        The created prim, so callers can author further attributes on it.
    """
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(pos)
    return prim


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
        Absolute prim paths of the created switches.
    """
    prim_paths: List[str] = []
    for i, pos in enumerate(positions):
        prim_path = f"/Root/LightSwitch_{i}"
        _reference_asset_at(stage, prim_path, usd_path, pos)
        prim_paths.append(prim_path)

    carb.log_info(f"[LightSwitch] Spawned {len(prim_paths)} switches.")
    return prim_paths


def _spawn_lamps(
    *,
    rng: random.Random,
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
        Absolute prim paths of the created lamps.
    """
    prim_paths: List[str] = []
    for i, pos in enumerate(positions):
        lamp_pos = Gf.Vec3d(pos[0], pos[1], rng.uniform(3.4, 5.0))
        prim_path = f"/Root/Lamp_{i}"

        _reference_asset_at(stage, prim_path, usd_path, lamp_pos)

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


def spawn_switches(*, stage: Usd.Stage, config: LightSwitchConfig) -> tuple[SwitchState, ...]:
    """Spawn switches and paired lamps from the configured deterministic seed.

    Args:
        stage: USD stage to mutate.
        config: Spawner configuration.

    Returns:
        One :class:`SwitchState` per successfully spawned switch, pairing the
        switch prim path with its lamp prim path.
    """
    switch_usd_path = str(load_switch_asset(model=config.model).path)
    missing = next(
        (
            (kind, path)
            for kind, path in (
                ("switch", switch_usd_path),
                ("lamp", config.lamp_usd_path),
            )
            if not os.path.isfile(path)
        ),
        None,
    )
    if missing is not None:
        kind, path = missing
        raise FileNotFoundError(f"{kind} USD asset not found: {path!r}")

    target_paths = tuple(
        path
        for index in range(config.num_switches)
        for path in (f"/Root/LightSwitch_{index}", f"/Root/Lamp_{index}")
    )
    occupied = [path for path in target_paths if stage.GetPrimAtPath(path).IsValid()]
    if occupied:
        raise RuntimeError("switch/lamp spawn paths already exist: " + ", ".join(occupied))
    try:
        rng = random.Random(config.spawn_seed)
        positions = _sample_positions(
            rng=rng,
            count=config.num_switches,
            bounds=[_to_range(b) for b in config.spawn_bounds],
            min_distance=config.min_spawn_dist,
            max_retries=config.max_spawn_retries,
        )
        switch_paths = _spawn_switches(stage, switch_usd_path, positions)
        lamp_paths = _spawn_lamps(
            rng=rng,
            stage=stage,
            usd_path=config.lamp_usd_path,
            positions=positions,
        )
        if len(switch_paths) != config.num_switches or len(lamp_paths) != config.num_switches:
            raise RuntimeError(
                "switch/lamp spawn count mismatch: "
                f"expected {config.num_switches}, got "
                f"{len(switch_paths)} switches and {len(lamp_paths)} lamps"
            )
        switches = tuple(
            SwitchState(prim_path=sp, lamp_prim_path=lp)
            for sp, lp in zip(switch_paths, lamp_paths, strict=True)
        )
        if not validate_switches(stage, switches):
            raise RuntimeError("spawned switch/lamp prim validation failed")
        return switches
    except Exception:
        for path in target_paths:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                stage.RemovePrim(Sdf.Path(path))
        raise


def despawn_switches(stage: Usd.Stage, switch_state: tuple[SwitchState, ...]) -> None:
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


def validate_switches(stage: Usd.Stage, switch_state: tuple[SwitchState, ...]) -> bool:
    """Verify that every tracked switch and paired lamp resolves on the stage.

    Args:
        stage: USD stage to inspect.
        switch_state: Per-switch state entries to validate.

    Returns:
        ``True`` if every switch and paired lamp prim is valid, ``False`` if
        any is missing.
    """
    for sw in switch_state:
        for kind, path in (("switch", sw.prim_path), ("lamp", sw.lamp_prim_path)):
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                carb.log_error(f"[LightSwitch] {kind} prim not found at '{path}'.")
                return False
    return True
