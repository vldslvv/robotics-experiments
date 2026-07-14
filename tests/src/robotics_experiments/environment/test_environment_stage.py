import importlib
import sys
import types
from types import SimpleNamespace


class _Vec3d(tuple):
    def __new__(cls, x, y, z):
        return super().__new__(cls, (x, y, z))


class _Range3d:
    def __init__(self, minimum, maximum):
        self.min = minimum
        self.max = maximum


def _import_stage(monkeypatch):
    carb = types.ModuleType("carb")
    carb.log_error = lambda message: None
    carb.log_info = lambda message: None
    carb.log_warn = lambda message: None

    pxr = types.ModuleType("pxr")
    pxr.Gf = SimpleNamespace(Vec3d=_Vec3d, Vec3f=_Vec3d, Range3d=_Range3d)
    pxr.Sdf = SimpleNamespace(Path=str, ValueTypeNames=SimpleNamespace())
    pxr.Usd = SimpleNamespace(Stage=object, Prim=object)
    pxr.UsdGeom = SimpleNamespace()
    pxr.UsdLux = SimpleNamespace()
    pxr.UsdShade = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    sys.modules.pop("robotics_experiments.environment.stage", None)
    return importlib.import_module("robotics_experiments.environment.stage")


def _config(seed):
    return SimpleNamespace(
        spawn_seed=seed,
        num_switches=3,
        spawn_bounds=[SimpleNamespace(min=(-2.0, -1.0, 0.5), max=(2.0, 1.0, 2.5))],
        min_spawn_dist=0.1,
        max_spawn_retries=10,
        switch_usd_path="switch.usd",
        lamp_usd_path="lamp.usd",
    )


def _spawn_layout(monkeypatch, module, seed):
    captured = {}

    def spawn_switches(stage, usd_path, positions):
        captured["switches"] = tuple(tuple(position) for position in positions)
        return [f"/Switch_{index}" for index in range(len(positions))]

    def spawn_lamps(*, rng, stage, usd_path, positions):
        captured["lamps"] = tuple(
            (position[0], position[1], rng.uniform(3.4, 5.0)) for position in positions
        )
        return [f"/Lamp_{index}" for index in range(len(positions))]

    monkeypatch.setattr(module, "_spawn_switches", spawn_switches)
    monkeypatch.setattr(module, "_spawn_lamps", spawn_lamps)
    module.spawn_switches(stage=object(), config=_config(seed))
    return captured


def test_switch_and_lamp_layout_is_reproducible_from_seed(monkeypatch):
    module = _import_stage(monkeypatch)

    first = _spawn_layout(monkeypatch, module, 17)
    repeated = _spawn_layout(monkeypatch, module, 17)
    different = _spawn_layout(monkeypatch, module, 18)

    assert first == repeated
    assert first != different
