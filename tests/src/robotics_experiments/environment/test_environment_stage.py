import importlib
import sys
import types
from types import SimpleNamespace

import pytest


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
    monkeypatch.setattr(module.os.path, "isfile", lambda path: True)

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
    monkeypatch.setattr(module, "validate_switches", lambda stage, switches: True)
    module.spawn_switches(stage=_Stage(), config=_config(seed))
    return captured


def test_switch_and_lamp_layout_is_reproducible_from_seed(monkeypatch):
    module = _import_stage(monkeypatch)

    first = _spawn_layout(monkeypatch, module, 17)
    repeated = _spawn_layout(monkeypatch, module, 17)
    different = _spawn_layout(monkeypatch, module, 18)

    assert first == repeated
    assert first != different


class _Prim:
    def __init__(self, valid):
        self._valid = valid

    def IsValid(self):
        return self._valid


class _Stage:
    def __init__(self, valid_paths=()):
        self.valid_paths = set(valid_paths)
        self.removed = []

    def GetPrimAtPath(self, path):
        return _Prim(path in self.valid_paths)

    def RemovePrim(self, path):
        self.removed.append(str(path))
        self.valid_paths.discard(str(path))


def test_spawn_preflights_both_assets_before_mutating(monkeypatch):
    module = _import_stage(monkeypatch)
    calls = []
    monkeypatch.setattr(module.os.path, "isfile", lambda path: path != "lamp.usd")
    monkeypatch.setattr(module, "_spawn_switches", lambda *args: calls.append(args))

    with pytest.raises(FileNotFoundError, match="lamp USD asset"):
        module.spawn_switches(stage=_Stage(), config=_config(1))

    assert calls == []


def test_partial_pair_spawn_is_cleaned_up_and_raises(monkeypatch):
    module = _import_stage(monkeypatch)
    stage = _Stage()
    monkeypatch.setattr(module.os.path, "isfile", lambda path: True)

    def spawn_switches(stage, usd_path, positions):
        paths = [f"/Root/LightSwitch_{index}" for index in range(3)]
        stage.valid_paths.update(paths)
        return paths

    def spawn_lamps(**kwargs):
        paths = [f"/Root/Lamp_{index}" for index in range(2)]
        kwargs["stage"].valid_paths.update(paths)
        return paths

    monkeypatch.setattr(
        module,
        "_spawn_switches",
        spawn_switches,
    )
    monkeypatch.setattr(module, "_spawn_lamps", spawn_lamps)

    with pytest.raises(RuntimeError, match="spawn count mismatch"):
        module.spawn_switches(stage=stage, config=_config(1))

    assert stage.valid_paths == set()


def test_spawn_rejects_occupied_target_without_removing_it(monkeypatch):
    module = _import_stage(monkeypatch)
    stage = _Stage({"/Root/LightSwitch_0"})
    monkeypatch.setattr(module.os.path, "isfile", lambda path: True)

    with pytest.raises(RuntimeError, match="spawn paths already exist"):
        module.spawn_switches(stage=stage, config=_config(1))

    assert stage.valid_paths == {"/Root/LightSwitch_0"}
    assert stage.removed == []


def test_validation_rejects_missing_paired_lamp(monkeypatch):
    module = _import_stage(monkeypatch)
    stage = _Stage({"/Root/LightSwitch_0"})
    switch = module.SwitchState("/Root/LightSwitch_0", "/Root/Lamp_0")

    assert not module.validate_switches(stage, (switch,))
