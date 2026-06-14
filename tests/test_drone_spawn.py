import importlib
import sys
import types
from types import SimpleNamespace

import pytest


def _install_fake_isaac_modules(monkeypatch, *, min_z=-0.06, empty_bound=False):
    added_references = []

    class _Vec3d(tuple):
        def __new__(cls, *values):
            return tuple.__new__(cls, values)

    class _Vec3f(tuple):
        def __new__(cls, *values):
            return tuple.__new__(cls, values)

    class _Path:
        def __init__(self, path):
            self.pathString = path

    class _Attribute:
        def __init__(self):
            self.value = None

        def Set(self, value):
            self.value = value

    class _Prim:
        def __init__(self, path, *, valid=True, rigid_body=False):
            self.path = path
            self.valid = valid
            self.rigid_body = rigid_body
            self.references = []
            self.xform_ops = []
            self.attrs = {}
            self.children = []

        def IsValid(self):
            return self.valid

        def GetPath(self):
            return _Path(self.path)

        def GetReferences(self):
            return SimpleNamespace(AddReference=lambda usd_path: self.references.append(usd_path))

        def HasAPI(self, api):
            return self.rigid_body and api is _RigidBodyAPI

        def GetAttribute(self, name):
            return self.attrs.setdefault(name, _Attribute())

    class _Stage:
        def __init__(self):
            self.prims = {"/World": _Prim("/World")}
            self.default_prim = self.prims["/World"]

        def GetDefaultPrim(self):
            return self.default_prim

        def GetPrimAtPath(self, path):
            return self.prims.get(path, _Prim(path, valid=False))

        def RemovePrim(self, path):
            self.prims.pop(path, None)

    class _Xformable:
        def __init__(self, prim):
            self.prim = prim

        def ClearXformOpOrder(self):
            self.prim.xform_ops.clear()

        def AddTranslateOp(self):
            attr = _Attribute()
            self.prim.xform_ops.append(("translate", attr))
            return attr

        def AddRotateZOp(self):
            attr = _Attribute()
            self.prim.xform_ops.append(("rotate_z", attr))
            return attr

        def AddScaleOp(self):
            attr = _Attribute()
            self.prim.xform_ops.append(("scale", attr))
            return attr

    class _Cube:
        @staticmethod
        def Define(stage, path):
            prim = _Prim(path)
            stage.prims[path] = prim
            return prim

    class _Imageable:
        def __init__(self, prim):
            self.prim = prim

        def CreatePurposeAttr(self, value):
            attr = _Attribute()
            attr.Set(value)
            self.prim.attrs["purpose"] = attr
            return attr

    class _Range:
        def IsEmpty(self):
            return empty_bound

        def GetMin(self):
            return (0.0, 0.0, min_z)

    class _WorldBound:
        def ComputeAlignedBox(self):
            return _Range()

    class _BBoxCache:
        def __init__(self, *args, **kwargs):
            pass

        def ComputeWorldBound(self, prim):
            return _WorldBound()

    class _TimeCode:
        @staticmethod
        def Default():
            return 0

    class _RigidBodyAPI:
        pass

    def _prim_range(root):
        yield root
        yield from root.children

    def add_reference_to_stage(usd_path, prim_path):
        added_references.append((usd_path, prim_path))
        root = stage.prims.setdefault(prim_path, _Prim(prim_path))
        body = _Prim(f"{prim_path}/body", rigid_body=True)
        root.children = [body]
        stage.prims[body.path] = body

    stage = _Stage()

    carb = types.ModuleType("carb")
    carb.logs = []
    carb.log_info = lambda message: carb.logs.append(("info", message))
    carb.log_warn = lambda message: carb.logs.append(("warn", message))

    isaacsim = types.ModuleType("isaacsim")
    isaacsim_core = types.ModuleType("isaacsim.core")
    isaacsim_core_utils = types.ModuleType("isaacsim.core.utils")
    isaacsim_core_utils_stage = types.ModuleType("isaacsim.core.utils.stage")
    isaacsim_core_utils_stage.add_reference_to_stage = add_reference_to_stage

    pxr = types.ModuleType("pxr")
    gf = types.ModuleType("pxr.Gf")
    gf.Vec3d = _Vec3d
    gf.Vec3f = _Vec3f
    usd = types.ModuleType("pxr.Usd")
    usd.Stage = _Stage
    usd.PrimRange = _prim_range
    usd.TimeCode = _TimeCode
    usd_geom = types.ModuleType("pxr.UsdGeom")
    usd_geom.Cube = _Cube
    usd_geom.Xformable = _Xformable
    usd_geom.Imageable = _Imageable
    usd_geom.BBoxCache = _BBoxCache
    usd_geom.Tokens = SimpleNamespace(
        default_="default", render="render", proxy="proxy", guide="guide"
    )
    usd_physics = types.ModuleType("pxr.UsdPhysics")
    usd_physics.RigidBodyAPI = _RigidBodyAPI

    modules = {
        "carb": carb,
        "isaacsim": isaacsim,
        "isaacsim.core": isaacsim_core,
        "isaacsim.core.utils": isaacsim_core_utils,
        "isaacsim.core.utils.stage": isaacsim_core_utils_stage,
        "pxr": pxr,
        "pxr.Gf": gf,
        "pxr.Usd": usd,
        "pxr.UsdGeom": usd_geom,
        "pxr.UsdPhysics": usd_physics,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    return stage, carb, added_references


def _import_spawn_module(monkeypatch, *, min_z=-0.06, empty_bound=False):
    for name in list(sys.modules):
        if name == "robotics_experiments.drone.spawn":
            sys.modules.pop(name, None)
    stage, carb, references = _install_fake_isaac_modules(
        monkeypatch,
        min_z=min_z,
        empty_bound=empty_bound,
    )
    module = importlib.import_module("robotics_experiments.drone.spawn")
    return module, stage, carb, references


def _config(**overrides):
    values = {
        "usd_path": "/assets/drone.usda",
        "prim_name": "Drone",
        "spawn_volume_prim_name": "DroneSpawnVolume",
        "spawn_volume_center": (1.0, 2.0, 3.0),
        "spawn_volume_half_extents": (0.0, 0.0, 0.5),
        "spawn_stationary": True,
        "stationary_ground_z": 0.0,
        "stationary_ground_clearance_m": 0.01,
        "stationary_spawn_z_fallback": 0.07,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _last_translate(stage):
    drone = stage.GetPrimAtPath("/World/Drone")
    for op_name, attr in reversed(drone.xform_ops):
        if op_name == "translate":
            return tuple(attr.value)
    raise AssertionError("translate op was not authored")


def test_stationary_spawn_computes_z_from_composed_bounds(monkeypatch):
    module, stage, _carb, _references = _import_spawn_module(monkeypatch, min_z=-0.06)

    module.spawn_drone(stage, _config())

    assert _last_translate(stage) == pytest.approx((1.0, 2.0, 0.07))


def test_stationary_spawn_uses_fallback_when_bounds_are_invalid(monkeypatch):
    module, stage, carb, _references = _import_spawn_module(monkeypatch, empty_bound=True)

    module.spawn_drone(stage, _config(stationary_spawn_z_fallback=0.123))

    assert _last_translate(stage) == (1.0, 2.0, 0.123)
    assert any("using fallback z=0.123" in message for _level, message in carb.logs)


def test_volume_spawn_keeps_randomized_z_when_stationary_disabled(monkeypatch):
    module, stage, _carb, _references = _import_spawn_module(monkeypatch)
    monkeypatch.setattr(module.random, "uniform", lambda low, high: high)

    module.spawn_drone(stage, _config(spawn_stationary=False))

    assert _last_translate(stage) == (1.0, 2.0, 3.5)
