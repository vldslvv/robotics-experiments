import importlib
import sys
import types

import pytest


class _FakeAttr:
    def __init__(self):
        self.value = None

    def Set(self, value):
        self.value = value


class _FakeScene:
    def __init__(self):
        self.gravity_direction = _FakeAttr()
        self.gravity_magnitude = _FakeAttr()

    def CreateGravityDirectionAttr(self):
        return self.gravity_direction

    def CreateGravityMagnitudeAttr(self):
        return self.gravity_magnitude

    def GetPrim(self):
        return object()


class _FakePhysxApi:
    last_instance = None

    def __init__(self):
        self.time_steps_per_second = _FakeAttr()
        self.solver_type = _FakeAttr()

    @classmethod
    def Apply(cls, prim):
        cls.last_instance = cls()
        return cls.last_instance

    def CreateTimeStepsPerSecondAttr(self):
        return self.time_steps_per_second

    def CreateSolverTypeAttr(self):
        return self.solver_type


class _FakeCarbSettings:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


@pytest.fixture()
def isaac_settings_module(monkeypatch):
    module_name = "robotics_experiments.simulation._isaac_settings"
    previous_module = sys.modules.pop(module_name, None)
    carb_settings = _FakeCarbSettings()
    carb = types.SimpleNamespace(
        log_info=lambda message: None,
        settings=types.SimpleNamespace(get_settings=lambda: carb_settings),
    )
    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace(Vec3f=lambda x, y, z: (x, y, z))
    pxr.PhysxSchema = types.SimpleNamespace(PhysxSceneAPI=_FakePhysxApi)
    pxr.Usd = types.SimpleNamespace(Stage=object)
    pxr.UsdPhysics = types.SimpleNamespace(Scene=object)

    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "pxr.Gf", pxr.Gf)
    monkeypatch.setitem(sys.modules, "pxr.PhysxSchema", pxr.PhysxSchema)
    monkeypatch.setitem(sys.modules, "pxr.Usd", pxr.Usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdPhysics", pxr.UsdPhysics)

    module = importlib.import_module(module_name)
    module._test_carb_settings = carb_settings
    yield module

    sys.modules.pop(module_name, None)
    if previous_module is not None:
        sys.modules[module_name] = previous_module


def test_apply_gravity_writes_direction_and_magnitude(isaac_settings_module):
    scene = _FakeScene()

    isaac_settings_module._apply_gravity(scene, (0.0, 0.0, -9.81))

    assert scene.gravity_direction.value == (0.0, 0.0, -1.0)
    assert scene.gravity_magnitude.value == 9.81


def test_apply_gravity_rejects_zero_magnitude(isaac_settings_module):
    scene = _FakeScene()

    with pytest.raises(ValueError, match="Gravity vector must have positive magnitude"):
        isaac_settings_module._apply_gravity(scene, (0.0, 0.0, 0.0))


@pytest.mark.parametrize("physics_hz", [0.0, -1.0])
def test_apply_physx_rejects_non_positive_rate(isaac_settings_module, physics_hz):
    with pytest.raises(ValueError, match="physics_hz must be positive"):
        isaac_settings_module._apply_physx(_FakeScene(), physics_hz, "TGS")


def test_apply_physx_writes_rate_and_solver(isaac_settings_module):
    isaac_settings_module._apply_physx(_FakeScene(), 500.9, "TGS")

    physx_api = _FakePhysxApi.last_instance
    assert physx_api.time_steps_per_second.value == 500
    assert physx_api.solver_type.value == "TGS"


@pytest.mark.parametrize("rendering_hz", [0.0, -1.0])
def test_apply_render_rate_rejects_non_positive_rate(isaac_settings_module, rendering_hz):
    with pytest.raises(ValueError, match="rendering_hz must be positive"):
        isaac_settings_module._apply_render_rate(rendering_hz)


def test_apply_render_rate_writes_carb_settings(isaac_settings_module):
    isaac_settings_module._apply_render_rate(60.9)

    assert isaac_settings_module._test_carb_settings.values == {
        "/app/runLoops/main/rateLimitEnabled": True,
        "/app/runLoops/main/rateLimitFrequency": 60,
    }
