import importlib
import sys
import types

import pytest

from robotics_experiments.config import SimSettings


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


class _FakeUsdPhysicsScene:
    @staticmethod
    def Define(stage, path):
        stage.defined_scene_path = path
        stage.defined_scene = _FakeScene()
        return stage.defined_scene


class _FakeStage:
    def __init__(self):
        self.defined_scene_path = None
        self.defined_scene = None

    def Traverse(self):
        return []

    def GetDefaultPrim(self):
        return None


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
def simulation_api(monkeypatch):
    previous_modules = {
        name: sys.modules.pop(name)
        for name in (
            "robotics_experiments.simulation",
            "robotics_experiments.simulation.settings",
            "robotics_experiments.simulation._isaac_settings",
        )
        if name in sys.modules
    }
    carb_settings = _FakeCarbSettings()
    carb = types.SimpleNamespace(
        log_info=lambda message: None,
        settings=types.SimpleNamespace(get_settings=lambda: carb_settings),
    )
    pxr = types.ModuleType("pxr")
    pxr.Gf = types.SimpleNamespace(Vec3f=lambda x, y, z: (x, y, z))
    pxr.PhysxSchema = types.SimpleNamespace(PhysxSceneAPI=_FakePhysxApi)
    pxr.Usd = types.SimpleNamespace(Stage=object)
    pxr.UsdPhysics = types.SimpleNamespace(Scene=_FakeUsdPhysicsScene)

    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "pxr", pxr)
    monkeypatch.setitem(sys.modules, "pxr.Gf", pxr.Gf)
    monkeypatch.setitem(sys.modules, "pxr.PhysxSchema", pxr.PhysxSchema)
    monkeypatch.setitem(sys.modules, "pxr.Usd", pxr.Usd)
    monkeypatch.setitem(sys.modules, "pxr.UsdPhysics", pxr.UsdPhysics)

    simulation = importlib.import_module("robotics_experiments.simulation")
    yield simulation.apply_settings, carb_settings

    for name in (
        "robotics_experiments.simulation",
        "robotics_experiments.simulation.settings",
        "robotics_experiments.simulation._isaac_settings",
    ):
        sys.modules.pop(name, None)
    sys.modules.update(previous_modules)


def test_apply_settings_writes_gravity_physx_and_render_rate(simulation_api):
    apply_settings, carb_settings = simulation_api
    stage = _FakeStage()

    apply_settings(
        stage=stage,
        settings=SimSettings(
            physics_hz=500.9,
            rendering_hz=60.9,
            solver_type="TGS",
            gravity=(0.0, 0.0, -9.81),
        ),
    )

    assert stage.defined_scene_path == "/World/physicsScene"
    assert stage.defined_scene.gravity_direction.value == (0.0, 0.0, -1.0)
    assert stage.defined_scene.gravity_magnitude.value == 9.81
    physx_api = _FakePhysxApi.last_instance
    assert physx_api.time_steps_per_second.value == 500
    assert physx_api.solver_type.value == "TGS"
    assert carb_settings.values == {
        "/app/runLoops/main/rateLimitEnabled": True,
        "/app/runLoops/main/rateLimitFrequency": 60,
    }


def test_apply_settings_rejects_zero_magnitude_gravity(simulation_api):
    apply_settings, _carb_settings = simulation_api

    with pytest.raises(ValueError, match="Gravity vector must have positive magnitude"):
        apply_settings(
            stage=_FakeStage(),
            settings=SimSettings(
                physics_hz=500.0,
                rendering_hz=60.0,
                solver_type="TGS",
                gravity=(0.0, 0.0, 0.0),
            ),
        )


@pytest.mark.parametrize("physics_hz", [0.0, -1.0])
def test_apply_settings_rejects_non_positive_physics_rate(simulation_api, physics_hz):
    apply_settings, _carb_settings = simulation_api

    with pytest.raises(ValueError, match="physics_hz must be positive"):
        apply_settings(
            stage=_FakeStage(),
            settings=SimSettings(
                physics_hz=physics_hz,
                rendering_hz=60.0,
                solver_type="TGS",
                gravity=(0.0, 0.0, -9.81),
            ),
        )


@pytest.mark.parametrize("rendering_hz", [0.0, -1.0])
def test_apply_settings_rejects_non_positive_render_rate(simulation_api, rendering_hz):
    apply_settings, _carb_settings = simulation_api

    with pytest.raises(ValueError, match="rendering_hz must be positive"):
        apply_settings(
            stage=_FakeStage(),
            settings=SimSettings(
                physics_hz=500.0,
                rendering_hz=rendering_hz,
                solver_type="TGS",
                gravity=(0.0, 0.0, -9.81),
            ),
        )
