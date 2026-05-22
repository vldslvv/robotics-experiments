"""Standalone Isaac Sim smoke test for PX4 bridge host wiring.

This test verifies that the scene-owning bridge host can be constructed,
started, ticked, and stopped inside a real headless Isaac process without a
live PX4 instance. The PX4 bridge object is faked here; live MAVLink behavior
is covered by the separate PX4 smoke script.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh \
        /isaac-sim/proj/isaac/runtime/standalone/tests/test_px4_bridge_host.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from _runtime import SceneTestContext, run_scene_test

DEFAULT_SCENE = "@isaac/assets/LightSwitchWarehouse/Warehouse.usd"
TEST_NAME = "px4_bridge_host"


class _FakePx4Bridge:
    starts = 0
    stops = 0
    ticks = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def start(self):
        type(self).starts += 1

    def stop(self):
        type(self).stops += 1

    def on_physics_step(self, dt: float, sim_time_us: int):
        type(self).ticks.append((dt, sim_time_us))


class _FakePx4MavlinkPort:
    def __init__(self, *, num_rotors: int, **kwargs):
        self.num_rotors = num_rotors
        self.kwargs = kwargs


def _extension_root() -> Path:
    return Path(__file__).resolve().parents[2] / "extensions" / "drone_spawner"


def _run_test(context: SceneTestContext) -> None:
    sys.path.insert(0, str(_extension_root()))

    from DroneSpawner_python.bridge_host import BridgeHost

    px4_bridge = importlib.import_module("robotics_experiments.px4_bridge")
    px4_bridge.Px4Bridge = _FakePx4Bridge
    px4_bridge.Px4MavlinkPort = _FakePx4MavlinkPort

    _FakePx4Bridge.starts = 0
    _FakePx4Bridge.stops = 0
    _FakePx4Bridge.ticks = []

    cfg = SimpleNamespace(
        enabled=True,
        articulation_path="",
        rotor_link_paths=[
            "propulsion/motor_1/prop",
            "propulsion/motor_3/prop",
            "propulsion/motor_2/prop",
            "propulsion/motor_4/prop",
        ],
        num_rotors=4,
        max_rotor_force_n=8.0,
        connection_type="tcpin",
        connection_ip="localhost",
        connection_baseport=4560,
        vehicle_id=0,
        px4_autolaunch=False,
        px4_dir="",
        px4_vehicle_model="gazebo-classic_iris",
        lockstep_timeout_s=0.1,
        gps_send_rate_hz=5.0,
        home_lat_deg=47.3977,
        home_lon_deg=8.5456,
        home_alt_m=488.0,
    )

    host = BridgeHost(articulation_path="/World/DroneSpawned", config=cfg)
    host.start()
    host._on_physics_step(0.004)
    host.stop()

    if _FakePx4Bridge.starts != 1:
        raise AssertionError(f"Expected one bridge start, got {_FakePx4Bridge.starts}")
    if _FakePx4Bridge.stops != 1:
        raise AssertionError(f"Expected one bridge stop, got {_FakePx4Bridge.stops}")
    if _FakePx4Bridge.ticks != [(0.004, 4_000)]:
        raise AssertionError(f"Unexpected bridge ticks: {_FakePx4Bridge.ticks}")

    context.report(f"{TEST_NAME}: verified fake bridge start/tick/stop")


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
        test_body=_run_test,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
