"""Standalone Isaac Sim smoke test for PX4 bridge host wiring.

This test verifies that the scene-owning bridge host can be constructed,
started, ticked, and stopped inside a real headless Isaac process while
connected to a live PX4 SITL instance running in the PX4 Docker container.
The test intentionally does not autolaunch or fake PX4.

Run inside the Isaac Sim container:
    /isaac-sim/python.sh \
        /isaac-sim/proj/isaac/runtime/standalone/tests/test_px4_bridge_host.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from _runtime import SceneTestContext, run_scene_test

DEFAULT_SCENE = "@isaac/assets/LightSwitchWarehouse/Warehouse.usd"
TEST_NAME = "px4_bridge_host"
PX4_CONTAINER_NAME = os.environ.get("PX4_TEST_CONTAINER_NAME", "vv-px4-sitl")
PHYSICS_DT_S = 0.004
HEARTBEAT_POLL_INTERVAL_S = 0.05


class _SyntheticArticulationIO:
    """Minimal simulator-side I/O so this test isolates live PX4 wiring."""

    is_initialized = True

    def __init__(self):
        self.applied_forces = []

    def initialize(self) -> None:
        return

    def read_state(self, sim_time_us: int):
        from robotics_experiments.px4_bridge import VehicleState

        return VehicleState.hover_at(sim_time_us=sim_time_us)

    def apply_forces(self, forces_n) -> None:
        self.applied_forces.append(forces_n)

    def shutdown(self) -> None:
        return


def _extension_root() -> Path:
    return Path(__file__).resolve().parents[2] / "extensions" / "drone_spawner"


def _px4_bridge_extension_root() -> Path:
    return Path(__file__).resolve().parents[2] / "extensions" / "px4_sitl_brigde"


def _running_px4_container_status() -> str | None:
    """Return Docker's running state for the PX4 container, if Docker is usable."""
    try:
        result = subprocess.run(
            [
                "docker",
                "inspect",
                "--format",
                "{{.State.Running}}",
                PX4_CONTAINER_NAME,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"Timed out checking PX4 Docker container {PX4_CONTAINER_NAME!r}"
        ) from exc

    if result.returncode == 0:
        return result.stdout.strip().lower()

    stderr = result.stderr.strip()
    lower_stderr = stderr.lower()
    if (
        "cannot connect to the docker daemon" in lower_stderr
        or "permission denied" in lower_stderr
        or "is the docker daemon running" in lower_stderr
    ):
        return None

    raise AssertionError(
        f"Expected running PX4 Docker container {PX4_CONTAINER_NAME!r}; "
        f"docker inspect failed: {stderr or result.stdout.strip()}"
    )


def _require_running_px4_container(context: SceneTestContext) -> None:
    context.report(f"{TEST_NAME}: checking PX4 Docker container {PX4_CONTAINER_NAME!r}")
    running = _running_px4_container_status()
    if running is None:
        context.report(
            f"{TEST_NAME}: Docker is not available here; requiring live PX4 MAVLink heartbeat"
        )
        return
    if running != "true":
        raise AssertionError(
            f"PX4 Docker container {PX4_CONTAINER_NAME!r} is not running; "
            "start PX4 before running this test"
        )
    context.report(f"{TEST_NAME}: verified PX4 Docker container is running")


def _bridge_received_first_heartbeat(host) -> bool:
    bridge = getattr(host, "_bridge", None)
    port = getattr(bridge, "_port", None)
    backend = getattr(port, "_backend", None)
    return bool(getattr(backend, "_received_first_hearbeat", False))


def _bridge_is_connected(host) -> bool:
    bridge = getattr(host, "_bridge", None)
    return bool(bridge is not None and bridge.is_connected)


def _wait_for_bridge_start(context: SceneTestContext, host, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    context.report(f"{TEST_NAME}: waiting up to {timeout_s:.1f}s for bridge host start")
    while time.monotonic() < deadline:
        host._on_physics_step(PHYSICS_DT_S)
        if host.is_running:
            context.report(f"{TEST_NAME}: bridge host started")
            return

    raise AssertionError("PX4 bridge host did not start")


def _install_synthetic_articulation_io(host) -> _SyntheticArticulationIO:
    bridge = getattr(host, "_bridge", None)
    if bridge is None:
        raise AssertionError("PX4 bridge host did not construct a bridge")
    io = _SyntheticArticulationIO()
    bridge._io = io
    return io


def _run_test(context: SceneTestContext) -> None:
    _require_running_px4_container(context)

    sys.path.insert(0, str(_extension_root()))
    sys.path.insert(0, str(_px4_bridge_extension_root()))

    from DroneSpawner_python.bridge_host import BridgeHost

    from robotics_experiments.config import get_config
    from robotics_experiments.drone import despawn_drone, spawn_drone

    context.report(f"{TEST_NAME}: loading experiment config")
    env = get_config()

    default_prim = context.stage.GetDefaultPrim()
    parent = default_prim.GetPath().pathString if default_prim else "/World"
    articulation_path = env.px4.articulation_path or f"{parent}/{env.drone.prim_name}"
    endpoint = (
        f"{env.px4.connection_type}:{env.px4.connection_ip}:"
        f"{env.px4.connection_baseport + env.px4.vehicle_id}"
    )
    context.report(f"{TEST_NAME}: using drone articulation path {articulation_path}")
    context.report(f"{TEST_NAME}: expecting live PX4 MAVLink on {endpoint}")

    context.report(f"{TEST_NAME}: constructing bridge host")
    host = BridgeHost(
        articulation_path=articulation_path,
        config=env.px4,
        physics_hz=env.sim.physics_hz,
    )
    startup_settle_time_s = max(0.0, float(env.px4.startup_settle_time_s))

    context.report(f"{TEST_NAME}: spawning configured drone")
    spawn_drone(context.stage, env.drone)
    try:
        context.report(f"{TEST_NAME}: starting bridge host")
        host.start()
        _wait_for_bridge_start(
            context,
            host,
            timeout_s=startup_settle_time_s + 1.0,
        )

        context.report(f"{TEST_NAME}: installing synthetic simulator-side articulation I/O")
        io = _install_synthetic_articulation_io(host)
        deadline = time.monotonic() + float(env.px4.first_heartbeat_timeout_s) + 1.0
        context.report(
            f"{TEST_NAME}: waiting up to {env.px4.first_heartbeat_timeout_s:.1f}s "
            "for live PX4 heartbeat"
        )
        while time.monotonic() < deadline:
            host._on_physics_step(PHYSICS_DT_S)
            if _bridge_received_first_heartbeat(host):
                context.report(f"{TEST_NAME}: received live PX4 heartbeat")
                break
            if not _bridge_is_connected(host):
                raise AssertionError("PX4 bridge disconnected before receiving a live heartbeat")
            time.sleep(HEARTBEAT_POLL_INTERVAL_S)
        else:
            raise AssertionError(
                "Timed out waiting for live PX4 heartbeat; ensure PX4 SITL is running "
                f"in container {PX4_CONTAINER_NAME!r} and reachable on {endpoint}"
            )
        if not io.applied_forces:
            raise AssertionError("PX4 bridge host did not apply any synthetic tick forces")
        context.report(f"{TEST_NAME}: bridge applied synthetic actuator forces")
    finally:
        context.report(f"{TEST_NAME}: stopping bridge host and despawning drone")
        host.stop()
        despawn_drone(context.stage, env.drone)

    context.report(f"{TEST_NAME}: verified real bridge start/tick/stop")


def main() -> None:
    run_scene_test(
        test_name=TEST_NAME,
        default_scene=DEFAULT_SCENE,
        test_body=_run_test,
        description=__doc__ or "",
    )


if __name__ == "__main__":
    main()
