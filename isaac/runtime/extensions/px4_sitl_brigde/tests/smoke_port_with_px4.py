"""Stage-2 smoke test: ``Px4MavlinkPort`` against a live PX4 SITL.

Requires:
  - PX4 SITL reachable on the configured ``connection_*`` (default
    ``tcpin:localhost:4560``). Bridge listens; PX4 must dial in.
  - ``pymavlink`` installed in the running Python interpreter.

Run inside the Isaac Sim Kit Python (or any env that has ``pymavlink`` +
``numpy`` + ``scipy``):

    python tests/smoke_port_with_px4.py

The script does NOT use Isaac Sim. It feeds the port a synthesized hovering
``VehicleState`` at ~250 Hz with a monotonic ``sim_time_us`` and prints the
actuator commands PX4 sends back, plus connection/armed status. Use this to
verify lockstep, GPS/mag synthesis, and that PX4's EKF converges before
wiring the bridge into the scene.

Stop with Ctrl-C; the port closes cleanly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

EXT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXT_ROOT))

from Px4SitlBridge_python import (  # noqa: E402
    LockstepSettings,
    Px4MavlinkPort,
)

from robotics_experiments.px4_bridge import VehicleState, read_bridge_status  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--num-rotors", type=int, default=4)
    p.add_argument("--connection-type", default="tcpin")
    p.add_argument("--connection-ip", default="localhost")
    p.add_argument("--port", type=int, default=4560)
    p.add_argument("--rate-hz", type=float, default=250.0)
    p.add_argument("--duration-s", type=float, default=30.0)
    p.add_argument("--lockstep-timeout-s", type=float, default=0.1)
    args = p.parse_args()

    port = Px4MavlinkPort(
        num_rotors=args.num_rotors,
        connection_type=args.connection_type,
        connection_ip=args.connection_ip,
        port=args.port,
        lockstep=LockstepSettings(lockstep_timeout_s=args.lockstep_timeout_s),
    )
    port.start()
    print(
        f"port started; waiting for PX4 to dial in on {args.connection_type}"
        f":{args.connection_ip}:{args.port}"
    )

    period_s = 1.0 / args.rate_hz
    sim_time_us = 0
    deadline = time.time() + args.duration_s
    last_print = 0.0
    disconnected = False
    try:
        while time.time() < deadline:
            tick_start = time.time()

            state = VehicleState.hover_at(sim_time_us=sim_time_us)
            commands = port.exchange(state)
            status = read_bridge_status(port=port)
            if not status.is_connected:
                disconnected = True

            now = time.time()
            if now - last_print > 1.0:
                print(
                    f"t={sim_time_us/1e6:7.3f}s  "
                    f"connected={status.is_connected}  armed={status.is_armed}  "
                    f"heartbeat={getattr(port, '_first_heartbeat_received_s', None) is not None}  "
                    f"sensors={getattr(port, '_sensor_send_count', 0)}  "
                    f"actuators={getattr(port, '_actuator_receive_count', 0)}  "
                    f"last_sensor_us={getattr(port, '_last_sensor_send_us', None)}  "
                    f"last_actuator_us={getattr(port, '_last_actuator_receive_us', None)}  "
                    f"disconnect={getattr(port, '_last_disconnect_reason', None)!r}  "
                    f"cmd={np.round(commands.controls, 3).tolist()}"
                )
                last_print = now
            if disconnected:
                break

            sim_time_us += int(period_s * 1_000_000)
            elapsed = time.time() - tick_start
            if elapsed < period_s:
                time.sleep(period_s - elapsed)
    except KeyboardInterrupt:
        print("interrupted")
    finally:
        port.stop()
        print("port stopped")
    if disconnected:
        print(
            "port disconnected before duration elapsed: "
            f"{getattr(port, '_last_disconnect_reason', None)!r}"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
