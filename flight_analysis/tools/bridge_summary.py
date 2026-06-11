"""Summarize key events from a PX4 bridge log."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

__all__ = ["main"]

_SENSOR_RE = re.compile(
    r"sensor stream: sensors=(?P<sensors>\d+) actuators=(?P<actuators>\d+) "
    r"sim_time_us=(?P<sim>\d+) last_actuator_us=(?P<actuator>\S+) "
    r"actuator_lag_us=(?P<lag>\S+) connected=(?P<connected>\S+)"
)
_STOP_RE = re.compile(
    r"stopped: sensors=(?P<sensors>\d+) actuators=(?P<actuators>\d+) "
    r"last_sensor_us=(?P<sensor>\S+) last_actuator_us=(?P<actuator>\S+) "
    r"last_disconnect=(?P<disconnect>.*)$"
)


def _append_first(events: dict[str, str], key: str, line: str) -> None:
    events.setdefault(key, line.strip())


def _print_section(title: str, rows: list[str]) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if rows:
        for row in rows:
            print(row)
    else:
        print("none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bridge_log", type=Path)
    args = parser.parse_args(argv)

    events: dict[str, str] = {}
    counters: Counter[str] = Counter()
    sensor_rows: list[str] = []
    actuator_rows: list[str] = []
    stop_rows: list[str] = []
    warning_rows: list[str] = []
    rotor_rows: list[str] = []

    with args.bridge_log.open("r", encoding="utf-8", errors="replace") as handle:
        for lineno, line in enumerate(handle, start=1):
            if "PX4 MAVLink port constructed" in line:
                _append_first(events, "constructed", f"{lineno}: {line}")
            if "PX4 first heartbeat received" in line:
                _append_first(events, "heartbeat", f"{lineno}: {line}")
            if "PX4 first HIL_SENSOR sent" in line:
                _append_first(events, "first_sensor", f"{lineno}: {line}")
            if "PX4 first HIL_ACTUATOR_CONTROLS received" in line:
                _append_first(events, "first_actuator", f"{lineno}: {line}")
            if "PX4 actuator bootstrap lag accepted" in line:
                _append_first(events, "bootstrap_lag", f"{lineno}: {line}")
            if "rotor mapping resolved" in line:
                rotor_rows.append(f"{lineno}: {line.strip()}")
            if "sensor stream:" in line:
                counters["sensor_stream"] += 1
                sensor_rows.append(f"{lineno}: {line.strip()}")
            if "HIL_ACTUATOR_CONTROLS received:" in line:
                counters["actuator_debug"] += 1
                actuator_rows.append(f"{lineno}: {line.strip()}")
            if "tilt cutoff" in line:
                counters["tilt_cutoff"] += 1
                warning_rows.append(f"{lineno}: {line.strip()}")
            if "disconnected" in line or "timed out" in line or "timestamp stale" in line:
                counters["disconnect_or_timeout"] += 1
                warning_rows.append(f"{lineno}: {line.strip()}")
            if "PX4 MAVLink port stopped" in line:
                stop_rows.append(f"{lineno}: {line.strip()}")

    print(f"Bridge log: {args.bridge_log}")
    for key in ["constructed", "heartbeat", "first_sensor", "first_actuator", "bootstrap_lag"]:
        print(f"{key}: {events.get(key, 'missing')}")

    _print_section("Rotor Mapping", rotor_rows[:2])

    last_sensor = sensor_rows[-5:]
    parsed_sensor = []
    for row in last_sensor:
        match = _SENSOR_RE.search(row)
        if match:
            parsed_sensor.append(
                "line={line} sensors={sensors} actuators={actuators} "
                "sim={sim} last_actuator={actuator} lag={lag} connected={connected}".format(
                    line=row.split(":", 1)[0],
                    **match.groupdict(),
                )
            )
        else:
            parsed_sensor.append(row)
    _print_section("Last Sensor Streams", parsed_sensor)
    _print_section("Last Actuator Debug Lines", actuator_rows[-5:])
    _print_section("Warnings And Disconnects", warning_rows[-20:])

    parsed_stops = []
    for row in stop_rows[-3:]:
        match = _STOP_RE.search(row)
        if match:
            parsed_stops.append(
                "line={line} sensors={sensors} actuators={actuators} "
                "last_sensor={sensor} last_actuator={actuator} disconnect={disconnect}".format(
                    line=row.split(":", 1)[0],
                    **match.groupdict(),
                )
            )
        else:
            parsed_stops.append(row)
    _print_section("Stop Counters", parsed_stops)

    print("\nCounts")
    print("------")
    for key in sorted(counters):
        print(f"{key}: {counters[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
