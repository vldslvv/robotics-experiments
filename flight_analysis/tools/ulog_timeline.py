"""Print a focused PX4 ULog flight timeline."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from pyulog import ULog

__all__ = ["main"]


def _dataset(ulog: ULog, name: str) -> dict:
    for item in ulog.data_list:
        if item.name == name:
            return item.data
    raise KeyError(f"ULog topic not found: {name}")


def _index_at(data: dict, seconds: float) -> int:
    timestamp = np.asarray(data["timestamp"], dtype=np.float64)
    return int(np.argmin(np.abs(timestamp - seconds * 1_000_000.0)))


def _value(data: dict, key: str, index: int, default: float = float("nan")) -> float:
    if key not in data:
        return default
    return float(data[key][index])


def _bool_value(data: dict, key: str, index: int) -> int:
    if key not in data:
        return -1
    return int(data[key][index])


def _euler_deg(q: list[float]) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _samples(args: argparse.Namespace) -> list[float]:
    if args.samples:
        return [float(part) for part in args.samples.split(",") if part.strip()]
    values = []
    current = args.start
    while current <= args.end + 1e-9:
        values.append(round(current, 6))
        current += args.step
    return values


def _print_topic_ranges(ulog: ULog, names: list[str]) -> None:
    print("Topic ranges")
    print("------------")
    for name in names:
        try:
            data = _dataset(ulog, name)
        except KeyError:
            print(f"{name}: missing")
            continue
        timestamp = data["timestamp"]
        print(
            f"{name}: n={len(timestamp)} "
            f"start={timestamp[0] / 1_000_000.0:.3f}s "
            f"end={timestamp[-1] / 1_000_000.0:.3f}s"
        )


def _print_commands(ulog: ULog) -> None:
    print("\nCommands")
    print("--------")
    try:
        commands = _dataset(ulog, "vehicle_command")
        for index, timestamp in enumerate(commands["timestamp"]):
            print(
                f"cmd t={timestamp / 1_000_000.0:.3f}s "
                f"command={commands['command'][index]} "
                f"p1={commands['param1'][index]} p2={commands['param2'][index]} "
                f"p7={commands['param7'][index]}"
            )
    except KeyError:
        print("vehicle_command: missing")

    try:
        acks = _dataset(ulog, "vehicle_command_ack")
        for index, timestamp in enumerate(acks["timestamp"]):
            print(
                f"ack t={timestamp / 1_000_000.0:.3f}s "
                f"command={acks['command'][index]} result={acks['result'][index]} "
                f"result_param1={acks['result_param1'][index]}"
            )
    except KeyError:
        print("vehicle_command_ack: missing")


def _print_timeline(ulog: ULog, sample_seconds: list[float]) -> None:
    att = _dataset(ulog, "vehicle_attitude")
    attitude_sp = _dataset(ulog, "vehicle_attitude_setpoint")
    rates_sp = _dataset(ulog, "vehicle_rates_setpoint")
    local_position = _dataset(ulog, "vehicle_local_position")
    trajectory = _dataset(ulog, "trajectory_setpoint")
    motors = _dataset(ulog, "actuator_motors")
    outputs = _dataset(ulog, "actuator_outputs")
    thrust_sp = _dataset(ulog, "vehicle_thrust_setpoint")
    torque_sp = _dataset(ulog, "vehicle_torque_setpoint")
    land = _dataset(ulog, "vehicle_land_detected")
    takeoff = _dataset(ulog, "takeoff_status")
    failsafe = _dataset(ulog, "failsafe_flags")
    estimator = _dataset(ulog, "estimator_status_flags")

    print("\nTimeline")
    print("--------")
    header = (
        "t  att_rpy  sp_rpy  rates_sp  thrust_z  torque_xyz  motors  outputs  "
        "pos_xyz  vel_xyz  traj_xyz  landed ground takeoff fs_att reject_yaw"
    )
    print(header)
    for seconds in sample_seconds:
        ia = _index_at(att, seconds)
        isp = _index_at(attitude_sp, seconds)
        ir = _index_at(rates_sp, seconds)
        il = _index_at(local_position, seconds)
        it = _index_at(trajectory, seconds)
        im = _index_at(motors, seconds)
        io = _index_at(outputs, seconds)
        ith = _index_at(thrust_sp, seconds)
        itq = _index_at(torque_sp, seconds)
        ild = _index_at(land, seconds)
        ito = _index_at(takeoff, seconds)
        ifs = _index_at(failsafe, seconds)
        ies = _index_at(estimator, seconds)

        attitude = _euler_deg([_value(att, f"q[{j}]", ia) for j in range(4)])
        setpoint = _euler_deg([_value(attitude_sp, f"q_d[{j}]", isp) for j in range(4)])
        motor_values = [_value(motors, f"control[{j}]", im) for j in range(4)]
        output_values = [_value(outputs, f"output[{j}]", io) for j in range(4)]
        print(
            f"{seconds:5.2f} "
            f"{attitude[0]:6.1f},{attitude[1]:6.1f},{attitude[2]:6.1f} "
            f"{setpoint[0]:6.1f},{setpoint[1]:6.1f},{setpoint[2]:6.1f} "
            f"{_value(rates_sp, 'roll', ir):5.2f},{_value(rates_sp, 'pitch', ir):5.2f},{_value(rates_sp, 'yaw', ir):5.2f} "
            f"{_value(thrust_sp, 'xyz[2]', ith):7.3f} "
            f"{_value(torque_sp, 'xyz[0]', itq):5.2f},{_value(torque_sp, 'xyz[1]', itq):5.2f},{_value(torque_sp, 'xyz[2]', itq):5.2f} "
            f"{','.join(f'{value:.3f}' for value in motor_values)} "
            f"{','.join(f'{value:.0f}' for value in output_values)} "
            f"{_value(local_position, 'x', il):5.2f},{_value(local_position, 'y', il):5.2f},{_value(local_position, 'z', il):5.2f} "
            f"{_value(local_position, 'vx', il):5.2f},{_value(local_position, 'vy', il):5.2f},{_value(local_position, 'vz', il):5.2f} "
            f"{_value(trajectory, 'position[0]', it):5.2f},{_value(trajectory, 'position[1]', it):5.2f},{_value(trajectory, 'position[2]', it):5.2f} "
            f"{_bool_value(land, 'landed', ild):3d} "
            f"{_bool_value(land, 'ground_contact', ild):3d} "
            f"{_bool_value(takeoff, 'takeoff_state', ito):3d} "
            f"{_bool_value(failsafe, 'attitude_invalid', ifs):3d} "
            f"{_bool_value(estimator, 'reject_yaw', ies):3d}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ulog", type=Path)
    parser.add_argument("--samples", help="Comma-separated sample seconds.")
    parser.add_argument("--start", type=float, default=10.0)
    parser.add_argument("--end", type=float, default=15.0)
    parser.add_argument("--step", type=float, default=0.5)
    parser.add_argument(
        "--no-ranges",
        action="store_true",
        help="Skip topic range output.",
    )
    args = parser.parse_args(argv)

    ulog = ULog(str(args.ulog))
    topic_names = [
        "vehicle_command",
        "vehicle_command_ack",
        "vehicle_status",
        "takeoff_status",
        "vehicle_land_detected",
        "trajectory_setpoint",
        "vehicle_local_position",
        "vehicle_attitude",
        "vehicle_attitude_setpoint",
        "vehicle_rates_setpoint",
        "vehicle_thrust_setpoint",
        "vehicle_torque_setpoint",
        "actuator_motors",
        "actuator_outputs",
        "failsafe_flags",
        "estimator_status_flags",
    ]
    if not args.no_ranges:
        _print_topic_ranges(ulog, topic_names)
    _print_commands(ulog)
    _print_timeline(ulog, _samples(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
