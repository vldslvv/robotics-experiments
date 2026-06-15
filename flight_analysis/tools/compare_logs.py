"""Compare two PX4 ULogs and report hover/dynamics divergence.

The comparison is symmetric and origin-agnostic: both inputs are just PX4 ULogs.
It works for any pairing (sim-vs-real, sim-vs-sim, real-vs-real). Neither log is a
privileged reference; swapping the two positional inputs only flips the sign of
each delta. The delta column is reported as ``B - A`` purely for presentation.

The tool only reads standard PX4 estimator/actuator topics that exist regardless
of whether a log came from SITL or hardware, so it never branches on log origin.
Note: ``vehicle_local_position`` is NED, so +z is down.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from _ulog_common import dataset, euler_deg
from pyulog import ULog

__all__ = ["main"]

# Aiding-source independent: only estimator/actuator outputs present in every log.
_ARM_COMMAND = 400
_TAKEOFF_COMMAND = 22


def _anchor_seconds(ulog: ULog, align: str) -> float:
    """Return the time origin (seconds) for a log given an alignment mode."""
    if align == "none":
        return 0.0
    command = _ARM_COMMAND if align == "arm" else _TAKEOFF_COMMAND
    acks = dataset(ulog, "vehicle_command_ack")
    for index, timestamp in enumerate(acks["timestamp"]):
        if int(acks["command"][index]) == command and int(acks["result"][index]) == 0:
            return float(timestamp) / 1_000_000.0
    raise KeyError(f"no accepted vehicle_command_ack for command={command} (align={align})")


def _rel_time(data: dict, anchor_s: float) -> np.ndarray:
    return np.asarray(data["timestamp"], dtype=np.float64) / 1_000_000.0 - anchor_s


def _window_mask(rel_t: np.ndarray, window: tuple[float, float]) -> np.ndarray:
    start, end = window
    return (rel_t >= start) & (rel_t <= end)


def _resolve_window(
    ulog: ULog, anchor_s: float, window: tuple[float, float] | None, hover_last: float
) -> tuple[float, float]:
    """Pick the comparison window, anchor-relative, for one log."""
    if window is not None:
        return window
    rel_t = _rel_time(dataset(ulog, "vehicle_local_position"), anchor_s)
    end = float(rel_t[-1])
    return (end - hover_last, end)


def _stats_for_log(ulog: ULog, anchor_s: float, window: tuple[float, float]) -> dict[str, float]:
    pos = dataset(ulog, "vehicle_local_position")
    mask = _window_mask(_rel_time(pos, anchor_s), window)
    x, y, z = pos["x"][mask], pos["y"][mask], pos["z"][mask]
    vx, vy, vz = pos["vx"][mask], pos["vy"][mask], pos["vz"][mask]
    radial = np.hypot(x - x.mean(), y - y.mean())
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    att = dataset(ulog, "vehicle_attitude")
    amask = _window_mask(_rel_time(att, anchor_s), window)
    q1, q2 = att["q[1]"][amask], att["q[2]"][amask]
    # Tilt = angle between body-up and world-up; frame-agnostic magnitude.
    tilt = np.degrees(np.arccos(np.clip(1.0 - 2.0 * (q1**2 + q2**2), -1.0, 1.0)))
    yaw = np.array(
        [euler_deg([att[f"q[{j}]"][i] for j in range(4)])[2] for i in np.flatnonzero(amask)]
    )

    stats = {
        "samples": float(mask.sum()),
        "horiz_x_range_m": float(x.max() - x.min()),
        "horiz_y_range_m": float(y.max() - y.min()),
        "horiz_radial_rms_m": float(np.sqrt((radial**2).mean())),
        "horiz_radial_max_m": float(radial.max()),
        "vert_z_range_m": float(z.max() - z.min()),
        "speed_max_mps": float(speed.max()),
        "speed_mean_mps": float(speed.mean()),
        "tilt_mean_deg": float(tilt.mean()),
        "tilt_range_deg": float(tilt.max() - tilt.min()),
        "tilt_max_deg": float(tilt.max()),
        "yaw_range_deg": float(yaw.max() - yaw.min()),
    }

    try:
        rates = dataset(ulog, "vehicle_angular_velocity")
        rmask = _window_mask(_rel_time(rates, anchor_s), window)
        for axis, name in enumerate(("roll", "pitch", "yaw")):
            deg = np.degrees(rates[f"xyz[{axis}]"][rmask])
            stats[f"rate_{name}_maxabs_dps"] = float(np.abs(deg).max())
            stats[f"rate_{name}_std_dps"] = float(deg.std())
    except KeyError:
        pass

    try:
        motors = dataset(ulog, "actuator_motors")
        mmask = _window_mask(_rel_time(motors, anchor_s), window)
        controls = np.array([motors[f"control[{j}]"][mmask] for j in range(4)])  # shape (4, n)
        stats["motor_mean"] = float(controls.mean())
        stats["motor_spread_mean"] = float((controls.max(axis=0) - controls.min(axis=0)).mean())
    except KeyError:
        pass

    return stats


def _print_table(
    labels: tuple[str, str],
    windows: tuple[tuple[float, float], tuple[float, float]],
    stats_a: dict[str, float],
    stats_b: dict[str, float],
) -> None:
    la, lb = labels
    print(f"window {la}: [{windows[0][0]:.2f}, {windows[0][1]:.2f}]s (anchor-relative)")
    print(f"window {lb}: [{windows[1][0]:.2f}, {windows[1][1]:.2f}]s (anchor-relative)")
    print()
    name_w = max(len(k) for k in stats_a)
    print(f"{'metric':<{name_w}}  {la:>14}  {lb:>14}  {'delta (B-A)':>14}")
    print("-" * (name_w + 3 * 16))
    for key in stats_a:
        a = stats_a[key]
        b = stats_b.get(key, float("nan"))
        print(f"{key:<{name_w}}  {a:>14.4f}  {b:>14.4f}  {b - a:>14.4f}")


def _write_csv(
    path: Path,
    labels: tuple[str, str],
    ulogs: tuple[ULog, ULog],
    anchors: tuple[float, float],
    windows: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Resample both logs' position onto a shared grid over their windows."""
    grid = np.linspace(0.0, 1.0, 200)  # normalized 0..1 across each window
    columns: list[tuple[str, np.ndarray]] = [("frac", grid)]
    for label, ulog, anchor, window in zip(labels, ulogs, anchors, windows, strict=False):
        pos = dataset(ulog, "vehicle_local_position")
        rel_t = _rel_time(pos, anchor)
        start, end = window
        t_norm = grid * (end - start) + start
        for field in ("x", "y", "z"):
            columns.append((f"{label}_{field}", np.interp(t_norm, rel_t, pos[field])))
    header = ",".join(name for name, _ in columns)
    rows = np.column_stack([values for _, values in columns])
    np.savetxt(path, rows, delimiter=",", header=header, comments="")
    print(f"\nwrote aligned series: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log_a", type=Path)
    parser.add_argument("log_b", type=Path)
    parser.add_argument(
        "--align",
        choices=("takeoff", "arm", "none"),
        default="takeoff",
        help="Flight event to zero each log's clock on (default takeoff).",
    )
    parser.add_argument(
        "--window",
        help="START,END seconds relative to the alignment anchor.",
    )
    parser.add_argument(
        "--hover-last",
        type=float,
        default=2.0,
        help="Window = last N seconds of each log when --window is unset.",
    )
    parser.add_argument("--labels", help="Comma-separated display names, e.g. sim,real.")
    parser.add_argument("--csv", type=Path, help="Write aligned resampled series.")
    args = parser.parse_args(argv)

    if args.labels:
        parts = [p.strip() for p in args.labels.split(",")]
        labels = (parts[0], parts[1])
    else:
        labels = (args.log_a.stem, args.log_b.stem)

    window = None
    if args.window:
        start, end = (float(p) for p in args.window.split(","))
        window = (start, end)

    ulog_a = ULog(str(args.log_a))
    ulog_b = ULog(str(args.log_b))
    anchor_a = _anchor_seconds(ulog_a, args.align)
    anchor_b = _anchor_seconds(ulog_b, args.align)
    window_a = _resolve_window(ulog_a, anchor_a, window, args.hover_last)
    window_b = _resolve_window(ulog_b, anchor_b, window, args.hover_last)

    stats_a = _stats_for_log(ulog_a, anchor_a, window_a)
    stats_b = _stats_for_log(ulog_b, anchor_b, window_b)

    print(f"align={args.align}  positions are NED (+z down)\n")
    _print_table(labels, (window_a, window_b), stats_a, stats_b)

    if args.csv:
        _write_csv(args.csv, labels, (ulog_a, ulog_b), (anchor_a, anchor_b), (window_a, window_b))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
