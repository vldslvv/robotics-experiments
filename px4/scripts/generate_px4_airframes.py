#!/usr/bin/env python3
"""Generate PX4 posix SITL airframes from repo airframe YAML configs."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__: list[str] = []

_PX4_AIRFRAME_TYPE_BY_MAV_TYPE = {
    2: "Quadrotor x",
    13: "Hexarotor x",
}


@dataclass(frozen=True)
class _RotorAllocation:
    index: int
    px: float
    py: float
    km: float


@dataclass(frozen=True)
class _GeneratedAirframe:
    source: str
    output_filename: str
    script_name: str
    sim_model: str
    sys_autostart: int
    mav_type: int
    rotors: tuple[_RotorAllocation, ...]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping")
    return data


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{path}.{key} is required")
    return mapping[key]


def _as_float_tuple(value: Any, length: int, path: str) -> tuple[float, ...]:
    if not isinstance(value, list | tuple) or len(value) != length:
        raise ValueError(f"{path} must be a {length}-item sequence")
    return tuple(float(item) for item in value)


def _round(value: float) -> float:
    rounded = round(value, 6)
    return 0.0 if rounded == -0.0 else rounded


def _rotor_geometry(
    data: dict[str, Any], *, km_abs: float, source: str
) -> tuple[_RotorAllocation, ...]:
    arm_geometry = _required(data, "arm_geometry", source)
    motor = _required(arm_geometry, "motor", f"{source}.arm_geometry")
    motor_position = _as_float_tuple(
        _required(motor, "position", f"{source}.arm_geometry.motor"),
        3,
        f"{source}.arm_geometry.motor.position",
    )
    arms = _required(data, "arms", source)
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"{source}.arms must be a non-empty list")

    rotors: dict[int, _RotorAllocation] = {}
    local_x, local_y, _local_z = motor_position
    for arm_index, arm in enumerate(arms):
        if not isinstance(arm, dict):
            raise ValueError(f"{source}.arms[{arm_index}] must be a mapping")
        motor_index = int(_required(arm, "px4_motor_index", f"{source}.arms[{arm_index}]"))
        if motor_index in rotors:
            raise ValueError(f"{source}.arms has duplicate px4_motor_index={motor_index}")
        azimuth_rad = math.radians(
            float(_required(arm, "azimuth_deg", f"{source}.arms[{arm_index}]"))
        )
        cos_a = math.cos(azimuth_rad)
        sin_a = math.sin(azimuth_rad)
        # Config geometry is body FLU. PX4 CA_ROTOR position uses body FRD.
        x_flu = local_x * cos_a - local_y * sin_a
        y_flu = local_x * sin_a + local_y * cos_a
        spin_dir = str(_required(arm, "spin_dir", f"{source}.arms[{arm_index}]")).upper()
        if spin_dir == "CW":
            km = km_abs
        elif spin_dir == "CCW":
            km = -km_abs
        else:
            raise ValueError(f"{source}.arms[{arm_index}].spin_dir must be CW or CCW")
        rotors[motor_index] = _RotorAllocation(
            index=motor_index,
            px=_round(x_flu),
            py=_round(-y_flu),
            km=_round(km),
        )

    expected = list(range(len(arms)))
    actual = sorted(rotors)
    if actual != expected:
        raise ValueError(
            f"{source}.arms px4_motor_index values must be contiguous {expected}, got {actual}"
        )
    return tuple(rotors[index] for index in expected)


def _airframe_from_config(path: Path) -> _GeneratedAirframe:
    data = _load_yaml(path)
    source = str(_required(data, "name", path.name))
    output_filename = str(_required(data, "output_filename", source))
    px4 = _required(data, "px4", source)
    sitl = _required(px4, "sitl", f"{source}.px4")
    sim_model = str(_required(sitl, "sim_model", f"{source}.px4.sitl"))
    sys_autostart = int(_required(sitl, "sys_autostart", f"{source}.px4.sitl"))
    mav_type = int(_required(sitl, "mav_type", f"{source}.px4.sitl"))
    km_abs = float(_required(sitl, "ca_rotor_km_abs", f"{source}.px4.sitl"))
    if not sim_model.startswith("none_"):
        raise ValueError(f"{source}.px4.sitl.sim_model must start with none_")
    if sys_autostart <= 0:
        raise ValueError(f"{source}.px4.sitl.sys_autostart must be positive")
    if km_abs <= 0.0:
        raise ValueError(f"{source}.px4.sitl.ca_rotor_km_abs must be positive")
    rotors = _rotor_geometry(data, km_abs=km_abs, source=source)
    return _GeneratedAirframe(
        source=source,
        output_filename=output_filename,
        script_name=f"{sys_autostart}_{sim_model}",
        sim_model=sim_model,
        sys_autostart=sys_autostart,
        mav_type=mav_type,
        rotors=rotors,
    )


def _airframes(config_dir: Path) -> tuple[_GeneratedAirframe, ...]:
    paths = sorted(config_dir.glob("*.yaml"))
    if not paths:
        raise ValueError(f"no airframe configs found in {config_dir}")
    airframes = tuple(_airframe_from_config(path) for path in paths)
    script_names = [airframe.script_name for airframe in airframes]
    if len(script_names) != len(set(script_names)):
        raise ValueError(f"duplicate PX4 airframe script names: {script_names}")
    return airframes


def _script_text(airframe: _GeneratedAirframe) -> str:
    airframe_type = _PX4_AIRFRAME_TYPE_BY_MAV_TYPE.get(
        airframe.mav_type, "External simulator multicopter"
    )
    lines = [
        "#!/bin/sh",
        "#",
        f"# @name {airframe.source}",
        "#",
        f"# @type {airframe_type}",
        "#",
        "# Generated from configs/airframes. Do not edit this file by hand.",
        "# Isaac/Pegasus provides dynamics over MAVLink; the none_* model uses",
        "# PX4's external simulator_mavlink path and only defines control allocation.",
        "#",
        "",
        ". ${R}etc/init.d/rc.mc_defaults",
        "",
        "param set-default CA_AIRFRAME 0",
        f"param set-default MAV_TYPE {airframe.mav_type}",
        "",
        f"param set-default CA_ROTOR_COUNT {len(airframe.rotors)}",
    ]
    for rotor in airframe.rotors:
        lines.extend(
            [
                f"param set-default CA_ROTOR{rotor.index}_PX {rotor.px:g}",
                f"param set-default CA_ROTOR{rotor.index}_PY {rotor.py:g}",
                f"param set-default CA_ROTOR{rotor.index}_KM {rotor.km:g}",
            ]
        )
    lines.append("")
    for rotor in airframe.rotors:
        lines.append(f"param set-default PWM_MAIN_FUNC{rotor.index + 1} {101 + rotor.index}")
    lines.append("")
    return "\n".join(lines)


def _manifest(airframes: tuple[_GeneratedAirframe, ...]) -> dict[str, Any]:
    return {
        "version": 1,
        "airframes": [
            {
                "source": airframe.source,
                "output_filename": airframe.output_filename,
                "script_name": airframe.script_name,
                "sim_model": airframe.sim_model,
                "sys_autostart": airframe.sys_autostart,
                "mav_type": airframe.mav_type,
                "rotors": [
                    {"index": rotor.index, "px": rotor.px, "py": rotor.py, "km": rotor.km}
                    for rotor in airframe.rotors
                ],
            }
            for airframe in airframes
        ],
    }


def _manifest_text(airframes: tuple[_GeneratedAirframe, ...]) -> str:
    return json.dumps(_manifest(airframes), indent=2, sort_keys=True) + "\n"


def _write_airframe_scripts(airframes: tuple[_GeneratedAirframe, ...], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for airframe in airframes:
        path = output_dir / airframe.script_name
        path.write_text(_script_text(airframe))
        path.chmod(0o755)


def _patch_cmake_list(cmake_list: Path, script_names: tuple[str, ...]) -> None:
    text = cmake_list.read_text()
    missing = [
        name
        for name in script_names
        if not re.search(rf"^\s*{re.escape(name)}\s*$", text, re.M)
    ]
    if not missing:
        return

    anchor = re.search(r"^(?P<indent>\s*)10016_none_iris\s*$", text, re.M)
    if anchor is None:
        raise ValueError(f"cannot find 10016_none_iris anchor in {cmake_list}")

    insert = "".join(f"{anchor.group('indent')}{name}\n" for name in missing)
    text = text[: anchor.end() + 1] + insert + text[anchor.end() + 1 :]
    for name in missing:
        if not re.search(rf"^\s*{re.escape(name)}\s*$", text, re.M):
            raise ValueError(f"failed to register {name} in {cmake_list}")
    cmake_list.write_text(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cmake-list", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--print-manifest", action="store_true")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    airframes = _airframes(args.config_dir)

    if args.print_manifest:
        print(_manifest_text(airframes), end="")

    if args.output_dir is not None:
        _write_airframe_scripts(airframes, args.output_dir)

    if args.cmake_list is not None:
        _patch_cmake_list(args.cmake_list, tuple(airframe.script_name for airframe in airframes))

    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(_manifest_text(airframes))


if __name__ == "__main__":
    _main()
