"""Summarize independent production-route press handoff pose samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

_ARTIFACT_KIND = "press-navigation-handoff-sample"
_SUMMARY_KIND = "press-navigation-handoff-summary"
_SCHEMA_VERSION = 2
_OBSERVATION_DIM = 34


def _read_sample(*, path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_kind") != _ARTIFACT_KIND
        or payload.get("schema_version") != _SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported press navigation handoff sample: {path}")
    observation = payload.get("observation")
    if (
        not isinstance(observation, list)
        or len(observation) != _OBSERVATION_DIM
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in observation
        )
        or not all(math.isfinite(float(value)) for value in observation)
    ):
        raise ValueError(f"press navigation sample has an invalid observation: {path}")
    relative_position = payload.get("relative_switch_pos_m")
    if relative_position != observation[:3]:
        raise ValueError(f"press navigation sample position disagrees with observation: {path}")
    relative_quaternion = payload.get("relative_switch_quat_xyzw")
    if relative_quaternion != observation[3:7]:
        raise ValueError(f"press navigation sample attitude disagrees with observation: {path}")
    quaternion = np.asarray(relative_quaternion, dtype=np.float64)
    if quaternion.shape != (4,) or not math.isclose(
        float(np.linalg.norm(quaternion)), 1.0, rel_tol=0.0, abs_tol=1.0e-5
    ):
        raise ValueError(f"press navigation sample attitude is not a normalized quaternion: {path}")
    return payload


def _summary(
    *,
    paths: list[Path],
    position_margin_m: float,
    rotation_margin_rad: float,
) -> dict[str, object]:
    if len(paths) < 10:
        raise ValueError("press navigation handoff measurement requires at least 10 samples")
    if not math.isfinite(position_margin_m) or position_margin_m <= 0.0:
        raise ValueError("position margin must be finite and positive")
    if not math.isfinite(rotation_margin_rad) or rotation_margin_rad <= 0.0:
        raise ValueError("rotation margin must be finite and positive")
    samples = [_read_sample(path=path) for path in paths]
    route = samples[0]["route"]
    if any(sample["route"] != route for sample in samples[1:]):
        raise ValueError("press navigation samples do not use one identical production route")
    positions = np.asarray(
        [sample["relative_switch_pos_m"] for sample in samples],
        dtype=np.float64,
    )
    lateral_errors = np.linalg.norm(positions[:, 1:3], axis=1)
    quaternions = np.asarray(
        [sample["relative_switch_quat_xyzw"] for sample in samples],
        dtype=np.float64,
    )
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    relative_attitude_errors = 2.0 * np.arccos(np.clip(np.abs(quaternions[:, 3]), 0.0, 1.0))
    observed_range_min_m = float(np.min(positions[:, 0]))
    observed_range_max_m = float(np.max(positions[:, 0]))
    observed_lateral_max_m = float(np.max(lateral_errors))
    observed_relative_attitude_error_max_rad = float(np.max(relative_attitude_errors))
    return {
        "artifact_kind": _SUMMARY_KIND,
        "schema_version": _SCHEMA_VERSION,
        "sample_count": len(samples),
        "position_margin_m": position_margin_m,
        "rotation_margin_rad": rotation_margin_rad,
        "route": route,
        "observed": {
            "relative_position_min_m": [float(value) for value in np.min(positions, axis=0)],
            "relative_position_max_m": [float(value) for value in np.max(positions, axis=0)],
            "switch_range_min_m": observed_range_min_m,
            "switch_range_max_m": observed_range_max_m,
            "lateral_align_max_m": observed_lateral_max_m,
            "relative_attitude_error_max_rad": observed_relative_attitude_error_max_rad,
        },
        "required_handoff_and_training_envelope": {
            "switch_range_min_m": observed_range_min_m - position_margin_m,
            "switch_range_max_m": observed_range_max_m + position_margin_m,
            "lateral_align_max_m": observed_lateral_max_m + position_margin_m,
            "relative_attitude_error_max_rad": (
                observed_relative_attitude_error_max_rad + rotation_margin_rad
            ),
        },
        "samples": [
            {
                "path": str(path),
                "relative_switch_pos_m": sample["relative_switch_pos_m"],
                "relative_switch_quat_xyzw": sample["relative_switch_quat_xyzw"],
                "relative_attitude_error_rad": float(relative_attitude_errors[index]),
            }
            for index, (path, sample) in enumerate(zip(paths, samples, strict=True))
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--position-margin-m", type=float, required=True)
    parser.add_argument("--rotation-margin-rad", type=float, required=True)
    parser.add_argument("samples", nargs="+", type=Path)
    args = parser.parse_args()
    summary = _summary(
        paths=list(args.samples),
        position_margin_m=float(args.position_margin_m),
        rotation_margin_rad=float(args.rotation_margin_rad),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
