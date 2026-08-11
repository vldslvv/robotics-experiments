"""Tests for production waypoint handoff measurement aggregation."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from tests.support.paths import REPO_ROOT


def _load_summarizer():
    path = REPO_ROOT / "make/scripts/summarize_press_navigation_handoff.py"
    spec = importlib.util.spec_from_file_location("summarize_press_navigation_handoff", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_sample(
    *,
    path: Path,
    position: tuple[float, float, float],
    route_y_m: float = 17.52,
    attitude_error_rad: float = 0.0,
    quaternion_scale: float = 1.0,
) -> None:
    quaternion = [
        quaternion_scale * math.sin(attitude_error_rad / 2.0),
        0.0,
        0.0,
        quaternion_scale * math.cos(attitude_error_rad / 2.0),
    ]
    observation = [*position, *quaternion, *(0.0 for _ in range(27))]
    path.write_text(
        json.dumps(
            {
                "artifact_kind": "press-navigation-handoff-sample",
                "schema_version": 2,
                "observation_schema_version": 1,
                "observation": observation,
                "relative_switch_pos_m": observation[:3],
                "relative_switch_quat_xyzw": observation[3:7],
                "route": {
                    "goto": {
                        "kind": "goto",
                        "x": 0.0,
                        "y": route_y_m,
                        "z": 2.5,
                        "ground_speed_m_s": 0.08,
                        "position_tol_m": 0.06,
                        "timeout_s": 20.0,
                    },
                    "settle_hold_s": 2.0,
                },
            }
        ),
        encoding="utf-8",
    )


def test_summary_covers_measured_position_extrema_with_margin(tmp_path: Path) -> None:
    module = _load_summarizer()
    paths = []
    for index in range(10):
        path = tmp_path / f"sample_{index:03d}.json"
        _write_sample(
            path=path,
            position=(0.46 + index * 0.001, index * 0.001, index * -0.002),
            attitude_error_rad=0.01 * index,
        )
        paths.append(path)

    summary = module._summary(  # noqa: SLF001
        paths=paths,
        position_margin_m=0.005,
        rotation_margin_rad=math.radians(0.25),
    )

    observed = summary["observed"]
    required = summary["required_handoff_and_training_envelope"]
    assert summary["sample_count"] == 10
    assert observed["switch_range_min_m"] == pytest.approx(0.46)
    assert observed["switch_range_max_m"] == pytest.approx(0.469)
    assert observed["lateral_align_max_m"] == pytest.approx((0.009**2 + 0.018**2) ** 0.5)
    assert observed["relative_attitude_error_max_rad"] == pytest.approx(0.09)
    assert required["switch_range_min_m"] == pytest.approx(0.455)
    assert required["switch_range_max_m"] == pytest.approx(0.474)
    assert required["lateral_align_max_m"] == pytest.approx((0.009**2 + 0.018**2) ** 0.5 + 0.005)
    assert required["relative_attitude_error_max_rad"] == pytest.approx(0.09 + math.radians(0.25))


def test_summary_requires_ten_identical_production_route_samples(tmp_path: Path) -> None:
    module = _load_summarizer()
    paths = []
    for index in range(10):
        path = tmp_path / f"sample_{index:03d}.json"
        _write_sample(
            path=path,
            position=(0.48, 0.0, 0.0),
            route_y_m=17.51 if index == 9 else 17.52,
        )
        paths.append(path)

    with pytest.raises(ValueError, match="identical production route"):
        module._summary(  # noqa: SLF001
            paths=paths,
            position_margin_m=0.005,
            rotation_margin_rad=math.radians(0.25),
        )
    with pytest.raises(ValueError, match="at least 10 samples"):
        module._summary(  # noqa: SLF001
            paths=paths[:9],
            position_margin_m=0.005,
            rotation_margin_rad=math.radians(0.25),
        )


def test_schema_two_rejects_a_non_normalized_relative_quaternion(tmp_path: Path) -> None:
    module = _load_summarizer()
    path = tmp_path / "sample.json"
    _write_sample(path=path, position=(0.48, 0.0, 0.0), quaternion_scale=2.0)

    with pytest.raises(ValueError, match="normalized quaternion"):
        module._read_sample(path=path)  # noqa: SLF001
