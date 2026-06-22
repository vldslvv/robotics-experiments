"""Shared runtime helpers for standalone Isaac Sim integration tests.

These helpers intentionally use the real Isaac standalone workflow: start a
``SimulationApp``, load a real USD stage through ``omni.usd``, reset the Hydra
config cache, and hand the loaded stage to an extension-specific test body.
Mocks should stay out of these tests unless a future extension dependency
cannot reasonably run inside standalone Isaac.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from isaacsim import SimulationApp

TEST_OUTPUT_PREFIX = "[isaac-standalone-test]"

# Sentinel ``--scene`` value meaning "load the scene from the Hydra ``scene``
# config" rather than an explicit path. The integration harness always injects
# ``--scene "$(SCENE)"``; a target that wants the config-driven scene sets
# ``SCENE := @config``.
SCENE_FROM_CONFIG = "@config"

__all__ = [
    "SceneTestContext",
    "TimingReporter",
    "format_vehicle_state_vectors",
    "format_xyz",
    "report",
    "run_scene_test",
]


def report(message: str) -> None:
    """Emit a line intended to survive Makefile log filtering."""
    print(f"{TEST_OUTPUT_PREFIX} {message}", file=sys.__stderr__, flush=True)


def format_xyz(label: str, values: Any, unit: str) -> str:
    """Format a 3-vector with stable x/y/z field names for integration logs."""
    return (
        f"{label}_{unit}="
        f"(x={float(values[0]):.2f}, y={float(values[1]):.2f}, z={float(values[2]):.2f})"
    )


def format_vehicle_state_vectors(state_or_sample: Any) -> str:
    """Format full ENU position and linear velocity vectors for flight triage."""
    return (
        f"{format_xyz('pos_enu', state_or_sample.position_enu, 'm')} "
        f"{format_xyz('vel_enu', state_or_sample.linear_velocity_enu, 'm_s')}"
    )


@dataclass(frozen=True)
class SceneTestContext:
    """Context passed to extension-specific standalone test bodies."""

    stage: object
    scene_path: Path
    simulation_app: SimulationApp
    report: Callable[[str], None]

    def report_vehicle_state(self, label: str, state_or_sample: Any) -> None:
        """Emit a standardized drone state vector log line."""
        self.report(f"{label}: {format_vehicle_state_vectors(state_or_sample)}")


class TimingReporter:
    """Emit elapsed wall-clock timings for standalone integration stages."""

    def __init__(self, test_name: str, *, emit: Callable[[str], None] | None = None):
        self._test_name = test_name
        self._emit = emit or report
        self._total_start_s = time.monotonic()

    def report_stage(self, stage: str, start_s: float, *, status: str = "ok") -> None:
        now_s = time.monotonic()
        self._emit(
            "TIMING "
            f"test={self._test_name} "
            f"stage={stage} "
            f"elapsed_s={now_s - start_s:.3f} "
            f"total_s={now_s - self._total_start_s:.3f} "
            f"status={status}"
        )

    @contextmanager
    def timed_stage(self, stage: str, *, sim_clock: object | None = None) -> Iterator[None]:
        """Report one timed stage, optionally including simulated elapsed time."""
        start_s = time.monotonic()
        start_us = int(getattr(sim_clock, "sim_time_us", 0)) if sim_clock is not None else None
        status = "ok"
        try:
            yield
        except BaseException:
            status = "fail"
            raise
        finally:
            now_s = time.monotonic()
            parts = [
                "TIMING",
                f"test={self._test_name}",
                f"stage={stage}",
                f"elapsed_s={now_s - start_s:.3f}",
                f"total_s={now_s - self._total_start_s:.3f}",
                f"status={status}",
            ]
            if start_us is not None:
                end_us = int(getattr(sim_clock, "sim_time_us", 0))
                parts.append(f"sim_elapsed_s={(end_us - start_us) * 1e-6:.3f}")
            self._emit(" ".join(parts))

    def start(self) -> float:
        return time.monotonic()

    def total_start(self) -> float:
        return self._total_start_s


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_scene_path(scene: str) -> Path:
    if scene.startswith("@"):
        return (_repo_root() / scene[1:]).resolve()
    return Path(scene).expanduser().resolve()


def _parse_scene_args(description: str, default_scene: str | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--scene",
        default=default_scene,
        help=(
            "USD scene path to load. Prefix with @ for a repository-relative "
            f"path. Pass {SCENE_FROM_CONFIG!r} (or omit) to load the scene from "
            "the Hydra scene config."
        ),
    )
    return parser.parse_args()


def _resolve_scene_from_config(test_name: str) -> str:
    """Return the experiment ``scene`` from the composed Hydra config.

    Used when no explicit ``--scene`` is given (or the sentinel is passed). The
    config composes without Isaac, so this is safe before ``SimulationApp``
    starts; the test body later resets and recomposes the cache.
    """
    from robotics_experiments.config import get_config, reset_config

    reset_config()
    scene = get_config().scene
    report(f"{test_name}: scene from scene config: {scene}")
    return scene


def run_scene_test(
    *,
    test_name: str,
    test_body: Callable[[SceneTestContext], None],
    description: str,
    default_scene: str | None = None,
) -> None:
    """Run a standalone Isaac scene test against Hydra-backed project config.

    Args:
        test_name: Short human-readable test name shown in filtered output.
        test_body: Extension-specific assertions to run after the stage loads.
        description: CLI description for the test script.
        default_scene: Default USD scene path when ``--scene`` is not passed.
            ``@`` prefixes are resolved relative to the repository root. When
            ``None`` (or when ``--scene`` is the :data:`SCENE_FROM_CONFIG`
            sentinel), the scene is read from the Hydra ``scene`` config.
    """
    args = _parse_scene_args(description, default_scene)
    scene = args.scene
    if scene in (None, SCENE_FROM_CONFIG):
        scene = _resolve_scene_from_config(test_name)
    scene_path = _resolve_scene_path(scene)
    report(f"{test_name}: scene argument: {scene}")
    report(f"{test_name}: resolved scene path: {scene_path}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene does not exist: {scene_path}")

    simulation_app = None
    error: BaseException | None = None
    timing = TimingReporter(test_name)
    try:
        stage_start_s = timing.start()
        report(f"{test_name}: starting SimulationApp")
        simulation_app = SimulationApp({"headless": True, "hide_ui": True})
        report(f"{test_name}: SimulationApp started")
        timing.report_stage("simulation_app_start", stage_start_s)

        stage_start_s = timing.start()
        report(f"{test_name}: importing omni.usd")
        import omni.usd

        timing.report_stage("import_omni_usd", stage_start_s)

        stage_start_s = timing.start()
        report(f"{test_name}: resetting Hydra config cache")
        from robotics_experiments.config import reset_config

        reset_config()
        timing.report_stage("hydra_reset", stage_start_s)

        stage_start_s = timing.start()
        report(f"{test_name}: getting USD context")
        usd_context = omni.usd.get_context()
        report(f"{test_name}: opening stage")
        if not usd_context.open_stage(str(scene_path)):
            raise RuntimeError(f"Failed to open stage: {scene_path}")

        simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError(f"Stage did not load: {scene_path}")
        timing.report_stage("stage_open", stage_start_s)

        stage_start_s = timing.start()
        test_body(
            SceneTestContext(
                stage=stage,
                scene_path=scene_path,
                simulation_app=simulation_app,
                report=report,
            )
        )
        timing.report_stage("test_body", stage_start_s)
        report(f"{test_name}: PASS")
    except SystemExit as exc:
        error = RuntimeError(f"{test_name}: Isaac test exited before completion: {exc}")
        report(f"ERROR: {error}")
    except BaseException as exc:
        error = exc
        report(f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        if simulation_app is not None:
            stage_start_s = timing.start()
            try:
                simulation_app.close()
                timing.report_stage("simulation_app_close", stage_start_s)
            except SystemExit as exc:
                timing.report_stage("simulation_app_close", stage_start_s, status="fail")
                if error is None:
                    error = RuntimeError(
                        f"{test_name}: SimulationApp.close exited unexpectedly: {exc}"
                    )
        timing.report_stage("total", timing.total_start(), status="fail" if error else "ok")
    if error is not None:
        raise error
