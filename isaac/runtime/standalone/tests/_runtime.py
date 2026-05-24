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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from isaacsim import SimulationApp

TEST_OUTPUT_PREFIX = "[isaac-standalone-test]"


@dataclass(frozen=True)
class SceneTestContext:
    """Context passed to extension-specific standalone test bodies."""

    stage: object
    scene_path: Path
    simulation_app: object
    report: Callable[[str], None]


def report(message: str) -> None:
    """Emit a line intended to survive Makefile log filtering."""
    print(f"{TEST_OUTPUT_PREFIX} {message}", file=sys.__stderr__, flush=True)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _resolve_scene_path(scene: str) -> Path:
    if scene.startswith("@"):
        return (_repo_root() / scene[1:]).resolve()
    return Path(scene).expanduser().resolve()


def _parse_scene_args(description: str, default_scene: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--scene",
        default=default_scene,
        help="USD scene path to load. Prefix with @ for a repository-relative path.",
    )
    return parser.parse_args()


def run_scene_test(
    *,
    test_name: str,
    default_scene: str,
    test_body: Callable[[SceneTestContext], None],
    description: str,
) -> None:
    """Run a standalone Isaac scene test against Hydra-backed project config.

    Args:
        test_name: Short human-readable test name shown in filtered output.
        default_scene: Default USD scene path. ``@`` prefixes are resolved
            relative to the repository root.
        test_body: Extension-specific assertions to run after the stage loads.
        description: CLI description for the test script.
    """
    args = _parse_scene_args(description, default_scene)
    scene_path = _resolve_scene_path(args.scene)
    report(f"{test_name}: scene argument: {args.scene}")
    report(f"{test_name}: resolved scene path: {scene_path}")
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene does not exist: {scene_path}")

    simulation_app = None
    error: BaseException | None = None
    try:
        report(f"{test_name}: starting SimulationApp")
        simulation_app = SimulationApp({"headless": True, "hide_ui": True})
        report(f"{test_name}: SimulationApp started")

        report(f"{test_name}: importing omni.usd")
        import omni.usd

        report(f"{test_name}: resetting Hydra config cache")
        from robotics_experiments.config import reset_config

        reset_config()
        report(f"{test_name}: getting USD context")
        usd_context = omni.usd.get_context()
        report(f"{test_name}: opening stage")
        if not usd_context.open_stage(str(scene_path)):
            raise RuntimeError(f"Failed to open stage: {scene_path}")

        simulation_app.update()
        stage = usd_context.get_stage()
        if stage is None:
            raise RuntimeError(f"Stage did not load: {scene_path}")

        test_body(
            SceneTestContext(
                stage=stage,
                scene_path=scene_path,
                simulation_app=simulation_app,
                report=report,
            )
        )
        report(f"{test_name}: PASS")
    except SystemExit as exc:
        error = RuntimeError(f"{test_name}: Isaac test exited before completion: {exc}")
        report(f"ERROR: {error}")
    except BaseException as exc:
        error = exc
        report(f"ERROR: {type(exc).__name__}: {exc}")
    finally:
        if simulation_app is not None:
            try:
                simulation_app.close()
            except SystemExit as exc:
                if error is None:
                    error = RuntimeError(
                        f"{test_name}: SimulationApp.close exited unexpectedly: {exc}"
                    )
    if error is not None:
        raise error
