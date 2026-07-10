import subprocess
import sys
from pathlib import Path

from tests.support.paths import REPO_ROOT


def _top_level_test_areas() -> set[str]:
    tests_root = REPO_ROOT / "tests"
    areas = set()
    for test_path in tests_root.rglob("test_*.py"):
        relative = test_path.relative_to(tests_root)
        if len(relative.parts) > 1:
            areas.add(relative.parts[0])
    return areas


def test_default_collection_includes_every_host_test_area(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            f"--basetemp={tmp_path / 'collection'}",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    missing = {area for area in _top_level_test_areas() if f"tests/{area}/" not in result.stdout}
    assert not missing, f"default pytest collection omitted host test areas: {sorted(missing)}"
