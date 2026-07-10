import subprocess
from pathlib import Path

from tests.support.paths import REPO_ROOT

_CHECKER = REPO_ROOT / "make/scripts/check_binary_files.py"


def _run_checker(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", str(_CHECKER), *(str(path) for path in paths)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_binary_asset_checker_accepts_text_and_grandfathered_asset(tmp_path: Path) -> None:
    text_file = tmp_path / "notes.txt"
    text_file.write_text("plain UTF-8 text\n")
    allowed_asset = REPO_ROOT / "isaac/assets/LightSwitchWarehouse/Lamp.usd"

    result = _run_checker(text_file, allowed_asset)

    assert result.returncode == 0, result.stdout + result.stderr


def test_binary_asset_checker_rejects_unlisted_binary(tmp_path: Path) -> None:
    binary_file = tmp_path / "new.usd"
    binary_file.write_bytes(b"PXR-USDC\0binary")

    result = _run_checker(binary_file)

    assert result.returncode == 1
    assert binary_file.relative_to(REPO_ROOT).as_posix() in result.stdout
