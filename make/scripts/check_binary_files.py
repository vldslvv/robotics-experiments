"""Reject binary files outside the repository's narrow asset allowlist."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ALLOWLIST_PATH = _REPO_ROOT / ".binary-asset-allowlist"


def _allowed_paths() -> set[str]:
    return {
        line
        for raw_line in _ALLOWLIST_PATH.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }


def _is_binary(path: Path) -> bool:
    data = path.read_bytes()
    if b"\0" in data:
        return True
    try:
        data.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _main(paths: list[str]) -> int:
    allowed = _allowed_paths()
    rejected: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if not path.is_file() or not _is_binary(path):
            continue
        try:
            relative = path.relative_to(_REPO_ROOT).as_posix()
        except ValueError:
            relative = str(path)
        if relative not in allowed:
            rejected.append(relative)

    if rejected:
        print("Binary files are not allowed outside .binary-asset-allowlist:")
        for rejected_path in rejected:
            print(f"  {rejected_path}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
