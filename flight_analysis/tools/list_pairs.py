"""List likely PX4 ULog and bridge-log pairs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

__all__ = ["main"]


@dataclass(frozen=True)
class _LogFile:
    path: Path
    stem: str
    seconds: int | None
    mtime: float


def _stem_seconds(stem: str) -> int | None:
    parts = stem.split("_")
    if len(parts) != 3:
        return None
    try:
        hour, minute, second = (int(part) for part in parts)
    except ValueError:
        return None
    return hour * 3600 + minute * 60 + second


def _collect_logs(root: Path) -> tuple[dict[Path, list[_LogFile]], dict[Path, list[_LogFile]]]:
    ulogs: dict[Path, list[_LogFile]] = {}
    bridges: dict[Path, list[_LogFile]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.endswith("_bridge.log"):
            stem = path.name[: -len("_bridge.log")]
            seconds = _stem_seconds(stem)
            if seconds is None:
                continue
            bridges.setdefault(path.parent, []).append(
                _LogFile(path=path, stem=stem, seconds=seconds, mtime=path.stat().st_mtime)
            )
        elif path.suffix == ".ulg":
            stem = path.stem
            seconds = _stem_seconds(stem)
            if seconds is None:
                continue
            ulogs.setdefault(path.parent, []).append(
                _LogFile(path=path, stem=stem, seconds=seconds, mtime=path.stat().st_mtime)
            )
    return ulogs, bridges


def _nearest(reference: _LogFile, candidates: list[_LogFile]) -> tuple[_LogFile | None, int | None]:
    exact = [candidate for candidate in candidates if candidate.stem == reference.stem]
    if exact:
        return exact[0], 0
    if reference.seconds is None:
        return None, None
    with_seconds = [candidate for candidate in candidates if candidate.seconds is not None]
    if not with_seconds:
        return None, None
    candidate = min(with_seconds, key=lambda item: abs(item.seconds - reference.seconds))
    return candidate, abs(candidate.seconds - reference.seconds) if candidate.seconds is not None else None


def _format_delta(seconds: int | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds}s"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="px4/logs", type=Path)
    parser.add_argument(
        "--max-stem-delta-s",
        type=int,
        default=2,
        help="Maximum HH_MM_SS delta to call a pair likely when stems differ.",
    )
    args = parser.parse_args(argv)

    ulogs, bridges = _collect_logs(args.root)
    rows: list[tuple[str, str, str, str]] = []
    for directory in sorted(set(ulogs) | set(bridges)):
        for bridge in sorted(bridges.get(directory, []), key=lambda item: item.stem):
            ulog, delta = _nearest(bridge, ulogs.get(directory, []))
            status = "paired" if ulog and (delta == 0 or (delta is not None and delta <= args.max_stem_delta_s)) else "check"
            rows.append(
                (
                    status,
                    _format_delta(delta),
                    str(ulog.path if ulog else ""),
                    str(bridge.path),
                )
            )

    if not rows:
        print(f"No bridge logs found under {args.root}")
        return 1

    print("status  delta  ulog  bridge")
    for status, delta, ulog, bridge in rows:
        print(f"{status:6}  {delta:5}  {ulog}  {bridge}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
