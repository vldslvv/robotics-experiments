"""Lossless storage and streaming access for producer-owned flight evidence."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, cast

__all__: list[str] = []


def resolve_flight_evidence(path: Path, /) -> Path:
    """Resolve a logical log name to its one stored representation."""

    compressed = path.with_name(path.name + ".gz")
    if path.is_file() and compressed.is_file():
        raise ValueError(f"ambiguous flight evidence: {path}")
    if path.is_file():
        return path
    if compressed.is_file():
        return compressed
    raise FileNotFoundError(path)


@contextmanager
def open_flight_evidence(path: Path, /) -> Iterator[BinaryIO]:
    """Open an exact evidence path, including gzip behind a consumer symlink."""

    with path.open("rb") as raw:
        if raw.peek(2)[:2] == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=raw, mode="rb") as decoded:
                yield cast(BinaryIO, decoded)
        else:
            yield raw


def flight_evidence_sha256(path: Path, /) -> str:
    """Hash the original contents, independent of their storage encoding."""

    digest = hashlib.sha256()
    with open_flight_evidence(path) as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compress_flight_evidence(*, path: Path, level: int) -> Path:
    """Atomically write gzip evidence; leave original removal to the caller."""

    if not 0 <= level <= 9:
        raise ValueError("gzip level must be between 0 and 9")
    destination = path.with_name(path.name + ".gz")
    pending = destination.with_name(destination.name + ".pending")
    with path.open("rb") as source, pending.open("wb") as stored:
        with gzip.GzipFile(
            filename="", fileobj=stored, mode="wb", compresslevel=level, mtime=0
        ) as target:
            shutil.copyfileobj(source, target)
        stored.flush()
        os.fsync(stored.fileno())
    pending.replace(destination)
    return destination
