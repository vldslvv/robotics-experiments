"""Evidence survives compression, interrupted writes and consumer symlinks."""

import gzip
import hashlib
import os
from pathlib import Path

import pytest

from robotics_experiments.flight_parameters import (
    compress_flight_evidence,
    flight_evidence_sha256,
    open_flight_evidence,
)


def test_lossless_evidence_and_consumer_symlink(tmp_path: Path) -> None:
    original = tmp_path / "flight.ulg"
    data = bytes(range(256)) * 4096
    original.write_bytes(data)
    compressed = compress_flight_evidence(path=original, level=6)
    assert original.read_bytes() == data
    assert compressed.stat().st_size < len(data)
    assert flight_evidence_sha256(compressed) == hashlib.sha256(data).hexdigest()
    assert compress_flight_evidence(path=original, level=6) == compressed
    original.unlink()
    consumer = tmp_path / "reference.ulg"
    consumer.symlink_to(compressed.name)
    with open_flight_evidence(consumer) as stream:
        assert stream.read() == data
    encoded = bytearray(compressed.read_bytes())
    encoded[-8] ^= 1
    compressed.write_bytes(encoded)
    with pytest.raises(gzip.BadGzipFile):
        flight_evidence_sha256(compressed)


def test_interrupted_compression_retains_original_and_can_retry(
    tmp_path: Path, monkeypatch
) -> None:
    original = tmp_path / "bridge.log"
    original.write_text("complete bridge evidence\n" * 100)
    before = original.read_bytes()

    def interrupted_fsync(fd: int, /) -> None:
        raise OSError("injected interrupted write")

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", interrupted_fsync)
        with pytest.raises(OSError, match="interrupted"):
            compress_flight_evidence(path=original, level=6)
    assert original.read_bytes() == before
    assert not original.with_suffix(".log.gz").exists()
    compressed = compress_flight_evidence(path=original, level=6)
    assert gzip.decompress(compressed.read_bytes()) == before
    assert not original.with_suffix(".log.gz.pending").exists()
