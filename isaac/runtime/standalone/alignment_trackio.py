"""Local TrackIO board for alignment progress, run through the Lab Docker image."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import trackio

from robotics_experiments.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", required=True, type=int)
    args = parser.parse_args()
    run = trackio.init(
        project="controller-alignment",
        name=args.run_dir.name,
        embed=False,
        auto_log_gpu=False,
        auto_log_cpu=False,
    )
    trackio.show(
        project="controller-alignment",
        open_browser=False,
        block_thread=False,
        host=args.host,
        server_port=args.port,
    )
    interval = load_config(args.config).alignment.execution.gpu_sample_interval_s
    previous = ""
    try:
        while True:
            try:
                progress = (args.run_dir / "progress.json").read_text()
                if progress != previous:
                    run.log(json.loads(progress))
                    previous = progress
            except FileNotFoundError:
                pass  # The board may start before the alignment run.
            except (OSError, ValueError) as exc:
                warnings.warn(
                    f"cannot read alignment progress: {exc}", RuntimeWarning, stacklevel=1
                )
            time.sleep(interval)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
