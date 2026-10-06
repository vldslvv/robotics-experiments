"""Finalize and publish one completed controller-alignment evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from robotics_experiments.alignment._artifact import finalize_alignment_bundle, publish_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    finalize_alignment_bundle(run_dir=args.run_dir)
    publish_controller(run_dir=args.run_dir)
    print(f"published controller evaluation {args.run_dir}")


if __name__ == "__main__":
    main()
