"""Prepare the unchanged alignment reference inputs from flight parameters."""

from __future__ import annotations

import argparse
from pathlib import Path

from robotics_experiments.alignment import (
    load_controller_definition,
    prepare_alignment_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flight-parameters", type=Path, required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare_alignment_reference(
            flight_parameters_artifact=args.flight_parameters,
            definition=load_controller_definition(args.definition),
            run_dir=args.run_dir,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"alignment input artifact validation failed: {exc}") from None


if __name__ == "__main__":
    main()
