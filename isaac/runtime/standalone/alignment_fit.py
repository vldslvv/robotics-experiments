"""Fit or finalize one controller-alignment bundle."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

from robotics_experiments.alignment import (
    align,
    load_controller_definition,
)
from robotics_experiments.alignment._artifact import (
    finalize_alignment_bundle,
    publish_controller,
    write_alignment_result,
)


def _diagnostic_image(path: Path, /) -> str:
    with path.open(encoding="utf-8") as diagnostics:
        header = diagnostics.read(64 * 1024)
    image_key = header.find('"image"')
    data_boundary = min(
        (index for key in ('"candidates"', '"traces"') if (index := header.find(key)) >= 0),
        default=len(header),
    )
    if 0 <= image_key < data_boundary:
        separator = header.find(":", image_key + len('"image"'))
        if 0 <= separator < data_boundary:
            payload = header[separator + 1 : data_boundary].lstrip()
            value, _end = json.JSONDecoder().raw_decode(payload)
            if isinstance(value, str) and value:
                return value
    raise ValueError(f"{path} has no image header")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize_alignment_bundle(run_dir=args.run_dir)
        publish_controller(run_dir=args.run_dir)
        print(f"finalized controller bundle {args.run_dir}")
        return
    if args.target is None or args.reference is None or args.run_id is None:
        parser.error("fitting requires --target, --reference, and --run-id")
    os.environ["ROBOTICS_ALIGNMENT_RUN_DIR"] = str(args.run_dir)
    target = replace(
        load_controller_definition(args.target),
        image=_diagnostic_image(args.run_dir / "target_candidates.json"),
    )
    reference = replace(
        load_controller_definition(args.reference),
        image=_diagnostic_image(args.run_dir / "reference.json"),
    )
    result = align(target, reference)
    result = replace(
        result,
        controller=replace(result.controller, run_id=args.run_id),
    )
    write_alignment_result(result=result, run_dir=args.run_dir)
    print(
        f"alignment {result.controller.alignment_id}/{result.controller.run_id}: "
        f"candidate={result.selected_candidate_index} "
        f"quality={result.controller.advisory_quality_passed}"
    )


if __name__ == "__main__":
    main()
