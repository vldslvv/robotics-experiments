"""Score captured alignment traces, write alignment.json, and publish links."""

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
from robotics_experiments.alignment._io import (
    publish_alignment,
    write_alignment_artifact,
)


def _diagnostic_image(path: Path, /) -> str:
    with path.open() as diagnostics:
        for line in diagnostics:
            stripped = line.strip()
            if stripped.startswith('"image":'):
                value = json.loads("{" + stripped.rstrip(",") + "}").get("image")
                if isinstance(value, str) and value:
                    return value
                break
            if stripped.startswith('"candidates":') or stripped.startswith('"traces":'):
                break
    raise ValueError(f"{path} has no image header")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    os.environ["ROBOTICS_ALIGNMENT_RUN_DIR"] = str(args.run_dir)
    target = replace(
        load_controller_definition(args.target),
        image=_diagnostic_image(args.run_dir / "target_candidates.json"),
    )
    reference = replace(
        load_controller_definition(args.reference),
        image=_diagnostic_image(args.run_dir / "reference.json"),
    )
    artifact = replace(
        align(
            target,
            reference,
        ),
        run_id=args.run_id,
    )
    write_alignment_artifact(artifact=artifact, run_dir=args.run_dir)
    publish_alignment(artifact=artifact, run_dir=args.run_dir)
    print(
        f"alignment {artifact.alignment_id}/{artifact.run_id}: "
        f"candidate={artifact.selected_candidate_index} "
        f"quality={artifact.advisory_quality_passed}"
    )


if __name__ == "__main__":
    main()
