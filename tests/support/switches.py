"""Small text producer bundles for host-only switch identity tests."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from robotics_experiments.config import SwitchModelConfig
from robotics_experiments.switches import SWITCH_GENERATOR_REVISION, SwitchAsset, switch_fingerprint

__all__ = ["write_switch_asset"]


def write_switch_asset(*, root: Path, model: SwitchModelConfig) -> SwitchAsset:
    fingerprint = switch_fingerprint(model)
    directory = root / ".build/assets/light-switches" / fingerprint
    directory.mkdir(parents=True)
    path = directory / "switch.usda"
    path.write_text('#usda 1.0\ndef Xform "TestSwitch" {}\n')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "generator_revision": SWITCH_GENERATOR_REVISION,
                "definition_fingerprint": fingerprint,
                "model": asdict(model),
                "usd_digest": digest,
            }
        )
    )
    return SwitchAsset(path=path, fingerprint=fingerprint, usd_digest=digest)
