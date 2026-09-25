"""Generated switch identity, read-only validation and shared timed activation."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import torch

from robotics_experiments.config import SwitchModelConfig

__all__ = [
    "SWITCH_GENERATOR_REVISION",
    "SwitchAsset",
    "switch_fingerprint",
    "load_switch_asset",
    "ButtonLatchState",
    "step_button_latch",
]

SWITCH_GENERATOR_REVISION = 1
_REPO_ROOT = Path(__file__).resolve().parents[3]


def switch_fingerprint(model: SwitchModelConfig, /) -> str:
    model.validate()
    definition = {"generator_revision": SWITCH_GENERATOR_REVISION, "model": asdict(model)}
    return hashlib.sha256(json.dumps(definition, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class SwitchAsset:
    """Exact immutable generated definition and USD bytes used by a consumer."""

    path: Path
    fingerprint: str
    usd_digest: str

    def record(self) -> dict[str, str]:
        return {
            "path": str(self.path),
            "fingerprint": self.fingerprint,
            "usd_digest": self.usd_digest,
        }


def load_switch_asset(
    *,
    model: SwitchModelConfig,
    repo_root: Path = _REPO_ROOT,
    expected: SwitchAsset | None = None,
) -> SwitchAsset:
    """Require prepared assets; never generate or repair producer files."""
    fingerprint = switch_fingerprint(model)
    directory = (
        repo_root / ".build/assets/light-switches" / fingerprint
        if expected is None
        else expected.path.parent
    )
    if directory.name != fingerprint:
        raise ValueError("switch asset path does not match its definition fingerprint")
    path = (directory / "switch.usda").resolve(strict=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if manifest != {
        "generator_revision": SWITCH_GENERATOR_REVISION,
        "model": asdict(model),
        "definition_fingerprint": fingerprint,
        "usd_digest": digest,
    }:
        raise ValueError(f"generated switch definition or USD changed: {directory}")
    result = SwitchAsset(path=path, fingerprint=fingerprint, usd_digest=digest)
    if expected is not None and result != expected:
        raise ValueError("switch asset changed after preparation")
    return result


_ArrayT = TypeVar("_ArrayT", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class ButtonLatchState(Generic[_ArrayT]):
    duration_s: _ArrayT
    latched: _ArrayT


def step_button_latch(
    *,
    state: ButtonLatchState[_ArrayT],
    displacement_m: _ArrayT,
    dt_s: float,
    model: SwitchModelConfig,
) -> tuple[ButtonLatchState[_ArrayT], _ArrayT]:
    """Sample each physics step; release rearms a confirmed, sustained press."""
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("button sample interval must be finite and positive")
    pressed = displacement_m >= model.activation_displacement_m
    duration = (state.duration_s + dt_s) * pressed
    # Allow only floating-point accumulation error at an exact duration boundary.
    toggled = (duration >= model.confirmation_s - dt_s * 1e-6) & ~state.latched
    return ButtonLatchState(duration, pressed & (state.latched | toggled)), toggled
