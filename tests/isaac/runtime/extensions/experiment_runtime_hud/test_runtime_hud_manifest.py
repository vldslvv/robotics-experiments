from __future__ import annotations

import tomllib

from tests.support.paths import REPO_ROOT


def test_hud_is_a_dependent_ui_only_extension() -> None:
    path = REPO_ROOT / "isaac/runtime/extensions/experiment_runtime_hud/config/extension.toml"
    manifest = tomllib.loads(path.read_text(encoding="utf-8"))

    assert set(manifest["dependencies"]) >= {
        "experiment_runtime",
        "isaacsim.gui.components",
        "omni.kit.uiapp",
    }
    assert manifest["python"]["module"] == [{"name": "ExperimentRuntimeHud_python"}]
    assert (
        manifest["settings"]["exts"]["experiment_runtime_hud"]["transition_history_capacity"] == 2
    )
    assert "omni.physx" not in manifest["dependencies"]

    runtime_path = REPO_ROOT / "isaac/runtime/extensions/experiment_runtime/config/extension.toml"
    runtime = tomllib.loads(runtime_path.read_text(encoding="utf-8"))
    assert "experiment_runtime_hud" not in runtime["dependencies"]


def test_manual_launcher_enables_the_hud_after_the_runtime() -> None:
    launcher = (REPO_ROOT / "isaac/runtime/standalone/runapp_custom.sh").read_text()

    runtime_enable = launcher.index('--enable "$EXPERIMENT_RUNTIME_EXT"')
    hud_enable = launcher.index('--enable "$EXPERIMENT_RUNTIME_HUD_EXT"')
    assert runtime_enable < hud_enable
