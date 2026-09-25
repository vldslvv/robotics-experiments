"""Asset preparation publishes deterministic files before dependent launches."""

import fcntl
import hashlib
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotics_experiments.config import load_switch_model
from robotics_experiments.switches import load_switch_asset
from tests.support.paths import REPO_ROOT


def _builders():
    pytest.importorskip("pxr.Usd")
    from isaac.runtime.standalone import build_assets_standalone, build_switch_usd_standalone

    return build_assets_standalone, build_switch_usd_standalone


def test_switch_publication_is_deterministic_immutable_and_has_consistent_joint_frames(*, tmp_path):
    _, builder = _builders()
    from pxr import Usd, UsdGeom, UsdPhysics

    for profile in ("easy", "nominal"):
        model = load_switch_model(REPO_ROOT / f"configs/experiment/switches/model/{profile}.yaml")
        path = builder.build_switch_usd(model=model, repo_root=tmp_path)
        before = (path.read_bytes(), path.stat().st_mtime_ns)
        assert builder.build_switch_usd(model=model, repo_root=tmp_path) == path.resolve()
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
        asset = load_switch_asset(model=model, repo_root=tmp_path)
        assert asset.usd_digest == hashlib.sha256(before[0]).hexdigest()
        stage = Usd.Stage.Open(str(path))
        joint = UsdPhysics.PrismaticJoint(
            stage.GetPrimAtPath("/Root/LightSwitch/ButtonSpringJoint")
        )
        assert tuple(joint.GetLocalPos0Attr().Get()) == pytest.approx((0, model.button_rest_y_m, 0))
        assert tuple(joint.GetLocalPos1Attr().Get()) == (0, 0, 0)
        assert joint.GetUpperLimitAttr().Get() == pytest.approx(model.travel_m)
        drive = UsdPhysics.DriveAPI(joint.GetPrim(), "linear")
        assert drive.GetStiffnessAttr().Get() == model.stiffness_n_m
        assert drive.GetDampingAttr().Get() == pytest.approx(model.damping_n_s_m)
        assert drive.GetTargetPositionAttr().Get() == 0
        housing = stage.GetPrimAtPath("/Root/LightSwitch/BaseMesh")
        assert len(housing.GetChildren()) == 5
        bounds = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_])
        for child in housing.GetChildren():
            box = bounds.ComputeWorldBound(child).ComputeAlignedBox()
            low, high = box.GetMin(), box.GetMax()
            button_lo = (
                -model.button_width_m / 2,
                model.button_rest_y_m - model.button_thickness_m / 2,
                -model.button_height_m / 2,
            )
            button_hi = (
                model.button_width_m / 2,
                model.button_rest_y_m + model.travel_m + model.button_thickness_m / 2,
                model.button_height_m / 2,
            )
            assert any(
                high[i] <= button_lo[i] - model.clearance_m + 1e-8
                or low[i] >= button_hi[i] + model.clearance_m - 1e-8
                for i in range(3)
            )
        path.write_text("changed")
        with pytest.raises(ValueError, match="changed"):
            builder.build_switch_usd(model=model, repo_root=tmp_path)


def test_aggregator_runs_generators_curriculum_variants_and_retains_completed_outputs(
    *, monkeypatch, tmp_path
):
    aggregator, _ = _builders()
    model = load_switch_model(REPO_ROOT / "configs/experiment/switches/model/nominal.yaml")
    variant = replace(model, confirmation_s=0.3)
    records = []
    completed = tmp_path / "drone.usda"

    def additional_generator():
        records.append("additional")
        return tmp_path / "extra.usda"

    def drones():
        records.append("drone")
        completed.write_text("drone")
        return [str(completed), str(additional_generator())]

    def switches(*, model, repo_root):
        records.append(model.confirmation_s)
        if model is variant:
            raise RuntimeError("switch failed")
        return repo_root / "switch.usda"

    monkeypatch.setattr(aggregator, "build_all_airframe_usds", drones)
    monkeypatch.setattr(aggregator, "build_switch_usd", switches)
    monkeypatch.setattr(
        aggregator,
        "load_config",
        lambda path: (_ for _ in ()).throw(
            ValueError("expected an experiment configuration using base_experiment")
        )
        if path.name == "curriculum.yaml"
        else SimpleNamespace(
            switches=SimpleNamespace(model=model if path.name == "first" else variant)
        ),
    )
    monkeypatch.setattr(
        aggregator,
        "load_curriculum_config",
        lambda _path: SimpleNamespace(
            stages=[SimpleNamespace(config_path="first"), SimpleNamespace(config_path="second")]
        ),
    )
    with pytest.raises(RuntimeError, match="switch failed"):
        aggregator.build_assets(config_path=Path("curriculum.yaml"), repo_root=tmp_path)
    assert records == ["drone", "additional", 0.25, 0.3]
    assert completed.read_text() == "drone"


def test_generation_lock_serializes_processes(*, tmp_path):
    _builders()
    lock_dir = tmp_path / ".build/assets"
    lock_dir.mkdir(parents=True)
    script = """
from pathlib import Path
import sys
from isaac.runtime.standalone import build_assets_standalone as builder
root = Path(sys.argv[1])
def drones():
    (root / 'entered').write_text('yes')
    return []
builder.build_all_airframe_usds = drones
builder.build_switch_usd = lambda **kwargs: root / 'switch.usda'
print('ready', flush=True)
builder.build_assets(config_path=Path(sys.argv[2]), repo_root=root)
"""
    with (lock_dir / "generation.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(tmp_path),
                str(REPO_ROOT / "configs/takeoff_quad.yaml"),
            ],
            cwd=REPO_ROOT,
            env=os.environ | {"PYTHONPATH": str(REPO_ROOT / "src")},
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert process.stdout.readline() == "ready\n"
            with pytest.raises(subprocess.TimeoutExpired):
                process.wait(timeout=0.3)
            assert not (tmp_path / "entered").exists()
            fcntl.flock(lock, fcntl.LOCK_UN)
            assert process.wait(timeout=30) == 0
            assert (tmp_path / "entered").read_text() == "yes"
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.parametrize("target", ["train-rl", "run-all"])
def test_generator_failure_prevents_launch(*, tmp_path, target):
    for name, code in (("uv", "exit 7"), ("docker", f"touch '{tmp_path / 'launched'}'")):
        executable = tmp_path / name
        executable.write_text(f"#!/bin/sh\n{code}\n")
        executable.chmod(0o755)
    result = subprocess.run(
        ["make", target, "HYDRA_CONFIG=configs/train/press_curriculum_quad.yaml"],
        cwd=REPO_ROOT,
        env=os.environ | {"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
    )
    assert result.returncode != 0
    assert not (tmp_path / "launched").exists()
