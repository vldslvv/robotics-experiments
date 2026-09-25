"""Author deterministic switch USDA using standalone OpenUSD, without Kit."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from robotics_experiments.config import SwitchModelConfig, load_switch_model
from robotics_experiments.switches import (
    SWITCH_GENERATOR_REVISION,
    load_switch_asset,
    switch_fingerprint,
)

__all__ = ["build_switch_usd"]


def _box(*, stage, path, size, center, model):
    box = UsdGeom.Cube.Define(stage, path)
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*center))
    box.AddScaleOp().Set(Gf.Vec3f(*size))
    prim = box.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
        model.contact_offset_m
    )
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(0.0)
    box.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.8, 0.8)])
    return prim


def build_switch_usd(*, model: SwitchModelConfig, repo_root: Path) -> Path:
    """Publish an immutable, content-addressed switch bundle atomically."""
    fingerprint = switch_fingerprint(model)
    parent = repo_root / ".build/assets/light-switches"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / fingerprint
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Root")
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    base = UsdGeom.Xform.Define(stage, "/Root/LightSwitch/BaseMesh")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Root/LightSwitch/BaseFixedJoint")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
    fixed.CreateLocalRot0Attr(Gf.Quatf(1))
    fixed.CreateLocalRot1Attr(Gf.Quatf(1))

    # Rails surround the entire swept volume; the back plate lies behind it.
    w, h = model.housing_width_m, model.housing_height_m
    inner_w = model.button_width_m + 2 * model.clearance_m
    inner_h = model.button_height_m + 2 * model.clearance_m
    front = model.button_rest_y_m + model.button_thickness_m / 2
    back = model.back_extent_m
    depth, center_y = back - front, (back + front) / 2
    for name, size, center in (
        ("Left", ((w - inner_w) / 2, depth, h), (-(w + inner_w) / 4, center_y, 0)),
        ("Right", ((w - inner_w) / 2, depth, h), ((w + inner_w) / 4, center_y, 0)),
        ("Top", (inner_w, depth, (h - inner_h) / 2), (0, center_y, (h + inner_h) / 4)),
        ("Bottom", (inner_w, depth, (h - inner_h) / 2), (0, center_y, -(h + inner_h) / 4)),
        (
            "Back",
            (w, model.back_plate_thickness_m, h),
            (0, back - model.back_plate_thickness_m / 2, 0),
        ),
    ):
        _box(stage=stage, path=f"{base.GetPath()}/{name}", size=size, center=center, model=model)
    button = UsdGeom.Xform.Define(stage, "/Root/LightSwitch/Button")
    button.AddTranslateOp().Set(Gf.Vec3d(0, model.button_rest_y_m, 0))
    UsdPhysics.RigidBodyAPI.Apply(button.GetPrim())
    UsdPhysics.MassAPI.Apply(button.GetPrim()).CreateMassAttr(model.button_mass_kg)
    _box(
        stage=stage,
        path=f"{button.GetPath()}/ButtonMesh",
        size=(model.button_width_m, model.button_thickness_m, model.button_height_m),
        center=(0, 0, 0),
        model=model,
    )
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/Root/LightSwitch/ButtonSpringJoint")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([button.GetPath()])
    joint.CreateAxisAttr("Y")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, model.button_rest_y_m, 0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLocalRot1Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(0.0)
    joint.CreateUpperLimitAttr(model.travel_m)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
    drive.CreateTypeAttr("force")
    drive.CreateTargetPositionAttr(0.0)
    drive.CreateTargetVelocityAttr(0.0)
    drive.CreateStiffnessAttr(model.stiffness_n_m)
    drive.CreateDampingAttr(model.damping_n_s_m)
    stage.SetDefaultPrim(root.GetPrim())
    content = (stage.GetRootLayer().ExportToString().rstrip() + "\n").encode()
    manifest = {
        "generator_revision": SWITCH_GENERATOR_REVISION,
        "model": asdict(model),
        "definition_fingerprint": fingerprint,
        "usd_digest": hashlib.sha256(content).hexdigest(),
    }
    if destination.exists():
        asset = load_switch_asset(model=model, repo_root=repo_root)
        if asset.usd_digest != manifest["usd_digest"]:
            raise ValueError("switch generator changed without a revision change")
        return asset.path
    temporary = Path(tempfile.mkdtemp(prefix=f".{fingerprint}.", dir=parent))
    (temporary / "switch.usda").write_bytes(content)
    (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    temporary.rename(destination)
    return destination / "switch.usda"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    print(build_switch_usd(model=load_switch_model(args.model), repo_root=Path.cwd()))
