"""Offline provenance guard; runtime asset values are checked by the Isaac probe."""

import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def test_official_configuration_matches_pinned_upstream_ast():
    source = (ROOT / "legged_lab/assets/unitree_g1/official_g1.py").read_text()
    assignment = next(node for node in ast.parse(source).body if isinstance(node, ast.Assign))
    evidence = json.loads((ROOT / "artifacts/portability/v4/official_source.json").read_text())
    digest = hashlib.sha256(json.dumps(_canonical(assignment), sort_keys=True).encode()).hexdigest()
    assert digest == evidence["config_canonical_sha256"]

    assert evidence["commit"] == "b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8"


def _canonical(value):
    if isinstance(value, ast.AST):
        return {
            "type": type(value).__name__,
            **{key: _canonical(item) for key, item in ast.iter_fields(value) if item is not None and item != []},
        }
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def test_unitree_velocity_config_matches_pinned_upstream():
    source = (ROOT / "legged_lab/assets/unitree_g1/official_velocity_g1.py").read_text()
    assignment = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "UNITREE_G1_29DOF_CFG" for target in node.targets)
    )
    evidence = json.loads((ROOT / "artifacts/portability/v5/official_source.json").read_text())
    digest = hashlib.sha256(json.dumps(_canonical(assignment), sort_keys=True).encode()).hexdigest()
    assert digest == evidence["config_canonical_sha256"]
    classes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}
    assert set(classes) == set(evidence["spawn_class_canonical_sha256"])
    for name, node in classes.items():
        digest = hashlib.sha256(json.dumps(_canonical(node), sort_keys=True).encode()).hexdigest()
        assert digest == evidence["spawn_class_canonical_sha256"][name]


def test_active_g1_has_exactly_29_joints_and_official_geometry():
    from legged_lab.assets.unitree_g1.locomotion import G1_LOCOMOTION

    asset_dir = ROOT / "legged_lab/assets/unitree_g1/urdf"
    urdf = asset_dir / "g1_29dof_rev_1_0.urdf"
    evidence = json.loads((ROOT / "artifacts/portability/v5/official_source.json").read_text())
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == evidence["urdf_sha256"]
    joints = [joint.get("name") for joint in ET.parse(urdf).findall("joint") if joint.get("type") != "fixed"]
    assert len(joints) == 29
    assert tuple(joints) == G1_LOCOMOTION.joint_names
    assert G1_LOCOMOTION.auxiliary_joint_names == ()
    for mesh in evidence["meshes"]:
        data = (asset_dir / "meshes" / mesh["name"]).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == mesh["upstream_blob"]
