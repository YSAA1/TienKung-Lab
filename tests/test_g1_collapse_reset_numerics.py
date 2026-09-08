"""Numerical contracts for G1 collapsed-pelvis termination (no Isaac runtime)."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from legged_lab.assets.unitree_g1.constants import G1_STANDING_JOINT_POS, G1_STANDING_PELVIS_Z
from tests.test_g1_asset_contract import _urdf_fk_and_collisions

_SIG_PATH = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "mdp" / "sparse_signals.py"
_spec = importlib.util.spec_from_file_location("t4_sparse_signals_g1", _SIG_PATH)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)

G1_URDF = Path(__file__).resolve().parents[1] / "legged_lab" / "assets" / "unitree_g1" / "urdf" / "g1_29dof_mode_15.urdf"
THRESHOLD = 0.20  # unitree_rl_lab G1 29DoF root_height_below_minimum
V8_OVERSIZE_THRESHOLD = 0.40


def _clearance(root_z: float, foot_zs: list[float], *, reducer: str) -> float:
    if reducer == "mean":
        foot_z = sum(foot_zs) / len(foot_zs)
    elif reducer == "min":
        foot_z = min(foot_zs)
    elif reducer == "max":
        foot_z = max(foot_zs)
    else:
        raise ValueError(reducer)
    return root_z - foot_z


def _collapsed(root_z: float, foot_zs: list[float], *, reducer: str) -> bool:
    if reducer == "mean":
        foot_z = sum(foot_zs) / len(foot_zs)
    else:
        foot_z = {"min": min, "max": max}[reducer](foot_zs)
    return sig.collapsed_pelvis_above_feet_mask(root_z, foot_z, THRESHOLD)


def test_g1_standing_fk_clearance_above_threshold() -> None:
    _, poses = _urdf_fk_and_collisions(G1_URDF)
    pelvis_z = poses["pelvis"][1][2]
    foot_zs = [poses["left_ankle_roll_link"][1][2], poses["right_ankle_roll_link"][1][2]]
    clearance = _clearance(pelvis_z, foot_zs, reducer="mean")
    assert pelvis_z == pytest.approx(G1_STANDING_PELVIS_Z)
    assert clearance == pytest.approx(0.722, abs=0.02)
    assert not _collapsed(pelvis_z, foot_zs, reducer="mean")
    assert not _collapsed(pelvis_z, foot_zs, reducer="min")


def test_v6_and_plantfix_sit_triggers_unitree_height() -> None:
    """v6/plantfix sit: pelvis z≈0.17 m, feet near ground — Unitree 0.2 m must fail."""
    pelvis_z = 0.170
    foot_zs = [0.038, 0.038]
    assert _clearance(pelvis_z, foot_zs, reducer="min") == pytest.approx(0.132, abs=0.02)
    assert sig.collapsed_pelvis_above_feet_mask(pelvis_z, min(foot_zs), THRESHOLD)
    assert _collapsed(pelvis_z, foot_zs, reducer="min")


def test_walking_crouch_survives_unitree_height_not_v8_040() -> None:
    """Clearance 0.35 is a crouch, not a sit. Unitree 0.2 keeps it; v8 0.40 did not."""
    pelvis_z = 0.50
    foot_zs = [0.15, 0.35]  # min clearance 0.35
    assert _clearance(pelvis_z, foot_zs, reducer="min") == pytest.approx(0.35, abs=1e-3)
    assert not sig.collapsed_pelvis_above_feet_mask(pelvis_z, min(foot_zs), THRESHOLD)
    assert sig.collapsed_pelvis_above_feet_mask(pelvis_z, min(foot_zs), V8_OVERSIZE_THRESHOLD)


def test_mean_foot_reducer_false_positive_on_single_support() -> None:
    """High swing foot + stance foot: mean foot z is too optimistic (bug shape)."""
    pelvis_z = 0.30
    foot_zs = [0.04, 0.40]
    mean_c = _clearance(pelvis_z, foot_zs, reducer="mean")
    min_c = _clearance(pelvis_z, foot_zs, reducer="min")
    assert mean_c == pytest.approx(0.08, abs=1e-3)
    assert min_c == pytest.approx(0.26, abs=1e-3)
    assert _collapsed(pelvis_z, foot_zs, reducer="mean")
    assert not _collapsed(pelvis_z, foot_zs, reducer="min")


def test_downstairs_walking_safe_with_min_and_mean() -> None:
    """Pelvis on upper step, feet split across levels — must not terminate."""
    pelvis_z = 0.20
    foot_zs = [-0.50, 0.10]
    assert not _collapsed(pelvis_z, foot_zs, reducer="min")
    assert not _collapsed(pelvis_z, foot_zs, reducer="mean")


def test_joint_scale_reset_does_not_put_standing_into_collapsed_fk() -> None:
    scaled = dict(G1_STANDING_JOINT_POS)
    for key, value in list(scaled.items()):
        if value != 0.0:
            scaled[key] = value * 1.5
    _, poses = _urdf_fk_and_collisions(G1_URDF)
    # FK keeps pelvis z fixed; re-run with modified joint dict via helper inline
    from tests.test_g1_asset_contract import _T_from_xyz_rpy, _compose, _axis_matrix, _rpy_matrix
    import xml.etree.ElementTree as ET

    root = ET.parse(G1_URDF).getroot()
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz = tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
        rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
        axis_elem = joint.find("axis")
        axis = (
            tuple(float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()) if axis_elem is not None else (0.0, 0.0, 1.0)
        )
        joints.append((joint.attrib["name"], joint.attrib.get("type"), parent, child, xyz, rpy, axis))
    pose_map = {"pelvis": (_rpy_matrix(0, 0, 0), (0.0, 0.0, G1_STANDING_PELVIS_Z))}
    pending = list(joints)
    for _ in range(256):
        if not pending:
            break
        nxt = []
        for item in pending:
            name, jtype, parent, child, xyz, rpy, axis = item
            if parent not in pose_map:
                nxt.append(item)
                continue
            joint_t = _T_from_xyz_rpy(xyz, rpy)
            angle = float(scaled.get(name, 0.0)) if jtype == "revolute" else 0.0
            rot = (_axis_matrix(axis, angle), (0.0, 0.0, 0.0))
            pose_map[child] = _compose(_compose(pose_map[parent], joint_t), rot)
        pending = nxt
    foot_zs = [pose_map["left_ankle_roll_link"][1][2], pose_map["right_ankle_roll_link"][1][2]]
    pelvis_z = pose_map["pelvis"][1][2]
    assert not _collapsed(pelvis_z, foot_zs, reducer="mean")


def test_replay_trace_world_z_not_same_as_clearance_gate() -> None:
    """World pelvis z<0.20 is Unitree's flat-ground form; sparse must use clearance."""
    import json

    trace_path = Path(__file__).resolve().parents[1] / "artifacts/replay/g1_plantfix_m6000/stones_d00.diagnostics.json"
    if not trace_path.exists():
        pytest.skip("plantfix replay diagnostics missing")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))["trace"]
    late = [row for row in trace if row["root_pos_w_m"][2] < 0.25]
    assert late, "plantfix replay never sat"
    pelvis_z = late[-1]["root_pos_w_m"][2]
    foot_z = 0.038
    assert (pelvis_z - foot_z) < THRESHOLD
    max_tilt = max(row["tilt_rad"] for row in trace)
    assert max_tilt < sig.LIGHTLP_TILT_LIMIT_RAD
    assert 0.8 > max_tilt  # unitree 0.8 rad would also miss this sit; height is the catcher


def test_non_ankle_contact_regex_matches_sit_bodies_not_feet() -> None:
    """Isaac Lab uses re.fullmatch; unitree's non-ankle pattern must hit thigh/pelvis/torso."""
    import re
    import xml.etree.ElementTree as ET

    pattern = re.compile(r"(?!.*ankle.*).*")
    links = [elem.attrib["name"] for elem in ET.parse(G1_URDF).getroot().findall("link")]
    matched = {name for name in links if pattern.fullmatch(name)}
    assert "pelvis" in matched
    assert "torso_link" in matched
    assert "left_hip_yaw_link" in matched
    assert "left_knee_link" in matched
    assert "left_ankle_roll_link" not in matched
    assert "right_ankle_pitch_link" not in matched
    assert matched

