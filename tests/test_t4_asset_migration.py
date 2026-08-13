from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from legged_lab.assets.t4 import vault_contract
from legged_lab.assets.t4.tracking_motion import body_indices, load_t4_tracking_motion

ROOT = Path(__file__).resolve().parents[1]
T4_ASSET = ROOT / "legged_lab/assets/t4"
T4_DATA = ROOT / "legged_lab/envs/t4/datasets/motion_source"
MOTION_TRACKING = ROOT / "legged_lab/envs/t4/datasets/motion_tracking"


EXPECTED_JOINT_ORDER = (
    "J_arm_l_01",
    "J_arm_l_02",
    "J_arm_l_03",
    "J_arm_l_04",
    "J_arm_l_05",
    "J_arm_l_06",
    "J_arm_l_07",
    "J_arm_r_01",
    "J_arm_r_02",
    "J_arm_r_03",
    "J_arm_r_04",
    "J_arm_r_05",
    "J_arm_r_06",
    "J_arm_r_07",
    "J_waist_yaw",
    "J_hip_l_pitch",
    "J_hip_l_roll",
    "J_hip_l_yaw",
    "J_knee_l_pitch",
    "J_ankle_l_pitch",
    "J_ankle_l_roll",
    "J_hip_r_pitch",
    "J_hip_r_roll",
    "J_hip_r_yaw",
    "J_knee_r_pitch",
    "J_ankle_r_pitch",
    "J_ankle_r_roll",
)


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"missing assignment {name} in {path}")


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def test_t4_asset_bundle_is_self_contained() -> None:
    urdf_path = T4_ASSET / "urdf/t4_std.urdf"
    assert urdf_path.is_file()
    assert (T4_ASSET / "mjcf/t4_std.xml").is_file()
    assert len(list((T4_ASSET / "meshes").glob("*"))) == 34
    urdf = ET.parse(urdf_path).getroot()
    mesh_paths = [urdf_path.parent / mesh.attrib["filename"] for mesh in urdf.findall(".//mesh")]
    assert mesh_paths
    assert all(path.is_file() for path in mesh_paths)


def test_t4_joint_order_matches_27dof_motion_columns() -> None:
    joint_order = _literal_assignment(T4_ASSET / "constants.py", "T4_JOINT_NAMES")
    assert joint_order == EXPECTED_JOINT_ORDER

    stand = T4_DATA / "t4_stand.csv"
    with stand.open(newline="") as stream:
        row = next(csv.reader(stream))
    assert len(row) == 7 + len(joint_order)


def test_t4_initial_joint_pose_has_unique_isaaclab_match_rules() -> None:
    tree = _module_ast(T4_ASSET / "t4.py")
    names = [
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]
    assert "T4_STANDING_JOINT_POS" in names

    source = (T4_ASSET / "t4.py").read_text()
    assert '"."*' not in source
    assert '" .*"' not in source
    assert "joint_pos=T4_STANDING_JOINT_POS" in source


def test_all_migrated_t4_motion_files_share_the_raw_contract() -> None:
    motion_files = sorted(T4_DATA.glob("*.csv"))
    assert len(motion_files) == 18
    for path in motion_files:
        with path.open(newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) >= 2, path
        assert {len(row) for row in rows} == {34}, path


def test_t4_motion_converter_declares_xyzw_and_27dof_contract() -> None:
    converter = ROOT / "legged_lab/scripts/t4_csv_motion_conversion.py"
    source = converter.read_text()
    assert "root_quat_xyzw" in source
    assert "T4_JOINT_NAMES" in source
    assert "FrameDuration" in source


def test_t4_motion_audit_script_generates_m0_contract_report() -> None:
    output = ROOT / "artifacts/test/t4_motion_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    script = ROOT / "legged_lab/scripts/audit_t4_motions.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--task",
            "t4_loco_teacher",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=True,
    )

    audit = json.loads(output.read_text())
    assert audit["task"] == "t4_loco_teacher"
    assert audit["schema_version"] == "t4_motion_audit.v1"
    assert audit["asset"]["joint_names"] == list(EXPECTED_JOINT_ORDER)
    assert audit["asset"]["mjcf_sites"]["forward_camera"]["pos"] == [0.085, 0.0, 0.42]
    assert audit["asset"]["required_sites_present"] is True
    assert len(audit["motions"]) == 18

    stand = audit["motions"]["t4_stand"]
    assert stand["machine_status"] == "accept"
    assert stand["human_playback_status"] == "pending"
    assert stand["raw"]["width"] == 34
    assert stand["visualization"]["frame_width"] == 66

    run = audit["motions"]["t4_run"]
    assert run["machine_status"] == "reject"
    assert any("holdout" in reason for reason in run["machine_reject_reasons"])


def test_t4_motion_playback_script_declares_isaaclab_m0_contract() -> None:
    script = ROOT / "legged_lab/scripts/playback_t4_motions.py"
    source = script.read_text()
    assert 'SCHEMA_VERSION = "t4_motion_playback.v1"' in source
    assert "T4_JOINT_NAMES" in source
    assert "_xyzw_to_wxyz" in source
    assert "_motion_to_sim_joint_indices" in source
    assert "motion_to_sim_joint_indices" in source
    assert "write_root_pose_to_sim" in source
    assert "write_joint_state_to_sim" in source
    assert "human_playback_status" in source
    assert "patch_physx_backward_compatibility_setting(AppLauncher)" in source
    assert "patch_missing_physx_material_attributes()" in source
    assert "AppLauncher.add_app_launcher_args" in source
    assert "spawn_ground_plane" not in source
    assert "--motion-dir" in source
    assert "--max-frames" in source


def test_t4_isaaclab_smoke_script_applies_runtime_compat_patch() -> None:
    source = (ROOT / "legged_lab/scripts/smoke_t4_asset.py").read_text()
    compat_source = (ROOT / "legged_lab/scripts/isaaclab_runtime_compat.py").read_text()

    assert "patch_physx_backward_compatibility_setting(AppLauncher)" in source
    assert "patch_missing_physx_material_attributes()" in source
    assert "spawn_ground_plane" not in source
    assert "set(robot.joint_names)" in source
    assert "SETTING_BACKWARD_COMPATIBILITY" in compat_source
    assert "improve_patch_friction" in compat_source


def _tracking_manifest() -> dict:
    return json.loads((MOTION_TRACKING / "_manifest.json").read_text())


def test_t4_tracking_motion_npz_matches_manifest() -> None:
    manifest = _tracking_manifest()
    assert manifest["schema_version"] == "t4_tracking_motion_manifest.v1"
    assert set(manifest["motions"]) == {"overbox_1m_t4_mjcf_fps50"}

    entry = manifest["motions"]["overbox_1m_t4_mjcf_fps50"]
    npz_path = MOTION_TRACKING / entry["file"]
    assert npz_path.is_file()
    assert hashlib.sha256(npz_path.read_bytes()).hexdigest() == entry["sha256"]

    motion = load_t4_tracking_motion(npz_path)
    assert motion["fps"] == entry["fps"] == 50.0
    assert motion["num_frames"] == entry["num_frames"] == 346
    assert len(motion["joint_names"]) == entry["num_joints"] == 27
    assert len(motion["body_names"]) == entry["num_bodies"] == 32
    assert entry["scene"]["box_size_xyz"] == [1.0, 1.0, 1.0]


def test_t4_tracking_motion_loader_reorders_to_t4_joint_names() -> None:
    entry = _tracking_manifest()["motions"]["overbox_1m_t4_mjcf_fps50"]
    npz_path = MOTION_TRACKING / entry["file"]
    motion = load_t4_tracking_motion(npz_path)
    assert motion["joint_names"] == EXPECTED_JOINT_ORDER
    # The npz keeps the MJCF BFS source order, so the reorder must be non-trivial.
    assert motion["source_joint_names"] != motion["joint_names"]

    with np.load(npz_path, allow_pickle=False) as raw:
        raw_names = [str(name) for name in raw["joint_names"]]
        raw_joint_pos = np.asarray(raw["joint_pos"])
        raw_joint_vel = np.asarray(raw["joint_vel"])
    for target_index, name in enumerate(EXPECTED_JOINT_ORDER):
        source_index = raw_names.index(name)
        assert np.array_equal(motion["joint_pos"][:, target_index], raw_joint_pos[:, source_index])
        assert np.array_equal(motion["joint_vel"][:, target_index], raw_joint_vel[:, source_index])


def test_t4_tracking_motion_bodies_cover_php_tracking_contract() -> None:
    entry = _tracking_manifest()["motions"]["overbox_1m_t4_mjcf_fps50"]
    motion = load_t4_tracking_motion(MOTION_TRACKING / entry["file"])

    contract = entry["tracking_contract"]
    tracked = contract["tracking_bodies"]
    # Pin the PHP recipe facts so a silent manifest edit cannot weaken the G1 contract.
    assert contract["anchor_body"] == "Trunk"
    assert contract["foot_bodies"] == ["left_foot_link", "right_foot_link"]
    assert contract["wrist_bodies"] == ["AL7", "AR7"]
    assert tracked == [
        "Trunk",
        "Hip_Roll_Left",
        "Shank_Left",
        "left_foot_link",
        "Hip_Roll_Right",
        "Shank_Right",
        "right_foot_link",
        "Waist_yaw",
        "AL2",
        "AL4",
        "AL7",
        "AR2",
        "AR4",
        "AR7",
    ]
    assert contract["anchor_body"] in tracked
    assert set(contract["foot_bodies"] + contract["wrist_bodies"]) <= set(tracked)

    urdf_links = {link.attrib["name"] for link in ET.parse(T4_ASSET / "urdf/t4_std.urdf").getroot().findall("link")}
    assert set(tracked) <= set(motion["body_names"])
    assert set(tracked) <= urdf_links
    assert set(entry["bodies_missing_in_urdf"]) == set(motion["body_names"]) - urdf_links

    assert len(body_indices(motion, tracked)) == len(tracked)
    with pytest.raises(KeyError):
        body_indices(motion, ["not_a_body"])

    quat_norm = np.linalg.norm(motion["body_quat_w"], axis=-1)
    assert np.allclose(quat_norm, 1.0, atol=1e-3)


def test_t4_tracking_motion_loader_fails_fast_on_contract_drift(tmp_path) -> None:
    entry = _tracking_manifest()["motions"]["overbox_1m_t4_mjcf_fps50"]
    with np.load(MOTION_TRACKING / entry["file"], allow_pickle=False) as raw:
        arrays = {key: np.asarray(raw[key]) for key in raw.files}

    missing_key = {key: value for key, value in arrays.items() if key != "joint_vel"}
    path = tmp_path / "missing_key.npz"
    np.savez(path, **missing_key)
    with pytest.raises(ValueError, match="missing required keys"):
        load_t4_tracking_motion(path)

    renamed = dict(arrays)
    joint_names = renamed["joint_names"].copy()
    joint_names[0] = "J_not_a_joint"
    renamed["joint_names"] = joint_names
    path = tmp_path / "renamed_joint.npz"
    np.savez(path, **renamed)
    with pytest.raises(ValueError, match="bijection"):
        load_t4_tracking_motion(path)

    poisoned = dict(arrays)
    joint_pos = poisoned["joint_pos"].copy()
    joint_pos[0, 0] = np.nan
    poisoned["joint_pos"] = joint_pos
    path = tmp_path / "nonfinite.npz"
    np.savez(path, **poisoned)
    with pytest.raises(ValueError, match="non-finite"):
        load_t4_tracking_motion(path)

    bad_fps = dict(arrays)
    bad_fps["fps"] = np.asarray([np.nan])
    path = tmp_path / "bad_fps.npz"
    np.savez(path, **bad_fps)
    with pytest.raises(ValueError, match="fps must be positive and finite"):
        load_t4_tracking_motion(path)


def test_t4_vault_contract_matches_manifest_and_stage_e_plant() -> None:
    entry = _tracking_manifest()["motions"]["overbox_1m_t4_mjcf_fps50"]
    contract = entry["tracking_contract"]
    assert vault_contract.T4_VAULT_ANCHOR_BODY_NAME == contract["anchor_body"]
    assert list(vault_contract.T4_VAULT_FOOT_BODY_NAMES) == contract["foot_bodies"]
    assert list(vault_contract.T4_VAULT_WRIST_BODY_NAMES) == contract["wrist_bodies"]
    assert list(vault_contract.T4_VAULT_TRACKING_BODY_NAMES) == contract["tracking_bodies"]
    assert vault_contract.T4_VAULT_END_EFFECTOR_BODY_NAMES == (
        vault_contract.T4_VAULT_FOOT_BODY_NAMES + vault_contract.T4_VAULT_WRIST_BODY_NAMES
    )

    scene = entry["scene"]
    assert list(vault_contract.T4_VAULT_BOX_SIZE) == scene["box_size_xyz"]
    assert list(vault_contract.T4_VAULT_BOX_POS) == scene["box_pos_w"]
    assert list(vault_contract.T4_VAULT_BOX_ROT) == scene["box_quat_wxyz"]

    assert vault_contract.T4_VAULT_MOTION_FILE == MOTION_TRACKING / entry["file"]
    assert vault_contract.T4_VAULT_MOTION_FILE.is_file()

    # The mimic teacher must share the frozen Stage E 0.25 action scale.
    teacher_source = (ROOT / "legged_lab/envs/t4/teacher_cfg.py").read_text()
    match = re.search(r"action_scale=([0-9.]+)", teacher_source)
    assert match is not None
    assert vault_contract.T4_VAULT_ACTION_SCALE == float(match.group(1)) == 0.25


def test_t4_urdf_carries_php_sphere_hand_collision() -> None:
    root = ET.parse(T4_ASSET / "urdf/t4_std.urdf").getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}

    for side, parent in (("left", "AL7"), ("right", "AR7")):
        hand = links[f"{side}_sphere_hand_link"]
        collisions = hand.findall("collision")
        assert len(collisions) == 1
        mesh = collisions[0].find("geometry/mesh")
        assert mesh is not None
        mesh_path = (T4_ASSET / "urdf" / mesh.attrib["filename"]).resolve()
        assert mesh_path == (T4_ASSET / "meshes/half_sphere.obj").resolve()
        assert mesh_path.is_file()
        assert links[f"{side}_sphere_hand_tip_link"].find("collision") is not None

        palm = joints[f"{side}_hand_palm_joint"]
        assert palm.attrib["type"] == "fixed"
        assert palm.find("parent").attrib["link"] == parent
        assert palm.find("origin").attrib["xyz"] == "0 0 -0.031"

        # Hand mass restores the reference plant's arm inertia after merging.
        mass = hand.find("inertial/mass")
        assert mass is not None and float(mass.attrib["value"]) == 0.124


def test_t4_vault_mimic_env_cfg_consumes_the_contract() -> None:
    source = (ROOT / "legged_lab/envs/t4/vault_mimic/vault_env_cfg.py").read_text()
    assert "from legged_lab.assets.t4.t4 import T4_CFG" in source
    assert "robot: ArticulationCfg = T4_CFG.replace" in source
    assert "scale=T4_VAULT_ACTION_SCALE" in source
    assert "motion_file=str(T4_VAULT_MOTION_FILE)" in source
    assert "wrist_body_pos" in source
    assert '"std": 0.15' in source
    # Vendored MotionLoader must keep the fail-fast named reorder.
    commands_source = (ROOT / "legged_lab/envs/t4/vault_mimic/mdp/commands.py").read_text()
    assert "reorder_named_axis" in commands_source
    assert "Named motion joints require robot_joint_names" in commands_source
