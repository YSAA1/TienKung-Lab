from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
T4_ASSET = ROOT / "legged_lab/assets/t4"
T4_DATA = ROOT / "legged_lab/envs/t4/datasets/motion_source"


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
    assert len(list((T4_ASSET / "meshes").glob("*"))) == 33
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
    assert 'joint_pos=T4_STANDING_JOINT_POS' in source


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
    assert "SCHEMA_VERSION = \"t4_motion_playback.v1\"" in source
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
