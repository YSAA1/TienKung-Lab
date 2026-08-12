from __future__ import annotations

import ast
import csv
from pathlib import Path
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
