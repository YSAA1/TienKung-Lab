from __future__ import annotations

import csv
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from legged_lab.assets.unitree_g1.constants import (
    G1_29DOF_JOINT_NAMES,
    G1_STANDING_JOINT_POS,
    G1_STANDING_PELVIS_Z,
    NUM_G1_29DOF_JOINTS,
)

ROOT = Path(__file__).resolve().parents[1]
G1_ASSET = ROOT / "legged_lab/assets/unitree_g1"


def test_g1_joint_contract_is_29dof() -> None:
    assert NUM_G1_29DOF_JOINTS == 29
    assert len(G1_29DOF_JOINT_NAMES) == 29
    assert len(set(G1_29DOF_JOINT_NAMES)) == 29


def test_g1_teacher_is_sparse_obstacle_task() -> None:
    text = (ROOT / "legged_lab/envs/g1/teacher_cfg.py").read_text(encoding="utf-8")
    assert "T4LocoSparseTeacherEnvCfg" in text
    assert "TienKungWalkFlatEnvCfg" not in text
    assert 'foot_scanner.body_names = ("left_ankle_roll_link", "right_ankle_roll_link")' in text
    assert 'terminate_contacts_body_names = ["torso_link"]' in text
    assert "self.collapse_reset_pelvis_above_feet_m = 0.20" in text
    assert "self.collapse_reset_pelvis_above_feet_m = 0.40" not in text
    assert "bad_orientation_limit_rad" not in text
    assert 'body_names=["(?!.*ankle.*).*"]' in text
    assert "cold_start_max_terrain_level" not in text
    assert "self.scene.max_init_terrain_level = 0" not in text
    assert "self.random_level_reset_fraction = 0.0" not in text
    assert "self.domain_rand.action_delay.enable = False" in text
    assert "randomize_actuator_gains" not in text
    assert "G1PlantEventCfg" not in text
    t4_sparse = (ROOT / "legged_lab/envs/t4/teacher_cfg.py").read_text(encoding="utf-8")
    assert "action_delay=ActionDelayCfg(enable=False" in t4_sparse
    assert "collapse_reset_pelvis_above_feet_m" not in t4_sparse
    signals = (ROOT / "legged_lab/envs/t4/mdp/sparse_signals.py").read_text(encoding="utf-8")
    assert "LIGHTLP_TILT_LIMIT_RAD = 63.0 * math.pi / 180.0" in signals
    assert "LIGHTLP_ACCEL_LIMIT = 40.0" in signals
    init_text = (ROOT / "legged_lab/envs/__init__.py").read_text(encoding="utf-8")
    assert 'task_registry.register("g1_loco_teacher", T4LocoEnv' in init_text


def test_g1_teacher_inherits_t4_sparse_curriculum() -> None:
    """G1 must not override the T4 sparse cold-start recipe (S6/S11 style)."""
    text = (ROOT / "legged_lab/envs/g1/teacher_cfg.py").read_text(encoding="utf-8")
    assert "self.scene.max_init_terrain_level" not in text
    assert "self.random_level_reset_fraction" not in text
    assert "self.commands.ranges.lin_vel_x" not in text
    t4 = (ROOT / "legged_lab/envs/t4/teacher_cfg.py").read_text(encoding="utf-8")
    assert "self.scene.max_init_terrain_level = 2" in t4
    assert "self.random_level_reset_fraction = 0.10" in t4
    assert "self.commands.ranges.lin_vel_x = (-0.6, 2.0)" in t4


def test_g1_actuators_use_urdf_effort_limits() -> None:
    src = (ROOT / "legged_lab/assets/unitree_g1/g1.py").read_text(encoding="utf-8")
    assert "effort_limit_sim=300.0" not in src
    assert "enabled_self_collisions=False" in src
    assert "effort_limit_sim=25.0" in src
    assert "effort_limit_sim=13.4" in src


def test_g1_teacher_uses_lafan_amp() -> None:
    text = (ROOT / "legged_lab/envs/g1/teacher_cfg.py").read_text(encoding="utf-8")
    assert "runner_class_name = \"AmpOnPolicyRunner\"" in text
    assert 'class_name="AMPPPO"' in text
    assert "self.enable_amp = True" in text
    assert "self.amp_terrain_schedule.enable = True" in text
    assert "amp_reward_coef = 0.3" in text
    assert "run_name = \"g1_sparse_teacher_g1term\"" in text
    assert "self.collapse_reset_pelvis_above_feet_m = 0.20" in text
    assert "cold_start_max_terrain_level" not in text
    assert "g1_amp_expert_files" in text
    env_text = (ROOT / "legged_lab/envs/t4/t4_env.py").read_text(encoding="utf-8")
    assert "G1AmpFeatureBuilder" in env_text
    t4_sparse = (ROOT / "legged_lab/envs/t4/teacher_cfg.py").read_text(encoding="utf-8")
    assert "action_delay=ActionDelayCfg(enable=False" in t4_sparse


def test_g1_amp_schema_is_70d_and_does_not_change_t4() -> None:
    from legged_lab.assets.t4 import schemas as t4_schemas
    from legged_lab.assets.unitree_g1 import schemas as g1_schemas

    assert t4_schemas.AMP_FRAME_DIM == 66
    assert g1_schemas.AMP_FRAME_DIM == 70
    assert g1_schemas.AMP_TRANSITION_DIM == 140
    assert g1_schemas.LAFAN1_CSV_WIDTH == 36
    assert tuple(g1_schemas.LAFAN1_G1_JOINT_NAMES) == G1_29DOF_JOINT_NAMES
    widths = {name: width for name, width in g1_schemas.AMP_FIELDS}
    assert widths == {
        "joint_pos": 29,
        "joint_vel": 29,
        "hand_pos_root": 6,
        "foot_pos_root": 6,
    }
    files = g1_schemas.amp_expert_files("some/dir")
    stems = [Path(path).stem for path in files]
    assert stems == sorted(g1_schemas.AMP_MOTION_CLASSES)
    assert "sprint1_subject2" not in stems
    for motion_class, weight in g1_schemas.AMP_MOTION_CLASS_WEIGHTS.items():
        members = [stem for stem, name in g1_schemas.AMP_MOTION_CLASSES.items() if name == motion_class]
        assert members
        assert g1_schemas.amp_motion_weight(members[0]) * len(members) == pytest.approx(weight)


def test_g1_lafan_source_csvs_match_joint_contract() -> None:
    from legged_lab.assets.unitree_g1.schemas import AMP_MOTION_SOURCE_DIR, LAFAN1_CSV_WIDTH, amp_expert_files

    source = ROOT / AMP_MOTION_SOURCE_DIR
    declared = {Path(path).stem for path in amp_expert_files()}
    available = {path.stem for path in source.glob("*.csv")}
    assert declared <= available, f"missing LAFAN CSVs: {sorted(declared - available)}"
    for stem in sorted(declared):
        rows = []
        with (source / f"{stem}.csv").open(newline="") as stream:
            for row in csv.reader(stream):
                if row:
                    rows.append([float(value) for value in row])
                    if len(rows) >= 3:
                        break
        assert len(rows) >= 3
        assert {len(row) for row in rows} == {LAFAN1_CSV_WIDTH}
        assert all(math.isfinite(value) for row in rows for value in row)


def test_g1_urdf_bundle_matches_joint_contract() -> None:
    urdf_path = G1_ASSET / "urdf/g1_29dof_mode_15.urdf"
    assert urdf_path.is_file()
    assert (G1_ASSET / "xmls/g1.xml").is_file()
    meshes = list((G1_ASSET / "urdf/meshes").glob("*"))
    assert len(meshes) >= 20
    urdf = ET.parse(urdf_path).getroot()
    revolute = [
        joint.attrib["name"]
        for joint in urdf.findall(".//joint")
        if joint.attrib.get("type") == "revolute"
    ]
    assert tuple(revolute) == G1_29DOF_JOINT_NAMES
    mesh_paths = [urdf_path.parent / mesh.attrib["filename"] for mesh in urdf.findall(".//mesh")]
    assert mesh_paths
    assert all(path.is_file() for path in mesh_paths)


def _xyz(elem: ET.Element, attr: str = "xyz") -> tuple[float, float, float]:
    origin = elem.find("origin")
    if origin is None or origin.attrib.get(attr) is None:
        raw = elem.attrib.get(attr, "0 0 0")
    else:
        raw = origin.attrib.get(attr, "0 0 0")
    x, y, z = (float(v) for v in raw.split())
    return x, y, z


def _rpy(elem: ET.Element) -> tuple[float, float, float]:
    origin = elem.find("origin")
    raw = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    r, p, y = (float(v) for v in raw.split())
    return r, p, y


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _axis_matrix(axis: tuple[float, float, float], angle: float) -> list[list[float]]:
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    x, y, z = x / n, y / n, z / n
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return [
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _matvec(m: list[list[float]], v: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def _add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _T_from_xyz_rpy(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> tuple[list[list[float]], tuple[float, float, float]]:
    return _rpy_matrix(*rpy), xyz


def _compose(a, b):
    ra, ta = a
    rb, tb = b
    r = _matmul(ra, rb)
    t = _add(ta, _matvec(ra, tb))
    return r, t


def _urdf_fk_and_collisions(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        xyz = tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
        rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
        axis_elem = joint.find("axis")
        axis = tuple(float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()) if axis_elem is not None else (0.0, 0.0, 1.0)
        joints.append((joint.attrib["name"], joint.attrib.get("type"), parent, child, xyz, rpy, axis))

    collisions: dict[str, list[tuple]] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        geoms = []
        for collision in link.findall("collision"):
            origin = collision.find("origin")
            xyz = tuple(float(v) for v in origin.attrib.get("xyz", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
            rpy = tuple(float(v) for v in origin.attrib.get("rpy", "0 0 0").split()) if origin is not None else (0.0, 0.0, 0.0)
            geom = collision.find("geometry")
            child = list(geom)[0] if geom is not None else None
            if child is None:
                continue
            kind = child.tag
            geoms.append((kind, xyz, rpy, child.attrib))
        collisions[name] = geoms

    poses = {"pelvis": (_rpy_matrix(0, 0, 0), (0.0, 0.0, G1_STANDING_PELVIS_Z))}
    pending = list(joints)
    guard = 0
    while pending and guard < 256:
        guard += 1
        leftover = []
        for item in pending:
            name, jtype, parent, child, xyz, rpy, axis = item
            if parent not in poses:
                leftover.append(item)
                continue
            joint_t = _T_from_xyz_rpy(xyz, rpy)
            angle = float(G1_STANDING_JOINT_POS.get(name, 0.0)) if jtype == "revolute" else 0.0
            rot = (_axis_matrix(axis, angle), (0.0, 0.0, 0.0))
            poses[child] = _compose(_compose(poses[parent], joint_t), rot)
        pending = leftover
    assert not pending, f"unresolved joints: {[j[0] for j in pending]}"
    return collisions, poses


def _geom_min_z(kind: str, xyz, rpy, attrib, link_pose) -> float:
    geom_t = _compose(link_pose, _T_from_xyz_rpy(xyz, rpy))
    rot, trans = geom_t
    if kind == "sphere":
        return trans[2] - float(attrib["radius"])
    if kind == "box":
        sx, sy, sz = (float(v) for v in attrib["size"].split())
        corners = [
            (x, y, z)
            for x in (-sx / 2, sx / 2)
            for y in (-sy / 2, sy / 2)
            for z in (-sz / 2, sz / 2)
        ]
        return min(_add(trans, _matvec(rot, c))[2] for c in corners)
    if kind == "cylinder":
        radius = float(attrib["radius"])
        length = float(attrib["length"])
        axis = _matvec(rot, (0.0, 0.0, 1.0))
        return trans[2] - abs(axis[2]) * (length / 2.0) - radius
    raise AssertionError(f"unsupported collision type {kind}")


def test_g1_standing_pose_matches_mjcf_keyframe() -> None:
    assert G1_STANDING_PELVIS_Z == pytest.approx(0.76)
    assert G1_STANDING_JOINT_POS["left_hip_pitch_joint"] == pytest.approx(-0.312)
    assert G1_STANDING_JOINT_POS["left_knee_joint"] == pytest.approx(0.669)
    assert G1_STANDING_JOINT_POS["left_ankle_pitch_joint"] == pytest.approx(-0.363)
    src = (G1_ASSET / "g1.py").read_text(encoding="utf-8")
    assert "G1_STANDING_JOINT_POS" in src
    assert "G1_STANDING_PELVIS_Z" in src
    assert '"left_hip_pitch_joint": -0.20' not in src


def test_g1_does_not_use_isaaclab_bundled_g1_cfg() -> None:
    src = (G1_ASSET / "g1.py").read_text(encoding="utf-8")
    assert "from isaaclab_assets" not in src
    assert "g1_29dof_mode_15.urdf" in src
    assert "G1_STANDING_JOINT_POS" in src


def test_g1_teacher_uses_mirror_symmetry() -> None:
    text = (ROOT / "legged_lab/envs/g1/teacher_cfg.py").read_text(encoding="utf-8")
    assert "use_mirror_loss=True" in text
    assert "mirror_loss_coeff=5.0" in text
    assert "legged_lab.envs.g1.symmetry:get_symmetric_states" in text


def test_g1_mirror_is_involution_on_sparse_obs_and_actions() -> None:
    import importlib.util

    import numpy as np

    path = ROOT / "legged_lab/envs/g1/symmetry.py"
    spec = importlib.util.spec_from_file_location("g1_symmetry", path)
    symmetry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(symmetry)

    rng = np.random.default_rng(0)
    actor = rng.normal(size=(2, symmetry.G1_SPARSE_ACTOR_OBS_DIM))
    critic = rng.normal(size=(2, symmetry.G1_SPARSE_CRITIC_OBS_DIM))
    actions = rng.normal(size=(2, NUM_G1_29DOF_JOINTS))
    assert np.allclose(symmetry.mirror_observations(symmetry.mirror_observations(actor, False), False), actor)
    assert np.allclose(symmetry.mirror_observations(symmetry.mirror_observations(critic, True), True), critic)
    assert np.allclose(symmetry.mirror_actions(symmetry.mirror_actions(actions)), actions)


def test_g1_mirror_swaps_legs_and_negates_roll() -> None:
    import importlib.util

    import numpy as np

    path = ROOT / "legged_lab/envs/g1/symmetry.py"
    spec = importlib.util.spec_from_file_location("g1_symmetry_swap", path)
    symmetry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(symmetry)

    obs = np.zeros((1, symmetry.G1_SPARSE_ACTOR_OBS_DIM))
    start = 9 + G1_29DOF_JOINT_NAMES.index("left_hip_pitch_joint")
    obs[0, start + G1_29DOF_JOINT_NAMES.index("left_hip_pitch_joint")] = 0.5
    obs[0, start + G1_29DOF_JOINT_NAMES.index("left_hip_roll_joint")] = 0.2
    mirrored = symmetry.mirror_observations(obs, False)
    assert mirrored[0, start + G1_29DOF_JOINT_NAMES.index("right_hip_pitch_joint")] == pytest.approx(0.5)
    assert mirrored[0, start + G1_29DOF_JOINT_NAMES.index("right_hip_roll_joint")] == pytest.approx(-0.2)


def test_g1_urdf_collision_matches_mjcf_not_mesh() -> None:
    import xml.etree.ElementTree as ET

    from legged_lab.assets.unitree_g1.sync_urdf_collision_from_mjcf import load_mjcf_collisions

    urdf_path = G1_ASSET / "urdf/g1_29dof_mode_15.urdf"
    mjcf_cols = load_mjcf_collisions(G1_ASSET / "xmls/g1_actuated.xml")
    root = ET.parse(urdf_path).getroot()
    mesh_cols = 0
    urdf_counts: dict[str, int] = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        cols = link.findall("collision")
        urdf_counts[name] = len(cols)
        for col in cols:
            geom = col.find("geometry/*")
            if geom is not None and geom.tag == "mesh":
                mesh_cols += 1
    assert mesh_cols == 0
    assert urdf_counts["left_ankle_roll_link"] == 7
    assert urdf_counts["right_ankle_roll_link"] == 7
    assert urdf_counts["left_hip_pitch_link"] == 0
    assert len(mjcf_cols["left_ankle_roll_link"]) == 7
    assert "pelvis" in mjcf_cols


def test_g1_standing_foot_collision_clears_ground() -> None:
    collisions, poses = _urdf_fk_and_collisions(G1_ASSET / "urdf/g1_29dof_mode_15.urdf")
    bottoms = []
    for side in ("left_ankle_roll_link", "right_ankle_roll_link"):
        assert collisions[side], f"{side} has no collision"
        for kind, xyz, rpy, attrib in collisions[side]:
            bottoms.append(_geom_min_z(kind, xyz, rpy, attrib, poses[side]))
    assert bottoms
    lowest = min(bottoms)
    assert 0.0 <= lowest <= 0.03, f"standing foot collision min z={lowest:.4f} m (want 0-3 cm above ground)"
