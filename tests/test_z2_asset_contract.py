# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""Z2 29DoF asset/AMP/teacher contract. Expected facts come from upstream sources."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from legged_lab.assets.z2.amp_source import (
    UPSTREAM_TXT_NOT_SOURCE,
    convert_file,
    load_motion,
    reorder_dofs,
)
from legged_lab.assets.z2.constants import (
    NUM_Z2_29DOF_JOINTS,
    Z2_20DOF_JOINT_NAMES,
    Z2_23DOF_JOINT_NAMES,
    Z2_29DOF_JOINT_NAMES,
    Z2_ASSET_DEFAULT_PELVIS_Z,
    Z2_FIXED_JOINT_NAMES,
    Z2_FOOT_SCAN_SIZE,
    Z2_NOMINAL_FEET_Y_DISTANCE,
    Z2_SOURCE_JOINT_ORDER,
    Z2_STANDING_JOINT_POS,
    Z2_STANDING_PELVIS_Z,
    Z2_UPSTREAM_AMP_JOINT_ORDER,
    Z2_URDF_REVOLUTE_JOINT_NAMES,
    Z2_URDF_TOTAL_MASS_KG,
)
from legged_lab.assets.z2.kinematic_amp import standing_amp_preview
from legged_lab.assets.z2.locomotion import Z2_LOCOMOTION
from legged_lab.assets.z2.schemas import (
    AMP_FIELDS,
    AMP_FOOT_BODIES,
    AMP_FRAME_DIM,
    AMP_HAND_BODIES,
    AMP_MOTION_SOURCE_DIR,
    UPSTREAM_64D_EXPERT_WIDTH,
    Z2_SOURCE_CSV_WIDTH,
    amp_expert_files,
    amp_field_slice,
)
from legged_lab.assets.z2.variants import Z2_29DOF_PD_RECIPES, Z2_TEACHER_PD_RECIPE
from legged_lab.locomotion.schemas import ObservationLayout
from rsl_rl.utils.motion_loader import AMPLoader

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT / "work/upstream-z2"
Z2_ASSET = ROOT / "legged_lab/assets/z2"


def _list_assign(src: str, name: str) -> list:
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name and isinstance(node.value, ast.List):
                    return [ast.literal_eval(elt) for elt in node.value.elts]
    raise KeyError(name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_policy_joints_match_upstream_z2_py() -> None:
    src = (UP / "legged_lab/assets/z2.py").read_text(encoding="utf-8")
    assert tuple(_list_assign(src, "Z2_29DOF_JOINT_NAMES")) == Z2_29DOF_JOINT_NAMES
    assert tuple(_list_assign(src, "Z2_23DOF_JOINT_NAMES")) == Z2_23DOF_JOINT_NAMES
    assert tuple(_list_assign(src, "Z2_20DOF_JOINT_NAMES")) == Z2_20DOF_JOINT_NAMES
    assert NUM_Z2_29DOF_JOINTS == 29
    assert len(set(Z2_29DOF_JOINT_NAMES)) == 29


def test_three_joint_orders_are_not_the_same() -> None:
    convert_src = (UP / "legged_lab/scripts/convert_z2_amp_data.py").read_text(encoding="utf-8")
    source = tuple(_list_assign(convert_src, "SOURCE_JOINT_ORDER_FALLBACK"))
    amp = tuple(_list_assign(convert_src, "AMP_JOINT_ORDER"))
    assert source == Z2_SOURCE_JOINT_ORDER == Z2_URDF_REVOLUTE_JOINT_NAMES
    assert amp == Z2_UPSTREAM_AMP_JOINT_ORDER
    assert source != Z2_29DOF_JOINT_NAMES
    assert amp != Z2_29DOF_JOINT_NAMES
    assert source != amp
    assert set(source) == set(Z2_29DOF_JOINT_NAMES) == set(amp)


def test_urdf_is_29_revolute_with_fixed_neck_and_sphere_hands() -> None:
    urdf = ET.parse(Z2_ASSET / "urdf/assembly.urdf").getroot()
    joints = [(j.attrib["name"], j.attrib.get("type")) for j in urdf.findall("joint")]
    revolute = [name for name, kind in joints if kind == "revolute"]
    fixed = [name for name, kind in joints if kind == "fixed"]
    assert tuple(revolute) == Z2_URDF_REVOLUTE_JOINT_NAMES
    assert tuple(fixed) == Z2_FIXED_JOINT_NAMES
    masses = [
        float(link.find("inertial/mass").attrib["value"])
        for link in urdf.findall("link")
        if link.find("inertial/mass") is not None
    ]
    assert sum(masses) == pytest.approx(Z2_URDF_TOTAL_MASS_KG, rel=0, abs=1e-9)


def test_copied_meshes_exist_and_are_not_lfs_pointers() -> None:
    urdf_path = Z2_ASSET / "urdf/assembly.urdf"
    urdf = ET.parse(urdf_path).getroot()
    for mesh in urdf.findall(".//mesh"):
        path = urdf_path.parent / mesh.attrib["filename"]
        assert path.is_file(), path
        head = path.read_bytes()[:64]
        assert not head.startswith(b"version https://git-lfs.github.com/spec/v1"), path
        assert path.stat().st_size > 1000, path


def test_training_urdf_keeps_upstream_left_foot_collision() -> None:
    """Source collision fidelity. L/R mesh SHA difference is not a plant defect."""

    def foot_collision(path: Path, link_name: str) -> list[str]:
        root = ET.parse(path).getroot()
        link = root.find(f"./link[@name='{link_name}']")
        return [
            col.find("geometry/mesh").attrib["filename"]
            for col in link.findall("collision")
            if col.find("geometry/mesh") is not None
        ]

    orig = UP / "legged_lab/assets/z2_description/assembly_urdf_29/assembly.urdf"
    kept = Z2_ASSET / "urdf/upstream_assembly.urdf"
    train = Z2_ASSET / "urdf/assembly.urdf"
    defined = {
        node.name
        for node in ast.walk(ast.parse((ROOT / "tests/test_z2_asset_contract.py").read_text(encoding="utf-8")))
        if isinstance(node, ast.FunctionDef)
    }
    assert "test_training_urdf_keeps_upstream_left_foot_collision" in defined
    assert "test_upstream_left_foot_collision_used_right_mesh_and_training_urdf_does_not" not in defined
    for path in (orig, kept, train):
        assert foot_collision(path, "L_ankle_roll_link") == ["meshes/R_ankle_roll_link.STL"]
        assert foot_collision(path, "R_ankle_roll_link") == ["meshes/R_ankle_roll_link.STL"]
    visual = ET.parse(train).getroot().find("./link[@name='L_ankle_roll_link']/visual/geometry/mesh")
    assert visual.attrib["filename"] == "meshes/L_ankle_roll_link.STL"
    note = train.read_text(encoding="utf-8")
    assert "L-mesh swap" in note and "rejected" in note
    left = _sha256(Z2_ASSET / "urdf/meshes/L_ankle_roll_link.STL")
    right = _sha256(Z2_ASSET / "urdf/meshes/R_ankle_roll_link.STL")
    assert left != right


def test_teacher_selects_walk_pose_damped_pd_not_generic_or_g1() -> None:
    text = (ROOT / "legged_lab/envs/z2/teacher_cfg.py").read_text(encoding="utf-8")
    assert "Z2LocoTeacherEnvCfg(LightLPLocomotionEnvCfg)" in text
    assert "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG" in text
    assert "Z2_29DOF_CFG.copy()" not in text
    assert "envs.g1" not in text
    assert "envs.t4" not in text
    assert "G1_29DOF" not in text
    recipe = Z2_29DOF_PD_RECIPES[Z2_TEACHER_PD_RECIPE]
    generic = Z2_29DOF_PD_RECIPES["Z2_29DOF_CFG"]
    assert recipe["ankle_effort_limit_sim"] == 150.0
    assert generic["ankle_effort_limit_sim"] == 75.0
    assert recipe != generic
    src = (Z2_ASSET / "z2.py").read_text(encoding="utf-8")
    assert "ankle_effort_limit_sim=150.0" in src
    assert "hip_pitch_knee_damping=6.0" in src
    assert "ankle_damping=4.0" in src
    assert "self.robot.action_scale = 0.25" in text
    assert "Z2UsdFileCfg(usd_path=str(Z2_29DOF_USD_PATH))" in src
    assert "pos=(0.0, 0.0, Z2_ASSET_DEFAULT_PELVIS_Z)" in src
    assert "Z2_STANDING_PELVIS_Z" in text
    assert Z2_ASSET_DEFAULT_PELVIS_Z == 0.8
    assert Z2_STANDING_PELVIS_Z == 0.75
    assert "self.robot.terminate_contacts_body_names = []" in text
    assert "self.scene.max_init_terrain_level = 0" in text
    assert "self.random_level_reset_fraction = 0.10" in text
    assert "collapse_reset_pelvis_above_feet_m" not in text
    init = (ROOT / "legged_lab/envs/__init__.py").read_text(encoding="utf-8")
    assert 'task_registry.register("z2_loco_teacher", LocomotionEnv' in init


def test_spec_is_z2_not_g1_morphology() -> None:
    Z2_LOCOMOTION.validate()
    assert Z2_LOCOMOTION.name == "z2"
    assert Z2_LOCOMOTION.feet == AMP_FOOT_BODIES == ("L_ankle_roll_link", "R_ankle_roll_link")
    assert Z2_LOCOMOTION.hands == AMP_HAND_BODIES == ("L_wrist_yaw_link", "R_wrist_yaw_link")
    assert Z2_LOCOMOTION.torso == "waist_roll_link"
    assert Z2_LOCOMOTION.nominal_feet_distance == pytest.approx(Z2_NOMINAL_FEET_Y_DISTANCE)
    from legged_lab.assets.unitree_g1.constants import G1_NOMINAL_FEET_Y_DISTANCE

    assert Z2_LOCOMOTION.nominal_feet_distance != pytest.approx(G1_NOMINAL_FEET_Y_DISTANCE)
    # G1 uses left_/right_ names and torso_link; Z2 must not.
    assert "torso_link" not in Z2_LOCOMOTION.diagnostic_bodies
    assert not any(name.startswith("left_") for name in Z2_LOCOMOTION.joint_names)
    mirrored = [Z2_29DOF_JOINT_NAMES[i] for i in Z2_LOCOMOTION.mirror_indices]
    assert mirrored[Z2_29DOF_JOINT_NAMES.index("L_hip_pitch_joint")] == "R_hip_pitch_joint"
    assert Z2_LOCOMOTION.mirror_signs[Z2_29DOF_JOINT_NAMES.index("L_hip_roll_joint")] == -1.0
    assert Z2_LOCOMOTION.mirror_signs[Z2_29DOF_JOINT_NAMES.index("L_hip_pitch_joint")] == 1.0
    assert Z2_LOCOMOTION.mirror_signs[Z2_29DOF_JOINT_NAMES.index("waist_yaw_joint")] == -1.0
    assert Z2_LOCOMOTION.amp_frame_dim == 70


def test_standing_pose_is_upstream_z2_not_g1_mimic() -> None:
    assert Z2_STANDING_PELVIS_Z == 0.75
    assert Z2_STANDING_JOINT_POS["L_hip_pitch_joint"] == pytest.approx(-0.2)
    assert Z2_STANDING_JOINT_POS["L_knee_joint"] == pytest.approx(0.42)
    assert Z2_STANDING_JOINT_POS["L_ankle_pitch_joint"] == pytest.approx(-0.23)
    assert Z2_STANDING_JOINT_POS["L_hip_pitch_joint"] != pytest.approx(-0.312)


def test_amp_schema_is_shared_70d_not_upstream_64d() -> None:
    assert AMP_FRAME_DIM == 70
    assert dict(AMP_FIELDS) == {"joint_pos": 29, "joint_vel": 29, "hand_pos_root": 6, "foot_pos_root": 6}
    assert amp_field_slice("foot_pos_root") == (64, 70)
    amp_cfg = (UP / "legged_lab/envs/z2/z2_switch_29dof_cfg.py").read_text(encoding="utf-8")
    assert '"endpoint_mode": "no_feet"' in amp_cfg
    assert UPSTREAM_64D_EXPERT_WIDTH == 64
    files = amp_expert_files("dir")
    assert [Path(path).stem for path in files] == ["run", "walk", "walk_l"]


def test_source_conversion_reorders_walk_pkl_from_upstream_names() -> None:
    raw = Z2_ASSET.parents[1] / "legged_lab/envs/z2/datasets/motion_source_raw/walk.pkl"
    if not raw.is_file():
        raw = UP / "legged_lab/envs/z2/datasets/z2_data/z2_data_pkl/walk.pkl"
    motion = load_motion(raw)
    assert motion["source_joint_names"] == list(Z2_SOURCE_JOINT_ORDER)
    reordered = reorder_dofs(motion["dof_pos"], motion["source_joint_names"])
    src_index = {name: i for i, name in enumerate(motion["source_joint_names"])}
    for j, name in enumerate(Z2_29DOF_JOINT_NAMES):
        np.testing.assert_allclose(reordered[:, j], motion["dof_pos"][:, src_index[name]])


def test_whitelist_csvs_match_policy_width_and_reject_g1() -> None:
    source = ROOT / AMP_MOTION_SOURCE_DIR
    declared = {Path(path).stem for path in amp_expert_files()}
    available = {path.stem for path in source.glob("*.csv")}
    assert declared <= available, f"missing CSVs {sorted(declared - available)}"
    for stem in sorted(declared):
        rows = []
        with (source / f"{stem}.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if row:
                    rows.append([float(v) for v in row])
                    if len(rows) >= 3:
                        break
        assert {len(row) for row in rows} == {Z2_SOURCE_CSV_WIDTH}
        assert all(math.isfinite(v) for row in rows for v in row)
    g1_csv = ROOT / "legged_lab/envs/g1/datasets/motion_source/walk1_subject1.csv"
    with pytest.raises(ValueError):
        convert_file(g1_csv, source / "from_g1.csv")


def test_amp_loader_rejects_g1_and_upstream_64d() -> None:
    from tempfile import TemporaryDirectory

    g1 = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v5/walk1_subject1.txt"
    upstream_64 = UP / "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert/walk_repaired.txt"
    with pytest.raises(ValueError):
        AMPLoader(
            device="cpu",
            time_between_frames=0.02,
            frame_dim=70,
            motion_files=[str(g1)],
            expected_joint_order=list(Z2_29DOF_JOINT_NAMES),
        )
    with pytest.raises(ValueError):
        AMPLoader(
            device="cpu",
            time_between_frames=0.02,
            frame_dim=70,
            motion_files=[str(upstream_64)],
            expected_joint_order=list(Z2_29DOF_JOINT_NAMES),
        )
    payload = json.loads(upstream_64.read_text(encoding="utf-8"))
    frames = np.asarray(payload["Frames"])
    assert frames.shape[1] == 64
    dt = float(payload["FrameDuration"])
    err = np.abs(np.diff(frames[:, :29], axis=0) / dt - frames[:-1, 29:58])
    run = UP / "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert/run.txt"
    run_frames = np.asarray(json.loads(run.read_text(encoding="utf-8"))["Frames"])
    run_dt = float(json.loads(run.read_text(encoding="utf-8"))["FrameDuration"])
    run_err = np.abs(np.diff(run_frames[:, :29], axis=0) / run_dt - run_frames[:-1, 29:58])
    assert float(err.max()) == pytest.approx(0.0)
    assert float(run_err.mean()) > 1.0
    with TemporaryDirectory() as tmp:
        good = Path(tmp) / "walk.txt"
        good.write_text(
            json.dumps(
                {
                    "LoopMode": "Wrap",
                    "FrameDuration": 0.0333,
                    "MotionWeight": 0.5,
                    "JointOrder": list(Z2_29DOF_JOINT_NAMES),
                    "Frames": np.zeros((4, 70)).tolist(),
                }
            ),
            encoding="utf-8",
        )
        loader = AMPLoader(
            device="cpu",
            time_between_frames=0.02,
            frame_dim=70,
            motion_files=[str(good)],
            expected_joint_order=list(Z2_29DOF_JOINT_NAMES),
        )
        assert loader.trajectories[0].shape[-1] == 70


def test_z2_layout_actor_matches_29dof_critic_uses_z2_foot_scan() -> None:
    layout = ObservationLayout(29, scan_history=5, actor_contact=True, critic_foot_scan=True, critic_immunity=True)
    assert layout.actor_dim == 1997
    foot = SimpleNamespace(resolution=0.04, size=Z2_FOOT_SCAN_SIZE)
    scan = SimpleNamespace(resolution=0.1, size=(1.4, 1.2))
    cfg = SimpleNamespace(
        robot_spec=Z2_LOCOMOTION,
        robot=SimpleNamespace(actor_obs_history_length=10, critic_obs_history_length=10),
        teacher_scan_history_length=5,
        scene=SimpleNamespace(height_scanner=scan, foot_scanner=foot),
        append_actor_feet_contact=True,
        append_critic_foot_scan=True,
        append_critic_immunity=True,
    )
    runtime = ObservationLayout.from_cfg(cfg)
    assert runtime.actor_dim == 1997
    assert runtime.foot_scan_shape == (7, 3)
    assert runtime.critic_dim == 2088


def test_incomplete_z2_specs_fail() -> None:
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(Z2_LOCOMOTION, joint_names=("duplicate",) * 29).validate()


def test_mjcf_zero_quat_was_illegal_and_copy_is_loadable() -> None:
    orig_up = UP / "legged_lab/assets/z2_description/assembly_mjcf_29/assembly.xml"
    orig_copy = Z2_ASSET / "mjcf/upstream_assembly.xml"
    ours = (Z2_ASSET / "mjcf/assembly.xml").read_text(encoding="utf-8")
    orig = orig_up.read_text(encoding="utf-8")
    assert orig_copy.read_bytes() == orig_up.read_bytes()
    assert 'quat="0 0 0 0"' in orig
    assert 'quat="0 0 0 0"' not in ours
    mujoco = pytest.importorskip("mujoco")
    with pytest.raises(ValueError, match="zero quaternion"):
        mujoco.MjModel.from_xml_path(str(orig_copy))
    model = mujoco.MjModel.from_xml_path(str(Z2_ASSET / "mjcf/assembly.xml"))
    assert int(model.nu) == 29
    assert int(model.nq) == 7 + 29
    data = mujoco.MjData(model)
    data.qpos[0:3] = [0.0, 0.0, Z2_STANDING_PELVIS_Z]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    name_to_adr = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        if name and name != "floating_base_joint":
            name_to_adr[name] = int(model.jnt_qposadr[i])
    for name, value in Z2_STANDING_JOINT_POS.items():
        data.qpos[name_to_adr[name]] = value
    mujoco.mj_forward(model, data)
    min_z = []
    for i in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[i]))
        if body not in AMP_FOOT_BODIES:
            continue
        gtype = int(model.geom_type[i])
        pos = data.geom_xpos[i]
        size = model.geom_size[i]
        mat = data.geom_xmat[i].reshape(3, 3)
        if gtype == 5:  # cylinder
            axis = mat[:, 2]
            min_z.append(float(pos[2] - abs(axis[2]) * size[1] - size[0]))
        elif gtype == 2:
            min_z.append(float(pos[2] - size[0]))
    assert min_z, "no foot collision geoms"
    lowest = min(min_z)
    assert 0.0 <= lowest <= 0.08, f"MJCF standing foot min_z={lowest:.4f}"


def test_upstream_amp_registration_uses_damped_not_generic_cfg() -> None:
    z2_py = (UP / "legged_lab/assets/z2.py").read_text(encoding="utf-8")
    cfg = (UP / "legged_lab/envs/z2/z2_switch_29dof_cfg.py").read_text(encoding="utf-8")
    registry = (UP / "legged_lab/envs/__init__.py").read_text(encoding="utf-8")
    base = (UP / "legged_lab/envs/z2/z2_switch_base_cfg.py").read_text(encoding="utf-8")
    ab = (UP / "legged_lab/envs/z2/z2_switch_ab_cfg.py").read_text(encoding="utf-8")
    assert "Z2_CFG = Z2_29DOF_CFG" in z2_py
    assert "hip_pitch_knee_damping=6.0" in z2_py
    assert "ankle_damping=4.0" in z2_py
    assert "class Z2SwitchFlatEnvCfg_Base:" in base
    assert "class Z2SwitchFlatEnvCfg_AB(Z2SwitchFlatEnvCfg_Base):" in ab
    assert "class Z2Switch29DofEnvCfg(Z2SwitchFlatEnvCfg_AB):" in cfg
    assert "class Z2Switch29DofAmpBaseEnvCfg(Z2Switch29DofEnvCfg)" in cfg
    assert "self.scene.robot = copy.deepcopy(Z2_29DOF_WALK_POSE_DAMPED_PD_CFG)" in cfg
    for name in (
        "Z2Switch29DofWalkAmpEnvCfg(Z2Switch29DofAmpBaseEnvCfg)",
        "Z2Switch29DofWalkAmpStopPhaseEnvCfg(Z2Switch29DofWalkAmpEnvCfg)",
        "Z2Switch29DofWalkAmpStopPhaseTorsoEnvCfg(Z2Switch29DofWalkAmpStopPhaseEnvCfg)",
        "Z2Switch29DofMixedAmpEnvCfg(Z2Switch29DofAmpBaseEnvCfg)",
        "Z2Switch29DofAmpEnvCfg(Z2Switch29DofMixedAmpEnvCfg)",
    ):
        assert name in cfg
    for task, cls in (
        ("z2_switch_29dof_amp_rl_symmetry", "Z2Switch29DofAmpEnvCfg()"),
        ("z2_switch_29dof_amp_walk_rl_symmetry", "Z2Switch29DofWalkAmpEnvCfg()"),
        ("z2_switch_29dof_amp_walk_stop_phase_rl_symmetry", "Z2Switch29DofWalkAmpStopPhaseEnvCfg()"),
        ("z2_switch_29dof_amp_walk_stop_phase_torso_rl_symmetry", "Z2Switch29DofWalkAmpStopPhaseTorsoEnvCfg()"),
        ("z2_switch_29dof_amp_mixed_rl_symmetry", "Z2Switch29DofMixedAmpEnvCfg()"),
    ):
        assert f'"{task}"' in registry
        assert cls in registry
    amp_base = cfg[cfg.find("class Z2Switch29DofAmpBaseEnvCfg") : cfg.find("class Z2Switch29DofWalkAmpEnvCfg")]
    assert "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG" in amp_base
    assert "physical.base_link_height" in amp_base
    for cls_src in (
        cfg[cfg.find("class Z2Switch29DofWalkAmpEnvCfg") : cfg.find("class Z2Switch29DofWalkAmpStopPhaseEnvCfg")],
        cfg[
            cfg.find("class Z2Switch29DofWalkAmpStopPhaseEnvCfg") : cfg.find(
                "class Z2Switch29DofWalkAmpStopPhaseTorsoEnvCfg"
            )
        ],
        cfg[cfg.find("class Z2Switch29DofWalkAmpStopPhaseTorsoEnvCfg") : cfg.find("class Z2Switch29DofMixedAmpEnvCfg")],
        cfg[cfg.find("class Z2Switch29DofMixedAmpEnvCfg") : cfg.find("class Z2Switch29DofAmpEnvCfg")],
        cfg[cfg.find("class Z2Switch29DofAmpEnvCfg") : cfg.find("class Z2Switch29DofWalkPoseZeroPhaseDampedEnvCfg")],
    ):
        assert "WALK_POSE_DAMPED_PD_CFG" not in cls_src
        assert "Z2_29DOF_CFG" not in cls_src
    assert "action_scale=0.25" in base
    assert "clip_actions=100.0" in base
    assert "dt=0.005, decimation=4" in base
    teacher = (ROOT / "legged_lab/envs/z2/teacher_cfg.py").read_text(encoding="utf-8")
    assert "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG.copy()" in teacher
    our_z2 = (Z2_ASSET / "z2.py").read_text(encoding="utf-8")
    assert "Z2_CFG = Z2_29DOF_CFG" in our_z2
    assert "Z2_29DOF_TEACHER_CFG = Z2_29DOF_WALK_POSE_DAMPED_PD_CFG" in our_z2


def test_source_waist_yaw_pitch_roll_is_not_amp_yaw_roll_pitch() -> None:
    convert_src = (UP / "legged_lab/scripts/convert_z2_amp_data.py").read_text(encoding="utf-8")
    source = _list_assign(convert_src, "SOURCE_JOINT_ORDER_FALLBACK")
    amp = _list_assign(convert_src, "AMP_JOINT_ORDER")
    assert source[:3] == ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"]
    assert amp[-3:] == ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    q_source = np.zeros(29)
    q_source[source.index("waist_pitch_joint")] = 0.11
    q_source[source.index("waist_roll_joint")] = 0.22
    q_source[source.index("waist_yaw_joint")] = 0.33
    q_policy = reorder_dofs(q_source[None, :], source)[0]
    assert q_policy[Z2_29DOF_JOINT_NAMES.index("waist_pitch_joint")] == pytest.approx(0.11)
    assert q_policy[Z2_29DOF_JOINT_NAMES.index("waist_roll_joint")] == pytest.approx(0.22)
    q_amp = np.array([q_policy[Z2_29DOF_JOINT_NAMES.index(name)] for name in amp])
    assert q_amp[-3:].tolist() == pytest.approx([0.33, 0.22, 0.11])


def test_csv_lineage_matches_pkl_not_same_named_txt() -> None:
    source_dir = ROOT / AMP_MOTION_SOURCE_DIR
    raw_dir = ROOT / "legged_lab/envs/z2/datasets/motion_source_raw"
    expert_dir = UP / "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert"
    vis_dir = UP / "legged_lab/envs/z2/datasets/z2_data/motion_visualization"

    def n_csv(stem: str) -> int:
        return sum(1 for row in (source_dir / f"{stem}.csv").read_text(encoding="utf-8").splitlines() if row.strip())

    def n_txt(path: Path) -> tuple[int, int]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload["Frames"]
        return len(frames), len(frames[0])

    walk_pkl = load_motion(raw_dir / "walk.pkl")
    walk_l_pkl = load_motion(raw_dir / "walk_l.pkl")
    run_pkl = load_motion(raw_dir / "run.pkl")
    assert n_csv("walk") == walk_pkl["dof_pos"].shape[0] == 75
    assert n_csv("walk_l") == walk_l_pkl["dof_pos"].shape[0] == 281
    assert n_csv("run") == run_pkl["dof_pos"].shape[0] == 40
    walk_n, walk_w = n_txt(expert_dir / "walk.txt")
    run_n, run_w = n_txt(expert_dir / "run.txt")
    vis_n, vis_w = n_txt(vis_dir / "run.txt")
    assert (walk_n, walk_w) == (74, 64)
    assert (run_n, run_w) == (39, 64)
    assert (vis_n, vis_w) == (171, 70)
    assert n_csv("run") != run_n
    assert n_csv("run") != vis_n
    assert n_csv("walk") != walk_n
    for paths in UPSTREAM_TXT_NOT_SOURCE.values():
        for rel in paths:
            assert (UP / rel).is_file(), rel


def test_unrepaired_experts_fail_qvel_repaired_walk_passes() -> None:
    expert = UP / "legged_lab/envs/z2/datasets/z2_data/motion_amp_expert"

    def err(name: str):
        payload = json.loads((expert / name).read_text(encoding="utf-8"))
        frames = np.asarray(payload["Frames"], dtype=np.float64)
        dt = float(payload["FrameDuration"])
        delta = np.abs(np.diff(frames[:, :29], axis=0) / dt - frames[:-1, 29:58])
        return float(delta.mean()), float(delta.max()), int(frames.shape[1])

    run_mae, run_max, run_w = err("run.txt")
    walk_mae, _, walk_w = err("walk.txt")
    walk_l_mae, _, walk_l_w = err("walk_l.txt")
    repaired_mae, repaired_max, repaired_w = err("walk_repaired.txt")
    assert run_w == walk_w == walk_l_w == repaired_w == 64
    assert run_mae == pytest.approx(4.245582816127713)
    assert run_max == pytest.approx(14.394983644838687)
    assert walk_mae > 1.0 and walk_l_mae > 1.0
    assert repaired_mae == pytest.approx(0.0)
    assert repaired_max == pytest.approx(0.0)
    subject_w = len(json.loads((expert / "walk1_subject1.txt").read_text(encoding="utf-8"))["Frames"][0])
    assert subject_w == 70
    g1_compat = list((UP / "legged_lab/envs/z2/datasets/g1_compat").rglob("*.txt"))
    assert len(g1_compat) >= 8


def test_urdf_fk_70d_preview_includes_feet_and_is_not_training_expert() -> None:
    standing = standing_amp_preview(0.0)
    bent = standing_amp_preview(0.4)
    assert standing.shape == bent.shape == (AMP_FRAME_DIM,)
    assert np.isfinite(standing).all() and np.isfinite(bent).all()
    feet = standing[-6:]
    hands = standing[-12:-6]
    assert np.linalg.norm(feet) > 0.05
    assert np.linalg.norm(hands) > 0.05
    assert float(np.max(np.abs(bent[-6:] - standing[-6:]))) > 1.0e-3
    generate = (ROOT / "legged_lab/scripts/generate_z2_amp_expert.py").read_text(encoding="utf-8")
    assert "AmpFeatureBuilder" in generate
    assert "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG" in generate
    assert "validate_z2_source_manifest" in generate
    assert "amp_training_manifest" in generate


def test_expert_lifecycle_pending_and_completed_states(tmp_path) -> None:
    from legged_lab.assets.z2.amp_manifest import (
        COMPLETED_EXPERT_STATUS,
        PENDING_EXPERT_STATUS,
        amp_training_manifest,
        assert_completed_expert_tree,
        assert_expert_tree_lifecycle,
        assert_pending_expert_tree,
        code_provenance,
        sha256_file,
        training_clip_record,
        validate_z2_source_manifest,
    )
    from legged_lab.assets.z2.schemas import AMP_MOTION_CLASSES

    source_dir = ROOT / "legged_lab/envs/z2/datasets/motion_source"
    live = ROOT / "legged_lab/envs/z2/datasets/motion_amp_expert"
    live_status = assert_expert_tree_lifecycle(live, source_dir=source_dir, root=ROOT)
    assert live_status in (PENDING_EXPERT_STATUS, COMPLETED_EXPERT_STATUS)

    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "_manifest.json").write_text(
        json.dumps({"status": PENDING_EXPERT_STATUS, "clips": {}, "motions": {}}) + "\n",
        encoding="utf-8",
    )
    assert_pending_expert_tree(pending)
    (pending / "walk.txt").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must not contain walk.txt"):
        assert_pending_expert_tree(pending)

    completed = tmp_path / "completed"
    completed.mkdir()
    source_manifest = json.loads((source_dir / "_manifest.json").read_text(encoding="utf-8"))
    checked = validate_z2_source_manifest(source_manifest, source_dir)
    clips = {}
    motions = {}
    for stem, source in checked.items():
        fps = float(source["fps"])
        n_frames = int(source["frames"]) - 1
        payload = {
            "LoopMode": "Wrap",
            "FrameDuration": 1.0 / fps,
            "MotionWeight": 0.5,
            "JointOrder": list(Z2_29DOF_JOINT_NAMES),
            "Frames": [[0.0] * AMP_FRAME_DIM for _ in range(n_frames)],
        }
        txt_path = completed / f"{stem}.txt"
        txt_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        clips[stem] = training_clip_record(
            txt_path=txt_path,
            source_csv=source["csv"],
            source_csv_sha256=source["csv_sha256"],
            source_raw_sha256=source["source_sha256"],
            frames=n_frames,
            fps=fps,
            motion_class=AMP_MOTION_CLASSES[stem],
            motion_weight=0.5,
            root=ROOT,
        )
        motions[stem] = {"frames": n_frames, "fps": fps}
        assert clips[stem]["sha256"] == sha256_file(txt_path)
    manifest = amp_training_manifest(
        clips=clips,
        motions=motions,
        kinematics_response=0.1,
        sim_dt=0.005,
        max_frames=0,
        converter={"asset_mode": "upstream_usd", "urdf_converter_applied": False},
        plant="Z2_29DOF_WALK_POSE_DAMPED_PD_CFG",
        joint_order=list(Z2_29DOF_JOINT_NAMES),
        held_out=[],
        frame_dim=AMP_FRAME_DIM,
        provenance=code_provenance(ROOT),
    )
    (completed / "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    assert_completed_expert_tree(completed, source_dir=source_dir, root=ROOT)
    broken = json.loads((completed / "_manifest.json").read_text(encoding="utf-8"))
    broken["clips"]["walk"]["sha256"] = "0" * 64
    (completed / "_manifest.json").write_text(json.dumps(broken) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clip sha256 mismatch"):
        assert_completed_expert_tree(completed, source_dir=source_dir, root=ROOT)


def test_z2_recipe_sets_max_radial_progress_without_g1_cli() -> None:
    from legged_lab.assets.z2.eval_contract import (
        LIGHTLP_TILE_SIZE_M,
        z2_training_progress_contract,
    )
    from legged_lab.locomotion.curriculum import lightlp_terrain_level_moves
    from legged_lab.locomotion.mdp.sparse_signals import (
        oob_linf_limit_m,
        promote_radius_m,
    )

    shared = (ROOT / "legged_lab/locomotion/teacher_cfg.py").read_text(encoding="utf-8")
    g1 = (ROOT / "legged_lab/envs/g1/teacher_cfg.py").read_text(encoding="utf-8")
    z2 = (ROOT / "legged_lab/envs/z2/teacher_cfg.py").read_text(encoding="utf-8")
    train = (ROOT / "legged_lab/scripts/train.py").read_text(encoding="utf-8")
    assert "class LightLPLocomotionEnvCfg(AmpLocomotionEnvCfg)" in shared
    assert 'lightlp_promotion_distance: str = "path_length"' in shared
    assert "progress_monitor_enabled: bool = False" in shared
    assert "lightlp_promotion_distance" not in g1
    assert "progress_monitor_enabled" not in g1
    assert 'self.lightlp_promotion_distance = "max_radial"' in z2
    assert "self.progress_monitor_enabled = True" in z2
    assert "envs.g1" not in z2
    assert "motion_experiment" not in z2
    assert 'env_class_name != "g1_loco_teacher"' in train
    contract = z2_training_progress_contract()
    assert contract["lightlp_promotion_distance"] == "max_radial"
    assert contract["progress_monitor_enabled"] is True
    assert contract["shared_default_promotion"] == "path_length"
    assert contract["promotion_distance_m"] == promote_radius_m(LIGHTLP_TILE_SIZE_M) == 4.0
    assert contract["oob_distance_m"] == oob_linf_limit_m(LIGHTLP_TILE_SIZE_M) == 4.25
    path = np.array([6.0])
    peak = np.array([0.05])
    tracking = np.array([0.8])
    cmd = np.array([0.6])
    legacy, _ = lightlp_terrain_level_moves(path, tracking, cmd, LIGHTLP_TILE_SIZE_M)
    radial, _ = lightlp_terrain_level_moves(path, tracking, cmd, LIGHTLP_TILE_SIZE_M, max_radial_dist=peak)
    assert bool(legacy[0]) and not bool(radial[0])
    assert contract["promotion_distance_m"] < contract["oob_distance_m"]


def test_visualization_70d_schema_is_not_shared_amp_70d() -> None:
    from legged_lab.assets.z2.schemas import (
        UPSTREAM_64D_EXPERT_WIDTH,
        UPSTREAM_VISUALIZATION_WIDTH,
    )

    convert = (UP / "legged_lab/scripts/convert_z2_amp_data.py").read_text(encoding="utf-8")
    assert "root_pos[:-1]" in convert
    assert "root_euler" in convert
    assert "dof_pos_amp[:-1]" in convert
    assert "root_lin_vel" in convert
    assert "root_ang_vel" in convert
    assert "dof_vel_amp" in convert
    vis = json.loads(
        (UP / "legged_lab/envs/z2/datasets/z2_data/motion_visualization/run.txt").read_text(encoding="utf-8")
    )
    vis_w = len(vis["Frames"][0])
    vis_layout = 3 + 3 + NUM_Z2_29DOF_JOINTS + 3 + 3 + NUM_Z2_29DOF_JOINTS
    amp_layout = sum(width for _, width in AMP_FIELDS)
    assert vis_w == vis_layout == UPSTREAM_VISUALIZATION_WIDTH == AMP_FRAME_DIM == amp_layout == 70
    assert vis["Frames"][0][2] > 0.5  # visualization root z; shared AMP 70D starts at q29
    assert "foot_pos_root" not in convert
    assert dict(AMP_FIELDS)["foot_pos_root"] == 6
    assert "hand_pos_root" in dict(AMP_FIELDS)
    assert UPSTREAM_64D_EXPERT_WIDTH == 64
    assert vis_layout != UPSTREAM_64D_EXPERT_WIDTH


def test_eval_locomotion_ignores_g1_cli_and_z2_helper_is_explicit(tmp_path) -> None:
    from legged_lab.assets.z2.eval_contract import (
        COMPLETED_EXPERT_RUNTIME,
        ISAAC_PROBE_EVIDENCE,
        PENDING_RUNTIME_CONFIG,
        PHYSICS_USD_EVIDENCE,
        eval_locomotion_ignored_g1_flags,
        reject_eval_g1_cli_flags,
        runtime_evidence_status,
        z2_eval_provenance,
    )

    eval_src = (ROOT / "legged_lab/scripts/eval_locomotion.py").read_text(encoding="utf-8")
    dump_src = (ROOT / "legged_lab/scripts/dump_z2_eval_contract.py").read_text(encoding="utf-8")
    assert "parse_known_args" in eval_src
    assert "task_registry.get_cfgs" in eval_src
    assert "apply_vital_motion_experiment" not in eval_src
    assert "g1_progress_ab" not in eval_src
    assert "g1_motion_experiment" not in eval_src
    assert 'startswith("--g1_")' in eval_src
    assert "does not apply train.py G1 experiment flags" in eval_src
    assert "legged_lab.envs.g1" not in eval_src
    assert "legged_lab.assets.z2" not in eval_src
    assert "legged_lab.envs.z2" not in eval_src
    assert "legged_lab.assets.z2.eval_contract" in dump_src
    assert "isaaclab" not in dump_src
    assert not (ROOT / "legged_lab/envs/z2/eval_contract.py").exists()
    assert eval_locomotion_ignored_g1_flags(["--task", "g1_loco_teacher", "--g1_motion_experiment", "vital_v1"]) == (
        "--g1_motion_experiment",
        "vital_v1",
    )
    with pytest.raises(ValueError, match="does not apply"):
        reject_eval_g1_cli_flags(["--g1_progress_ab", "B"])
    reject_eval_g1_cli_flags(["--task", "z2_loco_teacher"])
    provenance = z2_eval_provenance()
    assert provenance["task"] == "z2_loco_teacher"
    assert provenance["asset_mode"] == "upstream_usd"
    assert provenance["lightlp_promotion_distance"] == "max_radial"
    assert provenance["g1_comparison"]["eval_cli_flags_apply_vital_v1"] is False
    assert provenance["g1_comparison"]["eval_does_not_load_saved_run_env_yaml"] is True
    assert provenance["g1_comparison"]["requires"] == "frozen_run_env_yaml_or_explicit_profile_helper"
    status = provenance["resolved_runtime_config"]
    assert status == runtime_evidence_status(ROOT)
    if status not in (PENDING_RUNTIME_CONFIG, COMPLETED_EXPERT_RUNTIME):
        assert status in (ISAAC_PROBE_EVIDENCE, PHYSICS_USD_EVIDENCE)
        assert (ROOT / status).is_file()
    empty = tmp_path / "empty_root"
    empty.mkdir()
    assert runtime_evidence_status(empty) == PENDING_RUNTIME_CONFIG
    evidence = tmp_path / "with_probe"
    probe = evidence / ISAAC_PROBE_EVIDENCE
    probe.parent.mkdir(parents=True)
    probe.write_text("{}\n", encoding="utf-8")
    assert runtime_evidence_status(evidence) == ISAAC_PROBE_EVIDENCE


def test_generic_asset_root_is_0_8_teacher_overrides_0_75() -> None:
    teacher = (ROOT / "legged_lab/envs/z2/teacher_cfg.py").read_text(encoding="utf-8")
    asset = (Z2_ASSET / "z2.py").read_text(encoding="utf-8")
    assert Z2_ASSET_DEFAULT_PELVIS_Z == 0.8
    assert Z2_STANDING_PELVIS_Z == 0.75
    assert "pos=(0.0, 0.0, Z2_ASSET_DEFAULT_PELVIS_Z)" in asset
    assert "self.scene.robot.init_state.pos = (0.0, 0.0, Z2_STANDING_PELVIS_Z)" in teacher
    assert "Z2_29DOF_WALK_POSE_DAMPED_PD_CFG = _make_z2_cfg(" in asset
    up_asset = (UP / "legged_lab/assets/z2.py").read_text(encoding="utf-8")
    assert "pos=(0.0, 0.0, 0.8)" in up_asset


def test_training_spawn_uses_upstream_usd_not_urdf_reimport() -> None:
    from legged_lab.assets.z2.amp_manifest import usd_layer_shas
    from legged_lab.assets.z2.constants import Z2_UPSTREAM_USD_DIR, Z2_USD_LAYER_FILES

    src = (Z2_ASSET / "z2.py").read_text(encoding="utf-8")
    assert "class Z2UsdFileCfg(sim_utils.UsdFileCfg)" in src
    assert "spawn=Z2UsdFileCfg(usd_path=str(Z2_29DOF_USD_PATH))" in src
    assert "Z2UrdfFileCfg(asset_path" not in src
    yaml = (UP / "legged_lab/assets/z2_description/usd/assembly_29dof/config.yaml").read_text(encoding="utf-8")
    ensure = (UP / "legged_lab/assets/z2.py").read_text(encoding="utf-8")
    assert "make_instanceable: bool = False" in src
    assert "make_instanceable: false" in yaml
    assert "make_instanceable=False" in ensure
    probe = (ROOT / "legged_lab/scripts/probe_z2_isaac.py").read_text(encoding="utf-8")
    generate = (ROOT / "legged_lab/scripts/generate_z2_amp_expert.py").read_text(encoding="utf-8")
    assert 'asset_mode"] = "upstream_usd"' in probe or 'result["asset_mode"] = "upstream_usd"' in probe
    assert "make_instanceable must be False" not in probe
    assert "TraverseInstanceProxies" in probe
    assert "/World/envs/env_0/Robot" in probe
    assert "get_dof_max_velocities" in probe
    assert "get_dof_velocity_limits" not in probe
    assert "urdf_converter_applied" in generate
    layers = usd_layer_shas(Z2_ASSET)
    assert set(layers) == set(Z2_USD_LAYER_FILES)
    for rel in Z2_USD_LAYER_FILES:
        ours = Z2_ASSET / rel
        upstream = UP / Z2_UPSTREAM_USD_DIR / rel.removeprefix("usd/")
        assert ours.read_bytes() == upstream.read_bytes(), rel


def test_source_amp_manifest_hashes_and_training_clips_contract(tmp_path) -> None:
    from legged_lab.assets.z2.amp_manifest import (
        amp_training_manifest,
        code_provenance,
        sha256_file,
        training_clip_record,
        validate_z2_source_manifest,
    )

    source_dir = ROOT / "legged_lab/envs/z2/datasets/motion_source"
    manifest = json.loads((source_dir / "_manifest.json").read_text(encoding="utf-8"))
    checked = validate_z2_source_manifest(manifest, source_dir)
    assert set(checked) == {"walk", "walk_l", "run"}
    walk_csv = source_dir / "walk.csv"
    assert checked["walk"]["csv_sha256"] == sha256_file(walk_csv)
    txt = tmp_path / "walk.txt"
    txt.write_text('{"Frames": [[0.0, 1.0]]}\n', encoding="utf-8")
    clip = training_clip_record(
        txt_path=txt,
        source_csv=walk_csv,
        source_csv_sha256=checked["walk"]["csv_sha256"],
        source_raw_sha256=checked["walk"]["source_sha256"],
        frames=1,
        fps=30.0,
        motion_class="walk_forward",
        motion_weight=0.5,
        root=ROOT,
    )
    payload = amp_training_manifest(
        clips={"walk": clip},
        motions={"walk": {"frames": 1}},
        kinematics_response=0.1,
        sim_dt=0.005,
        max_frames=0,
        converter={"make_instanceable": False, "collider_type": "convex_hull"},
        plant="Z2_29DOF_WALK_POSE_DAMPED_PD_CFG",
        joint_order=list(Z2_29DOF_JOINT_NAMES),
        held_out=[],
        frame_dim=AMP_FRAME_DIM,
        provenance=code_provenance(ROOT),
    )
    assert payload["clips"]["walk"]["sha256"] == sha256_file(txt)
    assert payload["provenance"]["asset_mode"] == "upstream_usd"
    assert payload["provenance"]["urdf_sha256"] == sha256_file(ROOT / "legged_lab/assets/z2/urdf/assembly.urdf")
    assert payload["provenance"]["usd_layers"]["usd/assembly.usd"] == sha256_file(Z2_ASSET / "usd/assembly.usd")
    broken = dict(manifest)
    broken["motions"] = [{**manifest["motions"][0], "csv_sha256": "0" * 64}]
    with pytest.raises(ValueError, match="CSV hash mismatch"):
        validate_z2_source_manifest(broken, source_dir)

    from copy import deepcopy

    missing_raw = deepcopy(manifest)
    for item in missing_raw["motions"]:
        item["source"] = "nonexistent_review_raw.bin"
        item["source_sha256"] = "f" * 64
    with pytest.raises(FileNotFoundError):
        validate_z2_source_manifest(missing_raw, source_dir)
    missing_sha = deepcopy(manifest)
    for item in missing_sha["motions"]:
        item["source_sha256"] = ""
    with pytest.raises(ValueError, match="raw source SHA missing"):
        validate_z2_source_manifest(missing_sha, source_dir)
    mismatch = deepcopy(manifest)
    for item in mismatch["motions"]:
        item["source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="raw source hash mismatch"):
        validate_z2_source_manifest(mismatch, source_dir)


def test_probe_negative_amp_guard_rejects_schema_not_missing_file() -> None:
    from legged_lab.assets.z2.probe_contract import (
        G1_70D_FIXTURE,
        WIDTH64_FIXTURE,
        assert_command_target_vector,
        assert_pair_initial_match,
        assert_policy_maps_onto_sim,
        assert_signed_channel_response,
        reject_packaged_negative_amp,
        restore_level0_terrain,
    )
    from rsl_rl.utils.motion_loader import AMPLoader

    def fail(message: str) -> None:
        raise AssertionError(message)

    rejected = reject_packaged_negative_amp(loader_cls=AMPLoader, step_dt=0.02, fail=fail)
    assert rejected["g1_70d"]["rejected"] is True
    assert "ValueError" in rejected["g1_70d"]["error"]
    assert rejected["upstream_64d"]["rejected"] is True
    assert "64" in rejected["upstream_64d"]["error"]
    assert G1_70D_FIXTURE.is_file() and WIDTH64_FIXTURE.is_file()
    with pytest.raises(AssertionError, match="loaded as a Z2 expert"):
        reject_packaged_negative_amp(loader_cls=lambda **kwargs: object(), step_dt=0.02, fail=fail)
    with pytest.raises(AssertionError, match="fixture missing"):
        from legged_lab.assets.z2.probe_contract import reject_negative_amp_file

        reject_negative_amp_file(
            label="missing",
            path=ROOT / "does_not_exist_z2_negative.txt",
            expect="width",
            loader_cls=AMPLoader,
            frame_dim=AMP_FRAME_DIM,
            joint_order=Z2_29DOF_JOINT_NAMES,
            step_dt=0.02,
            fail=fail,
        )
    names = list(Z2_29DOF_JOINT_NAMES)
    ids = list(range(len(names)))
    assert_policy_maps_onto_sim(names, ids, names)
    with pytest.raises(AssertionError, match="policy-to-sim mapping"):
        assert_policy_maps_onto_sim(names[::-1], ids, names)
    null = [0.0] * 29
    plus = [0.0] * 29
    plus[3] = 0.05
    assert_signed_channel_response(null, plus, 3, expected_sign=1)
    minus = [0.0] * 29
    minus[3] = -0.05
    assert_signed_channel_response(null, minus, 3, expected_sign=-1)
    with pytest.raises(AssertionError, match="expected sign"):
        assert_signed_channel_response(null, minus, 3, expected_sign=1)
    drift = [0.01] * 29
    with pytest.raises(AssertionError, match="expected sign"):
        assert_signed_channel_response(drift, list(drift), 3, expected_sign=1)
    coupled = [0.0] * 29
    coupled[3] = 0.04
    coupled[7] = 0.20
    assert_signed_channel_response(null, coupled, 3, expected_sign=1)
    default = [0.1] * 29
    plus_target = list(default)
    plus_target[3] = 0.1 + 0.08 * 0.25
    assert_command_target_vector(
        readback_target=plus_target,
        default_joint_pos=default,
        policy_joint_ids=ids,
        policy_names=names,
        sim_names=names,
        channel=3,
        action=0.08,
        action_scale=0.25,
    )
    leaked = list(plus_target)
    leaked[5] = 0.4
    with pytest.raises(AssertionError, match="uncommanded"):
        assert_command_target_vector(
            readback_target=leaked,
            default_joint_pos=default,
            policy_joint_ids=ids,
            policy_names=names,
            sim_names=names,
            channel=3,
            action=0.08,
            action_scale=0.25,
        )
    left = {"q": [0.0] * 29, "qd": [0.0] * 29, "root_rel": [0.0, 0.0, 0.75], "root_vel": [0.0] * 6, "masses": [1.0]}
    right = dict(left)
    assert_pair_initial_match(left, right)
    right_q = dict(left)
    right_q["q"] = [0.01] + [0.0] * 28
    with pytest.raises(AssertionError, match="pair q"):
        assert_pair_initial_match(left, right_q)

    import torch

    terrain = SimpleNamespace(
        terrain_levels=torch.tensor([-1, -1, -1]),
        terrain_types=torch.zeros(3, dtype=torch.long),
        terrain_origins=torch.zeros(10, 1, 3),
        env_origins=torch.ones(3, 3),
    )
    terrain.terrain_origins[0, 0] = torch.tensor([1.0, 2.0, 0.0])
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(3, 3)), env_origins=torch.zeros(3, 3))
    restore_level0_terrain(terrain, env)
    assert torch.equal(terrain.terrain_levels, torch.zeros(3, dtype=terrain.terrain_levels.dtype))
    assert torch.allclose(terrain.env_origins[0], torch.tensor([1.0, 2.0, 0.0]))
    assert torch.allclose(env.scene.env_origins[0], torch.tensor([1.0, 2.0, 0.0]))
    assert torch.allclose(env.env_origins[0], torch.tensor([1.0, 2.0, 0.0]))
    port = (ROOT / "legged_lab/scripts/probe_locomotion_portability.py").read_text(encoding="utf-8")
    helper = (Z2_ASSET / "probe_contract.py").read_text(encoding="utf-8")
    shared_restore = "terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]"
    assert "terrain.terrain_levels.zero_()" in port and shared_restore in port
    assert "terrain.terrain_levels.zero_()" in helper and shared_restore in helper
    probe = (ROOT / "legged_lab/scripts/probe_z2_isaac.py").read_text(encoding="utf-8")
    assert "restore_level0_terrain(terrain, env)" in probe
    assert probe.find("restore_level0_terrain(terrain, env)") < probe.find("settle_steps")
    assert "restore_level0_terrain" in probe
    assert 'result["z2_self_collision"] = True' not in probe
    assert "except ValueError as exc:" in probe
    assert "except Exception" not in probe.split("g1_expert = G1_70D_FIXTURE", 1)[1]
    assert '_fail(f"{label} loaded as a Z2 expert: {path}")' in probe
    assert 'result["ready"] = False' in probe
    assert '["plant_support_proven"]' not in probe
    assert "landing_contact_series" in probe
    assert "physx_readback" in probe
    assert "physics_material = None" in probe
    assert "add_base_mass = None" in probe


def test_probe_loop_ast_accepting_loader_fails_open() -> None:
    import ast
    import hashlib

    from legged_lab.assets.z2.probe_contract import G1_70D_FIXTURE, WIDTH64_FIXTURE

    path = ROOT / "legged_lab/scripts/probe_z2_isaac.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    loop = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and [getattr(elt, "id", "") for elt in node.target.elts] == ["label", "path", "expect"]
    )
    captured = {"negatives": {}}

    def fail(message):
        captured["error"] = message
        raise AssertionError(message)

    namespace = {
        "AMPLoader": lambda **kwargs: object(),
        "env": SimpleNamespace(step_dt=0.02),
        "AMP_FRAME_DIM": 70,
        "Z2_29DOF_JOINT_NAMES": list(Z2_29DOF_JOINT_NAMES),
        "g1_expert": G1_70D_FIXTURE,
        "upstream_64": WIDTH64_FIXTURE,
        "result": captured,
        "_fail": fail,
        "Path": Path,
    }
    escaped = None
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), "exec"), namespace)
    except Exception as exc:
        escaped = f"{type(exc).__name__}:{exc}"
    false_rejection = escaped is None and all(item.get("rejected") for item in captured["negatives"].values())
    assert escaped is not None
    assert "loaded as a Z2 expert" in escaped
    assert false_rejection is False
    missing = {"negatives": {}}

    def fail_missing(message):
        missing["error"] = message
        raise AssertionError(message)

    missing_ns = dict(namespace)
    missing_ns["result"] = missing
    missing_ns["_fail"] = fail_missing
    missing_ns["g1_expert"] = ROOT / "does_not_exist_g1.txt"
    missing_ns["upstream_64"] = ROOT / "does_not_exist_64.txt"
    missing_escaped = None
    try:
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[loop], type_ignores=[])), str(path), "exec"), missing_ns
        )
    except Exception as exc:
        missing_escaped = f"{type(exc).__name__}:{exc}"
    assert missing_escaped is not None
    assert "fixture missing" in missing_escaped
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        != "7841f5613cfa0fd5adb672405f5872139c1763e430aaf048eb8064d84d4cc181"
    )


def test_standing_ankle_separation_matches_urdf_fk() -> None:
    from legged_lab.assets.z2.kinematic_amp import standing_ankle_separation_m

    measured = standing_ankle_separation_m()
    assert measured == pytest.approx(0.211715903, abs=1.0e-9)
    assert Z2_NOMINAL_FEET_Y_DISTANCE == pytest.approx(measured, abs=1.0e-9)


def test_z2_shell_launchers_are_lf() -> None:
    for rel in ("scripts/nubot_probe_z2.sh", "scripts/nubot_generate_z2_amp.sh"):
        raw = (ROOT / rel).read_bytes()
        assert b"\r\n" not in raw
        assert raw.startswith(b"#!/usr/bin/env bash\n")
