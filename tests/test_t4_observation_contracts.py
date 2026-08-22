"""Machine checks for the frozen T4 observation contracts.

These tests stay importable without IsaacLab so a schema regression is caught on
any machine, not only on the GPU runtime.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from legged_lab.assets.t4 import schemas
from legged_lab.assets.t4.constants import T4_JOINT_NAMES, T4_NOMINAL_FEET_Y_DISTANCE

ROOT = Path(__file__).resolve().parents[1]
MJCF = ROOT / "legged_lab/assets/t4/mjcf/t4_std.xml"
MOTION_SOURCE = ROOT / "legged_lab/envs/t4/datasets/motion_source"


def _field_widths(fields):
    return {name: width for name, width in fields}


def test_teacher_actor_dim_stays_1155_sparse_is_separate():
    assert schemas.TEACHER_ACTOR_OBS_DIM == 1155
    assert schemas.TEACHER_SPARSE_ACTOR_OBS_DIM == 1937
    assert schemas.TEACHER_SPARSE_ACTOR_OBS_DIM != schemas.TEACHER_ACTOR_OBS_DIM
    assert schemas.TEACHER_PAPER_ACTOR_OBS_DIM == schemas.TEACHER_ACTOR_OBS_DIM + 2
    start, end = schemas.sparse_teacher_latest_scan_range()
    assert end - start == schemas.TEACHER_SCAN_DIM
    assert end == schemas.TEACHER_SPARSE_ACTOR_OBS_DIM - schemas.TEACHER_SPARSE_CONTACT_DIM


def test_sparse_actor_mirror_swaps_trailing_contact_and_scan_history():
    import importlib.util

    import numpy as np

    path = ROOT / "legged_lab" / "envs" / "t4" / "symmetry.py"
    spec = importlib.util.spec_from_file_location("t4_symmetry_sparse", path)
    symmetry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(symmetry)
    obs = np.zeros((1, schemas.TEACHER_SPARSE_ACTOR_OBS_DIM), dtype=np.float32)
    obs[0, -2] = 1.0
    obs[0, -1] = 0.0
    mirrored = symmetry.mirror_observations(obs, is_critic=False)
    assert mirrored.shape == obs.shape
    assert mirrored[0, -2] == 0.0
    assert mirrored[0, -1] == 1.0


def test_paper_actor_mirror_swaps_trailing_contact():
    import importlib.util

    import numpy as np

    path = ROOT / "legged_lab" / "envs" / "t4" / "symmetry.py"
    spec = importlib.util.spec_from_file_location("t4_symmetry_paper", path)
    symmetry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(symmetry)
    obs = np.zeros((1, schemas.TEACHER_PAPER_ACTOR_OBS_DIM), dtype=np.float32)
    obs[0, -2] = 1.0
    obs[0, -1] = 0.0
    mirrored = symmetry.mirror_observations(obs, is_critic=False)
    assert mirrored.shape == obs.shape
    assert mirrored[0, -2] == 0.0
    assert mirrored[0, -1] == 1.0


def test_amp_frame_layout_matches_frozen_66d_schema():
    widths = _field_widths(schemas.AMP_FIELDS)
    assert widths == {
        "joint_pos": 27,
        "joint_vel": 27,
        "hand_pos_root": 6,
        "foot_pos_root": 6,
    }
    assert schemas.AMP_FRAME_DIM == 66
    assert schemas.AMP_TRANSITION_DIM == 132


def test_amp_field_slices_tile_the_frame_without_gaps():
    cursor = 0
    for name, width in schemas.AMP_FIELDS:
        start, end = schemas.amp_field_slice(name)
        assert start == cursor
        assert end - start == width
        cursor = end
    assert cursor == schemas.AMP_FRAME_DIM
    with pytest.raises(KeyError):
        schemas.amp_field_slice("root_height")


def test_every_declared_motion_has_a_weighted_class():
    for stem in schemas.AMP_MOTION_CLASSES:
        motion_class = schemas.amp_motion_class(stem)
        assert motion_class in schemas.AMP_MOTION_CLASS_WEIGHTS
        assert schemas.amp_motion_weight(stem) > 0.0
    with pytest.raises(KeyError):
        schemas.amp_motion_class("t4_unlabelled_clip")


def test_class_weights_are_independent_of_file_counts():
    """A class keeps its declared weight no matter how many clips it contains."""
    for motion_class, weight in schemas.AMP_MOTION_CLASS_WEIGHTS.items():
        members = [stem for stem, name in schemas.AMP_MOTION_CLASSES.items() if name == motion_class]
        assert members, f"class {motion_class!r} has no member motions"
        assert schemas.amp_motion_weight(members[0]) * len(members) == pytest.approx(weight)


def test_expert_file_list_excludes_held_out_motions_and_is_complete():
    files = schemas.amp_expert_files("some/dir")
    stems = [Path(path).stem for path in files]
    assert set(stems) == set(schemas.AMP_MOTION_CLASSES) - set(schemas.AMP_HELD_OUT_MOTIONS)
    assert stems == sorted(stems)
    for held_out in schemas.AMP_HELD_OUT_MOTIONS:
        assert held_out not in stems


def test_declared_motions_exist_in_the_migrated_source_set():
    available = {path.stem for path in MOTION_SOURCE.glob("*.csv")}
    declared = set(schemas.AMP_MOTION_CLASSES)
    assert declared <= available, f"declared but missing motions: {sorted(declared - available)}"
    unlabelled = available - declared - set(schemas.AMP_HELD_OUT_MOTIONS)
    assert not unlabelled, f"motions without a declared AMP class: {sorted(unlabelled)}"


def test_teacher_scan_window_is_forward_asymmetric_and_matches_its_shape():
    rows, cols = schemas.TEACHER_SCAN_SHAPE
    assert rows == round(schemas.TEACHER_SCAN_SIZE[0] / schemas.TEACHER_SCAN_RESOLUTION) + 1
    assert cols == round(schemas.TEACHER_SCAN_SIZE[1] / schemas.TEACHER_SCAN_RESOLUTION) + 1
    assert schemas.TEACHER_SCAN_DIM == rows * cols

    offset_x, offset_y = schemas.TEACHER_SCAN_OFFSET
    half_x, half_y = schemas.TEACHER_SCAN_SIZE[0] / 2, schemas.TEACHER_SCAN_SIZE[1] / 2
    assert schemas.TEACHER_SCAN_FORWARD_RANGE == pytest.approx((offset_x - half_x, offset_x + half_x))
    assert schemas.TEACHER_SCAN_LATERAL_RANGE == pytest.approx((offset_y - half_y, offset_y + half_y))
    # Forward-asymmetric: the window must look further ahead than behind.
    assert schemas.TEACHER_SCAN_FORWARD_RANGE[0] > 0.0
    assert schemas.TEACHER_SCAN_FORWARD_RANGE[1] >= 1.0


def test_teacher_scan_stays_within_the_depth_camera_reach():
    """The privilege must be recoverable from depth, not a global map."""
    assert schemas.TEACHER_SCAN_FORWARD_RANGE[1] <= schemas.DEPTH_CLIP_RANGE[1]
    assert schemas.DEPTH_NEAREST_VISIBLE_GROUND < schemas.TEACHER_SCAN_FORWARD_RANGE[1]
    assert schemas.TEACHER_SCAN_INVALID_VALUE == schemas.TEACHER_SCAN_CLIP[1]


def test_student_observation_rejects_privileged_fields():
    schemas.assert_no_privilege_leakage("teacher", ["teacher_scan", "base_ang_vel"])
    with pytest.raises(ValueError, match="leaks privileged fields"):
        schemas.assert_no_privilege_leakage("student", ["depth_history", "height_scan"])
    with pytest.raises(ValueError, match="non-transferable"):
        schemas.assert_no_privilege_leakage("teacher", ["global_map"])
    with pytest.raises(ValueError, match="unknown policy role"):
        schemas.assert_no_privilege_leakage("critic", ["base_lin_vel"])


def test_actor_observation_widths_follow_from_the_field_tables():
    assert schemas.PROPRIO_FRAME_DIM == 3 + 3 + 3 + 27 + 27 + 27 + 6
    assert schemas.CRITIC_FRAME_DIM == schemas.PROPRIO_FRAME_DIM + 5
    assert (
        schemas.TEACHER_ACTOR_OBS_DIM
        == schemas.PROPRIO_FRAME_DIM * schemas.PROPRIO_HISTORY_LENGTH + schemas.TEACHER_SCAN_DIM
    )
    depth_width = schemas.DEPTH_POLICY_SIZE[0] * schemas.DEPTH_POLICY_SIZE[1] * schemas.DEPTH_HISTORY_LENGTH
    assert schemas.STUDENT_ACTOR_OBS_DIM == schemas.PROPRIO_FRAME_DIM * schemas.PROPRIO_HISTORY_LENGTH + depth_width


def test_proprio_field_slices_tile_the_frame():
    cursor = 0
    for name, width in schemas.PROPRIO_FIELDS:
        start, end = schemas.proprio_field_slice(name)
        assert start == cursor
        assert end - start == width
        cursor = end
    assert cursor == schemas.PROPRIO_FRAME_DIM


def test_depth_preprocessing_contract_is_self_consistent():
    assert schemas.DEPTH_CLIP_RANGE[0] < schemas.DEPTH_CLIP_RANGE[1]
    assert schemas.DEPTH_NORMALIZED_RANGE == (0.0, 1.0)
    assert schemas.DEPTH_HISTORY_LENGTH >= 1
    assert schemas.DEPTH_UPDATE_DECIMATION >= 1
    assert schemas.DEPTH_POLICY_SIZE[0] < schemas.DEPTH_SENSOR_SIZE[0]
    assert schemas.DEPTH_POLICY_SIZE[1] < schemas.DEPTH_SENSOR_SIZE[1]


def test_depth_camera_is_head_height_and_pitched_down():
    right, down, look = schemas.depth_camera_ros_axes()
    pitch = math.radians(schemas.DEPTH_CAMERA_PITCH_DEG)
    assert schemas.DEPTH_CAMERA_SITE_POS[2] == pytest.approx(0.42)
    assert look[0] == pytest.approx(math.cos(pitch))
    assert look[2] == pytest.approx(-math.sin(pitch))
    assert right == pytest.approx((0.0, -1.0, 0.0))
    xyaxes = schemas.depth_camera_mujoco_xyaxes()
    assert xyaxes[:3] == pytest.approx(right)
    assert xyaxes[3:] == pytest.approx((-down[0], -down[1], -down[2]))


def test_manifest_is_json_serializable_and_pins_the_joint_order():
    manifest = schemas.observation_manifest()
    payload = json.loads(json.dumps(manifest))
    assert tuple(payload["joint_order"]) == T4_JOINT_NAMES
    assert payload["num_actions"] == 27
    assert payload["amp"]["schema_version"] == schemas.AMP_SCHEMA_VERSION
    assert payload["teacher_terrain"]["dim"] == schemas.TEACHER_SCAN_DIM
    assert payload["actor_obs_dim"]["teacher"] == schemas.TEACHER_ACTOR_OBS_DIM


def _mjcf_body_offsets() -> dict[str, list[float]]:
    root = ET.parse(MJCF).getroot()
    offsets = {}
    for body in root.findall(".//body"):
        name = body.attrib.get("name")
        pos = body.attrib.get("pos")
        if name and pos:
            offsets[name] = [float(value) for value in pos.split()]
    return offsets


def test_amp_end_effector_offsets_match_the_mjcf_sites():
    root = ET.parse(MJCF).getroot()
    sites = {
        site.attrib["name"]: [float(value) for value in site.attrib["pos"].split()]
        for site in root.findall(".//site")
        if "name" in site.attrib and "pos" in site.attrib
    }
    assert sites["left_palm"] == pytest.approx(list(schemas.AMP_HAND_SITE_OFFSET))
    assert sites["left_foot"] == pytest.approx(list(schemas.AMP_FOOT_SITE_OFFSET))


def test_nominal_feet_y_distance_matches_the_hip_chain():
    offsets = _mjcf_body_offsets()
    hip_y = offsets["Hip_Pitch_Left"][1] + offsets["Hip_Roll_Left"][1]
    assert T4_NOMINAL_FEET_Y_DISTANCE == pytest.approx(2 * hip_y, abs=1e-6)


# ---------------------------------------------------------------------------
# Sagittal mirror symmetry (legged_lab/envs/t4/symmetry.py)
# ---------------------------------------------------------------------------

import importlib.util  # noqa: E402

import numpy as np  # noqa: E402

# Load straight from the file: importing through the package would execute
# `legged_lab.envs.__init__`, which needs IsaacLab.
_SYMMETRY_PATH = ROOT / "legged_lab" / "envs" / "t4" / "symmetry.py"
_spec = importlib.util.spec_from_file_location("t4_symmetry", _SYMMETRY_PATH)
_symmetry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_symmetry)

_ACTOR_WIDTH = schemas.PROPRIO_FRAME_DIM * schemas.PROPRIO_HISTORY_LENGTH + schemas.TEACHER_SCAN_DIM
_CRITIC_WIDTH = schemas.CRITIC_FRAME_DIM * schemas.PROPRIO_HISTORY_LENGTH + schemas.TEACHER_SCAN_DIM


def test_mirror_is_an_involution_on_actor_critic_obs_and_actions():
    rng = np.random.default_rng(0)
    actor_obs = rng.normal(size=(4, _ACTOR_WIDTH))
    critic_obs = rng.normal(size=(4, _CRITIC_WIDTH))
    actions = rng.normal(size=(4, len(T4_JOINT_NAMES)))
    assert np.allclose(
        _symmetry.mirror_observations(_symmetry.mirror_observations(actor_obs, False), False), actor_obs
    )
    assert np.allclose(
        _symmetry.mirror_observations(_symmetry.mirror_observations(critic_obs, True), True), critic_obs
    )
    torch = pytest.importorskip("torch")
    critic_t = torch.as_tensor(critic_obs)
    mirrored_t = _symmetry.mirror_observations(_symmetry.mirror_observations(critic_t, True), True)
    assert torch.allclose(mirrored_t, critic_t)
    assert np.allclose(_symmetry.mirror_actions(_symmetry.mirror_actions(actions)), actions)


def test_mirror_negates_lateral_command_and_keeps_forward_command():
    obs = np.zeros((1, _ACTOR_WIDTH))
    start, _ = schemas.proprio_field_slice("velocity_command")
    obs[0, start : start + 3] = [0.7, 0.4, 0.3]  # vx, vy, wz
    mirrored = _symmetry.mirror_observations(obs, False)
    assert mirrored[0, start : start + 3] == pytest.approx([0.7, -0.4, -0.3])


def test_mirror_swaps_leg_joints_with_axis_correct_signs():
    joints = list(T4_JOINT_NAMES)
    start, _ = schemas.proprio_field_slice("joint_pos")
    obs = np.zeros((1, _ACTOR_WIDTH))
    obs[0, start + joints.index("J_hip_l_pitch")] = 0.5
    obs[0, start + joints.index("J_hip_l_roll")] = 0.2
    obs[0, start + joints.index("J_arm_l_02")] = 0.3
    obs[0, start + joints.index("J_waist_yaw")] = 0.1
    mirrored = _symmetry.mirror_observations(obs, False)
    # Pitch-like joints swap sides with sign +1; roll/yaw-like joints negate.
    assert mirrored[0, start + joints.index("J_hip_r_pitch")] == pytest.approx(0.5)
    assert mirrored[0, start + joints.index("J_hip_r_roll")] == pytest.approx(-0.2)
    assert mirrored[0, start + joints.index("J_arm_r_02")] == pytest.approx(-0.3)
    assert mirrored[0, start + joints.index("J_waist_yaw")] == pytest.approx(-0.1)
    assert mirrored[0, start + joints.index("J_hip_l_pitch")] == pytest.approx(0.0)


def test_mirror_flips_the_scan_laterally_and_keeps_forward_axis():
    num_x, num_y = schemas.TEACHER_SCAN_SHAPE
    scan_start = schemas.PROPRIO_FRAME_DIM * schemas.PROPRIO_HISTORY_LENGTH
    obs = np.zeros((1, _ACTOR_WIDTH))
    iy, ix = 0, 3  # one lateral edge, fixed forward position
    obs[0, scan_start + iy * num_x + ix] = 1.0
    mirrored = _symmetry.mirror_observations(obs, False)
    assert mirrored[0, scan_start + (num_y - 1) * num_x + ix] == pytest.approx(1.0)
    assert mirrored[0, scan_start + iy * num_x + ix] == pytest.approx(0.0)


def test_get_symmetric_states_returns_original_first_then_mirrored():
    rng = np.random.default_rng(1)
    obs = rng.normal(size=(3, _ACTOR_WIDTH))
    actions = rng.normal(size=(3, len(T4_JOINT_NAMES)))
    aug_obs, aug_actions = _symmetry.get_symmetric_states(obs=obs, actions=actions, obs_type="policy")
    assert aug_obs.shape == (6, _ACTOR_WIDTH)
    assert aug_actions.shape == (6, len(T4_JOINT_NAMES))
    assert np.allclose(aug_obs[:3], obs)
    assert np.allclose(aug_obs[3:], _symmetry.mirror_observations(obs, False))
    assert np.allclose(aug_actions[3:], _symmetry.mirror_actions(actions))
