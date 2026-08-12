"""Machine checks for the frozen T4 observation contracts.

These tests stay importable without IsaacLab so a schema regression is caught on
any machine, not only on the GPU runtime.
"""

from __future__ import annotations

import json
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
