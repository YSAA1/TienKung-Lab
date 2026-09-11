# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""G1 depth-student B2 contracts: observation dims, camera geometry, boundaries.

The env/cfg modules import IsaacLab, so these tests exercise the pure
``legged_lab.envs.g1.depth_student_contract`` module plus source-level
assertions on the Isaac-facing files (same pattern as the G1 asset contract).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

import importlib.util

_contract_spec = importlib.util.spec_from_file_location(
    "g1_depth_student_contract", ROOT / "legged_lab/envs/g1/depth_student_contract.py"
)
contract = importlib.util.module_from_spec(_contract_spec)
_contract_spec.loader.exec_module(contract)
G1_DEPTH_CAMERA_PITCH_DEG = contract.G1_DEPTH_CAMERA_PITCH_DEG
G1_DEPTH_CAMERA_SITE_POS = contract.G1_DEPTH_CAMERA_SITE_POS
G1_PROPRIO_FRAME_DIM = contract.G1_PROPRIO_FRAME_DIM
G1_SPARSE_TEACHER_ACTOR_OBS_DIM = contract.G1_SPARSE_TEACHER_ACTOR_OBS_DIM
G1_STUDENT_RENDER_SIZE = contract.G1_STUDENT_RENDER_SIZE
g1_depth_camera_ros_quat_wxyz = contract.g1_depth_camera_ros_quat_wxyz
g1_sparse_teacher_latest_scan_range = contract.g1_sparse_teacher_latest_scan_range
g1_sparse_teacher_scan_range = contract.g1_sparse_teacher_scan_range
from legged_lab.locomotion.schemas import (
    DEPTH_POLICY_SIZE,
    STUDENT_DEPTH_HISTORY_LENGTH,
    STUDENT_PROPRIO_HISTORY_LENGTH,
    TEACHER_SCAN_DIM,
    proprio_fields,
)


def test_g1_observation_arithmetic_matches_shared_schema():
    frame = sum(width for _, width in proprio_fields(29))
    assert frame == 3 + 3 + 3 + 29 * 3 + 2 + 2 + 2 == 102
    assert G1_PROPRIO_FRAME_DIM == frame
    # sparse teacher actor: proprio hist 10 + scan hist 5 + feet contact 2
    assert G1_SPARSE_TEACHER_ACTOR_OBS_DIM == 1020 + TEACHER_SCAN_DIM * 5 + 2 == 1997
    # T4 S12 (27DoF) stays 1937; the delta is exactly the two extra joints
    from legged_lab.assets.t4.schemas import TEACHER_SPARSE_ACTOR_OBS_DIM

    assert G1_SPARSE_TEACHER_ACTOR_OBS_DIM - TEACHER_SPARSE_ACTOR_OBS_DIM == 2 * 3 * 10


def test_g1_latest_scan_range_is_tail_of_teacher_obs():
    start, end = g1_sparse_teacher_latest_scan_range()
    block_start, block_end = g1_sparse_teacher_scan_range()
    assert (start, end) == (block_end - TEACHER_SCAN_DIM, block_end)
    assert end == G1_SPARSE_TEACHER_ACTOR_OBS_DIM - 2  # feet contact is appended last


def test_g1_camera_quat_is_unit_and_looks_forward_down():
    quat = g1_depth_camera_ros_quat_wxyz()
    norm = math.sqrt(sum(component * component for component in quat))
    assert norm == pytest.approx(1.0, abs=1e-9)
    # R(quat) maps camera axes onto (right, down, look) in the torso frame:
    # the third column must equal the look axis (cos(pitch), 0, -sin(pitch)).
    pitch = math.radians(G1_DEPTH_CAMERA_PITCH_DEG)
    look = (math.cos(pitch), 0.0, -math.sin(pitch))
    w, x, y, z = quat
    third_column = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    assert third_column[0] == pytest.approx(look[0], abs=1e-9)
    assert third_column[1] == pytest.approx(look[1], abs=1e-9)
    assert third_column[2] == pytest.approx(look[2], abs=1e-9)
    right_column = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    assert right_column[1] == pytest.approx(-1.0, abs=1e-9)  # right = (0, -1, 0)
    assert G1_DEPTH_CAMERA_SITE_POS[1] == 0.0  # centred laterally
    assert G1_STUDENT_RENDER_SIZE == DEPTH_POLICY_SIZE == (48, 64)


def test_g1_student_env_uses_shared_runtime_and_robot_spec_boundary():
    env_source = (ROOT / "legged_lab/envs/g1/depth_student_env.py").read_text(encoding="utf-8")
    cfg_source = (ROOT / "legged_lab/envs/g1/depth_student_cfg.py").read_text(encoding="utf-8")
    script_source = (ROOT / "legged_lab/scripts/train_g1_sparse_depth_student.py").read_text(encoding="utf-8")
    # runtime is the shared robot-neutral distillation env, not a T4 subclass
    for source in (env_source, cfg_source):
        assert "from legged_lab.envs.t4" not in source
        assert "T4Loco" not in source
    assert "G1LocoTeacherEnvCfg" in env_source
    assert "LightLPDepthDistillationEnv" in script_source
    assert "torso_link" in env_source
    # camera geometry comes from the pure contract, native policy resolution
    assert "G1_STUDENT_RENDER_SIZE" in env_source
    assert "270" not in env_source.replace("48x64", "")
    # the entry keeps the teacher gate: manifest or explicit waiver
    assert "--teacher_eval_manifest" in script_source
    assert "--allow_ungated_teacher" in script_source
    assert "G1_SPARSE_TEACHER_ACTOR_OBS_DIM" in script_source
    # repr-first recipe mirrors the T4 numbers via the shared algorithm
    assert "repr_first: bool = True" in cfg_source
    assert "SafeRecurrentDistillation" in cfg_source
    assert "STUDENT_DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE" in cfg_source.replace(" ", "") or (
        "(STUDENT_DEPTH_HISTORY_LENGTH, *DEPTH_POLICY_SIZE)" in cfg_source
    )
    assert STUDENT_DEPTH_HISTORY_LENGTH == 1 and STUDENT_PROPRIO_HISTORY_LENGTH == 1
