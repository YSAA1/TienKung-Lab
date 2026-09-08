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

"""Full T4 training coverage, sampling provenance and G1 transition regression."""

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from legged_lab.assets.t4.schemas import AMP_MOTION_CLASSES
from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_full17_v1"


def test_all_actual_t4_training_frames_and_weights_are_preserved():
    manifest = json.loads((DATA / "_manifest.json").read_text())
    actual = manifest["t4_runtime_manifest"]["clips"]
    assert len(actual) == len(manifest["clips"]) == 17
    assert set(actual) == set(AMP_MOTION_CLASSES) == set(manifest["sources"])
    assert {clip["source"] for clip in manifest["clips"].values()} == set(actual)
    xml = ET.parse(ROOT / "legged_lab/assets/unitree_g1/urdf/g1_29dof_rev_1_0.urdf").getroot()
    limits = [xml.find(f"joint[@name='{name}']/limit") for name in G1_29DOF_JOINT_NAMES]
    lower, upper, velocity = (
        np.array([float(limit.get(key)) for limit in limits]) for key in ("lower", "upper", "velocity")
    )
    class_weights = dict.fromkeys(manifest["class_probabilities"], 0.0)
    for name, clip in manifest["clips"].items():
        path = DATA / f"{name}.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == clip["sha256"]
        expert = json.loads(path.read_text())
        stem = clip["source"]
        source = manifest["sources"][stem]
        csv = ROOT / source["path"]
        assert hashlib.sha256(csv.read_bytes()).hexdigest() == source["sha256"]
        original = np.loadtxt(ROOT / source["original_path"], delimiter=",")
        rows = np.loadtxt(csv, delimiter=",")
        frames = np.asarray(expert["Frames"])
        assert len(rows) == len(original) == actual[stem]["frames"] + 1
        assert expert["SourceFrameStart"] == 0
        assert expert["SourceFrameStop"] == len(rows) - 1
        assert frames.shape == (len(rows) - 1, 70) and np.isfinite(frames).all()
        assert expert["JointOrder"] == list(G1_29DOF_JOINT_NAMES)
        assert expert["MotionWeight"] == actual[stem]["weight"] == clip["motion_weight"]
        assert expert["MotionClass"] == actual[stem]["metadata"]["MotionClass"]
        class_weights[expert["MotionClass"]] += expert["MotionWeight"]
        np.testing.assert_allclose(frames[:, :29], rows[:-1, 7:], atol=2e-7)
        np.testing.assert_allclose(frames[:, 29:58], np.diff(rows[:, 7:], axis=0) * 30, atol=2e-6)
        expected_root = original[:, :3] * manifest["leg_and_hip_scale"][0]
        expected_root[:, 2] += clip["root_vertical_shift_m"]
        np.testing.assert_allclose(rows[:, :3], expected_root, atol=1e-9)
        np.testing.assert_allclose(rows[:, 3:7], original[:, 3:7], atol=1e-9)
        assert np.all(frames[:, :29] >= lower - 1e-6) and np.all(frames[:, :29] <= upper + 1e-6)
        assert np.max(np.abs(frames[:, 29:58]) / velocity) < 1
        assert clip["max_sole_position_error_m"] < 0.025
    total = sum(class_weights.values())
    for cls, weight in class_weights.items():
        assert abs(weight / total - manifest["class_probabilities"][cls]) < 1e-12


def test_backward_jog_does_not_switch_to_hyperextended_knee_branch():
    from legged_lab.scripts.retarget_t4_g1_amp import RobotRetargeter

    source = np.loadtxt(ROOT / "legged_lab/envs/t4/datasets/motion_source/t4_jog_backward.csv", delimiter=",")
    retargeter = RobotRetargeter()
    rows, errors = retargeter.convert(source)
    assert np.all(rows[:, [7 + 3, 7 + 9]] >= 0)
    assert errors[:, :2].max() < 0.005
    assert errors[:, 4:].max() < 0.1
    assert np.max(np.abs(np.diff(rows[:, 7:], axis=0) * 30) / retargeter.velocity_limits) < 1
