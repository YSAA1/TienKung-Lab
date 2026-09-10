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

"""Verify the curated unitree_v6 AMP dataset: exact source slices, gates and mixture."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v6"


def _g1_joint_names():
    spec = importlib.util.spec_from_file_location("g1_constants", ROOT / "legged_lab/assets/unitree_g1/constants.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return list(module.G1_29DOF_JOINT_NAMES)


def test_v6_frames_preserve_real_source_transitions():
    manifest = json.loads((DATA / "_manifest.json").read_text())
    assert manifest["clips"], "v6 must contain curated clips"
    joint_names = _g1_joint_names()
    for name, clip in manifest["clips"].items():
        path = DATA / f"{name}.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == clip["sha256"]
        raw = json.loads(path.read_text())
        assert list(raw["JointOrder"]) == joint_names
        source = ROOT / raw["SourceMotion"]
        lineage = manifest["sources"][clip["source"]]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == lineage["sha256"]
        rows = np.loadtxt(source, delimiter=",")
        start, stop = raw["SourceFrameStart"], raw["SourceFrameStop"]
        frames = np.asarray(raw["Frames"])
        assert frames.shape == (stop - start, 70)
        assert np.isfinite(frames).all()
        np.testing.assert_allclose(frames[:, :29], rows[start:stop, 7:], atol=2e-7)
        # generator uses forward differences within the slice
        expected_dq = np.diff(rows[start : stop + 1, 7:], axis=0) * 30.0
        np.testing.assert_allclose(frames[:, 29:58], expected_dq, atol=1e-6)
        speed = np.linalg.norm(np.diff(rows[start : stop + 1, :2], axis=0) * 30, axis=1)
        assert (speed < 0.1).mean() <= 0.01
        # the T-pose calibration lead-in must be gone: slices start deep into the source
        assert start >= 60, f"{name} starts at frame {start}; calibration lead-in not trimmed"


def test_v6_mixture_and_selection_gates():
    manifest = json.loads((DATA / "_manifest.json").read_text())
    for cls, probability in {"walk_forward": 0.625, "run": 0.375}.items():
        clips = [c for c in manifest["clips"].values() if c["motion_class"] == cls]
        assert clips, f"v6 lost every {cls} clip"
        assert abs(sum(c["motion_weight"] for c in clips) - probability) < 1e-12
        for clip in clips:
            assert clip["full_stride_cycles"] >= 3
            assert clip["duration_s"] >= 2.5
            assert clip["hard_joint_limit_excess_rad"] == 0
            assert clip["max_joint_velocity_limit_ratio"] < 1
            assert clip["support_point_speed_proxy_p50_p95_m_s"][1] <= 0.5
            assert clip["lowest_sole_m"] > -0.025
            assert clip["max_root_tilt_rad"] < 0.45
    # foot-skating sources must remain excluded (walk1_subject1 rejected wholesale)
    rejected_sources = {entry["source"] for entry in manifest["rejected_segments"].values()}
    kept_sources = {entry["source"] for entry in manifest["clips"].values()}
    assert "walk1_subject1" in rejected_sources
    assert "walk1_subject1" not in kept_sources
