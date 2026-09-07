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

"""Verify selected expert transitions, rejection evidence and the sampling mixture."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "legged_lab/envs/g1/datasets/motion_amp_expert_t4_rob2rob_v1"


def test_curated_frames_preserve_real_source_transitions():
    manifest = json.loads((DATA / "_manifest.json").read_text())
    for name, clip in manifest["clips"].items():
        path = DATA / f"{name}.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == clip["sha256"]
        raw = json.loads(path.read_text())
        source = ROOT / raw["SourceMotion"]
        lineage = manifest["sources"][clip["source"]]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == lineage["sha256"]
        original = ROOT / lineage["original_path"]
        original_bytes = original.read_bytes()
        # Historical T4 CSVs follow repository autocrlf; the recorded source
        # bytes were read on Windows. Accept equivalent checkout line endings.
        source_hashes = {
            hashlib.sha256(original_bytes).hexdigest(),
            hashlib.sha256(original_bytes.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")).hexdigest(),
        }
        assert lineage["original_sha256"] in source_hashes
        t4_rows = np.loadtxt(original, delimiter=",")
        rows = np.loadtxt(source, delimiter=",")
        expected_root = t4_rows[:, :3] * manifest["leg_and_hip_scale"][0]
        expected_root[:, 2] += clip["root_vertical_shift_m"]
        np.testing.assert_allclose(rows[:, :3], expected_root, atol=1e-9)
        np.testing.assert_allclose(rows[:, 3:7], t4_rows[:, 3:7], atol=1e-9)
        np.testing.assert_allclose(rows[:, 7 + 12], t4_rows[:, 7 + 14], atol=1e-9)
        np.testing.assert_array_equal(rows[:, 7 + 13 : 7 + 15], 0)
        start, stop = raw["SourceFrameStart"], raw["SourceFrameStop"]
        frames = np.asarray(raw["Frames"])
        assert frames.shape == (stop - start, 70)
        assert np.isfinite(frames).all()
        np.testing.assert_allclose(frames[:, :29], rows[start:stop, 7:], atol=2e-7)
        np.testing.assert_allclose(frames[:, 29:58], np.diff(rows[start : stop + 1, 7:], axis=0) * 30, atol=1e-6)
        speed = np.linalg.norm(np.diff(rows[start : stop + 1, :2], axis=0) * 30, axis=1)
        assert (speed < 0.1).mean() <= 0.01


def test_curated_mixture_does_not_reweight_walk_versus_run():
    manifest = json.loads((DATA / "_manifest.json").read_text())
    for cls, probability in {"walk_forward": 0.625, "run": 0.375}.items():
        clips = [c for c in manifest["clips"].values() if c["motion_class"] == cls]
        assert abs(sum(c["motion_weight"] for c in clips) - probability) < 1e-12
        for clip in clips:
            assert clip["full_stride_cycles"] >= 3
            assert clip["duration_s"] >= 2.5
            assert clip["hard_joint_limit_excess_rad"] == 0
            assert clip["max_joint_velocity_limit_ratio"] < 1
            assert clip["support_point_speed_proxy_p50_p95_m_s"][1] <= 0.5
            assert clip["max_sole_position_error_m"] < 0.002
            assert clip["max_wrist_position_error_m"] < 0.05
            assert clip["max_sole_orientation_error_rad"] < 0.1
