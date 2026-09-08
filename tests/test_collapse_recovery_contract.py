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

"""Regression cases for transient collapse and impact-immunity recovery."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "legged_lab/locomotion/mdp/sparse_signals.py"
SPEC = importlib.util.spec_from_file_location("collapse_signals", PATH)
signals = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(signals)


def test_transient_pose_recovers_and_requires_a_new_full_window():
    steps = np.zeros(2, dtype=np.int64)
    immune = np.zeros(2, dtype=bool)
    for _ in range(9):
        steps, terminated, _ = signals.persistent_collapse_mask(np.ones(2, dtype=bool), steps, 10, immune)
        assert not terminated.any()
    steps, terminated, _ = signals.persistent_collapse_mask(np.array([False, True]), steps, 10, immune)
    np.testing.assert_array_equal(steps, [0, 10])
    np.testing.assert_array_equal(terminated, [False, True])
    steps, terminated, _ = signals.persistent_collapse_mask(np.array([True, False]), steps, 10, immune)
    np.testing.assert_array_equal(steps, [1, 0])
    assert not terminated.any()


def test_immunity_preserves_evidence_and_expiry_ends_sustained_collapse():
    low = np.ones(2, dtype=bool)
    steps, terminated, skipped = signals.persistent_collapse_mask(low, np.array([9, 9]), 10, low)
    assert not terminated.any()
    assert skipped.all()
    steps, terminated, skipped = signals.persistent_collapse_mask(low, steps, 10, np.array([True, False]))
    np.testing.assert_array_equal(terminated, [False, True])
    np.testing.assert_array_equal(skipped, [True, False])


def test_default_one_step_without_immunity_matches_original_height_gate():
    low = signals.collapsed_pelvis_above_feet_mask(np.array([0.19, 0.2, 0.5]), np.zeros(3), 0.2)
    _, terminated, _ = signals.persistent_collapse_mask(low, np.zeros(3, dtype=np.int64), 1, np.zeros(3, dtype=bool))
    np.testing.assert_array_equal(terminated, low)


def test_invalid_window_is_rejected():
    with pytest.raises(ValueError, match="required_steps"):
        signals.persistent_collapse_mask(np.array([True]), np.array([0]), 0, np.array([False]))
