# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free contracts for slack / illegal-footstep / paper actor dim."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_SIG_PATH = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "mdp" / "sparse_signals.py"
_spec = importlib.util.spec_from_file_location("t4_sparse_signals", _SIG_PATH)
sig = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sig)

from legged_lab.assets.t4 import schemas


def test_velocity_slack_band_and_standstill():
    assert sig.velocity_slack(0.8, 0.8) == 1.0
    assert sig.velocity_slack(0.8, 0.23) == 0.0
    assert sig.velocity_slack(0.8, 0.24) == 1.0
    assert sig.velocity_slack(0.8, 1.2) == 1.0
    assert sig.velocity_slack(0.8, 1.21) == 0.0
    assert sig.velocity_slack(0.0, 0.4) == 0.0


def test_illegal_footstep_fraction():
    assert sig.illegal_footstep_fraction(0.12, [0.12, 0.11, 0.12], in_contact=True) == 0.0
    assert sig.illegal_footstep_fraction(0.12, [-2.0, -2.0, 0.12], in_contact=True) == 2.0 / 3.0
    assert sig.illegal_footstep_fraction(0.12, [-2.0, -2.0], in_contact=False) == 0.0
    assert sig.illegal_footstep_fraction(0.12, [], in_contact=True) == 0.0
    assert sig.illegal_footstep_fraction(0.12, [float("nan"), 0.12], in_contact=True) == 0.5


def test_sparse_pit_fall_only_applies_to_sparse_terrains():
    fallen = sig.sparse_pit_fall_mask(
        root_z=np.array([-0.6, -0.6, -0.4]),
        origin_z=np.zeros(3),
        is_sparse=np.array([True, False, True]),
        drop_threshold=0.5,
    )
    np.testing.assert_array_equal(fallen, np.array([True, False, False]))


def test_sparse_curriculum_requires_timeout_success_and_demotes_early_falls():
    move_up, move_down = sig.sparse_curriculum_moves(
        move_up=np.array([True, True, False, False]),
        move_down=np.array([False, False, False, True]),
        is_sparse=np.array([True, True, True, False]),
        moving=np.array([True, True, True, True]),
        timed_out=np.array([True, False, False, False]),
        pit_fall=np.array([False, True, True, False]),
    )
    np.testing.assert_array_equal(move_up, np.array([True, False, False, False]))
    np.testing.assert_array_equal(move_down, np.array([False, True, True, True]))


def test_sparse_difficulty_bands_cover_all_ten_rows():
    assert [sig.terrain_difficulty_band(level, 9) for level in range(10)] == [
        "easy",
        "easy",
        "easy",
        "mid",
        "mid",
        "mid",
        "hard",
        "hard",
        "hard",
        "hard",
    ]


def test_compat_teacher_dim_unchanged_paper_is_plus_contact():
    assert schemas.TEACHER_ACTOR_OBS_DIM == 1155
    assert schemas.TEACHER_PAPER_CONTACT_DIM == 2
    assert schemas.TEACHER_PAPER_ACTOR_OBS_DIM == 1157
    assert "contact_truth" in schemas.TEACHER_FORBIDDEN_PRIVILEGE_FIELDS
