# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free contracts for LightLP sparse rewards and observation dims."""

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


def test_opposite_direction_and_foot_accel_ema():
    assert sig.opposite_direction(0.8, -0.2) == 1.0
    assert sig.opposite_direction(0.8, 0.2) == 0.0
    assert sig.opposite_direction(0.0, -1.0) == 0.0
    ema, excess = sig.foot_accel_ema_step(0.0, 40.0, tau_s=0.06, dt_s=0.02, threshold_mps2=30.0)
    assert ema > 0.0
    assert excess >= 0.0
    ema2, excess2 = sig.foot_accel_ema_step(ema, 40.0, tau_s=0.06, dt_s=0.02, threshold_mps2=30.0)
    assert ema2 >= ema
    assert excess2 >= 0.0


def test_sparse_pit_fall_only_applies_to_sparse_terrains():
    fallen = sig.sparse_pit_fall_mask(
        root_z=np.array([-0.6, -0.6, -0.4]),
        origin_z=np.zeros(3),
        is_sparse=np.array([True, False, True]),
        drop_threshold=0.5,
    )
    np.testing.assert_array_equal(fallen, np.array([True, False, False]))


def test_soft_terrain_disables_pit_fall():
    fallen = sig.sparse_pit_fall_mask(
        root_z=np.array([-0.6, -0.6]),
        origin_z=np.zeros(2),
        is_sparse=np.array([True, True]),
        drop_threshold=0.5,
        soft_terrain=True,
    )
    np.testing.assert_array_equal(fallen, np.array([False, False]))


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


def test_default_teacher_stays_1155_sparse_is_new_dim():
    assert schemas.TEACHER_ACTOR_OBS_DIM == 1155
    assert schemas.TEACHER_SPARSE_SCAN_HISTORY_LENGTH == 5
    assert schemas.TEACHER_SPARSE_ACTOR_OBS_DIM == 96 * 10 + 195 * 5 + 2
    assert schemas.TEACHER_SPARSE_ACTOR_OBS_DIM == 1937
    assert schemas.TEACHER_SPARSE_CRITIC_OBS_DIM == 101 * 10 + 195 * 5 + 30
    assert schemas.FOOT_SCAN_BOTH_DIM == 30
    assert "contact_truth" in schemas.TEACHER_FORBIDDEN_PRIVILEGE_FIELDS
