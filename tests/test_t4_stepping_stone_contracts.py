# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Layer-0 contracts for T4 stepping-stone / pillar layout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "legged_lab" / "terrains" / "stepping_stone_layout.py"
_spec = importlib.util.spec_from_file_location("t4_stepping_stone_layout", _LAYOUT_PATH)
layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layout)


def test_stone_difficulty_narrows_and_widens_gaps():
    assert layout.stone_width(0.0) == layout.T4_STONE_WIDTH_RANGE[0]
    assert layout.stone_width(1.0) == layout.T4_STONE_WIDTH_RANGE[1]
    assert layout.stone_width(1.0) >= 0.30
    assert layout.stone_width(0.0) > layout.stone_width(1.0)
    assert layout.stone_gap(0.0) < layout.stone_gap(1.0)
    assert layout.foothold_pitch(0.5) >= 0.65
    assert layout.stone_gap(0.5) >= 0.22


def test_pillar_first_step_is_a_step_not_a_jump():
    pillar_easy = layout.platform_to_first_gap(
        0.0, layout.pillar_diameter(0.0), pitch_range=layout.T4_PILLAR_PITCH_RANGE
    )
    pillar_hard = layout.platform_to_first_gap(
        1.0, layout.pillar_diameter(1.0), pitch_range=layout.T4_PILLAR_PITCH_RANGE
    )
    assert 0.03 <= pillar_easy <= 0.08
    assert pillar_easy < pillar_hard <= 0.20
    assert layout.first_foothold_center_offset(layout.foothold_pitch(0.0, layout.T4_PILLAR_PITCH_RANGE)) == pytest.approx(
        1.10
    )


def test_stone_and_pillar_height_curriculum_starts_reachable():
    assert 0.15 <= layout.stone_height(0.0) <= 0.20
    assert layout.stone_height(1.0) == 0.30
    assert 0.12 <= layout.pillar_height(0.0) <= 0.16
    assert layout.pillar_height(1.0) == 0.32
    assert layout.stone_height(0.0) < layout.stone_height(1.0)
    assert layout.pillar_height(0.0) < layout.pillar_height(1.0)


def test_pillar_difficulty_and_spacing():
    assert layout.pillar_diameter(1.0) >= 0.36
    assert layout.pillar_gap(0.0) <= 0.08
    assert layout.pillar_gap(0.0) < layout.pillar_gap(1.0)


def test_foothold_centers_skip_platform_and_stay_in_border():
    pitch = layout.foothold_pitch(0.5)
    centers = layout.foothold_centers(pitch)
    assert centers
    c = layout.T4_STONE_TILE_SIZE / 2.0
    half = layout.T4_STONE_PLATFORM_WIDTH / 2.0
    for x, y in centers:
        assert layout.T4_STONE_BORDER_WIDTH < x < layout.T4_STONE_TILE_SIZE - layout.T4_STONE_BORDER_WIDTH
        assert not (abs(x - c) <= half and abs(y - c) <= half)


def test_foothold_count_does_not_increase_with_difficulty():
    counts = [len(layout.foothold_centers(layout.foothold_pitch(d))) for d in (0.0, 0.5, 1.0)]
    assert counts[1] <= counts[0]
    assert counts[2] <= counts[1]
    assert counts[1] <= 80


def test_sparse_mix_is_large_enough_to_drive_learning():
    proportions = layout.T4_SPARSE_TERRAIN_PROPORTIONS
    assert abs(sum(proportions.values()) - 1.0) < 1.0e-9
    assert proportions["stepping_stones"] == 0.20
    assert proportions["raised_pillars"] == 0.20
    assert proportions["stepping_stones"] + proportions["raised_pillars"] == 0.40


def test_foothold_centers_reject_nonpositive_pitch():
    try:
        layout.foothold_centers(0.0)
    except ValueError:
        return
    raise AssertionError("pitch=0 must raise")


def test_rect_and_disk_hit_tests():
    assert layout.point_on_rect((1.0, 1.0), (1.0, 1.0), 0.4)
    assert not layout.point_on_rect((1.3, 1.0), (1.0, 1.0), 0.4)
    assert layout.point_on_disk((1.0, 1.0), (1.0, 1.0), 0.4)
    assert not layout.point_on_disk((1.3, 1.0), (1.0, 1.0), 0.4)
