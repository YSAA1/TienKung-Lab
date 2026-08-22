# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free contracts for IsaacLab curriculum column assignment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_COL_PATH = _ROOT / "legged_lab" / "envs" / "t4" / "terrain_columns.py"
_spec = importlib.util.spec_from_file_location("t4_terrain_columns", _COL_PATH)
cols = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cols)

_LAYOUT_PATH = _ROOT / "legged_lab" / "terrains" / "stepping_stone_layout.py"
_layout_spec = importlib.util.spec_from_file_location("t4_stone_layout", _LAYOUT_PATH)
layout = importlib.util.module_from_spec(_layout_spec)
_layout_spec.loader.exec_module(layout)
T4_SPARSE_TERRAIN_PROPORTIONS = layout.T4_SPARSE_TERRAIN_PROPORTIONS

ENV = _ROOT / "legged_lab" / "envs" / "t4" / "t4_env.py"
REWARDS = _ROOT / "legged_lab" / "mdp" / "rewards.py"


def test_sparse_20_col_assignment_matches_isaaclab_formula():
    names = cols.assign_curriculum_columns(T4_SPARSE_TERRAIN_PROPORTIONS, 20)
    assert names == [
        "flat",
        "random_rough",
        "random_rough",
        "boxes",
        "wave",
        "hurdles",
        "stepping_stones",
        "stepping_stones",
        "stepping_stones",
        "stepping_stones",
        "raised_pillars",
        "raised_pillars",
        "raised_pillars",
        "raised_pillars",
        "slope_up",
        "slope_down",
        "stairs_up_30",
        "stairs_up_34",
        "stairs_down_30",
        "stairs_down_34",
    ]
    assert cols.columns_named(names, "stepping_stones") == [6, 7, 8, 9]
    assert cols.columns_named(names, "raised_pillars") == [10, 11, 12, 13]
    assert cols.columns_named(names, "hurdles") == [5]
    assert cols.columns_named(names, *cols.SPARSE_FOOTHOLD_NAMES) == [6, 7, 8, 9, 10, 11, 12, 13]
    for col in (6, 7, 8, 9, 10, 11, 12):
        assert names[col + 1] in cols.SPARSE_FOOTHOLD_NAMES
    assert names[13] == "raised_pillars"
    assert names[14] != "stepping_stones"
    assert names[14] != "raised_pillars"


def test_name_index_is_not_a_column_id():
    names = cols.assign_curriculum_columns(T4_SPARSE_TERRAIN_PROPORTIONS, 20)
    sub_names = list(T4_SPARSE_TERRAIN_PROPORTIONS)
    assert sub_names.index("stepping_stones") == 5
    assert names[5] == "hurdles"
    assert sub_names.index("raised_pillars") == 6
    assert names[6] == "stepping_stones"


def test_single_type_play_marks_every_column():
    names = ["stepping_stones"] * 5
    assert cols.columns_named(names, *cols.SPARSE_FOOTHOLD_NAMES) == [0, 1, 2, 3, 4]
    assert cols.columns_named(names, *cols.HURDLE_TERRAIN_NAMES) == []
    names = ["hurdles"] * 5
    assert cols.columns_named(names, *cols.HURDLE_TERRAIN_NAMES) == [0, 1, 2, 3, 4]


def test_env_and_rewards_do_not_index_subterrain_names_as_columns():
    env_src = ENV.read_text(encoding="utf-8")
    reward_src = REWARDS.read_text(encoding="utf-8")
    assert "assign_curriculum_columns" in env_src
    assert "terrain_column_names" in env_src
    assert 'names.index("hurdles")' not in env_src
    assert 'names.index("stepping_stones")' not in env_src
    assert 'names.index("raised_pillars")' not in env_src
    assert "hurdle_terrain_type_ids" in env_src
    assert "hurdle_terrain_type_ids" in reward_src
    assert "len(sub) == 1" in env_src
    assert "if type_ids is not None" in reward_src
    assert "assign_curriculum_columns" in env_src
    assert "level_" in env_src and "_frac" in env_src
