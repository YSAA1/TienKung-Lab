# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Layer-0 contracts for the Stage E terrain-curriculum algebra.

These tests run without Isaac Sim or a GPU. They assert the MDP algebra of
``legged_lab.envs.t4.curriculum.terrain_level_moves`` - not whether any policy
can climb stairs - so a curriculum that is mathematically unreachable or has a
negative expected drift under mediocre tracking goes red before any training
starts. The `stage_e_prov1` lineage died exactly this way: terrain levels were
pinned at zero from iteration ~350 while reward and episode length looked
healthy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Load the curriculum module straight from its file: importing it through the
# package would execute `legged_lab.envs.__init__`, which needs IsaacLab.
_CURRICULUM_PATH = Path(__file__).resolve().parents[1] / "legged_lab" / "envs" / "t4" / "curriculum.py"
_spec = importlib.util.spec_from_file_location("t4_curriculum", _CURRICULUM_PATH)
_curriculum = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_curriculum)

STANDING_COMMAND_THRESHOLD = _curriculum.STANDING_COMMAND_THRESHOLD
terrain_level_moves = _curriculum.terrain_level_moves
lightlp_terrain_level_moves = _curriculum.lightlp_terrain_level_moves
gait_tracking_scale = _curriculum.gait_tracking_scale
LIGHTLP_TRACKING_WELL_THRESHOLD = _curriculum.LIGHTLP_TRACKING_WELL_THRESHOLD

# Stage E values (teacher_cfg.py / T4_STAGE_E_TERRAINS_CFG).
EPISODE_LENGTH_S = 20.0
TILE_SIZE = 8.0
PROMOTE_DIST = TILE_SIZE / 2

# Command distribution from T4LocoTeacherEnvCfg.commands.
VX_RANGE = (-0.6, 1.0)
VY_RANGE = (-0.5, 0.5)
REL_STANDING_ENVS = 0.2


def _sample_command_norms(rng: np.random.Generator, n: int) -> np.ndarray:
    vx = rng.uniform(*VX_RANGE, size=n)
    vy = rng.uniform(*VY_RANGE, size=n)
    norm = np.hypot(vx, vy)
    standing = rng.uniform(size=n) < REL_STANDING_ENVS
    return np.where(standing, 0.0, norm)


def test_demote_bar_is_strictly_below_promote_bar():
    """No legal command may demote an env that traversed (or nearly traversed) its tile."""
    cmd_norm = np.linspace(0.0, 2.5, 2001)  # beyond the legal range on purpose
    # An env that reached at least half the promotion distance must never demote.
    max_dist = np.full_like(cmd_norm, PROMOTE_DIST * 0.5)
    _, move_down = terrain_level_moves(max_dist, cmd_norm, EPISODE_LENGTH_S, TILE_SIZE)
    assert not move_down.any()


def test_full_traversal_always_promotes_moving_envs():
    cmd_norm = np.linspace(STANDING_COMMAND_THRESHOLD + 1e-6, 1.2, 500)
    max_dist = np.full_like(cmd_norm, PROMOTE_DIST + 1e-3)
    move_up, move_down = terrain_level_moves(max_dist, cmd_norm, EPISODE_LENGTH_S, TILE_SIZE)
    assert move_up.all()
    assert not move_down.any()


def test_standing_envs_never_promote_or_demote():
    cmd_norm = np.zeros(100)
    max_dist = np.concatenate([np.zeros(50), np.full(50, TILE_SIZE)])  # even absurd displacement
    move_up, move_down = terrain_level_moves(max_dist, cmd_norm, EPISODE_LENGTH_S, TILE_SIZE)
    assert not move_up.any()
    assert not move_down.any()


def test_moves_are_mutually_exclusive():
    rng = np.random.default_rng(0)
    cmd_norm = _sample_command_norms(rng, 10000)
    max_dist = rng.uniform(0.0, TILE_SIZE, size=10000)
    move_up, move_down = terrain_level_moves(max_dist, cmd_norm, EPISODE_LENGTH_S, TILE_SIZE)
    assert not (move_up & move_down).any()


def test_expected_level_drift_is_positive_under_mediocre_tracking():
    """Oracle Monte Carlo: a policy that tracks at half the commanded speed must
    still have a positive expected terrain-level drift, otherwise the curriculum
    ratchets everyone down to level 0 and locks there (the stage_e_prov1 bug)."""
    rng = np.random.default_rng(42)
    n = 200_000
    speed_scale = 0.5

    cmd_norm = _sample_command_norms(rng, n)
    # One command per episode, straight-line traversal at half the commanded speed.
    max_dist = cmd_norm * speed_scale * EPISODE_LENGTH_S

    move_up, move_down = terrain_level_moves(max_dist, cmd_norm, EPISODE_LENGTH_S, TILE_SIZE)
    drift = move_up.mean() - move_down.mean()
    assert drift > 0.0, f"expected positive level drift, got {drift:.4f}"


def test_early_falls_still_drain_to_easy_terrain():
    """Bootstrap behavior is intentional: an env that falls near the origin on a
    moving command must demote, so untrained policies start on easy terrain."""
    cmd_norm = np.full(100, 0.8)
    max_dist = np.full(100, 0.3)  # fell right after spawn
    move_up, move_down = terrain_level_moves(max_dist, cmd_norm, EPISODE_LENGTH_S, TILE_SIZE)
    assert not move_up.any()
    assert move_down.all()


def test_standing_command_zeros_gait_scale():
    cmd = np.zeros((3, 2))
    actual = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    scale = gait_tracking_scale(cmd, actual, tracking_std=0.5)
    assert np.all(scale == 0.0)


def test_perfect_tracking_gives_full_gait_scale():
    cmd = np.array([[0.3, 0.0], [1.0, 0.0], [0.4, 0.3]])
    scale = gait_tracking_scale(cmd, cmd, tracking_std=0.5)
    np.testing.assert_allclose(scale, np.ones(3))


def test_still_robot_on_fast_command_near_zero_gait_scale():
    cmd = np.array([[1.0, 0.0]])
    actual = np.zeros((1, 2))
    scale = gait_tracking_scale(cmd, actual, tracking_std=0.5)
    assert scale[0] < 0.05


def test_lightlp_promotes_on_path_length_and_tracking():
    path = np.array([4.01, 4.01, 3.0, 5.0])
    tracking = np.array([1.0, 0.2, 1.0, 1.0])
    cmd = np.array([0.8, 0.8, 0.8, 0.0])
    move_up, move_down = lightlp_terrain_level_moves(path, tracking, cmd, TILE_SIZE)
    np.testing.assert_array_equal(move_up, np.array([True, False, False, False]))
    np.testing.assert_array_equal(move_down, np.array([False, True, False, False]))


def test_lightlp_timeout_uses_episode_moving_not_fresh_command():
    """A horizon resample to standing must not erase a moving tracked episode."""
    path = np.array([5.0, 5.0])
    tracking = np.array([1.0, 1.0])
    # Fresh command after resample: first env now "standing", second still moving.
    # The env layer maps tracking_steps>0 to a moving command_norm before this call.
    cmd = np.array([0.5, 0.5])
    move_up, move_down = lightlp_terrain_level_moves(path, tracking, cmd, TILE_SIZE)
    assert move_up.all() and not move_down.any()


def test_lightlp_does_not_use_radial_displacement():
    # Long path that loops back (radial would be small) still promotes if tracking is good.
    path = np.array([5.0])
    tracking = np.array([LIGHTLP_TRACKING_WELL_THRESHOLD])
    cmd = np.array([0.5])
    move_up, move_down = lightlp_terrain_level_moves(path, tracking, cmd, TILE_SIZE)
    assert bool(move_up[0]) and not bool(move_down[0])
