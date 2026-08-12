# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Pure terrain-curriculum decision logic for the T4 Stage E teacher.

This module must stay importable without Isaac Sim / Isaac Lab so the promotion
and demotion algebra can be contract-tested on any machine (see
``tests/test_t4_terrain_curriculum.py``). ``t4_env.py`` only feeds tensors in.

The semantics are frozen with the Stage E MDP:

- Promotion means a completed tile traversal: the peak radial displacement from
  the terrain origin during the episode exceeded half the tile size (the full
  stair pyramid from the spawn platform to the tile edge). Judging the peak
  instead of the final position keeps "walked out over the stairs and came
  back" a success, and tolerates mid-episode pushes.
- Demotion keeps the "walked less than half the commanded distance" rule but is
  capped strictly below the promotion bar, so a fast command can never demote
  an env that nearly or actually traversed its tile.
- Standing envs (near-zero command) neither promote nor demote; standing is a
  capability bucket, not a traversal attempt.
"""

from __future__ import annotations

# Commands with a planar norm below this are standing envs (matches the
# `standing_env_fraction` logging convention).
STANDING_COMMAND_THRESHOLD = 0.1


def terrain_level_moves(
    max_radial_dist,
    command_lin_vel_norm,
    episode_length_s: float,
    tile_size: float,
    standing_threshold: float = STANDING_COMMAND_THRESHOLD,
):
    """Decide per-env terrain level moves at episode end.

    Works on any array type with elementwise ``>``, ``<``, ``&``, ``~`` and
    ``clip`` (torch tensors and numpy arrays both qualify).

    Args:
        max_radial_dist: Peak planar distance from the terrain origin reached
            during the episode, shape ``(N,)``.
        command_lin_vel_norm: Norm of the commanded planar velocity for the
            episode, shape ``(N,)``.
        episode_length_s: Episode duration in seconds.
        tile_size: Terrain tile edge length in meters.
        standing_threshold: Command norm below which an env counts as standing.

    Returns:
        Tuple ``(move_up, move_down)`` of boolean arrays, mutually exclusive.
    """
    promote_dist = tile_size / 2
    moving = command_lin_vel_norm > standing_threshold

    move_up = (max_radial_dist > promote_dist) & moving

    commanded_half_dist = command_lin_vel_norm * episode_length_s * 0.5
    demote_dist = commanded_half_dist.clip(max=promote_dist * 0.5)
    move_down = (max_radial_dist < demote_dist) & moving & ~move_up

    return move_up, move_down
