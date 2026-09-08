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

# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Pure terrain-curriculum decision logic for the AMP locomotion teacher.

This module must stay importable without Isaac Sim / Isaac Lab so the promotion
and demotion algebra can be contract-tested on any machine (see
``tests/test_t4_terrain_curriculum.py``). ``env.py`` only feeds tensors in.

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
- Periodic gait rewards scale with planar velocity-tracking accuracy and drop
  to zero for standing commands, so marching in place cannot farm the gait terms.
"""

from __future__ import annotations

# Commands with a planar norm below this are standing envs (matches the
# `standing_env_fraction` logging convention).
STANDING_COMMAND_THRESHOLD = 0.1
# LightLP §IV-C3: mean planar-tracking kernel above this counts as "tracking well".
LIGHTLP_TRACKING_WELL_THRESHOLD = 0.5


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


def lightlp_terrain_level_moves(
    path_length,
    tracking_mean,
    command_lin_vel_norm,
    tile_size: float,
    tracking_threshold: float = LIGHTLP_TRACKING_WELL_THRESHOLD,
    standing_threshold: float = STANDING_COMMAND_THRESHOLD,
    *,
    max_radial_dist=None,
):
    """LightLP §IV-C3 level moves: cumulative path length and command tracking.

    Promotion requires walking more than half a cell *and* tracking the command
    well. Poor tracking demotes. Standing commands do neither. Distance is
    episode-cumulative path length by default. The explicit traversal experiment
    supplies max_radial_dist instead, rejecting loops near the spawn platform.
    """
    promote_dist = tile_size / 2
    moving = command_lin_vel_norm > standing_threshold
    tracking_well = tracking_mean >= tracking_threshold
    distance = path_length if max_radial_dist is None else max_radial_dist
    move_up = (distance > promote_dist) & tracking_well & moving
    move_down = (~tracking_well) & moving & ~move_up
    return move_up, move_down


def gait_tracking_scale(
    command_lin_vel_xy,
    actual_lin_vel_xy,
    tracking_std: float = 0.5,
    standing_threshold: float = STANDING_COMMAND_THRESHOLD,
):
    """Scale periodic gait rewards by planar velocity-tracking accuracy.

    Standing commands get zero. Moving commands use the same exponential kernel
    as ``track_lin_vel_xy_exp`` (``exp(-||v_cmd - v_act||^2 / std^2)``), so a
    fast command that is ignored yields ~0 gait reward and a matched command
    yields 1.

    ``command_lin_vel_xy`` and ``actual_lin_vel_xy`` are ``(N, 2)`` arrays in the
    same yaw frame. Works on torch tensors and numpy arrays.
    """
    cmd_sq = (command_lin_vel_xy * command_lin_vel_xy).sum(-1)
    moving = cmd_sq**0.5 > standing_threshold
    err = command_lin_vel_xy - actual_lin_vel_xy
    err_sq = (err * err).sum(-1)
    scale = err_sq / max(1.0e-6, tracking_std * tracking_std)
    if type(scale).__module__.startswith("torch"):
        tracking = (-scale).exp()
    else:
        import numpy as np

        tracking = np.exp(-np.asarray(scale))
    return tracking * moving
