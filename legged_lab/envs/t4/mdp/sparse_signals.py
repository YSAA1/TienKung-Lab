# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free algebra for LightLP sparse-foothold and hurdle-contact rewards."""

from __future__ import annotations


def velocity_slack(vx_cmd: float, vx_act: float, lo: float = 0.3, hi: float = 1.5) -> float:
    """1 if commanded forward speed is nonzero and actual/cmd is in ``[lo, hi]``."""
    if abs(vx_cmd) < 1.0e-4:
        return 0.0
    ratio = vx_act / vx_cmd
    return 1.0 if lo <= ratio <= hi else 0.0


def illegal_footstep_fraction(foot_z: float, hit_zs: list[float], *, in_contact: bool, delta: float = 0.1) -> float:
    """Fraction of downward hits more than ``delta`` below the foot. Zero if swing."""
    if not in_contact or not hit_zs:
        return 0.0
    bad = 0
    for z in hit_zs:
        if z != z or (foot_z - z) > delta:  # NaN miss counts as a hole
            bad += 1
    return bad / float(len(hit_zs))


def sparse_pit_fall_mask(root_z, origin_z, is_sparse, drop_threshold: float = 0.5):
    """Terminate only sparse-terrain envs whose root fell below the tile origin."""
    return is_sparse & (root_z < origin_z - drop_threshold)


def sparse_curriculum_moves(move_up, move_down, is_sparse, moving, timed_out, pit_fall):
    """Require strict timeout traversal for sparse promotion and demote early falls."""
    strict_sparse_success = is_sparse & move_up & timed_out & ~pit_fall
    sparse_failure = is_sparse & moving & (~timed_out | pit_fall)
    adjusted_up = (move_up & ~is_sparse) | strict_sparse_success
    adjusted_down = move_down | (sparse_failure & ~strict_sparse_success)
    adjusted_down = adjusted_down & ~adjusted_up
    return adjusted_up, adjusted_down


def terrain_difficulty_band(level: int, max_level: int) -> str:
    """Map curriculum rows into stable easy/mid/hard monitoring buckets."""
    if max_level < 1:
        raise ValueError(f"max_level must be positive, got {max_level}")
    ratio = float(level) / float(max_level)
    if ratio < 1.0 / 3.0:
        return "easy"
    if ratio < 2.0 / 3.0:
        return "mid"
    return "hard"
