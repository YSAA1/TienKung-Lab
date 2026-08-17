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


def opposite_direction(vx_cmd: float, vx_act: float, cmd_threshold: float = 0.1) -> float:
    """1 when commanded forward progress and body-forward velocity have opposite sign."""
    if abs(vx_cmd) < cmd_threshold:
        return 0.0
    return 1.0 if vx_cmd * vx_act < 0.0 else 0.0


def foot_accel_ema_step(
    prev_ema: float,
    foot_accel_mps2: float,
    *,
    tau_s: float = 0.06,
    dt_s: float = 0.02,
    threshold_mps2: float = 30.0,
) -> tuple[float, float]:
    """LightLP filtered foot acceleration excess.

    Returns ``(new_ema, excess)`` where ``excess = max(0, ema - threshold)`` and
    ``ema`` tracks ``|a|`` with time constant ``tau_s``.
    """
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be positive, got {tau_s}")
    alpha = 1.0 - pow(2.718281828459045, -dt_s / tau_s)
    magnitude = abs(float(foot_accel_mps2))
    ema = (1.0 - alpha) * float(prev_ema) + alpha * magnitude
    excess = max(0.0, ema - threshold_mps2)
    return ema, excess


def sparse_pit_fall_mask(root_z, origin_z, is_sparse, drop_threshold: float = 0.5, soft_terrain: bool = False):
    """Terminate only sparse-terrain envs whose root fell below the tile origin.

    Soft stage fills collision, so pit-fall is disabled even on sparse tiles.
    """
    if soft_terrain:
        return is_sparse & (root_z != root_z)  # all-False, preserves numpy/torch type
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
