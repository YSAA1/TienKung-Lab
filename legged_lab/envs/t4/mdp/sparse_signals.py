# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free algebra for LightLP sparse-foothold and hurdle-contact rewards."""

from __future__ import annotations

import math
import random

# LightLP §IV-C2 numeric constants. ``t4_env`` must import these, not copy them.
LIGHTLP_TILT_LIMIT_RAD = 63.0 * math.pi / 180.0
LIGHTLP_FALL_PROB = 0.01
LIGHTLP_JOINT_VEL_LIMIT = 50.0
LIGHTLP_ACCEL_LIMIT = 40.0
LIGHTLP_ACCEL_WARMUP_S = 1.0
LIGHTLP_IMMUNITY_FRAC = 0.10
LIGHTLP_IMMUNITY_PERIOD = 200
LIGHTLP_OOB_MARGIN_M = 0.25


def _is_batch(x) -> bool:
    return hasattr(x, "shape") and getattr(x, "ndim", 0) > 0


def promote_radius_m(tile_size: float) -> float:
    """L2 promotion bar used by ``terrain_level_moves`` (half the tile)."""
    return float(tile_size) * 0.5


def oob_linf_limit_m(tile_size: float, margin: float = LIGHTLP_OOB_MARGIN_M) -> float:
    """L-inf timeout past the L2 promote bar so an axis-aligned crossing can promote."""
    return promote_radius_m(tile_size) + float(margin)


def out_of_bounds_linf(offset_xy, tile_size: float, margin: float = LIGHTLP_OOB_MARGIN_M):
    """True when Chebyshev distance from the tile origin exceeds ``oob_linf_limit_m``."""
    limit = oob_linf_limit_m(tile_size, margin)
    if hasattr(offset_xy, "abs"):
        return offset_xy.abs().amax(dim=-1) > limit
    import numpy as np

    return np.max(np.abs(np.asarray(offset_xy)), axis=-1) > limit


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


def opposite_direction(
    cmd_xy,
    vel_xy,
    cmd_threshold: float = 0.1,
) -> float:
    """LightLP Table I: max(0, -v · v̂) for a non-trivial planar command."""
    if _is_batch(cmd_xy):
        raise TypeError("opposite_direction helper is scalar; use the env reward for batches")
    cmd_x, cmd_y = float(cmd_xy[0]), float(cmd_xy[1])
    vel_x, vel_y = float(vel_xy[0]), float(vel_xy[1])
    cmd_norm = (cmd_x * cmd_x + cmd_y * cmd_y) ** 0.5
    if cmd_norm < cmd_threshold:
        return 0.0
    return max(0.0, -(vel_x * cmd_x + vel_y * cmd_y) / cmd_norm)


def illegal_from_foot_fractions(left_frac: float, right_frac: float) -> float:
    """Eq. (4): sum contacted-foot illegal fractions, not the mean."""
    return float(left_frac) + float(right_frac)


def foot_accel_ema_step(
    prev_e: float,
    foot_accels_mps2,
    *,
    tau_s: float = 0.06,
    dt_s: float = 0.02,
    threshold_mps2: float = 30.0,
) -> float:
    """LightLP Eq. (5): ẽ = α ẽ + Σ max(|a_i|-ā, 0), α = exp(-Δt/τ)."""
    if tau_s <= 0.0:
        raise ValueError(f"tau_s must be positive, got {tau_s}")
    if dt_s <= 0.0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")
    alpha = math.exp(-float(dt_s) / float(tau_s))
    if isinstance(foot_accels_mps2, (int, float)):
        magnitudes = [abs(float(foot_accels_mps2))]
    else:
        magnitudes = [abs(float(value)) for value in foot_accels_mps2]
    excess_sum = sum(max(mag - threshold_mps2, 0.0) for mag in magnitudes)
    return alpha * float(prev_e) + excess_sum


def random_level_reset_mask(draws, fraction: float = 0.10):
    """True for the fraction of resets placed at a random level (paper line 150)."""
    if isinstance(draws, (list, tuple)):
        return [float(draw) < fraction for draw in draws]
    return draws < fraction


def tilt_from_upright_rad(gx, gy, gz):
    """Angle from upright using body-frame projected gravity (down is −z).

    Accepts scalars or batched arrays/tensors. Inverted gravity ``(0, 0, +1)``
    is π, not 0.
    """
    if _is_batch(gx):
        g2 = gx * gx + gy * gy + gz * gz
        if hasattr(g2, "clamp"):
            norm = g2.clamp(min=1.0e-16).sqrt()
            cos_tilt = (-gz / norm).clamp(-1.0, 1.0)
            return cos_tilt.acos()
        import numpy as np

        norm = np.sqrt(np.maximum(g2, 1.0e-16))
        cos_tilt = np.clip(-np.asarray(gz) / norm, -1.0, 1.0)
        sin_tilt = np.clip(np.sqrt(np.maximum(np.asarray(gx) ** 2 + np.asarray(gy) ** 2, 0.0)) / norm, 0.0, 1.0)
        return np.arctan2(sin_tilt, cos_tilt)
    norm = (gx * gx + gy * gy + gz * gz) ** 0.5
    if norm <= 1.0e-8:
        return 0.0
    cos_tilt = max(-1.0, min(1.0, -gz / norm))
    sin_tilt = max(0.0, min(1.0, (gx * gx + gy * gy) ** 0.5 / norm))
    return math.atan2(sin_tilt, cos_tilt)


def stochastic_fall_over(tilt_rad, sample, *, limit_rad: float = LIGHTLP_TILT_LIMIT_RAD, prob: float = LIGHTLP_FALL_PROB):
    """LightLP fall-over: tilt past 63° and a Bernoulli draw (expected ~100-step window)."""
    over = tilt_rad > limit_rad
    drawn = sample < prob
    if _is_batch(over) or _is_batch(drawn):
        return over & drawn
    return bool(over and drawn)


def impact_immunity_from_draws(draws, fraction: float = LIGHTLP_IMMUNITY_FRAC):
    """Boolean immunity mask from uniform draws in ``[0, 1)``."""
    if isinstance(draws, (list, tuple)):
        return [float(d) < fraction for d in draws]
    return draws < fraction


def resample_impact_immunity(n: int, fraction: float = LIGHTLP_IMMUNITY_FRAC, draws=None) -> list[bool]:
    """Mark ``fraction`` of envs immune to torso-contact and excessive-accel resets."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if draws is None:
        draws = [random.random() for _ in range(n)]
    if len(draws) != n:
        raise ValueError("draws length must match n")
    flags = impact_immunity_from_draws(draws, fraction=fraction)
    return [bool(flag) for flag in flags]


def joint_velocity_timeout(max_abs_qd, limit: float = LIGHTLP_JOINT_VEL_LIMIT):
    """LightLP joint-speed guard: treat as time-out, not a behavioral failure."""
    over = abs(max_abs_qd) > limit
    if _is_batch(over):
        return over
    return bool(over)


def excessive_base_accel(
    accel_mps2, elapsed_s, *, limit: float = LIGHTLP_ACCEL_LIMIT, warmup_s: float = LIGHTLP_ACCEL_WARMUP_S
):
    """LightLP excessive-acceleration reset after a 1 s warmup."""
    warmed = elapsed_s >= warmup_s
    over = abs(accel_mps2) > limit
    if _is_batch(over) or _is_batch(warmed):
        return warmed & over
    return bool(warmed and over)


def lightlp_timeout_and_reset(
    *,
    episode_timeout,
    offset_xy,
    tile_size: float,
    max_abs_joint_vel,
    torso_hit,
    accel_mps2,
    elapsed_s,
    gravity_gx,
    gravity_gy,
    gravity_gz,
    fall_draws,
    immunity,
):
    """Shipped LightLP §IV-C2 timeout/reset flags. ``t4_env`` must call this."""
    oob = out_of_bounds_linf(offset_xy, tile_size)
    joint_to = joint_velocity_timeout(max_abs_joint_vel)
    time_out = episode_timeout | oob | joint_to
    hard_impact = excessive_base_accel(accel_mps2, elapsed_s)
    impact_reset = (torso_hit | hard_impact) & (~immunity)
    tilt = tilt_from_upright_rad(gravity_gx, gravity_gy, gravity_gz)
    fall_over = stochastic_fall_over(tilt, fall_draws)
    reset = time_out | impact_reset | fall_over
    reasons = {
        "horizon": episode_timeout,
        "oob": oob,
        "joint_vel": joint_to,
        "torso": torso_hit & (~immunity),
        "accel": hard_impact & (~immunity),
        "fall_over": fall_over,
        "immune_skip": (torso_hit | hard_impact) & immunity,
    }
    return reset, time_out, reasons


def wrap_heading_error(yaw: float, heading_target: float) -> float:
    """Absolute wrapped heading error |Δψ|."""
    return abs(math.atan2(math.sin(heading_target - yaw), math.cos(heading_target - yaw)))


def upright_orientation_reward(gx: float, gy: float) -> float:
    """LightLP Table I / Eq. (2): exp(-2||g_xy||^2) + 0.1 exp(-||g_xy||)."""
    n2 = gx * gx + gy * gy
    n1 = n2**0.5
    return math.exp(-2.0 * n2) + 0.1 * math.exp(-n1)


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
