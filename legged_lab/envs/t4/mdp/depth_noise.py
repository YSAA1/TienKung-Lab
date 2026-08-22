# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""LightLP §VI depth noise/latency algebra (Isaac-free).

Applied to metric depth in metres before the student normalizes and resizes.
Stage E distillation stays on clean simulated depth; sparse FT turns this on.
"""

from __future__ import annotations

LIGHTLP_DEPTH_NOISE_STD0 = 0.005
LIGHTLP_DEPTH_NOISE_STD_SLOPE = 0.02
LIGHTLP_DEPTH_SCALE_JITTER = 0.05
# 50 Hz control: 2 steps ≈ 40 ms, 3 steps ≈ 60 ms; hold-2 ≈ 25 Hz.
LIGHTLP_DEPTH_DELAY_STEPS = (2, 3)
LIGHTLP_DEPTH_HOLD_STEPS = 2
LIGHTLP_DEPTH_BLOCK_SIZE = 8
LIGHTLP_DEPTH_BLOCK_REFRESH = 4
LIGHTLP_DEPTH_BLOCK_COUNT = 6


def range_dependent_std(
    depth_m,
    std0: float = LIGHTLP_DEPTH_NOISE_STD0,
    slope: float = LIGHTLP_DEPTH_NOISE_STD_SLOPE,
):
    """σ = 0.005 + 0.02 d, matching LightLP §VI."""
    if hasattr(depth_m, "clamp_min"):
        return std0 + slope * depth_m.clamp_min(0.0)
    import numpy as np

    return std0 + slope * np.maximum(np.asarray(depth_m, dtype=float), 0.0)


def apply_metric_depth_noise(
    depth_m,
    *,
    gaussian,
    scale,
    dropout_mask=None,
    dropout_value: float = 3.0,
    std0: float = LIGHTLP_DEPTH_NOISE_STD0,
    slope: float = LIGHTLP_DEPTH_NOISE_STD_SLOPE,
):
    """Corrupt a metric depth map: range-dependent Gaussian, global scale, optional holes."""
    noisy = depth_m + gaussian * range_dependent_std(depth_m, std0=std0, slope=slope)
    noisy = noisy * scale
    if dropout_mask is not None:
        if hasattr(noisy, "masked_fill"):
            noisy = noisy.masked_fill(dropout_mask, dropout_value)
        else:
            noisy = noisy.copy()
            noisy[dropout_mask] = dropout_value
    return noisy


def sample_global_scale(draw, jitter: float = LIGHTLP_DEPTH_SCALE_JITTER) -> float:
    """Map a unit draw in ``[0, 1]`` to a multiplicative scale in ``[1-jitter, 1+jitter]``."""
    return (1.0 - float(jitter)) + 2.0 * float(jitter) * float(draw)


def scaled_block_hw(height: int, width: int, native_h: int = 270, native_w: int = 480):
    """Map LightLP 8x8 holes on 270x480 onto an arbitrary depth resolution."""
    block_h = max(1, round(LIGHTLP_DEPTH_BLOCK_SIZE * height / native_h))
    block_w = max(1, round(LIGHTLP_DEPTH_BLOCK_SIZE * width / native_w))
    return block_h, block_w


def edge_biased_block_rows(row_draw, height: int, block_h: int):
    """Bias holes toward the top and bottom bands. Matches the prior ``torch.where`` draws."""
    import torch

    quarter = height // 4
    row = torch.where(
        row_draw < 0.5,
        (row_draw * 2.0 * quarter).long(),
        (height - block_h - (row_draw - 0.5) * 2.0 * quarter).long(),
    )
    return row.clamp(0, height - block_h)


def stamp_rectangular_blocks(mask, env_ids, rows, cols, block_h: int, block_w: int):
    """OR True into ``mask[env, r:r+h, c:c+w]`` without a Python env loop."""
    import torch

    if env_ids.numel() == 0:
        return mask
    rr = torch.arange(block_h, device=mask.device)
    cc = torch.arange(block_w, device=mask.device)
    r_idx = rows.to(dtype=torch.long)[:, None, None] + rr[None, :, None]
    c_idx = cols.to(dtype=torch.long)[:, None, None] + cc[None, None, :]
    e_idx = env_ids.to(dtype=torch.long)[:, None, None].expand_as(r_idx)
    mask[e_idx, r_idx, c_idx] = True
    return mask


def apply_normalized_block_dropout(policy_depth, mask, fill_value: float = 1.0):
    """Fill holes on normalized policy depth. ``1.0`` is clip-range max (3.0 m)."""
    return policy_depth.masked_fill(mask, fill_value)


def depth_refresh_plan(counter: int, hold: int, reset_any: bool, noise_on: bool):
    """Decide whether to ingest camera/delay and whether to run clamp+resize.

    Non-refresh steps reuse the last policy frame; delay/noise still ingest every
    control step when ``noise_on``. Reset rows always materialize immediately.
    """
    hold = max(1, int(hold))
    write_all = int(counter) == 0 or int(counter) % hold == 0
    postprocess = write_all or bool(reset_any)
    ingest = bool(noise_on) or postprocess
    return ingest, postprocess, write_all
