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
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab project,
# and is distributed under the BSD-3-Clause license.

"""Privileged height-scan occlusion for the locomotion teacher.

RPL-style lateral band dropout: with per-episode probability an environment
loses a contiguous band of lateral scan columns. Masked points are filled with
uniform noise inside the scan clip range, i.e. the region reads as uninformative
rather than flat, so the teacher cannot treat an occluded hole as walkable
ground. Pure torch, no Isaac dependency.
"""

import torch


def sample_column_band_masks(
    num_envs: int,
    shape: tuple[int, int],
    band_fraction_range: tuple[float, float],
    probability: float,
    device,
) -> torch.Tensor:
    """Sample per-env occlusion masks over a flattened ``(rows, cols)`` scan.

    Each env draws Bernoulli(``probability``); occluded envs lose one contiguous
    band of lateral columns whose width is a uniform fraction of the column count.
    Returns a bool tensor ``[num_envs, rows * cols]`` where True means occluded.
    """
    rows, cols = int(shape[0]), int(shape[1])
    masks = torch.zeros(num_envs, rows * cols, dtype=torch.bool, device=device)
    occluded = torch.rand(num_envs, device=device) < float(probability)
    if not bool(torch.any(occluded)):
        return masks
    lo, hi = float(band_fraction_range[0]), float(band_fraction_range[1])
    widths = (torch.empty(num_envs, device=device).uniform_(lo, hi) * cols).long().clamp(min=1, max=cols)
    starts = (torch.rand(num_envs, device=device) * (cols - widths + 1).float()).long().clamp(min=0)
    col_idx = torch.arange(cols, device=device).unsqueeze(0).expand(num_envs, -1)
    in_band = (col_idx >= starts.unsqueeze(1)) & (col_idx < (starts + widths).unsqueeze(1))
    band = in_band & occluded.unsqueeze(1)
    masks.view(num_envs, rows, cols)[:] = band.unsqueeze(1).expand(-1, rows, -1)
    return masks


def apply_scan_occlusion(
    scan: torch.Tensor,
    masks: torch.Tensor,
    clip_lo: float,
    clip_hi: float,
) -> torch.Tensor:
    """Fill masked scan points with uniform noise inside ``[clip_lo, clip_hi]``."""
    if not bool(torch.any(masks)):
        return scan
    fill = torch.empty_like(scan).uniform_(float(clip_lo), float(clip_hi))
    return torch.where(masks, fill, scan)
