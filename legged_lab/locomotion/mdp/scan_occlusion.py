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
    """Sample per-env occlusion masks over a flattened ``(nx, ny)`` scan.

    The scanner grid flattens y-outer/x-inner (``flat = iy * nx + ix``, the
    layout asserted by ``tests/test_t4_observation_contracts.py``), so a
    lateral band is a contiguous run of ``iy`` spanning every forward column
    ``ix``. Each env draws Bernoulli(``probability``); occluded envs lose one
    such band whose width is a uniform fraction of the lateral count.
    Returns a bool tensor ``[num_envs, nx * ny]`` where True means occluded.
    """
    nx, ny = int(shape[0]), int(shape[1])
    masks = torch.zeros(num_envs, nx * ny, dtype=torch.bool, device=device)
    occluded = torch.rand(num_envs, device=device) < float(probability)
    if not bool(torch.any(occluded)):
        return masks
    lo, hi = float(band_fraction_range[0]), float(band_fraction_range[1])
    widths = (torch.empty(num_envs, device=device).uniform_(lo, hi) * ny).long().clamp(min=1, max=ny)
    starts = (torch.rand(num_envs, device=device) * (ny - widths + 1).float()).long().clamp(min=0)
    iy = torch.arange(ny, device=device).unsqueeze(0).expand(num_envs, -1)
    band = (iy >= starts.unsqueeze(1)) & (iy < (starts + widths).unsqueeze(1)) & occluded.unsqueeze(1)
    masks.view(num_envs, ny, nx)[:] = band.unsqueeze(-1)
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
