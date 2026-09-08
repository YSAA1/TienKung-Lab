# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Pure-Python layout truth for the T4 hurdle-ring sub-terrain.

The training terrain is a flat tile with concentric square "hurdle rings"
(thin bars) around the central spawn platform, mirroring the pyramid-stairs
convention so any commanded heading crosses bars. This module is the single
source of truth for the bar layout geometry; the IsaacLab mesh generator in
``terrain_generator_cfg.py`` and any future contact evaluator must both
consume it. It intentionally has no IsaacLab imports so the contract tests in
``tests/test_t4_hurdle_contracts.py`` run on any machine.

Coordinate frame: tile-local, x/y in ``[0, tile_size]``, ground at z=0, tile
center (= robot spawn origin) at ``(tile_size / 2, tile_size / 2, 0)``.
"""

from __future__ import annotations

# Canonical Stage E hurdle-bucket parameters. The rule contract (100m obstacle
# #2) uses 1.1m spacing and up to 0.35m bar height; the training tile keeps
# heights on-contract while widening spacing variety for robustness.
LIGHTLP_HURDLE_TILE_SIZE = 8.0
LIGHTLP_HURDLE_PLATFORM_WIDTH = 1.6
LIGHTLP_HURDLE_BORDER_WIDTH = 0.25
LIGHTLP_HURDLE_SPACING_RANGE = (0.9, 1.3)
LIGHTLP_HURDLE_BAR_HEIGHT_RANGE = (0.05, 0.35)
# Real hurdle top boards are ~7cm; 0.07m also keeps the bar visible to the
# 0.1m-resolution height scan often enough to be anticipated.
LIGHTLP_HURDLE_BAR_THICKNESS = 0.07


def hurdle_bar_height(difficulty: float, height_range: tuple[float, float] = LIGHTLP_HURDLE_BAR_HEIGHT_RANGE) -> float:
    """Interpolate the bar height from the terrain difficulty in [0, 1]."""
    d = min(max(float(difficulty), 0.0), 1.0)
    return height_range[0] + d * (height_range[1] - height_range[0])


def hurdle_ring_half_widths(
    spacing: float,
    tile_size: float = LIGHTLP_HURDLE_TILE_SIZE,
    platform_width: float = LIGHTLP_HURDLE_PLATFORM_WIDTH,
    border_width: float = LIGHTLP_HURDLE_BORDER_WIDTH,
) -> list[float]:
    """Half-widths (center to bar centerline) of the concentric square rings.

    Rings start one ``spacing`` outside the spawn platform half-width and
    repeat every ``spacing`` until the tile border. Returns an empty list if
    no ring fits.
    """
    if spacing <= 0.0:
        raise ValueError(f"spacing must be positive, got {spacing}")
    r_max = tile_size / 2.0 - border_width
    half_platform = platform_width / 2.0
    half_widths = []
    k = 1
    while half_platform + k * spacing <= r_max:
        half_widths.append(half_platform + k * spacing)
        k += 1
    return half_widths


def hurdle_bar_aabbs(
    ring_half_widths: list[float],
    bar_height: float,
    tile_size: float = LIGHTLP_HURDLE_TILE_SIZE,
    bar_thickness: float = LIGHTLP_HURDLE_BAR_THICKNESS,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
    """Axis-aligned bounding boxes of every bar, as ``(min_xyz, max_xyz)``.

    Each ring is four bars forming a closed square (corners overlap). This is
    the collision truth for strict "zero bar contact" evaluation.
    """
    c = tile_size / 2.0
    ht = bar_thickness / 2.0
    aabbs = []
    for r in ring_half_widths:
        lo, hi = c - r - ht, c + r + ht
        for s in (-1.0, 1.0):
            # bars normal to x at x = c + s*r, spanning the full ring in y
            aabbs.append(((c + s * r - ht, lo, 0.0), (c + s * r + ht, hi, bar_height)))
            # bars normal to y at y = c + s*r, spanning the full ring in x
            aabbs.append(((lo, c + s * r - ht, 0.0), (hi, c + s * r + ht, bar_height)))
    return aabbs


def point_hits_bar(
    point: tuple[float, float, float],
    aabbs: list[tuple[tuple[float, float, float], tuple[float, float, float]]],
    margin: float = 0.0,
) -> bool:
    """Whether a tile-local point is inside (or within ``margin`` of) any bar."""
    x, y, z = point
    for (x0, y0, z0), (x1, y1, z1) in aabbs:
        if x0 - margin <= x <= x1 + margin and y0 - margin <= y <= y1 + margin and z0 - margin <= z <= z1 + margin:
            return True
    return False

# Compatibility for historical external scripts and checkpoint recipes.
T4_HURDLE_TILE_SIZE = LIGHTLP_HURDLE_TILE_SIZE
T4_HURDLE_PLATFORM_WIDTH = LIGHTLP_HURDLE_PLATFORM_WIDTH
T4_HURDLE_BORDER_WIDTH = LIGHTLP_HURDLE_BORDER_WIDTH
T4_HURDLE_SPACING_RANGE = LIGHTLP_HURDLE_SPACING_RANGE
T4_HURDLE_BAR_HEIGHT_RANGE = LIGHTLP_HURDLE_BAR_HEIGHT_RANGE
T4_HURDLE_BAR_THICKNESS = LIGHTLP_HURDLE_BAR_THICKNESS
