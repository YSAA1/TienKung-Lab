# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Pure-Python layout truth for T4 sparse footholds (LightLP pillars + stones).

No IsaacLab imports: contract tests run on any machine. Tile frame matches
``hurdle_layout`` — x/y in ``[0, tile_size]``, ground z=0, spawn at center.
"""

from __future__ import annotations

T4_STONE_TILE_SIZE = 8.0
T4_STONE_PLATFORM_WIDTH = 1.6
T4_STONE_BORDER_WIDTH = 0.25
T4_FOOTHOLD_GRID_COUNT = 9
# Stones keep the v2 lattice: square tops are large enough that a 32 cm first
# step is already being learned. Pillars use a tighter pitch so the first
# circular landing is a step, not a jump onto a disk.
T4_FOOTHOLD_PITCH_RANGE = (0.68, 0.78)
T4_STONE_WIDTH_RANGE = (0.48, 0.32)
T4_STONE_HEIGHT_RANGE = (0.16, 0.30)
T4_STONE_HEIGHT_JITTER_RANGE = (0.00, 0.04)
T4_HOLE_DEPTH = -2.0

T4_PILLAR_PITCH_RANGE = (0.55, 0.58)
T4_PILLAR_DIAMETER_RANGE = (0.50, 0.38)
T4_PILLAR_HEIGHT_RANGE = (0.14, 0.32)

T4_SPARSE_TERRAIN_PROPORTIONS = {
    "flat": 0.04,
    "random_rough": 0.08,
    "boxes": 0.06,
    "wave": 0.04,
    "hurdles": 0.08,
    "stepping_stones": 0.20,
    "raised_pillars": 0.20,
    "slope_up": 0.04,
    "slope_down": 0.04,
    "stairs_up_30": 0.055,
    "stairs_up_34": 0.055,
    "stairs_down_30": 0.055,
    "stairs_down_34": 0.055,
}


def _clamp01(difficulty: float) -> float:
    return min(max(float(difficulty), 0.0), 1.0)


def _lerp(lo: float, hi: float, difficulty: float) -> float:
    d = _clamp01(difficulty)
    return lo + d * (hi - lo)


def stone_width(difficulty: float, width_range: tuple[float, float] = T4_STONE_WIDTH_RANGE) -> float:
    return _lerp(width_range[0], width_range[1], difficulty)


def foothold_pitch(difficulty: float, pitch_range: tuple[float, float] = T4_FOOTHOLD_PITCH_RANGE) -> float:
    return _lerp(pitch_range[0], pitch_range[1], difficulty)


def stone_gap(
    difficulty: float,
    width_range: tuple[float, float] = T4_STONE_WIDTH_RANGE,
    pitch_range: tuple[float, float] = T4_FOOTHOLD_PITCH_RANGE,
) -> float:
    return foothold_pitch(difficulty, pitch_range) - stone_width(difficulty, width_range)


def stone_height(difficulty: float, height_range: tuple[float, float] = T4_STONE_HEIGHT_RANGE) -> float:
    return _lerp(height_range[0], height_range[1], difficulty)


def stone_height_jitter(difficulty: float, jitter_range: tuple[float, float] = T4_STONE_HEIGHT_JITTER_RANGE) -> float:
    return _lerp(jitter_range[0], jitter_range[1], difficulty)


def pillar_diameter(difficulty: float, diameter_range: tuple[float, float] = T4_PILLAR_DIAMETER_RANGE) -> float:
    return _lerp(diameter_range[0], diameter_range[1], difficulty)


def pillar_gap(
    difficulty: float,
    diameter_range: tuple[float, float] = T4_PILLAR_DIAMETER_RANGE,
    pitch_range: tuple[float, float] = T4_PILLAR_PITCH_RANGE,
) -> float:
    return foothold_pitch(difficulty, pitch_range) - pillar_diameter(difficulty, diameter_range)


def pillar_height(difficulty: float, height_range: tuple[float, float] = T4_PILLAR_HEIGHT_RANGE) -> float:
    return _lerp(height_range[0], height_range[1], difficulty)


def first_foothold_center_offset(
    pitch: float,
    platform_width: float = T4_STONE_PLATFORM_WIDTH,
) -> float:
    """Distance from tile center to the first lattice center outside the pad."""
    if pitch <= 0.0:
        raise ValueError(f"pitch must be positive, got {pitch}")
    half_platform = 0.5 * platform_width
    ring = 1
    while ring * pitch <= half_platform + 1.0e-9:
        ring += 1
    return ring * pitch


def platform_to_first_gap(
    difficulty: float,
    support: float,
    pitch_range: tuple[float, float] = T4_FOOTHOLD_PITCH_RANGE,
    platform_width: float = T4_STONE_PLATFORM_WIDTH,
) -> float:
    """Void from the spawn-pad edge to the first foothold edge, along a cardinal axis."""
    pitch = foothold_pitch(difficulty, pitch_range)
    return first_foothold_center_offset(pitch, platform_width) - 0.5 * platform_width - 0.5 * support


def foothold_centers(
    pitch: float,
    tile_size: float = T4_STONE_TILE_SIZE,
    platform_width: float = T4_STONE_PLATFORM_WIDTH,
    border_width: float = T4_STONE_BORDER_WIDTH,
    grid_count: int = T4_FOOTHOLD_GRID_COUNT,
) -> list[tuple[float, float]]:
    """Centers of a fixed square lattice that skips the spawn platform."""
    if pitch <= 0.0:
        raise ValueError(f"pitch must be positive, got {pitch}")
    if grid_count < 2:
        raise ValueError(f"grid_count must be at least 2, got {grid_count}")
    c = tile_size / 2.0
    half_span = 0.5 * (grid_count - 1) * pitch
    lo = c - half_span
    hi = c + half_span
    if lo <= border_width or hi >= tile_size - border_width:
        raise ValueError(
            f"grid_count={grid_count} pitch={pitch} does not fit tile={tile_size} with border={border_width}"
        )
    half_platform = platform_width / 2.0
    centers: list[tuple[float, float]] = []
    for i in range(grid_count):
        x = lo + i * pitch
        for j in range(grid_count):
            y = lo + j * pitch
            if abs(x - c) <= half_platform and abs(y - c) <= half_platform:
                continue
            centers.append((x, y))
    return centers


def point_on_rect(point_xy: tuple[float, float], center: tuple[float, float], width: float) -> bool:
    hx = width * 0.5
    return abs(point_xy[0] - center[0]) <= hx and abs(point_xy[1] - center[1]) <= hx


def point_on_disk(point_xy: tuple[float, float], center: tuple[float, float], diameter: float) -> bool:
    dx = point_xy[0] - center[0]
    dy = point_xy[1] - center[1]
    r = diameter * 0.5
    return dx * dx + dy * dy <= r * r
