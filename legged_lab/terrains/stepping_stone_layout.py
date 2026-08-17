# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Pure-Python layout truth for T4 sparse footholds (LightLP pillars + stones).

No IsaacLab imports: contract tests run on any machine. Tile frame matches
``hurdle_layout`` — x/y in ``[0, tile_size]``, ground z=0, spawn at center.

v4 stones are intentionally narrow (slightly larger than the foot) so a 6 cm
lateral bias drives the foot scanner off the top and ``illegal_footstep`` fires.
"""

from __future__ import annotations

import math

T4_STONE_TILE_SIZE = 8.0
T4_STONE_PLATFORM_WIDTH = 1.6
T4_STONE_BORDER_WIDTH = 0.25
T4_FOOTHOLD_GRID_COUNT = 9
# Narrow tops: easy ~26 cm, hard ~22 cm. Pitch stays tight so the first gap is a
# step (3–10 cm easy, ≤18 cm hard), not a jump, and 9×9 still fits the 8 m tile.
T4_FOOTHOLD_PITCH_RANGE = (0.50, 0.53)
T4_STONE_WIDTH_RANGE = (0.26, 0.22)
T4_STONE_HEIGHT_RANGE = (0.16, 0.30)
T4_STONE_HEIGHT_JITTER_RANGE = (0.00, 0.04)
T4_HOLE_DEPTH = -2.0

# Pillars keep the v3 curriculum: easy first step ~5 cm onto a 50 cm disk.
T4_PILLAR_PITCH_RANGE = (0.55, 0.58)
T4_PILLAR_DIAMETER_RANGE = (0.50, 0.38)
T4_PILLAR_HEIGHT_RANGE = (0.14, 0.32)

# Foot sole scan used by illegal-footstep contracts (matches FootScannerCfg).
T4_FOOT_SCAN_SIZE = (0.16, 0.08)
T4_FOOT_SCAN_RESOLUTION = 0.04

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


def point_on_platform(
    point_xy: tuple[float, float],
    tile_size: float = T4_STONE_TILE_SIZE,
    platform_width: float = T4_STONE_PLATFORM_WIDTH,
) -> bool:
    c = tile_size / 2.0
    return point_on_rect(point_xy, (c, c), platform_width)


def point_on_support(
    point_xy: tuple[float, float],
    *,
    kind: str,
    difficulty: float,
    tile_size: float = T4_STONE_TILE_SIZE,
    platform_width: float = T4_STONE_PLATFORM_WIDTH,
) -> bool:
    """True when ``point_xy`` is on the spawn pad or a foothold (true-hole map)."""
    if point_on_platform(point_xy, tile_size=tile_size, platform_width=platform_width):
        return True
    if kind == "stepping_stones":
        pitch = foothold_pitch(difficulty, T4_FOOTHOLD_PITCH_RANGE)
        width = stone_width(difficulty, T4_STONE_WIDTH_RANGE)
        hit = point_on_rect
        size = width
    elif kind == "raised_pillars":
        pitch = foothold_pitch(difficulty, T4_PILLAR_PITCH_RANGE)
        size = pillar_diameter(difficulty, T4_PILLAR_DIAMETER_RANGE)
        hit = point_on_disk
    else:
        raise ValueError(f"unknown sparse kind {kind!r}")
    for center in foothold_centers(pitch, tile_size=tile_size, platform_width=platform_width):
        if hit(point_xy, center, size):
            return True
    return False


def foot_scan_local_offsets(
    size: tuple[float, float] = T4_FOOT_SCAN_SIZE,
    resolution: float = T4_FOOT_SCAN_RESOLUTION,
) -> list[tuple[float, float]]:
    """Yaw-frame (x, y) offsets for the downward foot grid, matching Isaac GridPattern."""
    num_x = int(round(size[0] / resolution)) + 1
    num_y = int(round(size[1] / resolution)) + 1
    xs = [i * resolution - 0.5 * size[0] for i in range(num_x)]
    ys = [j * resolution - 0.5 * size[1] for j in range(num_y)]
    return [(x, y) for y in ys for x in xs]


def foot_scan_world_points(
    foot_xy: tuple[float, float],
    yaw: float = 0.0,
    size: tuple[float, float] = T4_FOOT_SCAN_SIZE,
    resolution: float = T4_FOOT_SCAN_RESOLUTION,
) -> list[tuple[float, float]]:
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    points = []
    for lx, ly in foot_scan_local_offsets(size, resolution):
        wx = foot_xy[0] + cos_y * lx - sin_y * ly
        wy = foot_xy[1] + sin_y * lx + cos_y * ly
        points.append((wx, wy))
    return points


def illegal_footstep_fraction_layout(
    foot_xy: tuple[float, float],
    *,
    kind: str,
    difficulty: float,
    in_contact: bool = True,
    yaw: float = 0.0,
) -> float:
    """Algebraic illegal-footstep fraction from the true-hole map (soft-stage fallback)."""
    if not in_contact:
        return 0.0
    points = foot_scan_world_points(foot_xy, yaw=yaw)
    if not points:
        return 0.0
    bad = sum(1 for p in points if not point_on_support(p, kind=kind, difficulty=difficulty))
    return bad / float(len(points))


def soft_physics_supports(point_xy: tuple[float, float]) -> bool:
    """Soft stage: collision is filled, so every in-tile XY is walkable."""
    _ = point_xy
    return True


def soft_reports_hole(
    point_xy: tuple[float, float],
    *,
    kind: str,
    difficulty: float,
) -> bool:
    """Soft stage: eyes still see the true-hole map."""
    return not point_on_support(point_xy, kind=kind, difficulty=difficulty)
