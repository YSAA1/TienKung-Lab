# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Pure-Python layout truth for LightLP sparse footholds (pillars + stones).

No IsaacLab imports: contract tests run on any machine. Tile frame matches
``hurdle_layout`` — x/y in ``[0, tile_size]``, ground z=0, spawn at center.

s6 easy tops are wider than the foot so the first landing is a step, not a
24 cm void. Hard still narrows; ``illegal_footstep`` is expected to stay quiet
on easy and bite on a 6 cm bias only once the top is hard-narrow.

S12 adds a walkable rim so an axis-aligned crossing lands on solid ground
before Chebyshev OOB, instead of stepping into the pit past the last foothold.
"""

from __future__ import annotations

import math
import hashlib

LIGHTLP_STONE_TILE_SIZE = 8.0
LIGHTLP_STONE_PLATFORM_WIDTH = 1.6
LIGHTLP_STONE_BORDER_WIDTH = 0.25
# Walkable rim so the last foothold meets a landing instead of a pit. 0.75 m
# covers the hard-stone last edge (~3.37 m from origin) and the 0.25 m OOB
# overflow onto the neighboring tile.
LIGHTLP_SPARSE_RIM_WIDTH = 0.75
# Easy: 40 cm top, 10 cm inter-stone void, 9 cm rise. The 1.6 m pad meets the
# first stone edge (first_gap ≈ 0). Hard still narrows. The lattice count is
# derived from pitch so changing local step geometry cannot silently shorten the
# traversable course inside the fixed 8 m tile.
LIGHTLP_FOOTHOLD_PITCH_RANGE = (0.50, 0.54)
LIGHTLP_STONE_WIDTH_RANGE = (0.40, 0.26)
LIGHTLP_STONE_HEIGHT_RANGE = (0.09, 0.24)
LIGHTLP_STONE_HEIGHT_JITTER_RANGE = (0.00, 0.04)
LIGHTLP_HOLE_DEPTH = -2.0

# Pillars keep v3 plan-view (50→38 cm disks, ~5 cm easy void). Only the rise
# drops so the first foot does not have to clear a 14 cm curb.
LIGHTLP_PILLAR_PITCH_RANGE = (0.55, 0.58)
LIGHTLP_PILLAR_DIAMETER_RANGE = (0.50, 0.38)
LIGHTLP_PILLAR_HEIGHT_RANGE = (0.08, 0.28)

# Foot sole scan used by illegal-footstep contracts (matches FootScannerCfg).
LIGHTLP_FOOT_SCAN_SIZE = (0.16, 0.08)
LIGHTLP_FOOT_SCAN_RESOLUTION = 0.04
PINNED_SPARSE_SPAWN_MAX_ABS_Y_M = 0.10
PINNED_SPARSE_SPAWN_MAX_ABS_YAW_DEG = 10.0
PINNED_SPARSE_SPAWN_TERRAINS = frozenset({"stepping_stones", "raised_pillars"})

LIGHTLP_SPARSE_TERRAIN_PROPORTIONS = {
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


def stone_width(difficulty: float, width_range: tuple[float, float] = LIGHTLP_STONE_WIDTH_RANGE) -> float:
    return _lerp(width_range[0], width_range[1], difficulty)


def foothold_pitch(difficulty: float, pitch_range: tuple[float, float] = LIGHTLP_FOOTHOLD_PITCH_RANGE) -> float:
    return _lerp(pitch_range[0], pitch_range[1], difficulty)


def stone_gap(
    difficulty: float,
    width_range: tuple[float, float] = LIGHTLP_STONE_WIDTH_RANGE,
    pitch_range: tuple[float, float] = LIGHTLP_FOOTHOLD_PITCH_RANGE,
) -> float:
    return foothold_pitch(difficulty, pitch_range) - stone_width(difficulty, width_range)


def stone_height(difficulty: float, height_range: tuple[float, float] = LIGHTLP_STONE_HEIGHT_RANGE) -> float:
    return _lerp(height_range[0], height_range[1], difficulty)


def stone_height_jitter(difficulty: float, jitter_range: tuple[float, float] = LIGHTLP_STONE_HEIGHT_JITTER_RANGE) -> float:
    return _lerp(jitter_range[0], jitter_range[1], difficulty)


def pillar_diameter(difficulty: float, diameter_range: tuple[float, float] = LIGHTLP_PILLAR_DIAMETER_RANGE) -> float:
    return _lerp(diameter_range[0], diameter_range[1], difficulty)


def pillar_gap(
    difficulty: float,
    diameter_range: tuple[float, float] = LIGHTLP_PILLAR_DIAMETER_RANGE,
    pitch_range: tuple[float, float] = LIGHTLP_PILLAR_PITCH_RANGE,
) -> float:
    return foothold_pitch(difficulty, pitch_range) - pillar_diameter(difficulty, diameter_range)


def pillar_height(difficulty: float, height_range: tuple[float, float] = LIGHTLP_PILLAR_HEIGHT_RANGE) -> float:
    return _lerp(height_range[0], height_range[1], difficulty)


def targeted_layout_seed(difficulty: float, *, base_seed: int = 0, stream: int = 0) -> int:
    """Stable per-tile seed without the old 0.001 difficulty bucketing."""
    payload = f"{int(base_seed)}:{float(difficulty):.12f}:{int(stream)}".encode("ascii")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def pinned_spawn_stays_on_platform(
    y_offset_m: float,
    yaw_rad: float,
    *,
    feet_y_distance: float,
    foot_size: tuple[float, float],
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH,
) -> bool:
    """True when both feet stay inside the square spawn pad after the pinned pose."""
    half_pad = 0.5 * float(platform_width)
    half_stance = 0.5 * float(feet_y_distance)
    half_x = 0.5 * float(foot_size[0])
    half_y = 0.5 * float(foot_size[1])
    cos_y = math.cos(float(yaw_rad))
    sin_y = math.sin(float(yaw_rad))
    y0 = float(y_offset_m)
    for foot_y in (half_stance, -half_stance):
        for lx, ly in ((-half_x, -half_y), (-half_x, half_y), (half_x, -half_y), (half_x, half_y)):
            x = lx * cos_y - (ly + foot_y) * sin_y
            y = y0 + lx * sin_y + (ly + foot_y) * cos_y
            if abs(x) > half_pad + 1.0e-9 or abs(y) > half_pad + 1.0e-9:
                return False
    return True


def resolve_pinned_sparse_spawn(
    y_offset_m: float | None,
    yaw_deg: float | None,
    *,
    feet_y_distance: float,
    foot_size: tuple[float, float],
    terrain_type: str | None = None,
) -> dict | None:
    """Return reset ranges for a pad-safe pinned spawn, or None to keep random reset.

    Either offset or yaw being set pins both (unset axis is 0). The robot stays on
    the 1.6 m sparse pad; this is a first-step diagnostic, not a hole spawn.
    """
    if y_offset_m is None and yaw_deg is None:
        return None
    if terrain_type not in PINNED_SPARSE_SPAWN_TERRAINS:
        raise ValueError(
            f"pinned spawn is only valid on {sorted(PINNED_SPARSE_SPAWN_TERRAINS)}, got {terrain_type!r}"
        )
    y = 0.0 if y_offset_m is None else float(y_offset_m)
    yaw_d = 0.0 if yaw_deg is None else float(yaw_deg)
    if abs(y) > PINNED_SPARSE_SPAWN_MAX_ABS_Y_M + 1.0e-9:
        raise ValueError(
            f"pinned y offset {y} m exceeds {PINNED_SPARSE_SPAWN_MAX_ABS_Y_M} m first-step diagnostic cap"
        )
    if abs(yaw_d) > PINNED_SPARSE_SPAWN_MAX_ABS_YAW_DEG + 1.0e-9:
        raise ValueError(
            f"pinned yaw {yaw_d} deg exceeds {PINNED_SPARSE_SPAWN_MAX_ABS_YAW_DEG} deg first-step diagnostic cap"
        )
    yaw = math.radians(yaw_d)
    if not pinned_spawn_stays_on_platform(y, yaw, feet_y_distance=feet_y_distance, foot_size=foot_size):
        raise ValueError(f"pinned spawn y={y} m yaw={yaw_d} deg would leave the 1.6 m spawn pad")
    zero6 = {axis: (0.0, 0.0) for axis in ("x", "y", "z", "roll", "pitch", "yaw")}
    return {
        "y_offset_m": y,
        "yaw_deg": yaw_d,
        "yaw_rad": yaw,
        "pose_range": {"x": (0.0, 0.0), "y": (y, y), "yaw": (yaw, yaw)},
        "velocity_range": zero6,
        "joint_position_range": (1.0, 1.0),
        "joint_velocity_range": (0.0, 0.0),
    }


def first_foothold_center_offset(
    pitch: float,
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH,
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
    pitch_range: tuple[float, float] = LIGHTLP_FOOTHOLD_PITCH_RANGE,
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH,
) -> float:
    """Void from the spawn-pad edge to the first foothold edge, along a cardinal axis."""
    pitch = foothold_pitch(difficulty, pitch_range)
    return first_foothold_center_offset(pitch, platform_width) - 0.5 * platform_width - 0.5 * support


def foothold_centers(
    pitch: float,
    tile_size: float = LIGHTLP_STONE_TILE_SIZE,
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH,
    border_width: float = LIGHTLP_STONE_BORDER_WIDTH,
) -> list[tuple[float, float]]:
    """Centers of a tile-filling square lattice that skips the spawn platform."""
    if pitch <= 0.0:
        raise ValueError(f"pitch must be positive, got {pitch}")
    c = tile_size / 2.0
    max_ring = math.floor((c - border_width) / pitch)
    if max_ring < 1:
        raise ValueError(f"pitch={pitch} does not leave room for a foothold ring in tile={tile_size}")
    grid_count = 2 * max_ring + 1
    half_span = max_ring * pitch
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
    tile_size: float = LIGHTLP_STONE_TILE_SIZE,
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH,
) -> bool:
    c = tile_size / 2.0
    return point_on_rect(point_xy, (c, c), platform_width)


def rim_inner_offset(tile_size: float = LIGHTLP_STONE_TILE_SIZE, rim_width: float = LIGHTLP_SPARSE_RIM_WIDTH) -> float:
    """Distance from tile center to the inner edge of the landing rim."""
    return 0.5 * float(tile_size) - float(rim_width)


def rim_slab_centers_and_sizes(
    tile_size: float = LIGHTLP_STONE_TILE_SIZE,
    rim_width: float = LIGHTLP_SPARSE_RIM_WIDTH,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Four axis-aligned landing slabs: (center_xy, size_xy) south/north/west/east.

    Mesh generation must consume this so collision boxes cannot drift from the
    algebraic ``point_on_rim`` frame. Corners overlap; that is intentional.
    """
    tile = float(tile_size)
    rim = float(rim_width)
    return [
        ((0.5 * tile, 0.5 * rim), (tile, rim)),
        ((0.5 * tile, tile - 0.5 * rim), (tile, rim)),
        ((0.5 * rim, 0.5 * tile), (rim, tile)),
        ((tile - 0.5 * rim, 0.5 * tile), (rim, tile)),
    ]


def point_on_rim(
    point_xy: tuple[float, float],
    tile_size: float = LIGHTLP_STONE_TILE_SIZE,
    rim_width: float = LIGHTLP_SPARSE_RIM_WIDTH,
    *,
    overflow: bool = True,
) -> bool:
    """True on this tile's frame, or up to ``rim_width`` into the neighbor.

    ``overflow=True`` covers the 0.25 m Chebyshev OOB margin, which sits on the
    neighboring sparse tile's rim rather than in a pit.
    """
    x, y = float(point_xy[0]), float(point_xy[1])
    lo = -float(rim_width) if overflow else 0.0
    hi = float(tile_size) + (float(rim_width) if overflow else 0.0)
    if x < lo or y < lo or x > hi or y > hi:
        return False
    return x <= rim_width or y <= rim_width or x >= tile_size - rim_width or y >= tile_size - rim_width


_ISAAC_TILE_PLATFORM_THICKNESS = 0.10
_ISAAC_TILE_HALF = 0.5 * LIGHTLP_STONE_TILE_SIZE
_RIM_DIRECTION_NAMES = ("south", "north", "west", "east")


def isaac_sparse_tile_geoms(
    difficulty: float,
    kind: str,
    *,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    platform_name: str = "start_platform",
    finish_name: str | None = "finish_platform",
    name_prefix: str = "",
) -> list[dict]:
    """MuJoCo geoms for one Isaac sparse tile, origin at the spawn-pad center.

    Training tiles are an 8 m lattice around a 1.6 m pad plus a 0.75 m landing
    rim. The student viewer must not replace that with a 7-lane corridor over a
    pit — that is a different task than the policy was distilled on.
    """
    import random

    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    thick = _ISAAC_TILE_PLATFORM_THICKNESS
    half_thick = 0.5 * thick
    geoms: list[dict] = [
        {
            "name": platform_name,
            "kind": "platform",
            "shape": "box",
            "pos": (ox, oy, -half_thick),
            "size": (0.5 * LIGHTLP_STONE_PLATFORM_WIDTH, 0.5 * LIGHTLP_STONE_PLATFORM_WIDTH, half_thick),
            "rgba": (0.28, 0.31, 0.34, 1.0),
        }
    ]
    for direction, ((cx, cy), (sx, sy)) in zip(
        _RIM_DIRECTION_NAMES, rim_slab_centers_and_sizes(), strict=True
    ):
        if direction == "east" and finish_name is not None:
            rim_name = finish_name
        else:
            rim_name = f"{name_prefix}rim_{direction}"
        geoms.append(
            {
                "name": rim_name,
                "kind": "rim",
                "shape": "box",
                "pos": (cx - _ISAAC_TILE_HALF + ox, cy - _ISAAC_TILE_HALF + oy, -half_thick),
                "size": (0.5 * sx, 0.5 * sy, half_thick),
                "rgba": (0.28, 0.31, 0.34, 1.0),
            }
        )
    if kind == "stepping_stones":
        pitch = foothold_pitch(difficulty)
        support = stone_width(difficulty)
        height = stone_height(difficulty)
        jitter = stone_height_jitter(difficulty)
        shape = "box"
        stem = "stone"
        rgba = (0.72, 0.63, 0.42, 1.0)
        pitch_range = LIGHTLP_FOOTHOLD_PITCH_RANGE
    elif kind == "raised_pillars":
        pitch = foothold_pitch(difficulty, LIGHTLP_PILLAR_PITCH_RANGE)
        support = pillar_diameter(difficulty)
        height = pillar_height(difficulty)
        jitter = 0.0
        shape = "cylinder"
        stem = "pillar"
        rgba = (0.35, 0.58, 0.72, 1.0)
        pitch_range = LIGHTLP_PILLAR_PITCH_RANGE
    else:
        raise ValueError(f"unknown sparse kind {kind!r}")
    del pitch_range
    rng = random.Random(int(round(float(difficulty) * 1000.0)))
    c = _ISAAC_TILE_HALF
    max_ring = math.floor((c - LIGHTLP_STONE_BORDER_WIDTH) / pitch)
    lo = c - max_ring * pitch
    for tile_x, tile_y in foothold_centers(pitch):
        ix = int(round((tile_x - lo) / pitch))
        iy = int(round((tile_y - lo) / pitch))
        h = max(0.04, height + rng.uniform(-jitter, jitter))
        world_x = tile_x - _ISAAC_TILE_HALF + ox
        world_y = tile_y - _ISAAC_TILE_HALF + oy
        if shape == "box":
            size: tuple[float, ...] = (0.5 * support, 0.5 * support, 0.5 * h)
        else:
            size = (0.5 * support, 0.5 * h)
        geoms.append(
            {
                "name": f"{stem}_{ix}_{iy}",
                "kind": kind,
                "shape": shape,
                "pos": (world_x, world_y, 0.5 * h),
                "size": size,
                "rgba": rgba,
            }
        )
    return geoms


def last_cardinal_support_edge(
    kind: str,
    difficulty: float,
    tile_size: float = LIGHTLP_STONE_TILE_SIZE,
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH,
) -> float:
    """+x outer edge of the farthest foothold, in tile-local coordinates."""
    if kind == "stepping_stones":
        pitch = foothold_pitch(difficulty, LIGHTLP_FOOTHOLD_PITCH_RANGE)
        half = 0.5 * stone_width(difficulty, LIGHTLP_STONE_WIDTH_RANGE)
    elif kind == "raised_pillars":
        pitch = foothold_pitch(difficulty, LIGHTLP_PILLAR_PITCH_RANGE)
        half = 0.5 * pillar_diameter(difficulty, LIGHTLP_PILLAR_DIAMETER_RANGE)
    else:
        raise ValueError(f"unknown sparse kind {kind!r}")
    last_center = max(x for x, _ in foothold_centers(pitch, tile_size=tile_size, platform_width=platform_width))
    return last_center + half


def point_on_support(
    point_xy: tuple[float, float],
    *,
    kind: str,
    difficulty: float,
    tile_size: float = LIGHTLP_STONE_TILE_SIZE,
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH,
) -> bool:
    """True when ``point_xy`` is on the spawn pad, landing rim, or a foothold."""
    if point_on_platform(point_xy, tile_size=tile_size, platform_width=platform_width):
        return True
    if point_on_rim(point_xy, tile_size=tile_size):
        return True
    if kind == "stepping_stones":
        pitch = foothold_pitch(difficulty, LIGHTLP_FOOTHOLD_PITCH_RANGE)
        width = stone_width(difficulty, LIGHTLP_STONE_WIDTH_RANGE)
        hit = point_on_rect
        size = width
    elif kind == "raised_pillars":
        pitch = foothold_pitch(difficulty, LIGHTLP_PILLAR_PITCH_RANGE)
        size = pillar_diameter(difficulty, LIGHTLP_PILLAR_DIAMETER_RANGE)
        hit = point_on_disk
    else:
        raise ValueError(f"unknown sparse kind {kind!r}")
    for center in foothold_centers(pitch, tile_size=tile_size, platform_width=platform_width):
        if hit(point_xy, center, size):
            return True
    return False


def foot_scan_local_offsets(
    size: tuple[float, float] = LIGHTLP_FOOT_SCAN_SIZE,
    resolution: float = LIGHTLP_FOOT_SCAN_RESOLUTION,
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
    size: tuple[float, float] = LIGHTLP_FOOT_SCAN_SIZE,
    resolution: float = LIGHTLP_FOOT_SCAN_RESOLUTION,
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

# Compatibility for historical external scripts and checkpoint recipes.
T4_STONE_TILE_SIZE = LIGHTLP_STONE_TILE_SIZE
T4_STONE_PLATFORM_WIDTH = LIGHTLP_STONE_PLATFORM_WIDTH
T4_STONE_BORDER_WIDTH = LIGHTLP_STONE_BORDER_WIDTH
T4_SPARSE_RIM_WIDTH = LIGHTLP_SPARSE_RIM_WIDTH
T4_FOOTHOLD_PITCH_RANGE = LIGHTLP_FOOTHOLD_PITCH_RANGE
T4_STONE_WIDTH_RANGE = LIGHTLP_STONE_WIDTH_RANGE
T4_STONE_HEIGHT_RANGE = LIGHTLP_STONE_HEIGHT_RANGE
T4_STONE_HEIGHT_JITTER_RANGE = LIGHTLP_STONE_HEIGHT_JITTER_RANGE
T4_HOLE_DEPTH = LIGHTLP_HOLE_DEPTH
T4_PILLAR_PITCH_RANGE = LIGHTLP_PILLAR_PITCH_RANGE
T4_PILLAR_DIAMETER_RANGE = LIGHTLP_PILLAR_DIAMETER_RANGE
T4_PILLAR_HEIGHT_RANGE = LIGHTLP_PILLAR_HEIGHT_RANGE
T4_FOOT_SCAN_SIZE = LIGHTLP_FOOT_SCAN_SIZE
T4_FOOT_SCAN_RESOLUTION = LIGHTLP_FOOT_SCAN_RESOLUTION
T4_SPARSE_TERRAIN_PROPORTIONS = LIGHTLP_SPARSE_TERRAIN_PROPORTIONS
