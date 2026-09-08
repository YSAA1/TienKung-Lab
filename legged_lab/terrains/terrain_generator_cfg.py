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
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

"""
Configuration classes defining the different terrains available. Each configuration class must
inherit from ``isaaclab.terrains.terrains_cfg.TerrainConfig`` and define the following attributes:

- ``name``: Name of the terrain. This is used for the prim name in the USD stage.
- ``function``: Function to generate the terrain. This function must take as input the terrain difficulty
  and the configuration parameters and return a `tuple with the `trimesh`` mesh object and terrain origin.
"""

import numpy as np
import trimesh

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass

from .hurdle_layout import (
    LIGHTLP_HURDLE_BAR_HEIGHT_RANGE,
    LIGHTLP_HURDLE_BAR_THICKNESS,
    LIGHTLP_HURDLE_BORDER_WIDTH,
    LIGHTLP_HURDLE_PLATFORM_WIDTH,
    LIGHTLP_HURDLE_SPACING_RANGE,
    hurdle_bar_height,
    hurdle_ring_half_widths,
)
from .stepping_stone_layout import (
    LIGHTLP_FOOTHOLD_PITCH_RANGE,
    LIGHTLP_HOLE_DEPTH,
    LIGHTLP_PILLAR_DIAMETER_RANGE,
    LIGHTLP_PILLAR_HEIGHT_RANGE,
    LIGHTLP_PILLAR_PITCH_RANGE,
    LIGHTLP_STONE_BORDER_WIDTH,
    LIGHTLP_SPARSE_RIM_WIDTH,
    LIGHTLP_SPARSE_TERRAIN_PROPORTIONS,
    LIGHTLP_STONE_HEIGHT_JITTER_RANGE,
    LIGHTLP_STONE_HEIGHT_RANGE,
    LIGHTLP_STONE_PLATFORM_WIDTH,
    LIGHTLP_STONE_WIDTH_RANGE,
    foothold_centers,
    foothold_pitch,
    pillar_diameter,
    pillar_height,
    rim_slab_centers_and_sizes,
    stone_height,
    stone_height_jitter,
    stone_width,
    targeted_layout_seed,
)


def hurdle_rings_terrain(difficulty, cfg):
    """Flat tile with concentric square hurdle rings around the spawn platform.

    Bars are thin boxes merged into the static tile mesh, so the existing
    contact MDP (stumble/shank/trunk penalties, fall termination) and the
    height scan see them like any other terrain. Bar height follows the
    difficulty row; ring spacing is sampled per tile. Layout math lives in
    ``hurdle_layout.py`` (pure Python) so contract tests and the future
    strict-contact evaluator share one truth.
    """
    spacing = float(cfg.spacing_range[0] + float(difficulty) * (cfg.spacing_range[1] - cfg.spacing_range[0]))
    bar_height = hurdle_bar_height(difficulty, cfg.bar_height_range)
    half_widths = hurdle_ring_half_widths(
        spacing,
        tile_size=min(cfg.size),
        platform_width=cfg.platform_width,
        border_width=cfg.border_width,
    )
    meshes = [make_plane(cfg.size, height=0.0, center_zero=False)]
    cx, cy = 0.5 * cfg.size[0], 0.5 * cfg.size[1]
    thickness = cfg.bar_thickness
    for radius in half_widths:
        span = 2.0 * radius + thickness
        for side in (-1.0, 1.0):
            meshes.append(
                trimesh.creation.box(
                    (thickness, span, bar_height),
                    trimesh.transformations.translation_matrix((cx + side * radius, cy, bar_height / 2.0)),
                )
            )
            meshes.append(
                trimesh.creation.box(
                    (span, thickness, bar_height),
                    trimesh.transformations.translation_matrix((cx, cy + side * radius, bar_height / 2.0)),
                )
            )
    return meshes, np.array([cx, cy, 0.0])


@configclass
class MeshHurdleRingsTerrainCfg(SubTerrainBaseCfg):
    """Hurdle-ring sub-terrain (100m obstacle #2 as a Stage E curriculum bucket)."""

    function = hurdle_rings_terrain

    platform_width: float = LIGHTLP_HURDLE_PLATFORM_WIDTH
    border_width: float = LIGHTLP_HURDLE_BORDER_WIDTH
    spacing_range: tuple[float, float] = LIGHTLP_HURDLE_SPACING_RANGE
    bar_height_range: tuple[float, float] = LIGHTLP_HURDLE_BAR_HEIGHT_RANGE
    bar_thickness: float = LIGHTLP_HURDLE_BAR_THICKNESS


def _sparse_rim_meshes(tile: float, rim_width: float):
    """Four overlapping z=0 slabs forming a walkable frame around the pit."""
    thickness = 0.10
    z = -0.5 * thickness
    meshes = []
    for (cx, cy), (sx, sy) in rim_slab_centers_and_sizes(tile, rim_width):
        box = trimesh.creation.box((sx, sy, thickness))
        box.apply_translation((cx, cy, z))
        meshes.append(box)
    return meshes


def _sparse_base_meshes(cfg, tile: float):
    """Spawn pad plus either a deep pit (hard) or a filled floor (soft)."""
    platform = make_plane((cfg.platform_width, cfg.platform_width), height=0.0, center_zero=False)
    platform.apply_translation((0.5 * (tile - cfg.platform_width), 0.5 * (tile - cfg.platform_width), 0.0))
    rim_width = float(getattr(cfg, "rim_width", LIGHTLP_SPARSE_RIM_WIDTH))
    rim = _sparse_rim_meshes(tile, rim_width)
    if getattr(cfg, "soft_fill", False):
        # Soft stage: walkable floor at z=0. Scan / illegal still use the true-hole map.
        floor = trimesh.creation.box((tile, tile, 0.1))
        floor.apply_translation((0.5 * tile, 0.5 * tile, -0.05))
        return [floor, platform, *rim]
    pit = trimesh.creation.box((tile, tile, 0.1))
    pit.apply_translation((0.5 * tile, 0.5 * tile, cfg.hole_depth - 0.05))
    return [pit, platform, *rim]


def stepping_stones_terrain(difficulty, cfg):
    """Discrete rectangular stones over a deep hole, plus a spawn platform."""
    width = stone_width(difficulty, cfg.stone_width_range)
    pitch = foothold_pitch(difficulty, cfg.foothold_pitch_range)
    base_height = stone_height(difficulty, cfg.height_range)
    jitter = stone_height_jitter(difficulty, cfg.height_jitter_range)
    tile = float(min(cfg.size))
    meshes = _sparse_base_meshes(cfg, tile)
    if getattr(cfg, "targeted_layout_seed", False):
        rng_seed = targeted_layout_seed(difficulty, base_seed=getattr(cfg, "seed", 0) or 0, stream=1)
    else:
        rng_seed = int(round(float(difficulty) * 1000.0))
    rng = np.random.default_rng(rng_seed)
    for x, y in foothold_centers(pitch, tile_size=tile, platform_width=cfg.platform_width, border_width=cfg.border_width):
        h = base_height + float(rng.uniform(-jitter, jitter))
        h = max(0.04, h)
        box = trimesh.creation.box((width, width, h))
        box.apply_translation((x, y, 0.5 * h))
        meshes.append(box)
    origin = np.array([0.5 * tile, 0.5 * tile, 0.0])
    return meshes, origin


def raised_pillars_terrain(difficulty, cfg):
    """Cylindrical posts over a hole; LightLP Fig. 4 column 5."""
    diameter = pillar_diameter(difficulty, cfg.diameter_range)
    pitch = foothold_pitch(difficulty, cfg.foothold_pitch_range)
    height = pillar_height(difficulty, cfg.height_range)
    targeted = bool(getattr(cfg, "targeted_manufacturing_variation", False))
    rng = None
    if targeted:
        rng = np.random.default_rng(
            targeted_layout_seed(difficulty, base_seed=getattr(cfg, "seed", 0) or 0, stream=2)
        )
        diameter *= float(rng.uniform(1.0 - cfg.diameter_scale_jitter, 1.0 + cfg.diameter_scale_jitter))
        pitch *= float(rng.uniform(1.0 - cfg.pitch_scale_jitter, 1.0 + cfg.pitch_scale_jitter))
    tile = float(min(cfg.size))
    meshes = _sparse_base_meshes(cfg, tile)
    for x, y in foothold_centers(pitch, tile_size=tile, platform_width=cfg.platform_width, border_width=cfg.border_width):
        post_height = height
        post_x, post_y = x, y
        tilt_x = tilt_y = 0.0
        if targeted:
            post_x += float(rng.uniform(-cfg.xy_jitter_m, cfg.xy_jitter_m))
            post_y += float(rng.uniform(-cfg.xy_jitter_m, cfg.xy_jitter_m))
            post_height = max(0.04, height + float(rng.uniform(-cfg.height_jitter_m, cfg.height_jitter_m)))
            tilt_x = float(rng.uniform(-cfg.top_tilt_rad, cfg.top_tilt_rad))
            tilt_y = float(rng.uniform(-cfg.top_tilt_rad, cfg.top_tilt_rad))
        cyl = trimesh.creation.cylinder(radius=0.5 * diameter, height=post_height)
        if targeted:
            cyl.apply_transform(trimesh.transformations.euler_matrix(tilt_x, tilt_y, 0.0, axes="sxyz"))
        cyl.apply_translation((post_x, post_y, 0.5 * post_height))
        meshes.append(cyl)
    origin = np.array([0.5 * tile, 0.5 * tile, 0.0])
    return meshes, origin


@configclass
class MeshSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    function = stepping_stones_terrain
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH
    border_width: float = LIGHTLP_STONE_BORDER_WIDTH
    rim_width: float = LIGHTLP_SPARSE_RIM_WIDTH
    foothold_pitch_range: tuple[float, float] = LIGHTLP_FOOTHOLD_PITCH_RANGE
    stone_width_range: tuple[float, float] = LIGHTLP_STONE_WIDTH_RANGE
    height_range: tuple[float, float] = LIGHTLP_STONE_HEIGHT_RANGE
    height_jitter_range: tuple[float, float] = LIGHTLP_STONE_HEIGHT_JITTER_RANGE
    hole_depth: float = LIGHTLP_HOLE_DEPTH
    soft_fill: bool = False
    targeted_layout_seed: bool = False


@configclass
class MeshRaisedPillarsTerrainCfg(SubTerrainBaseCfg):
    function = raised_pillars_terrain
    platform_width: float = LIGHTLP_STONE_PLATFORM_WIDTH
    border_width: float = LIGHTLP_STONE_BORDER_WIDTH
    rim_width: float = LIGHTLP_SPARSE_RIM_WIDTH
    foothold_pitch_range: tuple[float, float] = LIGHTLP_PILLAR_PITCH_RANGE
    diameter_range: tuple[float, float] = LIGHTLP_PILLAR_DIAMETER_RANGE
    height_range: tuple[float, float] = LIGHTLP_PILLAR_HEIGHT_RANGE
    hole_depth: float = LIGHTLP_HOLE_DEPTH
    soft_fill: bool = False
    targeted_manufacturing_variation: bool = False
    xy_jitter_m: float = 0.0
    height_jitter_m: float = 0.0
    top_tilt_rad: float = 0.0
    diameter_scale_jitter: float = 0.0
    pitch_scale_jitter: float = 0.0


GRAVEL_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        )
    },
)

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs_28": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.28,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_32": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.32,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.0, 0.23),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(-0.02, 0.04), noise_step=0.02, border_width=0.25
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.15, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "high_platform": terrain_gen.MeshPitTerrainCfg(
            proportion=0.15, pit_depth_range=(0.0, 0.3), platform_width=2.0, double_pit=True
        ),
        # "star": terrain_gen.MeshStarTerrainCfg(
        #     proportion=0.15, num_bars=6, bar_width_range=(0.05, 0.05), bar_height_range=(0.0, 0.25), platform_width=1.0
        # ),
        # "gap": terrain_gen.MeshGapTerrainCfg(
        #     proportion=0.15, gap_width_range=(0.1, 0.4), platform_width=2.0
        # )
    },
)

# Stage E terrain for the T4 privileged teacher. Sub-terrain difficulty is
# interpolated across rows, so the same config covers the flat -> rough ->
# stairs curriculum; only the per-env terrain level moves.
#
# Traversal direction follows the IsaacLab origin convention: the robot spawns on
# the central platform, so an inverted pyramid puts it at the bottom of the
# stairs (ascending) and a pyramid puts it on top (descending). Both directions
# must be generated; a single variant only ever trains one of them.
AMP_LOCOMOTION_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.08),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.16, noise_range=(-0.02, 0.08), noise_step=0.02, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.0, 0.15), platform_width=2.0
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(proportion=0.06, amplitude_range=(0.0, 0.2), num_waves=5.0),
        "hurdles": MeshHurdleRingsTerrainCfg(proportion=0.10),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "stairs_up_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.0875,
            step_height_range=(0.0, 0.20),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.0875,
            step_height_range=(0.0, 0.20),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.0875,
            step_height_range=(0.0, 0.18),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.0875,
            step_height_range=(0.0, 0.18),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# New Stage E mix: LightLP stones/pillars + thicker training bars. Default
# ``AMP_LOCOMOTION_TERRAINS_CFG`` stays unchanged so old scripts keep the old mix.
LIGHTLP_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["flat"]),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["random_rough"],
            noise_range=(-0.02, 0.08),
            noise_step=0.02,
            border_width=0.25,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["boxes"],
            grid_width=0.45,
            grid_height_range=(0.0, 0.15),
            platform_width=2.0,
        ),
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["wave"], amplitude_range=(0.0, 0.2), num_waves=5.0
        ),
        "hurdles": MeshHurdleRingsTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["hurdles"], bar_thickness=0.10
        ),
        "stepping_stones": MeshSteppingStonesTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["stepping_stones"],
            soft_fill=False,
        ),
        "raised_pillars": MeshRaisedPillarsTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["raised_pillars"],
            soft_fill=False,
        ),
        "slope_up": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["slope_up"],
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["slope_down"],
            slope_range=(0.0, 0.3),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up_30": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["stairs_up_30"],
            step_height_range=(0.0, 0.20),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_up_34": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["stairs_up_34"],
            step_height_range=(0.0, 0.20),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_30": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["stairs_down_30"],
            step_height_range=(0.0, 0.18),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down_34": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=LIGHTLP_SPARSE_TERRAIN_PROPORTIONS["stairs_down_34"],
            step_height_range=(0.0, 0.18),
            step_width=0.34,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)

# Compatibility for historical external scripts and checkpoint recipes.
T4_STAGE_E_SPARSE_TERRAINS_CFG = LIGHTLP_TERRAINS_CFG
T4_STAGE_E_TERRAINS_CFG = AMP_LOCOMOTION_TERRAINS_CFG
