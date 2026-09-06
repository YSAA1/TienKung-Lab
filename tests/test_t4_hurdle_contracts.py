# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Layer-0 contracts for the Stage E hurdle-ring terrain layout.

These tests run without Isaac Sim or a GPU. They pin the bar layout algebra of
``legged_lab.terrains.hurdle_layout`` - ring placement, difficulty-to-height
mapping, and the bar AABB collision truth that the strict zero-contact
evaluator will consume. The IsaacLab mesh generator consumes the same module,
so a layout regression goes red here before any training starts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Load the layout module straight from its file: importing it through the
# package would execute `legged_lab.terrains.__init__`, which needs IsaacLab.
_LAYOUT_PATH = Path(__file__).resolve().parents[1] / "legged_lab" / "terrains" / "hurdle_layout.py"
_spec = importlib.util.spec_from_file_location("t4_hurdle_layout", _LAYOUT_PATH)
layout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(layout)


def test_bar_height_interpolates_and_clamps():
    lo, hi = layout.LIGHTLP_HURDLE_BAR_HEIGHT_RANGE
    assert layout.hurdle_bar_height(0.0) == lo
    assert layout.hurdle_bar_height(1.0) == hi
    assert abs(layout.hurdle_bar_height(0.5) - (lo + hi) / 2.0) < 1e-9
    # generator difficulty is defined on [0, 1]; out-of-range inputs must clamp
    assert layout.hurdle_bar_height(-1.0) == lo
    assert layout.hurdle_bar_height(2.0) == hi
    assert lo > 0.0 and hi <= 0.35  # rule contract: bar height never exceeds 0.35m


def test_ring_half_widths_geometry():
    for spacing in (layout.LIGHTLP_HURDLE_SPACING_RANGE[0], 1.1, layout.LIGHTLP_HURDLE_SPACING_RANGE[1]):
        rings = layout.hurdle_ring_half_widths(spacing)
        # canonical tile must always offer at least two consecutive bars so a
        # straight crossing rehearses continuous hurdling, not a single step
        assert len(rings) >= 2, f"spacing={spacing} yields {len(rings)} rings"
        # first ring clears the spawn platform by a full spacing
        assert rings[0] == layout.LIGHTLP_HURDLE_PLATFORM_WIDTH / 2.0 + spacing
        # constant spacing, strictly increasing
        for a, b in zip(rings, rings[1:]):
            assert abs((b - a) - spacing) < 1e-9
        # all rings stay inside the tile border
        assert rings[-1] <= layout.LIGHTLP_HURDLE_TILE_SIZE / 2.0 - layout.LIGHTLP_HURDLE_BORDER_WIDTH


def test_ring_half_widths_rejects_bad_spacing():
    try:
        layout.hurdle_ring_half_widths(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("spacing=0 must raise")


def test_bar_aabbs_cover_rings_and_only_rings():
    spacing = 1.1
    height = layout.hurdle_bar_height(1.0)
    rings = layout.hurdle_ring_half_widths(spacing)
    aabbs = layout.hurdle_bar_aabbs(rings, height)
    assert len(aabbs) == 4 * len(rings)
    c = layout.LIGHTLP_HURDLE_TILE_SIZE / 2.0
    r0 = rings[0]
    # a point on the first bar (any side, mid-height) is a hit
    for px, py in ((c + r0, c), (c - r0, c), (c, c + r0), (c, c - r0), (c + r0, c + r0)):
        assert layout.point_hits_bar((px, py, height / 2.0), aabbs)
    # spawn platform center, the gap between rings, and above the bar are clear
    assert not layout.point_hits_bar((c, c, 0.02), aabbs)
    assert not layout.point_hits_bar((c + r0 + spacing / 2.0, c, height / 2.0), aabbs)
    assert not layout.point_hits_bar((c + r0, c, height + 0.05), aabbs)
    # margin turns a near miss into a hit (clearance accounting for foot size)
    near = (c + r0 + layout.LIGHTLP_HURDLE_BAR_THICKNESS / 2.0 + 0.01, c, height / 2.0)
    assert not layout.point_hits_bar(near, aabbs)
    assert layout.point_hits_bar(near, aabbs, margin=0.02)


def test_canonical_constants_are_selfconsistent():
    # bars must be thinner than the tightest spacing, else rings merge
    assert 0.0 < layout.LIGHTLP_HURDLE_BAR_THICKNESS < layout.LIGHTLP_HURDLE_SPACING_RANGE[0]
    # platform plus one ring at max spacing still fits inside the border
    r_first_max = layout.LIGHTLP_HURDLE_PLATFORM_WIDTH / 2.0 + layout.LIGHTLP_HURDLE_SPACING_RANGE[1]
    assert r_first_max <= layout.LIGHTLP_HURDLE_TILE_SIZE / 2.0 - layout.LIGHTLP_HURDLE_BORDER_WIDTH
