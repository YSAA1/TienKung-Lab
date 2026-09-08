# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free column assignment matching IsaacLab 2.1.0 curriculum terrains.

``TerrainImporter.terrain_types`` is a column id in ``0 .. num_cols-1``, not an
index into the ``sub_terrains`` name list. Column ownership follows

    sub_index = min{ j | index / num_cols + 0.001 < cumsum(proportion)_j }

from ``TerrainGenerator._generate_curriculum_terrains``.
"""

from __future__ import annotations

import math

SPARSE_FOOTHOLD_NAMES = ("stepping_stones", "raised_pillars")
HURDLE_TERRAIN_NAMES = ("hurdles",)


def normalize_proportions(proportions: dict[str, float]) -> tuple[list[str], list[float]]:
    """Return name order and probabilities that sum to 1."""
    if not proportions:
        raise ValueError("proportions must not be empty")
    names = list(proportions)
    values = [float(proportions[name]) for name in names]
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError(f"proportions must sum to a positive value, got {total}")
    return names, [value / total for value in values]


def assign_curriculum_columns(proportions: dict[str, float], num_cols: int) -> list[str]:
    """Map each terrain column to a sub-terrain name (IsaacLab 2.1.0 rule)."""
    if num_cols < 1:
        raise ValueError(f"num_cols must be positive, got {num_cols}")
    names, probs = normalize_proportions(proportions)
    cdf: list[float] = []
    running = 0.0
    for prob in probs:
        running += prob
        cdf.append(running)
    columns: list[str] = []
    for index in range(num_cols):
        threshold = index / num_cols + 0.001
        for sub_index, edge in enumerate(cdf):
            if threshold < edge:
                columns.append(names[sub_index])
                break
        else:
            raise ValueError(f"column {index} (threshold={threshold}) is not covered by {proportions!r}")
    return columns


def columns_named(column_names: list[str], *wanted: str) -> list[int]:
    """Column ids whose assigned name is in ``wanted``, preserving column order."""
    wanted_set = set(wanted)
    return [index for index, name in enumerate(column_names) if name in wanted_set]


def unique_names(column_names: list[str], preferred_order: list[str] | None = None) -> list[str]:
    """Unique names that actually occupy at least one column."""
    occupied = {name for name in column_names}
    if preferred_order is None:
        seen: list[str] = []
        for name in column_names:
            if name not in seen:
                seen.append(name)
        return seen
    return [name for name in preferred_order if name in occupied]


def name_to_columns(column_names: list[str], names: list[str]) -> dict[str, list[int]]:
    """Map each requested name to the columns it owns."""
    return {name: columns_named(column_names, name) for name in names}
