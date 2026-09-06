"""Name-aware helpers for tracking motion arrays.

Vendored verbatim from the PHP vault recipe
(``whole_body_tracking.tasks.tracking.mdp.motion_schema``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_names(values: Sequence[object], *, label: str) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{label} names contain duplicate entries: {duplicates}")
    return names


def reorder_named_axis(
    values: np.ndarray,
    *,
    source_names: Sequence[object],
    target_names: Sequence[str],
    axis: int,
    label: str,
) -> np.ndarray:
    """Reorder one array axis by explicit names and fail on schema drift."""
    array = np.asarray(values)
    source = normalize_names(source_names, label=label)
    target = normalize_names(target_names, label=f"target {label}")
    if array.shape[axis] != len(source):
        raise ValueError(f"{label} axis has size {array.shape[axis]}, but metadata has {len(source)} names")
    source_index = {name: index for index, name in enumerate(source)}
    missing = [name for name in target if name not in source_index]
    if missing:
        raise ValueError(f"motion is missing required {label} names: {missing}")
    return np.take(array, [source_index[name] for name in target], axis=axis)


__all__ = ["normalize_names", "reorder_named_axis"]
