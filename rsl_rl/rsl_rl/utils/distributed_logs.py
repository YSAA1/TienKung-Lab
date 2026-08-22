# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Count-weighted reduction of per-reset TensorBoard extras.

The weighted-mean algebra is Isaac-free. Torch is imported only for the
distributed all-reduce path and when callers pass tensor values.
"""

from __future__ import annotations

from typing import Any

COUNT_SUFFIX = "__n"


def _as_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().reshape(()).item())
    return float(value)


def reduce_episode_log_dicts(
    ep_infos: list[dict],
    *,
    device: Any = "cpu",
    distributed: bool = False,
    process_group=None,
) -> dict[str, float]:
    """Return count-weighted means. Keys ending in ``__n`` are weights and are stripped.

    When ``distributed`` is true every rank must call this, including ranks with
    empty ``ep_infos``, so the all-reduce set stays aligned.
    """
    local_num: dict[str, float] = {}
    local_den: dict[str, float] = {}
    for info in ep_infos:
        for key, raw in info.items():
            if key.endswith(COUNT_SUFFIX):
                continue
            value = _as_float(raw)
            count_raw = info.get(key + COUNT_SUFFIX)
            count = _as_float(count_raw) if count_raw is not None else 1.0
            local_num[key] = local_num.get(key, 0.0) + value * count
            local_den[key] = local_den.get(key, 0.0) + count

    if distributed:
        import torch
        import torch.distributed as dist

        keys = sorted(local_num)
        world = dist.get_world_size(process_group)
        gathered: list[list[str] | None] = [None] * world
        dist.all_gather_object(gathered, keys, group=process_group)
        all_keys = sorted({key for item in gathered for key in (item or [])})
        pack = torch.zeros(len(all_keys) * 2, device=device, dtype=torch.float32)
        for index, key in enumerate(all_keys):
            if key in local_num:
                pack[2 * index] = local_num[key]
                pack[2 * index + 1] = local_den[key]
        dist.all_reduce(pack, group=process_group)
        reduced: dict[str, float] = {}
        for index, key in enumerate(all_keys):
            reduced[key] = float((pack[2 * index] / pack[2 * index + 1].clamp(min=1.0e-12)).item())
        return reduced

    return {key: local_num[key] / max(local_den[key], 1.0e-12) for key in local_num}
