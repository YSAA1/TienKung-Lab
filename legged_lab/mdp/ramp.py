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

"""Pure domain-randomization ramp math; importable without Isaac installed."""

from __future__ import annotations

import torch


def ramp_fraction(step_count: int, ramp_steps: int) -> float:
    """Linear ramp value in ``[0, 1]``; reaches 1 after ``ramp_steps`` steps."""
    if ramp_steps <= 0:
        raise ValueError("ramp_steps must be positive")
    return min(1.0, max(0.0, step_count / float(ramp_steps)))


def interpolate_range(source: tuple[float, float], target: tuple[float, float], fraction: float) -> tuple[float, float]:
    """Linearly interpolate both bounds of an absolute sampling range."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    lo = source[0] + (target[0] - source[0]) * fraction
    hi = source[1] + (target[1] - source[1]) * fraction
    return (lo, hi)


def scale_range_around_nominal(scale_range: tuple[float, float], fraction: float) -> tuple[float, float]:
    """Pull a multiplicative scale range toward nominal 1.0 by ``fraction``."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    lo = 1.0 + (scale_range[0] - 1.0) * fraction
    hi = 1.0 + (scale_range[1] - 1.0) * fraction
    return (lo, hi)


def covers_all_bodies(body_ids, num_bodies: int) -> bool:
    """Whether a resolved ``SceneEntityCfg.body_ids`` selects every body.

    ``body_names=".*"`` resolves to the full index list on some IsaacLab versions
    and to ``slice(None)`` on others; both spellings mean "all bodies" here.
    """
    if isinstance(body_ids, slice):
        return body_ids == slice(None)
    return len(body_ids) == num_bodies


def ramped_time_lags(draws: torch.Tensor, min_delay: int, max_delay: int, fraction: float) -> torch.Tensor:
    """Integer action delays whose ``P(lag > min_delay)`` grows with ``fraction``.

    Exact behavior for ``min_delay=0, max_delay=1``: ``P(lag=1) = max(0, 1 - 1/(2f))``,
    i.e. the first half of the ramp has no excess latency at all (deliberate
    slow-in), after which the probability rises to 0.5 at ``f = 1``.
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be within [0, 1]")
    span = max_delay + 1 - min_delay
    if span <= 0:
        raise ValueError("max_delay must be >= min_delay")
    lags = min_delay + (draws * span * fraction).floor().clamp(max=max_delay - min_delay)
    return lags.to(torch.int64)
