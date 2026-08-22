# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Isaac-free tests for episode-weighted TensorBoard log reduction."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_LOG_PATH = Path(__file__).resolve().parents[1] / "rsl_rl" / "rsl_rl" / "utils" / "distributed_logs.py"
_spec = importlib.util.spec_from_file_location("t4_distributed_logs", _LOG_PATH)
_logs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_logs)
COUNT_SUFFIX = _logs.COUNT_SUFFIX
reduce_episode_log_dicts = _logs.reduce_episode_log_dicts


def test_weighted_mean_prefers_larger_batches():
    infos = [
        {
            "Terrain/a/promotion_rate": 1.0,
            f"Terrain/a/promotion_rate{COUNT_SUFFIX}": 1.0,
        },
        {
            "Terrain/a/promotion_rate": 0.0,
            f"Terrain/a/promotion_rate{COUNT_SUFFIX}": 3.0,
        },
    ]
    out = reduce_episode_log_dicts(infos, distributed=False)
    assert out["Terrain/a/promotion_rate"] == pytest.approx(0.25)
    assert f"Terrain/a/promotion_rate{COUNT_SUFFIX}" not in out


def test_missing_count_is_unit_weight_and_empty_is_safe():
    infos = [
        {"Reset/accel": 1.0},
        {"Reset/accel": 0.0, f"Reset/accel{COUNT_SUFFIX}": 3.0},
    ]
    out = reduce_episode_log_dicts(infos, distributed=False)
    assert out["Reset/accel"] == pytest.approx(0.25)
    assert reduce_episode_log_dicts([], distributed=False) == {}


def test_amp_runner_reduces_before_rank0_write():
    source = (Path(__file__).resolve().parents[1] / "rsl_rl" / "rsl_rl" / "runners" / "amp_on_policy_runner.py").read_text()
    assert "reduce_episode_log_dicts" in source
    assert "reduced_ep_logs" in source
    assert "self.is_distributed" in source
    reduce_src = Path(__file__).resolve().parents[1] / "rsl_rl" / "rsl_rl" / "utils" / "distributed_logs.py"
    packed = reduce_src.read_text()
    assert "len(all_keys) * 2" in packed
    assert packed.count("dist.all_reduce(") == 1
