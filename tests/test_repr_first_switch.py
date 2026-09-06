from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RSL_RL_ROOT = ROOT / "rsl_rl"
if str(RSL_RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RSL_RL_ROOT))

from rsl_rl.algorithms.repr_first_switch import (  # noqa: E402
    ReprSwitchState,
    ScanProbe,
    observe_probe,
    probe_meets_switch_rule,
)


def _good_probe(**kwargs) -> ScanProbe:
    payload = dict(
        mse_global=0.20,
        mse_stepping_stones=0.10,
        mse_raised_pillars=0.10,
        occ_agree_stepping_stones=0.9,
        occ_agree_raised_pillars=0.9,
        count_global=32,
        count_stepping_stones=16,
        count_raised_pillars=16,
    )
    payload.update(kwargs)
    return ScanProbe(**payload)


def test_global_recon_alone_does_not_meet_switch_rule():
    probe = _good_probe(mse_global=0.05, mse_stepping_stones=0.40, mse_raised_pillars=0.40)
    assert probe_meets_switch_rule(probe, baseline_stones=0.80, baseline_pillars=0.80) is False


def test_sparse_improvement_and_ratio_meets_switch_rule():
    probe = _good_probe(mse_global=0.20, mse_stepping_stones=0.10, mse_raised_pillars=0.10)
    assert probe_meets_switch_rule(probe, baseline_stones=0.40, baseline_pillars=0.40) is True


def test_insufficient_sparse_samples_are_invalid():
    probe = _good_probe(count_stepping_stones=2, count_raised_pillars=16)
    assert probe.valid(min_count=8) is False


def test_invalid_probe_does_not_count_toward_patience():
    state = ReprSwitchState(baseline_stone_mses=[0.4], baseline_pillar_mses=[0.4], consecutive_hits=2)
    state = observe_probe(state, iteration=500, probe=_good_probe(count_stepping_stones=1), patience=3)
    assert state.phase == "representation"
    assert state.consecutive_hits == 0
    state = observe_probe(
        state,
        iteration=600,
        probe=_good_probe(mse_global=0.20, mse_stepping_stones=0.10, mse_raised_pillars=0.10),
        patience=3,
    )
    assert state.consecutive_hits == 1


def test_three_consecutive_metric_hits_switch_once():
    state = ReprSwitchState()
    baseline = _good_probe(mse_global=0.40, mse_stepping_stones=0.40, mse_raised_pillars=0.40)
    improved = _good_probe(mse_global=0.20, mse_stepping_stones=0.10, mse_raised_pillars=0.10)
    for iteration in (200, 300, 400):
        state = observe_probe(state, iteration=iteration, probe=baseline, patience=3, cap_iters=4000)
    assert state.phase == "representation"
    for iteration in (500, 600, 700):
        state = observe_probe(state, iteration=iteration, probe=improved, patience=3, cap_iters=4000)
    assert state.phase == "action"
    assert state.switch_reason == "metric"
    assert state.switch_iter == 700
    frozen = observe_probe(state, iteration=800, probe=improved, patience=3)
    assert frozen.phase == "action"
    assert frozen.switch_iter == 700


def test_cap_without_baseline_is_baseline_invalid():
    state = observe_probe(ReprSwitchState(), iteration=4000, probe=None, cap_iters=4000)
    assert state.phase == "action"
    assert state.switch_reason == "baseline_invalid"


def test_cap_with_baseline_is_cap_not_metric():
    state = ReprSwitchState(baseline_stone_mses=[0.5], baseline_pillar_mses=[0.5])
    bad = _good_probe(mse_global=0.05, mse_stepping_stones=0.4, mse_raised_pillars=0.4)
    state = observe_probe(state, iteration=4000, probe=bad, cap_iters=4000, patience=3)
    assert state.phase == "action"
    assert state.switch_reason == "cap"
