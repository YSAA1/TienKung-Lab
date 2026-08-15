"""Pure-torch contracts for the G3 phase-switched dual AMP prior."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.utils.dual_amp import (  # noqa: E402
    amp_phase_loss_masks,
    amp_skill_phase,
    select_by_phase,
    vault_amp_trigger_x,
)

from legged_lab.assets.t4.vault_contract import T4_VAULT_BOX_POS, T4_VAULT_BOX_SIZE  # noqa: E402


def test_phase_is_loco_before_trigger_and_skill_at_or_after():
    trigger = torch.tensor(0.0)
    x = torch.tensor([-0.5, -1e-6, 0.0, 0.25, 1.5])
    phase = amp_skill_phase(x, trigger)
    assert phase.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_trigger_defaults_to_box_front_edge():
    box_front = T4_VAULT_BOX_POS[0] - T4_VAULT_BOX_SIZE[0] / 2.0
    assert vault_amp_trigger_x(box_front) == box_front
    assert vault_amp_trigger_x(box_front, offset=0.10) == box_front + 0.10


def test_select_by_phase_hard_switches_reward_and_features():
    loco = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    skill = torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])
    phase = torch.tensor([0.0, 1.0, 0.0])
    chosen = select_by_phase(loco, skill, phase)
    assert torch.equal(chosen[0], loco[0])
    assert torch.equal(chosen[1], skill[1])
    assert torch.equal(chosen[2], loco[2])
    rewards = select_by_phase(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([8.0, 9.0, 10.0]), phase)
    assert torch.allclose(rewards, torch.tensor([0.1, 9.0, 0.3]))


def test_loss_masks_split_samples_by_trigger():
    phase = amp_skill_phase(torch.tensor([-0.2, 0.0, 0.4, -0.05]), trigger_x=0.0)
    loco_mask, skill_mask = amp_phase_loss_masks(phase)
    assert loco_mask.tolist() == [True, False, False, True]
    assert skill_mask.tolist() == [False, True, True, False]
    assert not bool((loco_mask & skill_mask).any())
    assert bool((loco_mask | skill_mask).all())
