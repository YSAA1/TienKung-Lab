"""Pure-torch contracts for G3 multi-expert DAgger routing.

These tests pin group → expert query, DAgger loss masks, critic-only group
one-hot, and per-group reward/termination scales. They must not import IsaacLab.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rsl_rl"))

from rsl_rl.utils.multi_expert import (  # noqa: E402
    GROUP_LOCO,
    GROUP_SKILL,
    GROUP_TRANSITION,
    NUM_GROUPS,
    MultiExpertRouter,
    attach_group_one_hot_to_critic,
    dagger_supervision_mask,
    group_one_hot,
    scale_by_group,
)


class _ConstExpert(nn.Module):
    def __init__(self, value: float, act_dim: int = 27):
        super().__init__()
        self.register_buffer("value", torch.tensor(value))
        self.act_dim = act_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.new_full((obs.shape[0], self.act_dim), float(self.value))


def test_group_one_hot_is_identity_rows():
    ids = torch.tensor([GROUP_LOCO, GROUP_SKILL, GROUP_TRANSITION, GROUP_LOCO])
    oh = group_one_hot(ids)
    assert oh.shape == (4, NUM_GROUPS)
    assert torch.equal(oh.argmax(dim=-1), ids)
    assert torch.allclose(oh.sum(dim=-1), torch.ones(4))


def test_one_hot_appends_to_critic_only():
    actor = torch.randn(5, 1155)
    critic = torch.randn(5, 276)
    ids = torch.tensor([0, 1, 2, 1, 0])
    actor_out, critic_out = attach_group_one_hot_to_critic(actor, critic, ids)
    assert actor_out.shape == (5, 1155)
    assert torch.equal(actor_out, actor)
    assert critic_out.shape == (5, 276 + NUM_GROUPS)
    assert torch.equal(critic_out[:, :276], critic)
    assert torch.equal(critic_out[:, 276:], group_one_hot(ids))


def test_dagger_mask_supervises_loco_and_skill_only():
    ids = torch.tensor([GROUP_LOCO, GROUP_SKILL, GROUP_TRANSITION, GROUP_SKILL])
    mask = dagger_supervision_mask(ids)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [True, True, False, True]


def test_router_queries_matching_expert_and_zeros_transition():
    router = MultiExpertRouter(
        {
            GROUP_LOCO: _ConstExpert(1.0),
            GROUP_SKILL: _ConstExpert(2.0),
        }
    )
    obs = torch.zeros(4, 8)
    ids = torch.tensor([GROUP_LOCO, GROUP_SKILL, GROUP_TRANSITION, GROUP_LOCO])
    actions, mask = router.query(obs, ids)
    assert actions.shape == (4, 27)
    assert torch.allclose(actions[0], torch.ones(27))
    assert torch.allclose(actions[1], torch.full((27,), 2.0))
    assert torch.allclose(actions[2], torch.zeros(27))
    assert torch.allclose(actions[3], torch.ones(27))
    assert mask.tolist() == [True, True, False, True]


def test_dagger_loss_ignores_transition_group():
    student = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [9.0, 9.0],
        ]
    )
    expert = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [0.0, 0.0],
        ]
    )
    ids = torch.tensor([GROUP_LOCO, GROUP_SKILL, GROUP_TRANSITION])
    loss = MultiExpertRouter.supervised_mse(student, expert, ids)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)
    # If transition were included, 9 vs 0 would make the loss large.
    unmasked = torch.nn.functional.mse_loss(student, expert)
    assert unmasked > 10.0


def test_scale_by_group_can_zero_foreign_rewards_or_terms():
    values = torch.tensor([1.0, 1.0, 1.0])
    ids = torch.tensor([GROUP_LOCO, GROUP_SKILL, GROUP_TRANSITION])
    # Loco keeps tracking reward, skill/transition drop it.
    loco_only = scale_by_group(values, ids, torch.tensor([1.0, 0.0, 0.0]))
    assert loco_only.tolist() == [1.0, 0.0, 0.0]
    skill_only = scale_by_group(values, ids, torch.tensor([0.0, 1.0, 0.0]))
    assert skill_only.tolist() == [0.0, 1.0, 0.0]
    # Transition keeps its own sparse crossing term.
    trans_only = scale_by_group(values, ids, torch.tensor([0.0, 0.0, 1.0]))
    assert trans_only.tolist() == [0.0, 0.0, 1.0]
