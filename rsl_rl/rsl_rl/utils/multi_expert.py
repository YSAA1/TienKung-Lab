"""Group routing for multi-expert DAgger (LightLP V-C / T4 G2-G3).

Actor observations stay label-free. Critic observations may append a group
one-hot. Loco and skill groups query a frozen expert; the transition group has
no expert and is excluded from the imitation loss.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
import torch.nn.functional as F

GROUP_LOCO = 0
GROUP_SKILL = 1
GROUP_TRANSITION = 2
NUM_GROUPS = 3
GROUP_NAMES = ("loco", "skill", "transition")
DAGGER_GROUPS = (GROUP_LOCO, GROUP_SKILL)

ExpertFn = Callable[[torch.Tensor], torch.Tensor]


def group_one_hot(group_ids: torch.Tensor, num_groups: int = NUM_GROUPS) -> torch.Tensor:
    """Return ``(N, num_groups)`` one-hot rows for integer group ids."""
    if group_ids.ndim != 1:
        raise ValueError(f"group_ids must be 1-D, got shape {tuple(group_ids.shape)}")
    return F.one_hot(group_ids.long(), num_classes=num_groups).to(dtype=torch.float32)


def attach_group_one_hot_to_critic(
    actor_obs: torch.Tensor,
    critic_obs: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: int = NUM_GROUPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Append the group one-hot to critic obs only. Actor obs is returned unchanged."""
    one_hot = group_one_hot(group_ids, num_groups=num_groups).to(device=critic_obs.device, dtype=critic_obs.dtype)
    return actor_obs, torch.cat([critic_obs, one_hot], dim=-1)


def dagger_supervision_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """True on loco/skill rows; False on transition (no expert)."""
    return (group_ids == GROUP_LOCO) | (group_ids == GROUP_SKILL)


def scale_by_group(values: torch.Tensor, group_ids: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Multiply per-env values by ``scales[group_id]`` (per-group reward / termination)."""
    return values * scales.to(device=values.device, dtype=values.dtype)[group_ids.long()]


class MultiExpertRouter:
    """Query the frozen expert that owns each environment group."""

    def __init__(self, experts: Mapping[int, ExpertFn], action_dim: int | None = None):
        if GROUP_LOCO not in experts or GROUP_SKILL not in experts:
            raise ValueError("MultiExpertRouter requires experts for loco and skill groups")
        self.experts = dict(experts)
        self.action_dim = action_dim

    def query(self, observations: torch.Tensor, group_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if observations.ndim != 2:
            raise ValueError(f"observations must be (N, obs_dim), got {tuple(observations.shape)}")
        if group_ids.shape[0] != observations.shape[0]:
            raise ValueError("group_ids and observations must share the batch dimension")

        if self.action_dim is None:
            probe_group = GROUP_LOCO if GROUP_LOCO in self.experts else next(iter(self.experts))
            probe = self.experts[probe_group](observations[:1])
            action_dim = int(probe.shape[-1])
        else:
            action_dim = self.action_dim

        actions = observations.new_zeros((observations.shape[0], action_dim))
        for group, expert in self.experts.items():
            if group == GROUP_TRANSITION:
                continue
            selected = group_ids == group
            if bool(selected.any()):
                actions[selected] = expert(observations[selected])
        return actions, dagger_supervision_mask(group_ids)

    @staticmethod
    def supervised_mse(
        student_actions: torch.Tensor, expert_actions: torch.Tensor, group_ids: torch.Tensor
    ) -> torch.Tensor:
        """Mean squared imitation loss over loco/skill rows only."""
        mask = dagger_supervision_mask(group_ids)
        if not bool(mask.any()):
            return student_actions.new_zeros(())
        return F.mse_loss(student_actions[mask], expert_actions[mask])
