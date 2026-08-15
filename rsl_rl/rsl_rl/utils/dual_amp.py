"""Phase-switched dual AMP prior for the G3 transition group.

Before the robot reaches the skill trigger (box front + fixed offset) the
locomotion discriminator is the motion prior; at or past the trigger the skill
discriminator takes over. The switch is hard, matching LightLP §V-C.
"""

from __future__ import annotations

import torch


def vault_amp_trigger_x(box_front_x: float, offset: float = 0.0) -> float:
    """Trigger at the box front edge plus a fixed offset (plan: 箱前缘固定偏移)."""
    return float(box_front_x) + float(offset)


def amp_skill_phase(anchor_x: torch.Tensor, trigger_x: float | torch.Tensor) -> torch.Tensor:
    """Return 0 before the trigger (loco prior) and 1 at/after it (skill prior)."""
    return (anchor_x >= trigger_x).to(dtype=anchor_x.dtype)


def select_by_phase(loco_value: torch.Tensor, skill_value: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Hard-switch ``loco_value`` / ``skill_value`` using the 0/1 phase mask."""
    if loco_value.shape != skill_value.shape:
        raise ValueError(
            f"phase branches must share shape, got {tuple(loco_value.shape)} vs {tuple(skill_value.shape)}"
        )
    view = [-1] + [1] * (loco_value.ndim - 1)
    skill = phase.reshape(*view) >= 0.5
    return torch.where(skill, skill_value, loco_value)


def amp_phase_loss_masks(phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a batch into loco-prior vs skill-prior rows for discriminator losses."""
    skill = phase >= 0.5
    return ~skill, skill
