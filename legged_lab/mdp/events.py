# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""Wrappers around Isaac Lab interval events. Do not shadow the upstream names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.events import push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_by_setting_velocity_tagged(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Isaac Lab push, then stamp ``env._push_step_marker`` for the accel gate."""
    push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg=asset_cfg)
    marker = getattr(env, "_push_step_marker", None)
    if marker is not None:
        marker[env_ids] = env.sim_step_counter
