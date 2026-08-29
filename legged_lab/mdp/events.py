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


def randomize_joint_effort_limits(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    effort_distribution_params: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    operation: str = "scale",
    distribution: str = "uniform",
):
    """Randomize simulator and actuator-model effort limits together."""
    if operation != "scale" or distribution != "uniform":
        raise ValueError("randomize_joint_effort_limits currently supports uniform scale only")
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    nominal = asset.data.joint_effort_limits[env_ids][:, joint_ids].clone()
    lo, hi = map(float, effort_distribution_params)
    scales = torch.empty_like(nominal).uniform_(lo, hi)
    asset.write_joint_effort_limit_to_sim(nominal * scales, joint_ids=joint_ids, env_ids=env_ids)
    for actuator in asset.actuators.values():
        group_limits = asset.data.joint_effort_limits[:, actuator.joint_indices]
        actuator.effort_limit_sim[env_ids] = group_limits[env_ids]
        actuator.effort_limit[env_ids] = group_limits[env_ids]
