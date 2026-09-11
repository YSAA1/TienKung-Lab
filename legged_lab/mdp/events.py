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

"""Wrappers around Isaac Lab interval events. Do not shadow the upstream names."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp.events import push_by_setting_velocity
from isaaclab.managers import SceneEntityCfg

# IsaacLab 2.1.0 releases do not ship ``randomize_rigid_body_com`` (it only
# exists on newer main); re-export the repo's own implementation so
# ``legged_lab.mdp`` resolves it on every contract environment.
from legged_lab.mdp.ramp import (
    covers_all_bodies,
    interpolate_range,
    ramp_fraction,
    ramped_time_lags,
    scale_range_around_nominal,
)
from legged_lab.motion_tracking.mdp.events import randomize_rigid_body_com  # noqa: F401

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


def _policy_step_count(env: ManagerBasedEnv) -> int:
    decimation = getattr(env.cfg.sim, "decimation", 1)
    return int(getattr(env, "sim_step_counter", 0)) // max(1, int(decimation))


def _ramp_fraction_from_env(env: ManagerBasedEnv, ramp_steps: int) -> float:
    return ramp_fraction(_policy_step_count(env), ramp_steps)


def randomize_rigid_body_material_ramped(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_source: tuple[float, float],
    static_friction_target: tuple[float, float],
    dynamic_friction_source: tuple[float, float],
    dynamic_friction_target: tuple[float, float],
    restitution_range: tuple[float, float],
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*"),
):
    """Interval-mode friction DR linearly ramped between two uniform ranges.

    Unlike the upstream bucketed term, ranges are re-read from the schedule on every
    call so the sampled distribution widens as training progresses. Assignment
    semantics follow the upstream all-bodies branch (``body_names=".*"``). The
    45-60s refresh relies on this repo's reset path not calling
    ``EventManager.reset()`` (which would resample per-env interval timers).
    """
    asset = env.scene[asset_cfg.name]
    if not covers_all_bodies(asset_cfg.body_ids, asset.num_bodies):
        # the all-shapes assignment below is only correct for body_names=".*"
        raise ValueError("randomize_rigid_body_material_ramped supports body_names='.*' only")
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    fraction = _ramp_fraction_from_env(env, ramp_steps)
    static_range = interpolate_range(static_friction_source, static_friction_target, fraction)
    dynamic_range = interpolate_range(dynamic_friction_source, dynamic_friction_target, fraction)
    num_shapes = asset.root_physx_view.max_shapes
    materials = asset.root_physx_view.get_material_properties()
    for prop, (lo, hi) in enumerate((static_range, dynamic_range, restitution_range)):
        sample = torch.empty(len(env_ids), num_shapes, device="cpu").uniform_(lo, hi)
        materials[env_ids, :, prop] = sample
    asset.root_physx_view.set_material_properties(materials, env_ids)


def randomize_actuator_gains_ramped(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    stiffness_distribution_params: tuple[float, float],
    damping_distribution_params: tuple[float, float],
    ramp_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*"),
    operation: str = "scale",
    distribution: str = "uniform",
):
    """Reset-mode actuator gain DR whose multiplicative range ramps from nominal 1.0."""
    if operation != "scale" or distribution != "uniform":
        raise ValueError("randomize_actuator_gains_ramped supports uniform scale only")
    fraction = _ramp_fraction_from_env(env, ramp_steps)
    if fraction <= 0.0:
        return
    from isaaclab.envs.mdp.events import randomize_actuator_gains

    randomize_actuator_gains(
        env,
        env_ids,
        asset_cfg=asset_cfg,
        stiffness_distribution_params=scale_range_around_nominal(stiffness_distribution_params, fraction),
        damping_distribution_params=scale_range_around_nominal(damping_distribution_params, fraction),
        operation=operation,
        distribution=distribution,
    )


def randomize_action_delay_ramped(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    min_delay: int,
    max_delay: int,
    ramp_steps: int,
):
    """Reset-mode per-episode action latency with a ramp on the excess-delay probability."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    fraction = _ramp_fraction_from_env(env, ramp_steps)
    draws = torch.rand(len(env_ids), device=env_ids.device)
    lags = ramped_time_lags(draws, min_delay, max_delay, fraction)
    # DelayBuffer keeps int32 lags; pass int32 explicitly instead of relying on
    # the implicit downcast of an int64 sample.
    env.action_buffer.set_time_lag(lags.to(torch.int32), env_ids)
