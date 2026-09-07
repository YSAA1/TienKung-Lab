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

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv, reference_scale: float | None = None) -> torch.Tensor:
    """Action change, optionally expressed in units of reference_scale radians.

    Position-action scaling must not silently change the physical smoothing cost.
    The buffer is policy-ordered while the actuator scale is simulator-ordered.
    """
    delta = (
        env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
    )
    if reference_scale is not None:
        scale = env.action_scale
        if torch.is_tensor(scale) and scale.ndim > 0:
            scale = scale[..., env.policy_joint_ids]
        delta = delta * (scale / reference_scale)
    return torch.sum(torch.square(delta), dim=1)


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward = (reward - threshold).clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def upright_orientation(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """LightLP Table I / Eq. (2) upright bonus on projected gravity xy."""
    asset: Articulation = env.scene[asset_cfg.name]
    gravity_b = math_utils.quat_rotate_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    g_xy = gravity_b[:, :2]
    n2 = torch.sum(torch.square(g_xy), dim=1)
    n1 = torch.sqrt(n2 + 1.0e-8)
    return torch.exp(-2.0 * n2) + 0.1 * torch.exp(-n1)


def heading_error(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """LightLP Table I |Δψ| between commanded heading and base yaw."""
    target = getattr(env.command_generator, "heading_target", None)
    if target is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    heading_w = getattr(asset.data, "heading_w", None)
    if heading_w is None:
        _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
        heading_w = yaw
    delta = math_utils.wrap_to_pi(target - heading_w)
    return torch.abs(delta)


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    penalty = torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    ).to(dtype=torch.float)
    # LightLP sparse mix: zero stumble on sparse tiles (disk rims look like stumbles).
    mask = getattr(env, "sparse_tile_mask", None)
    if mask is not None:
        penalty = penalty * (~mask).float()
    return penalty


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: TienKungEnv, reference_scale: float | None = None) -> torch.Tensor:
    """Ankle target offset, optionally in units of reference_scale radians."""
    action = env.action
    if reference_scale is not None:
        action = action * (env.action_scale / reference_scale)
    return torch.sum(torch.abs(action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def velocity_slack(
    env: BaseEnv | TienKungEnv,
    lo: float = 0.3,
    hi: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """LightLP slack: 1 when actual/cmd forward speed is in ``[lo, hi]``."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    cmd = env.command_generator.command[:, 0]
    standing = cmd.abs() < 1.0e-4
    ratio = vel_yaw[:, 0] / torch.where(standing, torch.ones_like(cmd), cmd)
    return ((ratio >= lo) & (ratio <= hi) & (~standing)).to(dtype=vel_yaw.dtype)


def illegal_footstep(
    env: BaseEnv | TienKungEnv,
    delta: float = 0.1,
    contact_threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_foot_link"),
) -> torch.Tensor:
    """Eq. (4): sum over contacted feet of the fraction of rays that miss support.

    Soft-stage sparse tiles use the algebraic true-hole map when the env sets
    ``use_algebraic_sparse_scan`` (filled collision would otherwise silence rays).
    """
    algebraic = getattr(env, "algebraic_illegal_footstep", None)
    if algebraic is not None and getattr(env, "use_algebraic_sparse_scan", False):
        return algebraic(contact_threshold=contact_threshold)

    scanners = []
    sensors = env.scene.sensors
    for name in ("left_foot_scanner", "right_foot_scanner"):
        if name in sensors:
            scanners.append(sensors[name])
    if not scanners:
        return torch.zeros(env.num_envs, device=env.device)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_force_w = contact_sensor.data.net_forces_w
    if feet_force_w.ndim == 4:
        feet_force_w = feet_force_w[:, -1]
    feet_force = torch.norm(feet_force_w[:, env.feet_body_ids], dim=-1)
    in_contact = feet_force > contact_threshold
    penalties = []
    for i, scanner in enumerate(scanners):
        hits = scanner.data.ray_hits_w[..., 2]
        foot_z = scanner.data.pos_w[:, 2].unsqueeze(-1)
        miss = torch.isnan(hits) | torch.isinf(hits)
        depth = foot_z - hits
        bad = miss | (depth > delta)
        frac = bad.float().mean(dim=-1)
        contact_i = in_contact[:, min(i, in_contact.shape[1] - 1)]
        penalties.append(frac * contact_i.float())
    penalty = torch.stack(penalties, dim=-1).sum(dim=-1)
    type_ids = getattr(env, "sparse_foothold_type_ids", None)
    types = getattr(getattr(env.scene, "terrain", None), "terrain_types", None)
    if type_ids is not None:
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if type_ids and types is not None:
            for type_id in type_ids:
                mask |= types == type_id
        penalty = penalty * mask.float()
    return penalty


def opposite_direction(
    env: BaseEnv | TienKungEnv,
    cmd_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """LightLP Table I: max(0, -v · v̂) against the commanded planar direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    cmd_xy = env.command_generator.command[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=-1)
    active = cmd_norm >= cmd_threshold
    vhat = cmd_xy / cmd_norm.clamp(min=1.0e-6).unsqueeze(-1)
    against = (-(vel_yaw[:, :2] * vhat).sum(dim=-1)).clamp(min=0.0)
    return against * active.to(dtype=against.dtype)


def foot_acceleration_penalty(
    env: BaseEnv | TienKungEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*_foot_link"]),
    tau_s: float = 0.06,
    threshold_mps2: float = 30.0,
) -> torch.Tensor:
    """LightLP filtered foot-acceleration excess (replaces touchdown-impact on sparse)."""
    update = getattr(env, "update_foot_accel_penalty", None)
    if update is not None:
        return update(tau_s=tau_s, threshold_mps2=threshold_mps2, asset_cfg=asset_cfg)
    asset: Articulation = env.scene[asset_cfg.name]
    # Fallback without EMA state: instantaneous |a| excess of both feet.
    accel = asset.data.body_lin_acc_w[:, asset_cfg.body_ids, :]
    magnitude = torch.norm(accel, dim=-1)
    return torch.sum(torch.clamp(magnitude - threshold_mps2, min=0.0), dim=-1)


def hurdle_bar_contact(
    env: BaseEnv | TienKungEnv,
    z_range: tuple[float, float] = (0.06, 0.38),
    contact_threshold: float = 0.5,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor", body_names=".*_foot_link"),
) -> torch.Tensor:
    """1 on hurdle tiles when a stance foot is in the bar height band (plow/clip)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w
    if forces.ndim == 4:
        forces = forces[:, -1]
    foot_ids = env.feet_body_ids
    force = torch.norm(forces[:, foot_ids], dim=-1)
    in_contact = force > contact_threshold
    foot_z = env.robot.data.body_pos_w[:, foot_ids, 2]
    mid_air_contact = in_contact & (foot_z > z_range[0]) & (foot_z < z_range[1])
    hit = mid_air_contact.any(dim=-1)
    type_ids = getattr(env, "hurdle_terrain_type_ids", None)
    if type_ids is None:
        legacy_id = getattr(env, "hurdle_terrain_type_id", None)
        type_ids = [legacy_id] if legacy_id is not None else None
    if type_ids is None or not hasattr(env.scene, "terrain"):
        return hit.float()
    types = getattr(env.scene.terrain, "terrain_types", None)
    mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if type_ids and types is not None:
        for type_id in type_ids:
            mask |= types == type_id
    return hit.float() * mask.float()


def feet_y_distance(env: TienKungEnv, target: float = 0.299) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing.

    ``target`` is the robot's nominal lateral foot spacing and must be set per robot.
    """
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - target)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


def foot_touchdown_impact_penalty(
    env: BaseEnv | TienKungEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 20.0,
    safe_downward_speed: float = 0.25,
    contact_time_window: float = 0.04,
) -> torch.Tensor:
    """Penalize excessive downward foot velocity only at fresh foot contact.

    Ported from the VITAL T4_27 task: it teaches soft touchdowns on stairs and
    drops without punishing normal stance loading.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_force = torch.norm(net_forces, dim=-1).amax(dim=1)
    in_contact = contact_force > force_threshold
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    fresh_touchdown = in_contact & (contact_time > 0.0) & (contact_time <= contact_time_window)
    foot_vz = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, 2]
    downward_excess = (-foot_vz - safe_downward_speed).clamp(min=0.0)
    return torch.sum(fresh_touchdown.float() * downward_excess.square(), dim=1)


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_reward_scale(env: BaseEnv | TienKungEnv) -> torch.Tensor | float:
    """Per-env weight on the periodic gait rewards.

    Envs that relax the gait clock on hard terrain expose ``gait_reward_scale``;
    envs on a strictly fixed clock do not and keep the full weight.
    """
    return getattr(env, "gait_reward_scale", 1.0)


def gait_feet_frc_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return (left_frc_score + right_frc_score) * gait_reward_scale(env)


def gait_feet_spd_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return (left_spd_score + right_spd_score) * gait_reward_scale(env)


def gait_feet_frc_support_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return (left_frc_score + right_frc_score) * gait_reward_scale(env)
