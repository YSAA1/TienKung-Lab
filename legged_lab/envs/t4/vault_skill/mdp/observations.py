"""Student 1155D actor observation for the G2 vault skill.

Layout matches Stage E teacher actor obs: 10 stacked 96D proprio frames plus
the 195D forward height scan. The G1 tracking observation lives in a separate
teacher group and is not built here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.buffers import CircularBuffer

from legged_lab.assets.t4.vault_contract import T4_VAULT_BOX_POS, T4_VAULT_BOX_SIZE
from legged_lab.assets.t4.vault_skill_contract import (
    G2_GAIT_AIR_RATIO,
    G2_GAIT_CYCLE,
    G2_GAIT_PHASE_OFFSET,
    G2_PROPRIO_HISTORY_LENGTH,
    G2_SKILL_LIN_VEL_X,
    analytic_vault_height_scan,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _ensure_proprio_buffer(env: ManagerBasedEnv) -> CircularBuffer:
    buf = getattr(env, "_g2_proprio_buf", None)
    if buf is None:
        buf = CircularBuffer(max_len=G2_PROPRIO_HISTORY_LENGTH, batch_size=env.num_envs, device=env.device)
        env._g2_proprio_buf = buf
    reset_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
        buf.reset(reset_ids)
    return buf


def g2_proprio_frame(env: ManagerBasedEnv) -> torch.Tensor:
    """One 96D Stage E proprio frame. Velocity command is a fixed forward skill cmd."""
    robot = env.scene["robot"]
    ang_vel = robot.data.root_ang_vel_b
    projected_gravity = robot.data.projected_gravity_b
    command = ang_vel.new_zeros((env.num_envs, 3))
    command[:, 0] = G2_SKILL_LIN_VEL_X
    joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
    joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
    previous_action = env.action_manager.action
    cycle = max(float(G2_GAIT_CYCLE), 1.0e-6)
    gait_time = env.episode_length_buf.to(dtype=ang_vel.dtype) * env.step_dt / cycle
    offsets = ang_vel.new_tensor(list(G2_GAIT_PHASE_OFFSET))
    phase = (gait_time.unsqueeze(-1) + offsets) % 1.0
    air_ratio = ang_vel.new_tensor(list(G2_GAIT_AIR_RATIO)).expand(env.num_envs, 2)
    return torch.cat(
        (
            ang_vel,
            projected_gravity,
            command,
            joint_pos,
            joint_vel,
            previous_action,
            torch.sin(2.0 * torch.pi * phase),
            torch.cos(2.0 * torch.pi * phase),
            air_ratio,
        ),
        dim=-1,
    )


def g2_height_scan(env: ManagerBasedEnv) -> torch.Tensor:
    """Analytic forward scan of the flat ground plus the 1 m box (no RayCaster mesh)."""
    robot = env.scene["robot"]
    trunk_ids, _ = robot.find_bodies("Trunk")
    trunk_pos = robot.data.body_pos_w[:, trunk_ids[0]]
    trunk_quat = robot.data.body_quat_w[:, trunk_ids[0]]
    w, x, y, z = trunk_quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    box_center = env.scene.env_origins + trunk_pos.new_tensor(T4_VAULT_BOX_POS)
    return analytic_vault_height_scan(
        trunk_pos[:, :2],
        yaw,
        trunk_pos[:, 2],
        box_center,
        box_half_xyz=(T4_VAULT_BOX_SIZE[0] / 2.0, T4_VAULT_BOX_SIZE[1] / 2.0, T4_VAULT_BOX_SIZE[2] / 2.0),
    )


def g2_actor_obs(env: ManagerBasedEnv) -> torch.Tensor:
    """1155D student observation: proprio history then scan."""
    frame = g2_proprio_frame(env)
    history = _ensure_proprio_buffer(env)
    history.append(frame)
    stacked = history.buffer.reshape(env.num_envs, -1)
    return torch.cat((stacked, g2_height_scan(env)), dim=-1)
