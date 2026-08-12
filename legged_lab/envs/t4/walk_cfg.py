# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""TienKung-native T4 walk baseline: proprio-only PPO+AMP on gravel.

This is Route W, a control lineage against Stage E. It reuses the original
TienKung ``walk`` recipe (gravel, no height scan, 10s command resampling,
constant AMP style weight) on the T4 27DoF asset. It is not a deployment
policy and must not grow HeightScan, stairs, or depth inputs.
"""

from __future__ import annotations

import math

from isaaclab.utils import configclass

from legged_lab.assets.t4.schemas import (
    PROPRIO_HISTORY_LENGTH,
    WALK_AMP_TERRAIN_SCHEDULE_ENABLE,
    WALK_COMMAND_RESAMPLING_S,
    WALK_EXPERIMENT_NAME,
    WALK_GAIT_MODE,
    WALK_TERMINATE_CONTACTS,
)
from legged_lab.assets.t4.t4 import T4_CFG
from legged_lab.envs.base.base_config import (
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    HeightScannerCfg,
    RobotCfg,
)
from legged_lab.envs.t4.teacher_cfg import (
    T4AmpTerrainScheduleCfg,
    T4GaitCfg,
    T4LocoTeacherAgentCfg,
    T4LocoTeacherEnvCfg,
)
from legged_lab.terrains import GRAVEL_TERRAINS_CFG


@configclass
class T4WalkEnvCfg(T4LocoTeacherEnvCfg):
    policy_role: str = "walk"
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=1024,
        env_spacing=2.5,
        robot=T4_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="Trunk",
            resolution=0.1,
            size=(1.6, 1.0),
            offset=(0.0, 0.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
    )
    robot: RobotCfg = RobotCfg(
        actor_obs_history_length=PROPRIO_HISTORY_LENGTH,
        critic_obs_history_length=PROPRIO_HISTORY_LENGTH,
        action_scale=0.25,
        terminate_contacts_body_names=list(WALK_TERMINATE_CONTACTS),
        feet_body_names=[".*_foot_link"],
    )
    gait: T4GaitCfg = T4GaitCfg(mode=WALK_GAIT_MODE)
    amp_terrain_schedule: T4AmpTerrainScheduleCfg = T4AmpTerrainScheduleCfg(enable=WALK_AMP_TERRAIN_SCHEDULE_ENABLE)
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(WALK_COMMAND_RESAMPLING_S, WALK_COMMAND_RESAMPLING_S),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.57, 1.57), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class T4WalkAgentCfg(T4LocoTeacherAgentCfg):
    max_iterations = 50000
    experiment_name = WALK_EXPERIMENT_NAME
    neptune_project = WALK_EXPERIMENT_NAME
    wandb_project = WALK_EXPERIMENT_NAME
