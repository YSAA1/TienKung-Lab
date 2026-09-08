"""T4 recipes for shared AMP locomotion and LightLP implementations."""

import os
from isaaclab.utils import configclass

from legged_lab.assets.t4.constants import T4_JOINT_NAMES
from legged_lab.assets.t4.locomotion import T4_LOCOMOTION
from legged_lab.assets.t4.schemas import AMP_FORMAL_EXPERT_DIR, AMP_FRAME_DIM, amp_expert_files
from legged_lab.assets.t4.t4 import T4_CFG
from legged_lab.locomotion.config_binding import bind_reward_roles
from legged_lab.locomotion.teacher_cfg import (
    AmpLocomotionAgentCfg, AmpLocomotionEnvCfg, AmpLocomotionRewardCfg,
    AmpTerrainScheduleCfg, GaitCfg, LightLPLocomotionAgentCfg,
    LightLPLocomotionEnvCfg, LightLPRewardCfg,
)

# Historical public imports remain valid; the implementation is algorithm-owned.
T4GaitCfg = GaitCfg
T4AmpTerrainScheduleCfg = AmpTerrainScheduleCfg


@configclass
class T4TeacherRewardCfg(AmpLocomotionRewardCfg):
    def __post_init__(self):
        bind_reward_roles(self, T4_LOCOMOTION)


@configclass
class T4SparseTeacherRewardCfg(LightLPRewardCfg):
    def __post_init__(self):
        bind_reward_roles(self, T4_LOCOMOTION)


@configclass
class T4LocoTeacherEnvCfg(AmpLocomotionEnvCfg):
    robot_spec = T4_LOCOMOTION
    amp_terrain_schedule = T4AmpTerrainScheduleCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = T4_CFG


@configclass
class T4LocoSparseTeacherEnvCfg(LightLPLocomotionEnvCfg):
    robot_spec = T4_LOCOMOTION
    amp_terrain_schedule = T4AmpTerrainScheduleCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = T4_CFG


@configclass
class T4LocoTeacherAgentCfg(AmpLocomotionAgentCfg):
    experiment_name = "t4_loco_teacher"
    neptune_project = "t4_loco_teacher"
    wandb_project = "t4_loco_teacher"
    amp_frame_dim = AMP_FRAME_DIM
    amp_joint_order = list(T4_JOINT_NAMES)
    amp_expert_dir = os.environ.get("T4_AMP_EXPERT_DIR", AMP_FORMAL_EXPERT_DIR)
    amp_motion_files = amp_expert_files(os.environ.get("T4_AMP_EXPERT_DIR", AMP_FORMAL_EXPERT_DIR))
    min_normalized_std = [0.05] * len(T4_JOINT_NAMES)


@configclass
class T4LocoSparseTeacherAgentCfg(LightLPLocomotionAgentCfg):
    experiment_name = "t4_loco_teacher_sparse"
    run_name = "t_sparse_lightlp_s12_rim_yaw40"
    neptune_project = "t4_loco_teacher_sparse"
    wandb_project = "t4_loco_teacher_sparse"
    amp_frame_dim = AMP_FRAME_DIM
    amp_joint_order = list(T4_JOINT_NAMES)
    amp_expert_dir = os.environ.get("T4_AMP_EXPERT_DIR", AMP_FORMAL_EXPERT_DIR)
    amp_motion_files = amp_expert_files(os.environ.get("T4_AMP_EXPERT_DIR", AMP_FORMAL_EXPERT_DIR))
    min_normalized_std = [0.05] * len(T4_JOINT_NAMES)
