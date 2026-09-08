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

"""G1 recipe for the shared LightLP + AMP locomotion algorithm."""

from isaaclab.utils import configclass

from legged_lab.assets.unitree_g1.constants import (
    G1_29DOF_JOINT_NAMES,
    NUM_G1_29DOF_JOINTS,
)
from legged_lab.assets.unitree_g1.g1 import G1_29DOF_CFG
from legged_lab.assets.unitree_g1.locomotion import G1_LOCOMOTION
from legged_lab.assets.unitree_g1.schemas import (
    AMP_FORMAL_EXPERT_DIR,
    AMP_FRAME_DIM,
    amp_expert_files,
)
from legged_lab.locomotion.config_binding import bind_reward_roles
from legged_lab.locomotion.teacher_cfg import (
    LightLPLocomotionAgentCfg,
    LightLPLocomotionEnvCfg,
    LightLPRewardCfg,
)


@configclass
class G1SparseTeacherRewardCfg(LightLPRewardCfg):
    def __post_init__(self):
        bind_reward_roles(self, G1_LOCOMOTION)
        self.undesired_contacts.weight = -1.0


@configclass
class G1LocoTeacherEnvCfg(LightLPLocomotionEnvCfg):
    robot_spec = G1_LOCOMOTION

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_29DOF_CFG.copy()
        self.reward = G1SparseTeacherRewardCfg()
        # Official G1 enables self-collision. Torso net force also includes
        # internal impacts, so it cannot serve as a ground-fall detector.
        # Keep soft contact costs and geometric/impact failure conditions.
        self.robot.terminate_contacts_body_names = []
        # Match the official G1 velocity task's uniform position-action scale.
        self.robot.action_scale = 0.25
        self.robot.action_scale_effort_fraction = None
        self.scene.max_init_terrain_level = 0
        # 10% resets sample all ten rows, independently of initial terrain levels.
        self.random_level_reset_fraction = 0.10
        self.random_level_reset_min_level = None
        self.random_level_reset_max_level = None
        # Restore the T4 sparse command range at every terrain difficulty.
        self.sparse_command_min_speed_scale = 1.0
        self.collapse_reset_pelvis_above_feet_m = 0.20
        # A brief low pose is a recovery opportunity, not an immediate failure.
        # Preserve the LightLP impact-immunity contract for sustained collapse.
        self.collapse_reset_grace_s = 0.20
        self.collapse_reset_respects_impact_immunity = True
        self.amp_terrain_schedule.enable = True
        self.commands.debug_vis = False
        self.domain_rand.action_delay.enable = False
        # Honor the official initial joint pose; independent angle scaling can
        # introduce foot penetration without adjusting root height.
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.domain_rand.events.reset_base.params["velocity_range"] = {}


@configclass
class G1LocoTeacherAgentCfg(LightLPLocomotionAgentCfg):
    experiment_name = "g1_loco_teacher_sparse"
    run_name = "g1_unitree_29dof_teacher_v6"
    max_iterations = 30000
    neptune_project = "g1_loco_teacher_sparse"
    wandb_project = "g1_loco_teacher_sparse"
    min_normalized_std = [0.05] * NUM_G1_29DOF_JOINTS
    amp_frame_dim = AMP_FRAME_DIM
    amp_expert_dir = AMP_FORMAL_EXPERT_DIR
    amp_motion_files = amp_expert_files()
    amp_joint_order = list(G1_29DOF_JOINT_NAMES)
