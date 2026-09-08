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

"""Z2 recipe for the shared LightLP + AMP locomotion algorithm."""

from isaaclab.utils import configclass

from legged_lab.assets.z2.constants import (
    NUM_Z2_29DOF_JOINTS,
    Z2_29DOF_JOINT_NAMES,
    Z2_FOOT_SCAN_SIZE,
    Z2_STANDING_PELVIS_Z,
)
from legged_lab.assets.z2.locomotion import Z2_LOCOMOTION
from legged_lab.assets.z2.schemas import (
    AMP_FORMAL_EXPERT_DIR,
    AMP_FRAME_DIM,
    amp_expert_files,
)
from legged_lab.assets.z2.z2 import Z2_29DOF_WALK_POSE_DAMPED_PD_CFG
from legged_lab.locomotion.config_binding import bind_reward_roles
from legged_lab.locomotion.teacher_cfg import (
    LightLPLocomotionAgentCfg,
    LightLPLocomotionEnvCfg,
    LightLPRewardCfg,
)


@configclass
class Z2SparseTeacherRewardCfg(LightLPRewardCfg):
    def __post_init__(self):
        bind_reward_roles(self, Z2_LOCOMOTION)


@configclass
class Z2LocoTeacherEnvCfg(LightLPLocomotionEnvCfg):
    robot_spec = Z2_LOCOMOTION

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = Z2_29DOF_WALK_POSE_DAMPED_PD_CFG.copy()
        self.scene.robot.init_state.pos = (0.0, 0.0, Z2_STANDING_PELVIS_Z)
        self.scene.foot_scanner.size = Z2_FOOT_SCAN_SIZE
        self.reward = Z2SparseTeacherRewardCfg()
        # Upstream 29DoF enables self-collision. Torso net force then includes
        # internal impacts, so it cannot serve as a ground-fall detector.
        self.robot.terminate_contacts_body_names = []
        self.robot.action_scale = 0.25
        self.robot.action_scale_effort_fraction = None
        self.scene.max_init_terrain_level = 0
        self.random_level_reset_fraction = 0.10
        self.random_level_reset_min_level = None
        self.random_level_reset_max_level = None
        self.sparse_command_min_speed_scale = 1.0
        # Shared LightLP default is path_length, which promotes in-place looping.
        # Use the repaired radial traversal contract in this recipe, not a G1 CLI flag.
        # Promotion >4 m from tile origin is independent of OOB 4.25 m and of reach2m.
        self.lightlp_promotion_distance = "max_radial"
        self.progress_monitor_enabled = True
        # Collapse height is not copied from G1 0.20 m. Isaac probe must measure
        # Z2 sitting clearance before enabling a relative-to-feet threshold.
        self.amp_terrain_schedule.enable = True
        self.commands.debug_vis = False
        self.domain_rand.action_delay.enable = False
        self.domain_rand.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.domain_rand.events.reset_base.params["velocity_range"] = {}


@configclass
class Z2LocoTeacherAgentCfg(LightLPLocomotionAgentCfg):
    experiment_name = "z2_loco_teacher_sparse"
    run_name = "z2_29dof_amp_damped_teacher_v1"
    max_iterations = 30000
    neptune_project = "z2_loco_teacher_sparse"
    wandb_project = "z2_loco_teacher_sparse"
    min_normalized_std = [0.05] * NUM_Z2_29DOF_JOINTS
    amp_frame_dim = AMP_FRAME_DIM
    amp_expert_dir = AMP_FORMAL_EXPERT_DIR
    amp_motion_files = amp_expert_files()
    amp_joint_order = list(Z2_29DOF_JOINT_NAMES)
