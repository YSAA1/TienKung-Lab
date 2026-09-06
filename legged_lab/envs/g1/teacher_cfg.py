"""G1 sparse/obstacle teacher: T4 LightLP 越障 MDP on Unitree G1 29-DoF."""

from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

import legged_lab.mdp as mdp
from legged_lab.assets.unitree_g1.constants import (
    G1_29DOF_JOINT_NAMES,
    G1_NOMINAL_FEET_Y_DISTANCE,
    NUM_G1_29DOF_JOINTS,
)
from legged_lab.assets.unitree_g1.g1 import G1_29DOF_CFG
from legged_lab.assets.unitree_g1.schemas import AMP_FORMAL_EXPERT_DIR as G1_AMP_EXPERT_DIR
from legged_lab.assets.unitree_g1.schemas import AMP_FRAME_DIM as G1_AMP_FRAME_DIM
from legged_lab.assets.unitree_g1.schemas import amp_expert_files as g1_amp_expert_files
from legged_lab.envs.t4.teacher_cfg import T4LocoSparseTeacherAgentCfg, T4LocoSparseTeacherEnvCfg, T4SparseTeacherRewardCfg


@configclass
class G1SparseTeacherRewardCfg(T4SparseTeacherRewardCfg):
    body_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=0.0
    )
    upright_orientation = RewTerm(
        func=mdp.upright_orientation, params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")}, weight=1.0
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_sensor",
                # unitree_rl_lab G1 29DoF: all non-ankle bodies, not a T4 Trunk subset.
                body_names=["(?!.*ankle.*).*"],
            ),
            "threshold": 1.0,
        },
    )
    shank_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_knee_link"]), "threshold": 1.0},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    feet_force = RewTerm(
        func=mdp.body_force,
        weight=-3e-3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link"),
            "threshold": 500,
            "max_reward": 400,
        },
    )
    feet_too_near = RewTerm(
        func=mdp.feet_too_near_humanoid,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]), "threshold": 0.2},
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=[".*_ankle_roll_link"])},
    )
    foot_touchdown_impact = RewTerm(
        func=mdp.foot_acceleration_penalty,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "tau_s": 0.06,
            "threshold_mps2": 30.0,
        },
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )
    joint_deviation_wrist_waist = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_wrist_.*_joint", "waist_.*_joint"])},
    )
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.02,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            )
        },
    )
    feet_y_distance = RewTerm(func=mdp.feet_y_distance, weight=-2.0, params={"target": G1_NOMINAL_FEET_Y_DISTANCE})
    illegal_footstep = RewTerm(
        func=mdp.illegal_footstep,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=".*_ankle_roll_link")},
    )


@configclass
class G1LocoTeacherEnvCfg(T4LocoSparseTeacherEnvCfg):
    """Same sparse 越障 teacher as T4, G1 plant, with LAFAN1 walk/run AMP."""

    enable_amp: bool = True
    policy_joint_names: tuple = G1_29DOF_JOINT_NAMES
    feet_link_names: list = ["left_ankle_roll_link", "right_ankle_roll_link"]
    diagnostic_contact_body_names: tuple = ("torso_link", "left_knee_link", "right_knee_link")
    left_leg_joint_names: list = [
        "left_hip_roll_joint",
        "left_hip_pitch_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
    ]
    right_leg_joint_names: list = [
        "right_hip_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
    ]
    ankle_joint_names: list = [
        "left_ankle_pitch_joint",
        "right_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_ankle_roll_joint",
    ]

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = G1_29DOF_CFG
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.foot_scanner.body_names = ("left_ankle_roll_link", "right_ankle_roll_link")
        # Keep LightLP task resets (timeout/oob/pit/accel/63°/torso contact).
        # Add unitree_rl_lab G1 0.2 m root-height vs the support foot so a sit
        # that never hits the chest capsule cannot farm episode length. Do not
        # replace 63° with unitree's 0.8 rad; do not clear torso contact.
        self.robot.terminate_contacts_body_names = ["torso_link"]
        self.collapse_reset_pelvis_above_feet_m = 0.20
        self.robot.feet_body_names = [".*_ankle_roll_link"]
        self.reward = G1SparseTeacherRewardCfg()
        self.enable_amp = True
        self.amp_terrain_schedule.enable = True
        self.commands.debug_vis = False
        self.domain_rand.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="torso_link")
        # Same as T4 sparse teacher that walked: delay/actuator jitter stay off.
        self.domain_rand.action_delay.enable = False


@configclass
class G1LocoTeacherAgentCfg(T4LocoSparseTeacherAgentCfg):
    experiment_name = "g1_loco_teacher_sparse"
    run_name = "g1_sparse_teacher_g1term"
    neptune_project = "g1_loco_teacher_sparse"
    wandb_project = "g1_loco_teacher_sparse"
    runner_class_name = "AmpOnPolicyRunner"
    save_interval = 500
    max_iterations = 40000
    min_normalized_std = [0.05] * NUM_G1_29DOF_JOINTS
    amp_frame_dim = G1_AMP_FRAME_DIM
    amp_expert_dir = G1_AMP_EXPERT_DIR
    amp_motion_files = g1_amp_expert_files()
    amp_joint_order = list(G1_29DOF_JOINT_NAMES)
    amp_reward_coef = 0.3
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func="legged_lab.envs.g1.symmetry:get_symmetric_states",
            mirror_loss_coeff=5.0,
        ),
        rnd_cfg=None,
    )
