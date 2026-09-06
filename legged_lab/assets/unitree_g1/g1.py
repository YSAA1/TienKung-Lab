"""IsaacLab configuration for Unitree G1 29-DoF mode 15.

Uses the Unitree ``g1_29dof_mode_15.urdf`` with collisions synced from
``xmls/g1_actuated.xml`` (not mesh collision). Not Isaac Lab bundled ``G1_CFG``.
Standing pose is the Unitree 29-DoF MIMIC keyframe.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .constants import G1_STANDING_JOINT_POS, G1_STANDING_PELVIS_Z

G1_ASSET_DIR = Path(__file__).resolve().parent
G1_URDF_PATH = G1_ASSET_DIR / "urdf" / "g1_29dof_mode_15.urdf"

G1_29DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(G1_URDF_PATH),
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        force_usd_conversion=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Self-collision + torso termination treats arm/torso hits as falls.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
            fix_root_link=False,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, G1_STANDING_PELVIS_Z),
        joint_pos=G1_STANDING_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 139.0,
                ".*_knee_joint": 139.0,
                "waist_yaw_joint": 88.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 20.0,
                ".*_knee_joint": 20.0,
                "waist_yaw_joint": 32.0,
            },
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "waist_yaw_joint": 200.0,
            },
            damping=5.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=150.0,
            damping=5.0,
            armature=0.01,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness={
                ".*_shoulder_.*": 40.0,
                ".*_elbow_joint": 40.0,
                ".*_wrist_roll_joint": 20.0,
            },
            damping=10.0,
            armature=0.01,
        ),
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=13.4,
            velocity_limit_sim=27.0,
            stiffness=20.0,
            damping=10.0,
            armature=0.01,
        ),
    },
)
