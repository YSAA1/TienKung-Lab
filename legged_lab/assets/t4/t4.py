"""IsaacLab configuration for the T4 27-DoF humanoid."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .constants import T4_JOINT_NAMES

T4_ASSET_DIR = Path(__file__).resolve().parent
T4_URDF_PATH = T4_ASSET_DIR / "urdf" / "t4_std.urdf"

T4_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(T4_URDF_PATH),
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
        pos=(0.0, 0.0, 0.85),
        joint_pos={
            ".*": 0.0,
            "J_arm_l_01": 0.20,
            "J_arm_l_02": 0.13,
            "J_arm_l_04": -0.43,
            "J_arm_r_01": 0.20,
            "J_arm_r_02": -0.13,
            "J_arm_r_04": -0.43,
            "J_hip_l_pitch": -0.20,
            "J_knee_l_pitch": 0.42,
            "J_ankle_l_pitch": -0.24,
            "J_hip_r_pitch": -0.20,
            "J_knee_r_pitch": 0.42,
            "J_ankle_r_pitch": -0.24,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["J_arm_[lr]_0[1-7]"],
            effort_limit_sim={"J_arm_[lr]_0[1-4]": 36.0, "J_arm_[lr]_0[5-7]": 12.0},
            velocity_limit_sim=18.84,
            stiffness={"J_arm_[lr]_0[1-5]": 20.0, "J_arm_[lr]_0[6-7]": 10.0},
            damping=1.0,
            armature={"J_arm_[lr]_0[1-5]": 0.0236, "J_arm_[lr]_0[6-7]": 0.0055},
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["J_waist_yaw"],
            effort_limit_sim=120.0,
            velocity_limit_sim=10.88,
            stiffness=50.0,
            damping=2.0,
            armature=0.0943,
        ),
        "hips": ImplicitActuatorCfg(
            joint_names_expr=["J_hip_[lr]_(pitch|roll|yaw)"],
            effort_limit_sim={"J_hip_[lr]_pitch": 130.0, "J_hip_[lr]_(roll|yaw)": 120.0},
            velocity_limit_sim={"J_hip_[lr]_pitch": 12.5, "J_hip_[lr]_(roll|yaw)": 10.9},
            stiffness={"J_hip_[lr]_pitch": 100.0, "J_hip_[lr]_(roll|yaw)": 50.0},
            damping={"J_hip_[lr]_pitch": 4.0, "J_hip_[lr]_(roll|yaw)": 2.0},
            armature={"J_hip_[lr]_pitch": 0.0625, "J_hip_[lr]_(roll|yaw)": 0.0943},
        ),
        "knees": ImplicitActuatorCfg(
            joint_names_expr=["J_knee_[lr]_pitch"],
            effort_limit_sim=130.0,
            velocity_limit_sim=11.7,
            stiffness=100.0,
            damping=4.0,
            armature=0.0625,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=["J_ankle_[lr]_(pitch|roll)"],
            effort_limit_sim=72.0,
            velocity_limit_sim={"J_ankle_[lr]_pitch": 18.8, "J_ankle_[lr]_roll": 12.4},
            stiffness=10.0,
            damping=0.5,
            armature=0.0472,
        ),
    },
)
