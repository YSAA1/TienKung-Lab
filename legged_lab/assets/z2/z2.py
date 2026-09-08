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

"""Z2 29-DoF Isaac Lab articulation.

Source: nubot-zhixing/z2-lab-stable-AMP ``legged_lab/assets/z2.py`` at
c78eb1f8e31b7f7872733110c10276b7b2159414. Actuator numbers are copied from
``_make_z2_cfg``. This file does not import another robot package.

The default ``Z2_29DOF_CFG`` is the generic 29DoF plant (ankle 75 Nm). The
registered 29DoF AMP tasks use ``Z2_29DOF_WALK_POSE_DAMPED_PD_CFG``. The
LightLP teacher selects the damped plant explicitly.

Training spawn is the original compiled USD (``UsdFileCfg``), matching upstream
``z2.py``. ``Z2UrdfFileCfg`` keeps archival converter flags only; it is not the
runtime plant. URDF remains for source/kinematics; MJCF remains an explicit adapter.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from legged_lab.assets.z2.constants import (
    Z2_29DOF_JOINT_NAMES,
    Z2_ASSET_DEFAULT_PELVIS_Z,
    Z2_STANDING_JOINT_POS,
)

ASSET_DIR = Path(__file__).resolve().parent
Z2_29DOF_URDF_PATH = ASSET_DIR / "urdf" / "assembly.urdf"
Z2_29DOF_USD_PATH = ASSET_DIR / "usd" / "assembly.usd"


@configclass
class Z2ArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] = None
    soft_joint_pos_limit_factor = 0.9


@configclass
class Z2UsdFileCfg(sim_utils.UsdFileCfg):
    """Runtime plant: original compiled 29DoF USD, same props as upstream z2.py."""

    activate_contact_sensors: bool = True
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


@configclass
class Z2UrdfFileCfg(sim_utils.UrdfFileCfg):
    """Archival converter flags from upstream usd/assembly_29dof/config.yaml.

    Not used for training spawn. Kept so the original import recipe remains
    auditable after switching the plant to the compiled USD.
    """

    fix_base: bool = False
    make_instanceable: bool = False
    force_usd_conversion: bool = False
    merge_fixed_joints: bool = True
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = False
    collider_type: str = "convex_hull"
    collision_from_visuals: bool = False
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=120.0, damping=3.0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


def _make_z2_cfg(
    *,
    ankle_effort_limit_sim: float = 75.0,
    ankle_velocity_limit_sim: float = 10.0,
    hip_pitch_knee_damping: float = 4.0,
    hip_yaw_stiffness: float = 60.0,
    hip_yaw_damping: float = 2.5,
    ankle_stiffness: float = 60.0,
    ankle_damping: float = 2.0,
) -> Z2ArticulationCfg:
    actuators = {
        "Z2-waist-yaw": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint"],
            effort_limit_sim=90.0,
            velocity_limit_sim=16.44,
            stiffness=200.0,
            damping=5.0,
            armature=0.01,
        ),
        "Z2-waist-pitch-roll": ImplicitActuatorCfg(
            joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
            effort_limit_sim=75.0,
            velocity_limit_sim=12.25,
            stiffness=80.0,
            damping=4.0,
            armature=0.01,
        ),
        "Z2-hip-pitch-knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_knee_joint"],
            effort_limit_sim=130.0,
            velocity_limit_sim=14.66,
            stiffness={
                ".*_hip_pitch_joint": 120.0,
                ".*_knee_joint": 120.0,
            },
            damping={
                ".*_hip_pitch_joint": hip_pitch_knee_damping,
                ".*_knee_joint": hip_pitch_knee_damping,
            },
            armature=0.01,
        ),
        "Z2-hip-roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint"],
            effort_limit_sim=132.0,
            velocity_limit_sim=12.46,
            stiffness=120.0,
            damping=4.0,
            armature=0.01,
        ),
        "Z2-hip-yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint"],
            effort_limit_sim=70.0,
            velocity_limit_sim=13.40,
            stiffness=hip_yaw_stiffness,
            damping=hip_yaw_damping,
            armature=0.01,
        ),
        "Z2-ankle": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_.*_joint"],
            effort_limit_sim=ankle_effort_limit_sim,
            velocity_limit_sim=ankle_velocity_limit_sim,
            stiffness=ankle_stiffness,
            damping=ankle_damping,
            armature=0.01,
        ),
        "Z2-shoulder-pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint"],
            effort_limit_sim=70.0,
            velocity_limit_sim=13.40,
            stiffness=50.0,
            damping=1.5,
            armature=0.01,
        ),
        "Z2-shoulder-roll-yaw-elbow": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"],
            effort_limit_sim=36.0,
            velocity_limit_sim=9.32,
            stiffness=40.0,
            damping=1.0,
            armature=0.01,
        ),
        "Z2-wrist-roll": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_roll_joint"],
            effort_limit_sim=36.0,
            velocity_limit_sim=9.32,
            stiffness=40,
            damping=1,
            armature=0.01,
        ),
        "Z2-wrist-pitch-yaw": ImplicitActuatorCfg(
            joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
            effort_limit_sim=12.0,
            velocity_limit_sim=23.04,
            stiffness=40,
            damping=1,
            armature=0.01,
        ),
    }
    standing = {name: value for name, value in Z2_STANDING_JOINT_POS.items() if value != 0.0}
    return Z2ArticulationCfg(
        spawn=Z2UsdFileCfg(usd_path=str(Z2_29DOF_USD_PATH)),
        init_state=Z2ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, Z2_ASSET_DEFAULT_PELVIS_Z),
            joint_pos={
                ".*_hip_pitch_joint": standing["L_hip_pitch_joint"],
                ".*_knee_joint": standing["L_knee_joint"],
                ".*_ankle_pitch_joint": standing["L_ankle_pitch_joint"],
                ".*_shoulder_pitch_joint": standing["L_shoulder_pitch_joint"],
                "L_shoulder_roll_joint": standing["L_shoulder_roll_joint"],
                "R_shoulder_roll_joint": standing["R_shoulder_roll_joint"],
                ".*_elbow_joint": standing["L_elbow_joint"],
            },
            joint_vel={".*": 0.0},
        ),
        actuators=actuators,
        joint_sdk_names=list(Z2_29DOF_JOINT_NAMES),
    )


Z2_29DOF_CFG = _make_z2_cfg()
Z2_29DOF_RELAXED_ANKLE_CFG = _make_z2_cfg(ankle_effort_limit_sim=150.0, ankle_velocity_limit_sim=12.0)
Z2_29DOF_STRICT_ACTION_RATE_PD_CFG = _make_z2_cfg(
    ankle_effort_limit_sim=150.0,
    ankle_velocity_limit_sim=12.0,
    hip_yaw_stiffness=75.0,
    hip_yaw_damping=3.5,
    ankle_stiffness=50.0,
    ankle_damping=3.0,
)
Z2_29DOF_WALK_POSE_DAMPED_PD_CFG = _make_z2_cfg(
    ankle_effort_limit_sim=150.0,
    ankle_velocity_limit_sim=12.0,
    hip_pitch_knee_damping=6.0,
    hip_yaw_stiffness=75.0,
    hip_yaw_damping=3.5,
    ankle_stiffness=50.0,
    ankle_damping=4.0,
)

Z2_CFG = Z2_29DOF_CFG
Z2_NUM_ACTIONS = len(Z2_29DOF_JOINT_NAMES)
Z2_29DOF_TEACHER_CFG = Z2_29DOF_WALK_POSE_DAMPED_PD_CFG
