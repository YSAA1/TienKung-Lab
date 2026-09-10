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

"""Z2 29-DoF joint contract copied from upstream ``legged_lab/assets/z2.py``.

Policy order is ``Z2_29DOF_JOINT_NAMES`` (the upstream ``joint_sdk_names`` list).
It is not URDF appearance order and not the AMP visualization limb order.
Fixed neck and sphere-hand joints are not policy DOF.
"""

from __future__ import annotations

# Upstream z2.py::Z2_29DOF_JOINT_NAMES / joint_sdk_names. 29 actuated joints.
Z2_29DOF_JOINT_NAMES: tuple[str, ...] = (
    "L_hip_pitch_joint",
    "R_hip_pitch_joint",
    "waist_yaw_joint",
    "L_hip_roll_joint",
    "R_hip_roll_joint",
    "waist_pitch_joint",
    "L_hip_yaw_joint",
    "R_hip_yaw_joint",
    "waist_roll_joint",
    "L_knee_joint",
    "R_knee_joint",
    "L_shoulder_pitch_joint",
    "R_shoulder_pitch_joint",
    "L_ankle_pitch_joint",
    "R_ankle_pitch_joint",
    "L_shoulder_roll_joint",
    "R_shoulder_roll_joint",
    "L_ankle_roll_joint",
    "R_ankle_roll_joint",
    "L_shoulder_yaw_joint",
    "R_shoulder_yaw_joint",
    "L_elbow_joint",
    "R_elbow_joint",
    "L_wrist_roll_joint",
    "R_wrist_roll_joint",
    "L_wrist_pitch_joint",
    "R_wrist_pitch_joint",
    "L_wrist_yaw_joint",
    "R_wrist_yaw_joint",
)

NUM_Z2_29DOF_JOINTS = len(Z2_29DOF_JOINT_NAMES)

# URDF revolute appearance in assembly_urdf_29/assembly.urdf. Isaac load order
# is expected to follow this list; the shared env remaps policy order onto it.
Z2_URDF_REVOLUTE_JOINT_NAMES: tuple[str, ...] = (
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_joint",
    "R_wrist_roll_joint",
    "R_wrist_pitch_joint",
    "R_wrist_yaw_joint",
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_joint",
    "L_wrist_roll_joint",
    "L_wrist_pitch_joint",
    "L_wrist_yaw_joint",
    "R_hip_pitch_joint",
    "R_hip_roll_joint",
    "R_hip_yaw_joint",
    "R_knee_joint",
    "R_ankle_pitch_joint",
    "R_ankle_roll_joint",
    "L_hip_pitch_joint",
    "L_hip_roll_joint",
    "L_hip_yaw_joint",
    "L_knee_joint",
    "L_ankle_pitch_joint",
    "L_ankle_roll_joint",
)

# convert_z2_amp_data.py::SOURCE_JOINT_ORDER_FALLBACK. PKL/NPZ qpos after the
# free-root 7 uses this order when link_body_list is missing.
Z2_SOURCE_JOINT_ORDER: tuple[str, ...] = Z2_URDF_REVOLUTE_JOINT_NAMES

# convert_z2_amp_data.py::AMP_JOINT_ORDER. Visualization / upstream 64D AMP
# concatenation order (legs, arms+wrists, waist yaw/roll/pitch). Not policy order.
Z2_UPSTREAM_AMP_JOINT_ORDER: tuple[str, ...] = (
    "L_hip_pitch_joint",
    "L_hip_roll_joint",
    "L_hip_yaw_joint",
    "L_knee_joint",
    "L_ankle_pitch_joint",
    "L_ankle_roll_joint",
    "R_hip_pitch_joint",
    "R_hip_roll_joint",
    "R_hip_yaw_joint",
    "R_knee_joint",
    "R_ankle_pitch_joint",
    "R_ankle_roll_joint",
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_joint",
    "L_wrist_roll_joint",
    "L_wrist_pitch_joint",
    "L_wrist_yaw_joint",
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_joint",
    "R_wrist_roll_joint",
    "R_wrist_pitch_joint",
    "R_wrist_yaw_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

Z2_FIXED_JOINT_NAMES: tuple[str, ...] = (
    "neck_fixed",
    "R_sphere_hand_joint",
    "L_sphere_hand_joint",
)

# Asset ArticulationCfg default is z=0.8. Every registered 29DoF env then
# overwrites init_state.pos z with physical.base_link_height=0.75.
Z2_ASSET_DEFAULT_PELVIS_Z: float = 0.8
Z2_STANDING_PELVIS_Z: float = 0.75

Z2_STANDING_JOINT_POS: dict[str, float] = dict.fromkeys(Z2_29DOF_JOINT_NAMES, 0.0)
Z2_STANDING_JOINT_POS.update(
    {
        "L_hip_pitch_joint": -0.2,
        "R_hip_pitch_joint": -0.2,
        "L_knee_joint": 0.42,
        "R_knee_joint": 0.42,
        "L_ankle_pitch_joint": -0.23,
        "R_ankle_pitch_joint": -0.23,
        "L_shoulder_pitch_joint": 0.35,
        "R_shoulder_pitch_joint": 0.35,
        "L_shoulder_roll_joint": 0.18,
        "R_shoulder_roll_joint": -0.18,
        "L_elbow_joint": 0.87,
        "R_elbow_joint": 0.87,
    }
)

# Independent standing URDF FK of ankle-roll origins: 0.211715903 m (was rounded 0.2114).
# Upstream Z2PhysicalParamsCfg.foot_parallel_distance is 0.22 m (reward target).
# Spec uses the measured origin spacing, not G1's 0.22 and not the reward field.
Z2_NOMINAL_FEET_Y_DISTANCE: float = 0.211715903

# Foot mesh AABB in the ankle-roll frame (L_ankle_roll_link.STL).
Z2_FOOT_MESH_AABB_LENGTH_M: float = 0.2205
Z2_FOOT_MESH_AABB_WIDTH_M: float = 0.0776
Z2_FOOT_SCAN_SIZE: tuple[float, float] = (0.24, 0.08)

Z2_URDF_TOTAL_MASS_KG: float = 36.690941

# Training plant is the original compiled USD. Layers are hashed for provenance.
Z2_USD_LAYER_FILES: tuple[str, ...] = (
    "usd/assembly.usd",
    "usd/configuration/assembly_base.usd",
    "usd/configuration/assembly_physics.usd",
    "usd/configuration/assembly_robot.usd",
    "usd/configuration/assembly_sensor.usd",
)
Z2_UPSTREAM_USD_DIR: str = "legged_lab/assets/z2_description/usd/assembly_29dof"

# Documented 20/23 DoF policy lists from the same upstream z2.py. Not used by
# the 29DoF teacher. Kept so tests can prove we did not mix variants.
Z2_23DOF_JOINT_NAMES: tuple[str, ...] = Z2_29DOF_JOINT_NAMES[:23]
Z2_20DOF_JOINT_NAMES: tuple[str, ...] = (
    "R_shoulder_pitch_joint",
    "R_shoulder_roll_joint",
    "R_shoulder_yaw_joint",
    "R_elbow_joint",
    "L_shoulder_pitch_joint",
    "L_shoulder_roll_joint",
    "L_shoulder_yaw_joint",
    "L_elbow_joint",
    "R_hip_pitch_joint",
    "R_hip_roll_joint",
    "R_hip_yaw_joint",
    "R_knee_joint",
    "R_ankle_pitch_joint",
    "R_ankle_roll_joint",
    "L_hip_pitch_joint",
    "L_hip_roll_joint",
    "L_hip_yaw_joint",
    "L_knee_joint",
    "L_ankle_pitch_joint",
    "L_ankle_roll_joint",
)
