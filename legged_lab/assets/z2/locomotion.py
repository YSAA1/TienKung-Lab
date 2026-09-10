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

"""Z2 morphology adapter; it has no dependency on another robot."""

from legged_lab.assets.z2.constants import (
    Z2_29DOF_JOINT_NAMES,
    Z2_NOMINAL_FEET_Y_DISTANCE,
)
from legged_lab.assets.z2.schemas import (
    AMP_FOOT_BODIES,
    AMP_FOOT_SITE_OFFSET,
    AMP_HAND_BODIES,
    AMP_HAND_SITE_OFFSET,
)
from legged_lab.locomotion.robot_spec import LocomotionRobotSpec


def _mirror():
    indices, signs = [], []
    for name in Z2_29DOF_JOINT_NAMES:
        if name.startswith("L_"):
            other = "R_" + name[2:]
        elif name.startswith("R_"):
            other = "L_" + name[2:]
        else:
            other = name
        indices.append(Z2_29DOF_JOINT_NAMES.index(other))
        if name.endswith(("_pitch_joint", "_knee_joint", "_elbow_joint")):
            signs.append(1.0)
        elif name.endswith(("_roll_joint", "_yaw_joint")):
            signs.append(-1.0)
        else:
            raise ValueError(f"undeclared Z2 mirror axis for {name}")
    return tuple(indices), tuple(signs)


Z2_LOCOMOTION = LocomotionRobotSpec(
    name="z2",
    joint_names=Z2_29DOF_JOINT_NAMES,
    feet=AMP_FOOT_BODIES,
    hands=AMP_HAND_BODIES,
    torso="waist_roll_link",
    diagnostic_bodies=("waist_roll_link", "L_knee_link", "R_knee_link"),
    left_leg=(
        "L_hip_roll_joint",
        "L_hip_pitch_joint",
        "L_hip_yaw_joint",
        "L_knee_joint",
        "L_ankle_pitch_joint",
        "L_ankle_roll_joint",
    ),
    right_leg=(
        "R_hip_roll_joint",
        "R_hip_pitch_joint",
        "R_hip_yaw_joint",
        "R_knee_joint",
        "R_ankle_pitch_joint",
        "R_ankle_roll_joint",
    ),
    ankles=("L_ankle_pitch_joint", "R_ankle_pitch_joint", "L_ankle_roll_joint", "R_ankle_roll_joint"),
    hand_site_offset=AMP_HAND_SITE_OFFSET,
    foot_site_offset=AMP_FOOT_SITE_OFFSET,
    mirror_indices=_mirror()[0],
    mirror_signs=_mirror()[1],
    nominal_feet_distance=Z2_NOMINAL_FEET_Y_DISTANCE,
    reward_bodies={
        "undesired": ("(?!.*ankle.*).*",),
        "sparse_undesired": ("(?!.*ankle.*).*",),
        "shanks": (".*_knee_link",),
    },
    reward_joints={
        "hip": (".*_hip_yaw_joint", ".*_hip_roll_joint"),
        "arms": (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"),
        "wrist_waist": (".*_wrist_.*_joint", "waist_.*_joint"),
        "legs": (".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    },
)
Z2_LOCOMOTION.validate()
