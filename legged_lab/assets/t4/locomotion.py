"""T4 morphology adapter for the shared locomotion algorithms."""

from legged_lab.assets.t4.constants import T4_JOINT_NAMES, T4_NOMINAL_FEET_Y_DISTANCE
from legged_lab.assets.t4.schemas import (
    AMP_FOOT_BODIES, AMP_FOOT_SITE_OFFSET, AMP_HAND_BODIES, AMP_HAND_SITE_OFFSET,
)
from legged_lab.locomotion.robot_spec import LocomotionRobotSpec


def _mirror():
    arm_signs = {"01": 1.0, "02": -1.0, "03": -1.0, "04": 1.0, "05": -1.0, "06": 1.0, "07": -1.0}
    indices, signs = [], []
    for name in T4_JOINT_NAMES:
        other = name.replace("_l_", "_r_") if "_l_" in name else name.replace("_r_", "_l_")
        indices.append(T4_JOINT_NAMES.index(other))
        signs.append(arm_signs[name.rsplit("_", 1)[1]] if name.startswith("J_arm_")
                     else 1.0 if name.endswith("_pitch") else -1.0)
    return tuple(indices), tuple(signs)


T4_LOCOMOTION = LocomotionRobotSpec(
    name="t4", joint_names=T4_JOINT_NAMES,
    feet=AMP_FOOT_BODIES, hands=AMP_HAND_BODIES, torso="Trunk",
    diagnostic_bodies=("Trunk", "Shank_Left", "Shank_Right"),
    left_leg=("J_hip_l_roll", "J_hip_l_pitch", "J_hip_l_yaw", "J_knee_l_pitch", "J_ankle_l_pitch", "J_ankle_l_roll"),
    right_leg=("J_hip_r_roll", "J_hip_r_pitch", "J_hip_r_yaw", "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll"),
    ankles=("J_ankle_l_pitch", "J_ankle_r_pitch", "J_ankle_l_roll", "J_ankle_r_roll"),
    hand_site_offset=AMP_HAND_SITE_OFFSET, foot_site_offset=AMP_FOOT_SITE_OFFSET,
    mirror_indices=_mirror()[0], mirror_signs=_mirror()[1], nominal_feet_distance=T4_NOMINAL_FEET_Y_DISTANCE,
    reward_bodies={
        "undesired": ("A[LR]2", "A[LR]4", "Trunk"),
        "sparse_undesired": ("A[LR]2", "A[LR]4", "Trunk", "Shank_.*"),
        "shanks": ("Shank_.*",),
    },
    reward_joints={
        "hip": ("J_hip_[lr]_yaw", "J_hip_[lr]_roll", "J_arm_[lr]_01", "J_arm_[lr]_04"),
        "arms": ("J_arm_[lr]_02", "J_arm_[lr]_03"),
        "wrist_waist": ("J_arm_[lr]_0[567]", "J_waist_yaw"),
        "legs": ("J_hip_[lr]_pitch", "J_knee_[lr]_pitch", "J_ankle_[lr]_pitch", "J_ankle_[lr]_roll"),
    },
)
T4_LOCOMOTION.validate()
