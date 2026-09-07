"""G1 morphology adapter; it has no dependency on another robot."""

from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES, G1_NOMINAL_FEET_Y_DISTANCE
from legged_lab.assets.unitree_g1.schemas import (
    AMP_FOOT_BODIES, AMP_FOOT_SITE_OFFSET, AMP_HAND_BODIES, AMP_HAND_SITE_OFFSET,
)
from legged_lab.locomotion.robot_spec import LocomotionRobotSpec


def _mirror():
    indices, signs = [], []
    for name in G1_29DOF_JOINT_NAMES:
        other = ("right_" + name[5:] if name.startswith("left_")
                 else "left_" + name[6:] if name.startswith("right_") else name)
        indices.append(G1_29DOF_JOINT_NAMES.index(other))
        if name.endswith(("_pitch_joint", "_knee_joint", "_elbow_joint")):
            signs.append(1.0)
        elif name.endswith(("_roll_joint", "_yaw_joint")):
            signs.append(-1.0)
        else:
            raise ValueError(f"undeclared G1 mirror axis for {name}")
    return tuple(indices), tuple(signs)


G1_LOCOMOTION = LocomotionRobotSpec(
    name="g1", joint_names=G1_29DOF_JOINT_NAMES,
    auxiliary_joint_names=tuple(
        f"{side}_hand_{finger}_{index}_joint"
        for side in ("left", "right")
        for finger, count in (("index", 2), ("middle", 2), ("thumb", 3))
        for index in range(count)
    ),
    feet=AMP_FOOT_BODIES, hands=AMP_HAND_BODIES, torso="torso_link",
    diagnostic_bodies=("torso_link", "left_knee_link", "right_knee_link"),
    left_leg=("left_hip_roll_joint", "left_hip_pitch_joint", "left_hip_yaw_joint", "left_knee_joint",
              "left_ankle_pitch_joint", "left_ankle_roll_joint"),
    right_leg=("right_hip_roll_joint", "right_hip_pitch_joint", "right_hip_yaw_joint", "right_knee_joint",
               "right_ankle_pitch_joint", "right_ankle_roll_joint"),
    ankles=("left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint"),
    hand_site_offset=AMP_HAND_SITE_OFFSET, foot_site_offset=AMP_FOOT_SITE_OFFSET,
    mirror_indices=_mirror()[0], mirror_signs=_mirror()[1], nominal_feet_distance=G1_NOMINAL_FEET_Y_DISTANCE,
    reward_bodies={
        "undesired": ("(?!.*ankle.*).*",), "sparse_undesired": ("(?!.*ankle.*).*",),
        "shanks": (".*_knee_link",),
    },
    reward_joints={
        "hip": (".*_hip_yaw_joint", ".*_hip_roll_joint"),
        "arms": (".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint"),
        "wrist_waist": (".*_wrist_.*_joint", "waist_.*_joint"),
        "legs": (".*_hip_pitch_joint", ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    },
)
G1_LOCOMOTION.validate()
