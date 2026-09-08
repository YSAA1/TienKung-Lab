"""Unitree G1 29-DoF (mode 15) joint contract.

Actuated joints only. Order is the URDF appearance of revolute joints in
``urdf/g1_29dof_mode_15.urdf``.
"""

G1_29DOF_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

NUM_G1_29DOF_JOINTS = len(G1_29DOF_JOINT_NAMES)

# Legacy mode15 pose only; the active Unitree velocity pose is in official_velocity_g1.py.
# Unitree RL / ``xmls/g1_actuated.xml`` keyframe. Isaac Lab bundled G1_CFG uses a
# different robot (old 23-DoF-style USD) and a shallower walk pose that puts this
# URDF's 5 mm foot spheres ~18 mm below ground.
G1_STANDING_PELVIS_Z: float = 0.76
G1_STANDING_JOINT_POS: dict[str, float] = dict.fromkeys(G1_29DOF_JOINT_NAMES, 0.0)
G1_STANDING_JOINT_POS.update(
    {
        "left_hip_pitch_joint": -0.312,
        "right_hip_pitch_joint": -0.312,
        "left_knee_joint": 0.669,
        "right_knee_joint": 0.669,
        "left_ankle_pitch_joint": -0.363,
        "right_ankle_pitch_joint": -0.363,
        "left_shoulder_pitch_joint": 0.2,
        "right_shoulder_pitch_joint": 0.2,
        "left_shoulder_roll_joint": 0.2,
        "right_shoulder_roll_joint": -0.2,
        "left_elbow_joint": 0.6,
        "right_elbow_joint": 0.6,
    }
)

# Nominal lateral foot spacing in the default standing pose.
G1_NOMINAL_FEET_Y_DISTANCE: float = 0.22
