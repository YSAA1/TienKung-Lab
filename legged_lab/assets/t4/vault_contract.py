"""Pure-Python contract for the T4 1 m box-vault tracking task (G1).

Values are ported verbatim from the proven PHP vault recipe
(``whole_body_tracking.robots.t4_contract``) and must stay consistent with
``legged_lab/envs/t4/datasets/motion_tracking/_manifest.json``. This module
must stay importable without IsaacLab so the contract tests run anywhere.
"""

from __future__ import annotations

from pathlib import Path

_LEGGED_LAB_DIR = Path(__file__).resolve().parents[2]

T4_VAULT_MOTION_FILE = _LEGGED_LAB_DIR / "envs/t4/datasets/motion_tracking/overbox_1m_t4_mjcf_fps50.npz"

T4_VAULT_ANCHOR_BODY_NAME = "Trunk"
T4_VAULT_FOOT_BODY_NAMES = ("left_foot_link", "right_foot_link")
T4_VAULT_WRIST_BODY_NAMES = ("AL7", "AR7")
T4_VAULT_END_EFFECTOR_BODY_NAMES = T4_VAULT_FOOT_BODY_NAMES + T4_VAULT_WRIST_BODY_NAMES

T4_VAULT_TRACKING_BODY_NAMES = (
    "Trunk",
    "Hip_Roll_Left",
    "Shank_Left",
    "left_foot_link",
    "Hip_Roll_Right",
    "Shank_Right",
    "right_foot_link",
    "Waist_yaw",
    "AL2",
    "AL4",
    "AL7",
    "AR2",
    "AR4",
    "AR7",
)

T4_VAULT_BOX_SIZE = (1.0, 1.0, 1.0)
T4_VAULT_BOX_POS = (0.38, 0.2, 0.5)
T4_VAULT_BOX_ROT = (1.0, 0.0, 0.0, 0.0)

# Stage E locomotion plant contract: teacher and every derived policy share
# the uniform 0.25 rad position-target scale.
T4_VAULT_ACTION_SCALE = 0.25

T4_VAULT_ANCHOR_TERMINATION_THRESHOLD = 0.35
T4_VAULT_ANCHOR_ORI_TERMINATION_THRESHOLD = 0.8
T4_VAULT_FOOT_TERMINATION_THRESHOLD = 0.35
T4_VAULT_WRIST_TERMINATION_THRESHOLD = 0.50

T4_VAULT_ADAPTIVE_KERNEL_SIZE = 3
T4_VAULT_ADAPTIVE_KERNEL_LAMBDA = 0.8

# Play / eval start at the first reference frame with no RSI jitter. Training
# keeps the non-zero pose / joint / velocity ranges on MotionCommand.
T4_VAULT_PLAY_SAMPLING_STRATEGY = "zero"
T4_VAULT_PLAY_JOINT_POSITION_RANGE = (0.0, 0.0)
T4_VAULT_PLAY_POSE_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}
T4_VAULT_PLAY_VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}

__all__ = [name for name in globals() if name.startswith("T4_VAULT_")]
