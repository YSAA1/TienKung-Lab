"""G1 AMP observation contract.

Expert clips come from the public LAFAN1 G1 retarget
(``lvhaidong/LAFAN1_Retargeting_Dataset``, 30 FPS CSV:
``root_xyz + root_quat_xyzw + q29``). Joint order matches
``G1_29DOF_JOINT_NAMES``. Dance / fight / fall / jump clips are not part of this
locomotion mixture.

The AMP frame is ``q29 + dq29 + hands_root6 + feet_root6`` (70D). Hands and feet
are body origins of the wrist-yaw and ankle-roll links in the root frame. Expert
generation and the training env must use :class:`G1AmpFeatureBuilder`.
"""

from __future__ import annotations

from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES, NUM_G1_29DOF_JOINTS

AMP_SCHEMA_VERSION = "g1_amp.v1"

AMP_FIELDS: tuple[tuple[str, int], ...] = (
    ("joint_pos", NUM_G1_29DOF_JOINTS),
    ("joint_vel", NUM_G1_29DOF_JOINTS),
    ("hand_pos_root", 6),
    ("foot_pos_root", 6),
)

AMP_FRAME_DIM = sum(width for _, width in AMP_FIELDS)
AMP_TRANSITION_DIM = 2 * AMP_FRAME_DIM

# Wrist-yaw is the last actuated hand body after URDF ``merge_fixed_joints``
# folds the rubber-hand geoms into it. Ankle-roll is the G1 foot body.
AMP_HAND_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
AMP_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
AMP_HAND_SITE_OFFSET = (0.0, 0.0, 0.0)
AMP_FOOT_SITE_OFFSET = (0.0, 0.0, 0.0)

LAFAN1_SOURCE_DATASET = "lvhaidong/LAFAN1_Retargeting_Dataset"
LAFAN1_SOURCE_FPS = 30.0
LAFAN1_CSV_WIDTH = 7 + NUM_G1_29DOF_JOINTS
LAFAN1_G1_JOINT_NAMES = G1_29DOF_JOINT_NAMES

AMP_MOTION_CLASS_WEIGHTS: dict[str, float] = {
    "walk_forward": 1.0,
    "run": 0.6,
}

AMP_MOTION_CLASSES: dict[str, str] = {
    "walk1_subject1": "walk_forward",
    "walk2_subject1": "walk_forward",
    "walk3_subject1": "walk_forward",
    "walk4_subject1": "walk_forward",
    "run1_subject2": "run",
    "run2_subject1": "run",
}

AMP_HELD_OUT_MOTIONS: tuple[str, ...] = ("sprint1_subject2",)

AMP_FORMAL_EXPERT_DIR = "legged_lab/envs/g1/datasets/motion_amp_expert_unitree_v5"
AMP_MOTION_SOURCE_DIR = "legged_lab/envs/g1/datasets/motion_source"


def amp_field_slice(name: str) -> tuple[int, int]:
    """Return the ``[start, end)`` column range of one AMP field."""
    start = 0
    for field_name, width in AMP_FIELDS:
        if field_name == name:
            return start, start + width
        start += width
    raise KeyError(f"unknown AMP field {name!r}; known={[field for field, _ in AMP_FIELDS]}")


def amp_motion_class(motion_stem: str) -> str:
    """Return the declared behaviour class of a motion stem."""
    try:
        return AMP_MOTION_CLASSES[motion_stem]
    except KeyError as error:
        raise KeyError(
            f"motion {motion_stem!r} has no declared G1 AMP behaviour class; add it to AMP_MOTION_CLASSES "
            "or hold it out explicitly"
        ) from error


def amp_motion_weight(motion_stem: str) -> float:
    """Return the per-file sampling weight implied by the explicit class weights."""
    motion_class = amp_motion_class(motion_stem)
    class_members = [stem for stem, name in AMP_MOTION_CLASSES.items() if name == motion_class]
    return AMP_MOTION_CLASS_WEIGHTS[motion_class] / len(class_members)


def amp_expert_files(expert_dir: str = AMP_FORMAL_EXPERT_DIR) -> list[str]:
    """Return the AMP expert files implied by the declared motion classes."""
    stems = sorted(stem for stem in AMP_MOTION_CLASSES if stem not in AMP_HELD_OUT_MOTIONS)
    return [f"{expert_dir}/{stem}.txt" for stem in stems]
