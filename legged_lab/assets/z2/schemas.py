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

"""Z2 AMP observation contract.

Upstream 29DoF AMP runtime is ``q29 + dq29 + wrists_root6`` (64D, no feet,
``endpoint_mode=no_feet``). This project's shared ``AmpFeatureBuilder`` is
``qN + dqN + hands_root6 + feet_root6``. Z2 training experts are regenerated
into the shared 70D schema and must not reuse G1 70D tensors or upstream 64D
files.

Source clips are Z2 PKL/NPZ with ``root_xyz + root_quat_xyzw + q29`` after
conversion. Source joint order is ``Z2_SOURCE_JOINT_ORDER``; policy/AMP joint
order is ``Z2_29DOF_JOINT_NAMES``.
"""

from __future__ import annotations

from legged_lab.assets.z2.constants import NUM_Z2_29DOF_JOINTS, Z2_29DOF_JOINT_NAMES

AMP_SCHEMA_VERSION = "z2_amp.v1"

AMP_FIELDS: tuple[tuple[str, int], ...] = (
    ("joint_pos", NUM_Z2_29DOF_JOINTS),
    ("joint_vel", NUM_Z2_29DOF_JOINTS),
    ("hand_pos_root", 6),
    ("foot_pos_root", 6),
)

AMP_FRAME_DIM = sum(width for _, width in AMP_FIELDS)
AMP_TRANSITION_DIM = 2 * AMP_FRAME_DIM

# Wrist-yaw is the last actuated hand body. Sphere hands are fixed children and
# are already merged in the original compiled USD.
AMP_HAND_BODIES = ("L_wrist_yaw_link", "R_wrist_yaw_link")
AMP_FOOT_BODIES = ("L_ankle_roll_link", "R_ankle_roll_link")
# Upstream 29DoF AMP uses wrist/ankle body origins, not sphere-hand or sole sites.
AMP_HAND_SITE_OFFSET = (0.0, 0.0, 0.0)
AMP_FOOT_SITE_OFFSET = (0.0, 0.0, 0.0)

Z2_SOURCE_DATASET = "nubot-zhixing/z2-lab-stable-AMP@c78eb1f8e31b7f7872733110c10276b7b2159414"
Z2_SOURCE_CSV_WIDTH = 7 + NUM_Z2_29DOF_JOINTS
Z2_SOURCE_JOINT_NAMES = Z2_29DOF_JOINT_NAMES

# Equal per-file weights match upstream MotionWeight=0.5 on each clip. Do not
# copy G1 class weights.
AMP_MOTION_CLASS_WEIGHTS: dict[str, float] = {
    "walk_forward": 1.0,
    "run": 0.5,
}

AMP_MOTION_CLASSES: dict[str, str] = {
    "walk": "walk_forward",
    "walk_l": "walk_forward",
    "run": "run",
    # 2026-09-12 promoted out of the hold-out list: the v1 set turned out to
    # contain two in-place clips (``run``/``walk`` move < 0.02 m/s net); these
    # three are the clean forward-motion replacement members of curated v2.
    "run2": "run",
    "run_l": "run",
    "run_140_l": "run",
}

AMP_HELD_OUT_MOTIONS: tuple[str, ...] = (
    "walk1_subject1",
    "run1_subject2",
    "walk_l_unrepaired",
)

# Formal dataset versions. v1 keeps the migration lineage (including the two
# in-place clips); curated v2 drops them and is the forward-motion set.
AMP_FORMAL_V1_STEMS: tuple[str, ...] = ("run", "walk", "walk_l")
AMP_CURATED_V2_STEMS: tuple[str, ...] = ("run2", "run_l", "run_140_l", "walk_l")

AMP_FORMAL_EXPERT_DIR = "legged_lab/envs/z2/datasets/motion_amp_expert"
AMP_MOTION_SOURCE_DIR = "legged_lab/envs/z2/datasets/motion_source"
AMP_MOTION_SOURCE_RAW_DIR = "legged_lab/envs/z2/datasets/motion_source_raw"
AMP_CURATED_V2_SOURCE_DIR = "legged_lab/envs/z2/datasets/motion_source_z2_v2"
AMP_CURATED_V2_EXPERT_DIR = "legged_lab/envs/z2/datasets/motion_amp_expert_z2_v2"

UPSTREAM_64D_EXPERT_WIDTH = 64
UPSTREAM_VISUALIZATION_WIDTH = 70


def amp_field_slice(name: str) -> tuple[int, int]:
    start = 0
    for field_name, width in AMP_FIELDS:
        if field_name == name:
            return start, start + width
        start += width
    raise KeyError(f"unknown AMP field {name!r}; known={[field for field, _ in AMP_FIELDS]}")


def amp_motion_class(motion_stem: str) -> str:
    try:
        return AMP_MOTION_CLASSES[motion_stem]
    except KeyError as error:
        raise KeyError(
            f"motion {motion_stem!r} has no declared Z2 AMP behaviour class; add it to AMP_MOTION_CLASSES "
            "or hold it out explicitly"
        ) from error


def amp_motion_weight(motion_stem: str, stems: tuple[str, ...] = AMP_FORMAL_V1_STEMS) -> float:
    motion_class = amp_motion_class(motion_stem)
    class_members = [stem for stem in stems if AMP_MOTION_CLASSES[stem] == motion_class]
    return AMP_MOTION_CLASS_WEIGHTS[motion_class] / len(class_members)


def amp_expert_files(
    expert_dir: str = AMP_FORMAL_EXPERT_DIR, stems: tuple[str, ...] = AMP_FORMAL_V1_STEMS
) -> list[str]:
    return [f"{expert_dir}/{stem}.txt" for stem in sorted(stems)]
