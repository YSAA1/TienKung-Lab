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

"""Isaac-free URDF FK preview of the shared 70D Z2 AMP frame.

This is not a training expert. Isaac ``generate_z2_amp_expert.py`` remains the
only path that writes ``motion_amp_expert/*.txt``. Offsets are body origins,
matching upstream 29DoF wrist/ankle usage and this project's site offsets of 0.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from legged_lab.assets.z2.constants import (
    NUM_Z2_29DOF_JOINTS,
    Z2_29DOF_JOINT_NAMES,
    Z2_STANDING_JOINT_POS,
    Z2_STANDING_PELVIS_Z,
)
from legged_lab.assets.z2.schemas import AMP_FOOT_BODIES, AMP_FRAME_DIM, AMP_HAND_BODIES

ASSET_DIR = Path(__file__).resolve().parent
URDF_PATH = ASSET_DIR / "urdf" / "assembly.urdf"


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis / (np.linalg.norm(axis) or 1.0)
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def _quat_xyzw_to_R(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _parse_floats(text: str | None, default: str = "0 0 0") -> np.ndarray:
    return np.array([float(v) for v in (text or default).split()], dtype=np.float64)


def load_urdf_joints(path: Path = URDF_PATH):
    root = ET.parse(path).getroot()
    joints = []
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints.append(
            {
                "name": joint.attrib["name"],
                "type": joint.attrib.get("type"),
                "parent": joint.find("parent").attrib["link"],
                "child": joint.find("child").attrib["link"],
                "xyz": _parse_floats(None if origin is None else origin.attrib.get("xyz")),
                "rpy": _parse_floats(None if origin is None else origin.attrib.get("rpy"), "0 0 0"),
                "axis": _parse_floats(None if axis is None else axis.attrib.get("xyz"), "0 0 1"),
            }
        )
    return joints


def fk_link_poses(
    joint_pos: dict[str, float],
    root_xyz: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joints=None,
):
    joints = joints if joints is not None else load_urdf_joints()
    poses = {"base_link": (_quat_xyzw_to_R(root_quat_xyzw), np.asarray(root_xyz, dtype=np.float64))}
    pending = list(joints)
    guard = 0
    while pending and guard < 512:
        guard += 1
        leftover = []
        for joint in pending:
            if joint["parent"] not in poses:
                leftover.append(joint)
                continue
            Rp, tp = poses[joint["parent"]]
            Rj = _rpy_matrix(*joint["rpy"])
            angle = float(joint_pos.get(joint["name"], 0.0)) if joint["type"] == "revolute" else 0.0
            Ra = _axis_matrix(joint["axis"], angle)
            R = Rp @ Rj @ Ra
            t = tp + Rp @ joint["xyz"]
            poses[joint["child"]] = (R, t)
        pending = leftover
    if pending:
        raise RuntimeError(f"unresolved URDF joints {[j['name'] for j in pending]}")
    return poses


def _in_root(root_xyz, root_R, world_xyz) -> np.ndarray:
    return root_R.T @ (world_xyz - root_xyz)


def amp_frame_from_q(
    q_policy: np.ndarray,
    dq_policy: np.ndarray,
    root_xyz: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joints=None,
) -> np.ndarray:
    if q_policy.shape != (NUM_Z2_29DOF_JOINTS,) or dq_policy.shape != (NUM_Z2_29DOF_JOINTS,):
        raise ValueError(f"q/dq must be length {NUM_Z2_29DOF_JOINTS}")
    joint_pos = {name: float(q_policy[i]) for i, name in enumerate(Z2_29DOF_JOINT_NAMES)}
    poses = fk_link_poses(joint_pos, root_xyz, root_quat_xyzw, joints=joints)
    root_R = _quat_xyzw_to_R(root_quat_xyzw)
    sites = []
    for body in AMP_HAND_BODIES + AMP_FOOT_BODIES:
        sites.append(_in_root(root_xyz, root_R, poses[body][1]))
    frame = np.concatenate([q_policy, dq_policy, *sites])
    if frame.shape != (AMP_FRAME_DIM,):
        raise RuntimeError(f"preview AMP width {frame.shape} != {AMP_FRAME_DIM}")
    return frame


def standing_ankle_separation_m() -> float:
    """Planar Y distance of ankle-roll origins in the standing URDF pose."""
    joint_pos = dict(Z2_STANDING_JOINT_POS)
    poses = fk_link_poses(
        joint_pos,
        np.array([0.0, 0.0, Z2_STANDING_PELVIS_Z]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )
    return float(abs(poses["L_ankle_roll_link"][1][1] - poses["R_ankle_roll_link"][1][1]))


def standing_amp_preview(knee_delta: float = 0.0) -> np.ndarray:
    q = np.array([Z2_STANDING_JOINT_POS[name] for name in Z2_29DOF_JOINT_NAMES], dtype=np.float64)
    q[list(Z2_29DOF_JOINT_NAMES).index("L_knee_joint")] += knee_delta
    q[list(Z2_29DOF_JOINT_NAMES).index("R_knee_joint")] += knee_delta
    root = np.array([0.0, 0.0, Z2_STANDING_PELVIS_Z])
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    return amp_frame_from_q(q, np.zeros_like(q), root, quat)
