# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""G1 depth-student pure contract: dims, camera geometry, scan ranges.

No Isaac imports so local tests can verify the G1 student observation and
camera contracts directly. ``depth_student_env.py`` / ``depth_student_cfg.py``
consume these values.
"""

from __future__ import annotations

import math

from legged_lab.locomotion.schemas import DEPTH_POLICY_SIZE, TEACHER_SCAN_DIM, proprio_fields

# Torso-frame depth camera mount for the G1 student. Same head-height
# down-looking geometry contract as T4; tuned for G1's torso_link height.
# These numbers become the G1 deployment contract (sim2sim mirrors them).
G1_DEPTH_CAMERA_SITE_POS = (0.10, 0.0, 0.25)
G1_DEPTH_CAMERA_PITCH_DEG = 35.0

G1_STUDENT_RENDER_SIZE = DEPTH_POLICY_SIZE

# G1 (29DoF) observation arithmetic over the shared schema:
#   proprio frame = 3+3+3 + 29*3 + 2+2+2 = 102
#   sparse teacher actor = 102*10 (proprio hist) + 195*5 (scan hist) + 2 (contact)
G1_PROPRIO_FRAME_DIM = sum(width for _, width in proprio_fields(29))
G1_SPARSE_TEACHER_ACTOR_OBS_DIM = G1_PROPRIO_FRAME_DIM * 10 + TEACHER_SCAN_DIM * 5 + 2


def g1_depth_camera_ros_quat_wxyz(pitch_deg: float = G1_DEPTH_CAMERA_PITCH_DEG) -> tuple[float, float, float, float]:
    """ROS optical axes in the torso frame as an Isaac ``OffsetCfg.rot`` (wxyz).

    Same construction as the T4 schema camera (right/down/look columns from a
    pitch rotation about Y); implemented locally so G1 does not import T4.
    """
    pitch = math.radians(pitch_deg)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    m00, m10, m20 = 0.0, -1.0, 0.0  # right
    m01, m11, m21 = -sin_p, 0.0, -cos_p  # down
    m02, m12, m22 = cos_p, 0.0, -sin_p  # look
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return (0.25 * scale, (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    if m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        return ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    if m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        return ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
    return ((m10 - m01) / scale, (m02 - m20) / scale, (m12 + m21) / scale, 0.25 * scale)


def g1_sparse_teacher_scan_range() -> tuple[int, int]:
    """``[start, end)`` of the stacked HeightScan block in the G1 teacher actor obs."""
    start = G1_PROPRIO_FRAME_DIM * 10
    end = start + TEACHER_SCAN_DIM * 5
    return start, end


def g1_sparse_teacher_latest_scan_range() -> tuple[int, int]:
    """``[start, end)`` of the newest HeightScan frame (recon target)."""
    start, end = g1_sparse_teacher_scan_range()
    return end - TEACHER_SCAN_DIM, end
