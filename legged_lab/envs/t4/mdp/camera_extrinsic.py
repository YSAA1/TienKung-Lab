# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""LightLP Table II camera extrinsic jitter (Isaac-free algebra).

Student cameras are reset-randomized by ±1 cm position and ±0.025 rad
orientation because a fixed extrinsic calibration will not match the robot.
"""

from __future__ import annotations

from typing import Any

LIGHTLP_CAMERA_POS_JITTER_M = 0.01
LIGHTLP_CAMERA_ORI_JITTER_RAD = 0.025


def euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """XYZ intrinsic Euler (radians) to wxyz quaternion."""
    import math

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def quat_mul_wxyz(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product ``first ⊗ second`` in wxyz order."""
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def compose_camera_offset(
    nominal_pos,
    nominal_quat_wxyz,
    delta_pos,
    delta_rpy,
):
    """Apply a position delta and XYZ Euler jitter onto a ROS camera offset."""
    pos = [float(nominal_pos[i]) + float(delta_pos[i]) for i in range(3)]
    jitter = euler_xyz_to_quat_wxyz(float(delta_rpy[0]), float(delta_rpy[1]), float(delta_rpy[2]))
    quat = quat_mul_wxyz(tuple(float(v) for v in nominal_quat_wxyz), jitter)
    return pos, quat


def capture_nominal_camera_pose(camera: Any):
    """Read the spawned per-env pose so jitter stays in the camera's native frame."""
    if camera is None:
        return None
    if hasattr(camera, "_offset_pos") and hasattr(camera, "_offset_quat"):
        return (
            camera._offset_pos[0].detach().clone(),
            camera._offset_quat[0].detach().clone(),
            "offset_buffers",
        )
    view = getattr(camera, "_view", None)
    if view is not None and hasattr(view, "get_local_poses"):
        pos, quat = view.get_local_poses()
        return pos[0].detach().clone(), quat[0].detach().clone(), "local_poses"
    return None


def apply_camera_local_offset(camera: Any, env_ids, pos, quat_wxyz, *, mode: str) -> str:
    """Write per-env local camera poses using the same API that captured the nominal pose."""
    if camera is None:
        raise RuntimeError("depth camera is missing; cannot apply extrinsic jitter")
    ids = env_ids
    if mode == "offset_buffers":
        if not (hasattr(camera, "_offset_pos") and hasattr(camera, "_offset_quat")):
            raise RuntimeError("camera offset buffers disappeared after spawn")
        camera._offset_pos[ids] = pos
        camera._offset_quat[ids] = quat_wxyz
        return mode
    if mode == "local_poses":
        view = getattr(camera, "_view", None)
        if view is None or not hasattr(view, "set_local_poses"):
            raise RuntimeError("camera local-pose view disappeared after spawn")
        view.set_local_poses(pos, quat_wxyz, ids)
        return mode
    raise RuntimeError(
        f"unsupported camera pose mode {mode!r}; refusing to train without Table II extrinsic DR"
    )
