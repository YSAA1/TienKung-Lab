"""Offline named reference-motion validation and ordering for any robot."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from legged_lab.motion_tracking.schema import normalize_names

TRACKING_MOTION_SCHEMA_VERSION = "named_tracking_motion.v1"

REQUIRED_KEYS = (
    "fps",
    "joint_names",
    "body_names",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_BODY_ARRAY_TRAILING_DIMS = {
    "body_pos_w": 3,
    "body_quat_w": 4,
    "body_lin_vel_w": 3,
    "body_ang_vel_w": 3,
}


def load_tracking_motion(
    path: Path | str, *, joint_names: Sequence[str], schema_version: str = TRACKING_MOTION_SCHEMA_VERSION,
) -> dict:
    """Validate a named motion and reorder its joint columns to the requested robot order."""
    joint_names = normalize_names(joint_names, label="target joint")
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path.name}: missing required keys {missing}")

        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{path.name}: fps must be positive and finite, got {fps}")

        source_joint_names = normalize_names(data["joint_names"], label="joint")
        if sorted(source_joint_names) != sorted(joint_names):
            raise ValueError(f"{path.name}: joint names are not a bijection with joint_names")
        body_names = normalize_names(data["body_names"], label="body")

        joint_pos = np.asarray(data["joint_pos"])
        joint_vel = np.asarray(data["joint_vel"])
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(source_joint_names):
            raise ValueError(
                f"{path.name}: joint_pos must be (frames, {len(source_joint_names)}), got {joint_pos.shape}"
            )
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"{path.name}: joint_vel shape {joint_vel.shape} != joint_pos shape {joint_pos.shape}")
        num_frames = int(joint_pos.shape[0])

        permutation = tuple(source_joint_names.index(name) for name in joint_names)
        motion = {
            "schema_version": schema_version,
            "fps": fps,
            "num_frames": num_frames,
            "joint_names": tuple(joint_names),
            "source_joint_names": source_joint_names,
            "joint_pos": joint_pos[:, permutation],
            "joint_vel": joint_vel[:, permutation],
            "body_names": body_names,
        }
        for key, trailing_dim in _BODY_ARRAY_TRAILING_DIMS.items():
            array = np.asarray(data[key])
            expected_shape = (num_frames, len(body_names), trailing_dim)
            if array.shape != expected_shape:
                raise ValueError(f"{path.name}: {key} must have shape {expected_shape}, got {array.shape}")
            motion[key] = array

    for key in ("joint_pos", "joint_vel", *_BODY_ARRAY_TRAILING_DIMS):
        if not np.isfinite(motion[key]).all():
            raise ValueError(f"{path.name}: {key} contains non-finite values")
    return motion


def body_indices(motion: dict, names: Sequence[str]) -> tuple[int, ...]:
    """Resolve body names to indices into the motion body arrays, failing fast."""
    body_names = motion["body_names"]
    missing = [name for name in names if name not in body_names]
    if missing:
        raise KeyError(f"bodies not present in motion: {missing}")
    return tuple(body_names.index(name) for name in names)
