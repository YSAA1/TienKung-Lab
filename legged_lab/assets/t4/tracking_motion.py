"""Offline loader contract for T4 whole-body tracking reference motions.

The vault reference npz keeps its source (PHP/MJCF BFS) joint order on disk so the
committed bytes stay identical to the upstream lineage artifact. Every consumer
must go through :func:`load_t4_tracking_motion`, which validates the schema and
reorders joint columns into ``T4_JOINT_NAMES``. The module stays pure numpy so
contract tests and offline tools run without IsaacLab.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import numpy as np

from legged_lab.assets.t4.constants import T4_JOINT_NAMES

TRACKING_MOTION_SCHEMA_VERSION = "t4_tracking_motion.v1"
TRACKING_MOTION_DIR = Path(__file__).resolve().parents[2] / "envs" / "t4" / "datasets" / "motion_tracking"

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


def load_t4_tracking_motion(path: Path | str) -> dict:
    """Load one tracking reference motion and reorder joints to ``T4_JOINT_NAMES``.

    Returns a dict with ``joint_pos``/``joint_vel`` columns in ``T4_JOINT_NAMES``
    order, the untouched per-body world arrays, and the source metadata. Raises
    ``ValueError`` on any contract drift (missing key, bad shape, joint name set
    mismatch, non-finite values) instead of silently continuing.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED_KEYS if key not in data.files]
        if missing:
            raise ValueError(f"{path.name}: missing required keys {missing}")

        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError(f"{path.name}: fps must be positive and finite, got {fps}")

        source_joint_names = tuple(str(name) for name in data["joint_names"])
        if sorted(source_joint_names) != sorted(T4_JOINT_NAMES):
            raise ValueError(f"{path.name}: joint names are not a bijection with T4_JOINT_NAMES")
        body_names = tuple(str(name) for name in data["body_names"])

        joint_pos = np.asarray(data["joint_pos"])
        joint_vel = np.asarray(data["joint_vel"])
        if joint_pos.ndim != 2 or joint_pos.shape[1] != len(source_joint_names):
            raise ValueError(
                f"{path.name}: joint_pos must be (frames, {len(source_joint_names)}), got {joint_pos.shape}"
            )
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"{path.name}: joint_vel shape {joint_vel.shape} != joint_pos shape {joint_pos.shape}")
        num_frames = int(joint_pos.shape[0])

        permutation = tuple(source_joint_names.index(name) for name in T4_JOINT_NAMES)
        motion = {
            "schema_version": TRACKING_MOTION_SCHEMA_VERSION,
            "fps": fps,
            "num_frames": num_frames,
            "joint_names": tuple(T4_JOINT_NAMES),
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
