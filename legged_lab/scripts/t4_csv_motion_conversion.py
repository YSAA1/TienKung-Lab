"""Convert raw T4 steering CSV clips into TienKung-style visualization JSON.

Input rows use the frozen schema:
  root_xyz(3) + root_quat_xyzw(4) + T4_JOINT_NAMES(27)

The output contains:
  root_xyz + root_euler_xyz + q(27) + root_linear_velocity(3)
  + root_angular_velocity(3) + dq(27)

AMP expert files are intentionally not produced here. Their end-effector
features must be recomputed from the migrated T4 asset inside IsaacLab so the
expert and runtime observations share exactly the same kinematics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from legged_lab.assets.t4.constants import T4_JOINT_NAMES


RAW_WIDTH = 3 + 4 + len(T4_JOINT_NAMES)


def _angular_velocity(root_quat_xyzw: np.ndarray, dt: float) -> np.ndarray:
    rotations = Rotation.from_quat(root_quat_xyzw)
    delta = rotations[:-1].inv() * rotations[1:]
    return delta.as_rotvec() / dt


def convert_csv(input_csv: Path, output_json: Path, fps: float) -> None:
    raw = np.loadtxt(input_csv, delimiter=",")
    if raw.ndim != 2 or raw.shape[1] != RAW_WIDTH:
        raise ValueError(f"{input_csv} must have shape [T, {RAW_WIDTH}], got {raw.shape}")
    if raw.shape[0] < 2:
        raise ValueError(f"{input_csv} needs at least two frames")
    if not np.isfinite(raw).all():
        raise ValueError(f"{input_csv} contains non-finite values")

    root_pos = raw[:, :3]
    root_quat_xyzw = raw[:, 3:7]
    quat_norm = np.linalg.norm(root_quat_xyzw, axis=1)
    if not np.allclose(quat_norm, 1.0, atol=1.0e-4):
        raise ValueError(f"{input_csv} contains non-unit root quaternions")
    joint_pos = raw[:, 7:]
    dt = 1.0 / fps

    root_euler = Rotation.from_quat(root_quat_xyzw[:-1]).as_euler("XYZ")
    root_euler = np.unwrap(root_euler, axis=0)
    root_lin_vel = np.diff(root_pos, axis=0) / dt
    root_ang_vel = _angular_velocity(root_quat_xyzw, dt)
    joint_vel = np.diff(joint_pos, axis=0) / dt
    frames = np.concatenate(
        (
            root_pos[:-1],
            root_euler,
            joint_pos[:-1],
            root_lin_vel,
            root_ang_vel,
            joint_vel,
        ),
        axis=1,
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "LoopMode": "Wrap",
                "FrameDuration": dt,
                "EnableCycleOffsetPosition": True,
                "EnableCycleOffsetRotation": True,
                "MotionWeight": 1.0,
                "JointOrder": list(T4_JOINT_NAMES),
                "SourceSchema": "root_xyz + root_quat_xyzw + 27 T4 joints",
                "Frames": frames.tolist(),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="CSV file or directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    inputs = sorted(args.input.glob("*.csv")) if args.input.is_dir() else [args.input]
    if not inputs:
        raise ValueError(f"no CSV files found under {args.input}")
    for input_csv in inputs:
        output = args.output_dir / f"{input_csv.stem}.txt"
        convert_csv(input_csv, output, args.fps)
        print(f"converted {input_csv} -> {output}")


if __name__ == "__main__":
    main()
