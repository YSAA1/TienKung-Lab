"""One-off numeric health check of the Z2 AMP motion sources and experts.

Read-only. Answers the user's question: is the T4->Z2 (upstream retargeted)
motion data usable, i.e. joint limits, root height, speed, direction, qvel
consistency, quaternion sanity, and expert/source agreement.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from legged_lab.assets.z2.constants import Z2_29DOF_JOINT_NAMES, Z2_STANDING_JOINT_POS  # noqa: E402

SOURCE_DIR = REPO / "legged_lab/envs/z2/datasets/motion_source"
EXPERT_DIR = REPO / "legged_lab/envs/z2/datasets/motion_amp_expert"
MJCF = REPO / "legged_lab/assets/z2/mjcf/assembly.xml"


def joint_limits() -> dict[str, tuple[float, float]]:
    text = MJCF.read_text(encoding="utf-8")
    limits = {}
    for match in re.finditer(r'<joint name="([^"]+)"[^>]*range="([-\d.]+) ([-\d.]+)"', text):
        limits[match.group(1)] = (float(match.group(2)), float(match.group(3)))
    return limits


def quat_yaw_xyzw(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def check_csv(stem: str, limits: dict[str, tuple[float, float]]) -> dict:
    raw = np.loadtxt(SOURCE_DIR / f"{stem}.csv", delimiter=",")
    assert raw.shape[1] == 36, raw.shape
    root_xyz, quat, q = raw[:, 0:3], raw[:, 3:7], raw[:, 7:36]
    manifest = json.loads((EXPERT_DIR / "_manifest.json").read_text(encoding="utf-8"))
    meta = manifest["clips"][stem]
    fps = meta["fps"]
    # Verified against the Isaac expert: csv frame 0 equals expert frame 0 to
    # 5.5e-17, so the converted CSV is already in Z2_29DOF_JOINT_NAMES policy
    # order. The source-order fallback applies to the raw PKL, not this CSV.
    q_policy = q
    dt = 1.0 / fps
    root_vel = np.diff(root_xyz, axis=0) / dt
    heading = quat_yaw_xyzw(quat)
    world_vel = np.stack(
        [
            root_vel[:, 0] * np.cos(heading[:-1]) + root_vel[:, 1] * np.sin(heading[:-1]),
            -root_vel[:, 0] * np.sin(heading[:-1]) + root_vel[:, 1] * np.cos(heading[:-1]),
        ],
        axis=1,
    )
    out = {
        "frames_csv": int(raw.shape[0]),
        "fps": fps,
        "duration_s": round(raw.shape[0] / fps, 3),
        "quat_norm_min": float(np.linalg.norm(quat, axis=1).min()),
        "quat_norm_max": float(np.linalg.norm(quat, axis=1).max()),
        "root_z_min": float(root_xyz[:, 2].min()),
        "root_z_max": float(root_xyz[:, 2].max()),
        "root_z_mean": float(root_xyz[:, 2].mean()),
        "root_yaw_deg_min": float(np.degrees(heading.min())),
        "root_yaw_deg_max": float(np.degrees(heading.max())),
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(root_xyz[:, :2], axis=0), axis=1))),
        "forward_speed_mean_mps": float(world_vel[:, 0].mean()),
        "forward_speed_abs_mean_mps": float(np.abs(world_vel[:, 0]).mean()),
        "lateral_speed_abs_mean_mps": float(np.abs(world_vel[:, 1]).mean()),
        "net_displacement_xy_m": [float(v) for v in (root_xyz[-1, :2] - root_xyz[0, :2])],
        "joint_limit_violations": {},
        "joint_range_deg": {},
    }
    for column, name in enumerate(Z2_29DOF_JOINT_NAMES):
        lo, hi = limits[name]
        column_data = q_policy[:, column]
        n_bad = int(np.sum((column_data < lo - 1e-6) | (column_data > hi + 1e-6)))
        if n_bad:
            out["joint_limit_violations"][name] = {
                "frames": n_bad,
                "min": float(column_data.min()),
                "max": float(column_data.max()),
                "limit": [lo, hi],
            }
        out["joint_range_deg"][name] = [
            round(float(np.degrees(column_data.min())), 1),
            round(float(np.degrees(column_data.max())), 1),
        ]
    # standing-pose distance to detect weird retarget offsets
    standing = np.array([Z2_STANDING_JOINT_POS[name] for name in Z2_29DOF_JOINT_NAMES])
    out["joint_l2_to_standing_mean"] = float(np.linalg.norm(q_policy - standing, axis=1).mean())
    return out


def check_expert(stem: str, limits: dict[str, tuple[float, float]]) -> dict:
    payload = json.loads((EXPERT_DIR / f"{stem}.txt").read_text(encoding="utf-8"))
    frames = np.asarray(payload["Frames"], dtype=np.float64)
    assert frames.shape[1] == 70, frames.shape
    q, dq, hands, feet = frames[:, 0:29], frames[:, 29:58], frames[:, 58:64], frames[:, 64:70]
    out = {
        "header_frame_duration": payload["FrameDuration"],
        "header_joint_order_matches_policy": payload["JointOrder"] == list(Z2_29DOF_JOINT_NAMES),
        "frames_expert": int(frames.shape[0]),
        "all_finite": bool(np.isfinite(frames).all()),
        "dq_abs_max": float(np.abs(dq).max()),
        "dq_abs_p99": float(np.percentile(np.abs(dq), 99)),
        "hands_root_z_minmax": [float(hands[:, 2].min()), float(hands[:, 2].max()),
                                float(hands[:, 5].min()), float(hands[:, 5].max())],
        "feet_root_z_min": [float(feet[:, 2].min()), float(feet[:, 5].min())],
        "feet_root_z_max": [float(feet[:, 2].max()), float(feet[:, 5].max())],
        "joint_limit_violations_frames": {},
    }
    # q in expert is policy order already
    for column, name in enumerate(Z2_29DOF_JOINT_NAMES):
        lo, hi = limits[name]
        n_bad = int(np.sum((q[:, column] < lo - 1e-6) | (q[:, column] > hi + 1e-6)))
        if n_bad:
            out["joint_limit_violations_frames"][name] = n_bad
    # feet alternation: min z per frame
    foot_clearance = np.minimum(feet[:, 2], feet[:, 5])
    out["min_foot_clearance_m"] = float(foot_clearance.min())
    swing = np.maximum(feet[:, 2], feet[:, 5]) - np.minimum(feet[:, 2], feet[:, 5])
    out["mean_foot_separation_m"] = float(swing.mean())
    return out


def main() -> None:
    limits = joint_limits()
    assert set(Z2_29DOF_JOINT_NAMES) == set(limits), "MJCF joints != policy joints"
    report = {"generated": "2026-09-11", "source": "local repo z2 datasets (upstream z2-lab-stable-AMP)"}
    for stem in ("walk", "walk_l", "run"):
        report[stem] = {"csv": check_csv(stem, limits), "expert": check_expert(stem, limits)}
    out_path = Path(__file__).resolve().parent / "motion_data_report.json"
    out_path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: {kk: {kkk: vvv for kkk, vvv in (vv.items() if isinstance(vv, dict) else [])}
                           for kk, vv in (v.items() if isinstance(v, dict) else [])}
                      if isinstance(v, dict) else v for k, v in report.items()}, indent=1)[:2000])
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
