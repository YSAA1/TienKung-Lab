"""Quantitative re-check of the Z2 AMP expert clips (G1 v6 acceptance criteria).

Pure numpy; reads the UE-JSON experts plus the source CSVs. Outputs one JSON
with per-clip stats: hash integrity, T-pose/frozen-arm fractions, gait cycles,
root speed from source, foot-stillness slip proxy, dq sanity.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPERT_DIR = ROOT / "legged_lab/envs/z2/datasets/motion_amp_expert"
SOURCE_DIR = ROOT / "legged_lab/envs/z2/datasets/motion_source"
OUT = Path(__file__).resolve().parent / "z2_amp_quantitative.json"

ARM_TOKENS = ("shoulder", "elbow", "wrist")
KNEE_TOKEN = "knee"


def _frozen_fraction(q_arm: np.ndarray, window: int, tol: float) -> tuple[float, float]:
    head = q_arm[: min(window, len(q_arm))]
    ref = head[0]
    frozen_head = float(np.mean(np.max(np.abs(head - ref), axis=1) < tol))
    frozen_all = float(np.mean(np.max(np.abs(q_arm - q_arm[0]), axis=1) < tol))
    return frozen_head, frozen_all


def _knee_cycles(q_knee: np.ndarray, prominence: float) -> int:
    peaks = 0
    for i in range(1, len(q_knee) - 1):
        if q_knee[i] > q_knee[i - 1] and q_knee[i] >= q_knee[i + 1] and q_knee[i] > prominence:
            peaks += 1
    return peaks


def main() -> None:
    manifest = json.loads((EXPERT_DIR / "_manifest.json").read_text())
    report = {"clips": {}, "criteria": {"frozen_head_max": 0.10, "cycles_min": 1, "dq_p95_max": 12.0}}
    for name, entry in manifest["clips"].items():
        path = EXPERT_DIR / f"{name}.txt"
        payload = json.loads(path.read_text())
        frames = np.asarray(payload["Frames"], dtype=float)
        order = payload["JointOrder"]
        dt = float(payload["FrameDuration"])
        arm_idx = [i for i, j in enumerate(order) if any(t in j for t in ARM_TOKENS)]
        knee_idx = [i for i, j in enumerate(order) if KNEE_TOKEN in j]
        q, dq = frames[:, :29], frames[:, 29:58]
        feet = frames[:, 64:70]
        frozen_head, frozen_all = _frozen_fraction(q[:, arm_idx], window=60, tol=0.05)
        cycles = _knee_cycles(q[:, min(knee_idx)], prominence=0.15)
        foot_xy_speed = np.linalg.norm(np.diff(feet[:, [0, 1, 3, 4]], axis=0), axis=1) / dt
        both_moving = float(np.mean(foot_xy_speed > 0.35))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        src = np.loadtxt(SOURCE_DIR / entry["source_csv"].split("/")[-1], delimiter=",")
        xy_disp = np.linalg.norm(np.diff(src[:, :2], axis=0), axis=1)
        speed_mean = float(xy_disp.mean() / dt)
        speed_net = float(np.linalg.norm(src[-1, :2] - src[0, :2]) / (len(src) * dt))
        report["clips"][name] = {
            "frames": int(len(frames)),
            "duration_s": round(len(frames) * dt, 2),
            "sha256_ok": digest == entry["sha256"],
            "frozen_arm_first60": frozen_head,
            "frozen_arm_all": frozen_all,
            "arm_q_std_mean": round(float(q[:, arm_idx].std(axis=0).mean()), 4),
            "knee_cycles_left": cycles,
            "dq_p95_rad_s": round(float(np.percentile(np.abs(dq), 95, axis=0).max()), 2),
            "nonfinite": bool(not np.isfinite(frames).all()),
            "source_speed_mean_m_s": round(speed_mean, 3),
            "source_speed_net_m_s": round(speed_net, 3),
            "feet_never_still_frac": round(both_moving, 3),
        }
    OUT.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
