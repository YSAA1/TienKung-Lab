"""Diff the Isaac and MuJoCo paired rollout npz files channel by channel.

Locates the first divergent observation channel and the leading physical
quantities, and computes the analytic PD torque both sides commanded (same
gains), including effort saturation, to separate policy-input divergence from
plant-response divergence.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from legged_lab.assets.unitree_g1.constants import G1_29DOF_JOINT_NAMES  # noqa: E402
from legged_lab.assets.unitree_g1.mujoco_sim2sim import isaac_pd_gains  # noqa: E402
from legged_lab.locomotion.schemas import ObservationLayout, proprio_fields  # noqa: E402

PROPRIO_FIELDS = proprio_fields(29)
LAYOUT = ObservationLayout(29, actor_history=10, scan_history=5, actor_contact=True)
KP = np.array([isaac_pd_gains(name)[0] for name in G1_29DOF_JOINT_NAMES])
KD = np.array([isaac_pd_gains(name)[1] for name in G1_29DOF_JOINT_NAMES])
EFF = np.array([isaac_pd_gains(name)[2] for name in G1_29DOF_JOINT_NAMES])


def channel_slices():
    """Named slices into the 1997D actor obs vector."""
    slices = []
    frame = LAYOUT.proprio_dim
    for hist in range(10):
        start = hist * frame
        for name, width in PROPRIO_FIELDS:
            slices.append((f"hist{hist}.{name}", start, start + width))
            start += width
    scan_dim = LAYOUT.scan_dim
    for hist in range(5):
        begin = 10 * frame + hist * scan_dim
        slices.append((f"scan{hist}", begin, begin + scan_dim))
    begin = 10 * frame + 5 * scan_dim
    slices.append(("contact", begin, begin + 2))
    return slices


def analytic_torque(ctrl, qpos, qvel):
    raw = KP * (ctrl - qpos) - KD * qvel
    return np.clip(raw, -EFF, EFF), raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", required=True)
    parser.add_argument("--mujoco", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tol", type=float, default=0.02)
    args = parser.parse_args()

    isaac = np.load(args.isaac)
    mujoco = np.load(args.mujoco)
    isaac_meta = json.loads(Path(args.isaac).with_suffix(".meta.json").read_text(encoding="utf-8"))
    mujoco_meta = json.loads(Path(args.mujoco).with_suffix(".meta.json").read_text(encoding="utf-8"))
    if isaac_meta["joint_names"] != mujoco_meta["joint_names"]:
        raise SystemExit("joint order mismatch between the two rollouts")
    steps = min(isaac["obs"].shape[0], mujoco["obs"].shape[0])
    out_dir = Path(args.output_dir or Path(args.mujoco).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    obs_i, obs_m = isaac["obs"][:steps], mujoco["obs"][:steps]
    act_i, act_m = isaac["action"][:steps], mujoco["action"][:steps]
    root_i, root_m = isaac["root"][:steps], mujoco["root"][:steps]
    joint_i, joint_m = isaac["joint"][:steps], mujoco["joint"][:steps]
    ctrl_i, ctrl_m = isaac["ctrl"][:steps], mujoco["ctrl"][:steps]
    con_i, con_m = isaac["contact"][:steps], mujoco["contact"][:steps]

    lines: list[str] = []
    report: dict = {"steps": steps, "vx": isaac_meta.get("vx"), "tol": args.tol}

    # Step 0 parity: identical inputs must give identical outputs.
    t0_obs = np.abs(obs_i[0] - obs_m[0])
    lines.append(f"[t0 obs] max|diff|={t0_obs.max():.6g}")
    for name, s, e in channel_slices():
        d = t0_obs[s:e].max()
        if d > 1.0e-6:
            lines.append(f"[t0 obs] {name}: max|diff|={d:.6g}")
    t0_act = np.abs(act_i[0] - act_m[0]).max()
    lines.append(f"[t0 action] max|diff|={t0_act:.6g} (obs parity implies policy parity)")

    # First divergence per channel group (latest frame only: hist9 / scan4).
    first = {}
    for name, s, e in channel_slices():
        if not (name.startswith("hist9.") or name.startswith("scan4.") or name == "contact"):
            continue
        d = np.abs(obs_i[:, s:e] - obs_m[:, s:e]).max(axis=1)
        hit = np.nonzero(d > args.tol)[0]
        first[name] = (int(hit[0]), float(d.max())) if len(hit) else (None, float(d.max()))
    ranked = sorted(first.items(), key=lambda kv: (kv[1][0] is None, kv[1][0] if kv[1][0] is not None else 10**9))
    lines.append("[first step |Δ|>tol, per channel (latest frame)]")
    for name, (step, peak) in ranked:
        lines.append(f"  {name:32s} first={step if step is not None else 'never':>6}  peak={peak:.4g}")
    report["first_divergence"] = {k: {"step": v[0], "peak": v[1]} for k, v in first.items()}

    # Per-step growth curves.
    d_obs = np.abs(obs_i - obs_m).max(axis=1)
    d_act = np.abs(act_i - act_m).max(axis=1)
    d_z = np.abs(root_i[:, 2] - root_m[:, 2])
    d_x = np.abs(root_i[:, 0] - root_m[:, 0])
    with (out_dir / "per_step_diff.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "max_d_obs", "max_d_action", "d_root_z", "d_root_x", "z_isaac", "z_mujoco"])
        for t in range(steps):
            writer.writerow(
                [t, f"{d_obs[t]:.5g}", f"{d_act[t]:.5g}", f"{d_z[t]:.5g}", f"{d_x[t]:.5g}",
                 f"{root_i[t, 2]:.4f}", f"{root_m[t, 2]:.4f}"]
            )

    # Joint-level breakdown at probe steps, including analytic PD torque.
    # Reconstruct the nominal default joint positions from the Isaac t=0 state.
    default_pos = joint_i[0, 0].copy()
    probe_steps = sorted({0, 1, 2, 5, 10, 25, 50, 100, steps - 1} & set(range(steps)))
    lines.append("[joint breakdown] Δq, Δdq, Δctrl, analytic τ (Isaac / MuJoCo) at probe steps")
    breakdown_rows = []
    for t in probe_steps:
        tau_i, raw_i = analytic_torque(ctrl_i[t], joint_i[t, 0], joint_i[t, 1])
        tau_m, raw_m = analytic_torque(ctrl_m[t], joint_m[t, 0], joint_m[t, 1])
        sat_i = np.abs(raw_i) > EFF + 1e-9
        sat_m = np.abs(raw_m) > EFF + 1e-9
        for j, name in enumerate(G1_29DOF_JOINT_NAMES):
            breakdown_rows.append(
                {
                    "step": t,
                    "joint": name,
                    "dq": float(joint_i[t, 0, j] - joint_m[t, 0, j]),
                    "ddq": float(joint_i[t, 1, j] - joint_m[t, 1, j]),
                    "dctrl": float(ctrl_i[t, j] - ctrl_m[t, j]),
                    "tau_isaac": float(tau_i[j]),
                    "tau_mujoco": float(tau_m[j]),
                    "sat_isaac": bool(sat_i[j]),
                    "sat_mujoco": bool(sat_m[j]),
                }
            )
    with (out_dir / "joint_breakdown.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(breakdown_rows[0].keys()))
        writer.writeheader()
        writer.writerows(breakdown_rows)
    for t in probe_steps:
        rows = [r for r in breakdown_rows if r["step"] == t]
        worst = max(rows, key=lambda r: abs(r["dq"]))
        sat = sum(1 for r in rows if r["sat_isaac"] or r["sat_mujoco"])
        lines.append(
            f"  t={t:4d} worst Δq={worst['dq']:+.4f} ({worst['joint']}), "
            f"saturated joints (either side)={sat}, |Δaction|max={d_act[t]:.4f}"
        )

    # Contact comparison.
    mismatch = int((con_i != con_m).any(axis=1).sum())
    lines.append(f"[contact] steps with any contact-flag mismatch: {mismatch}/{steps}")
    first_contact = np.nonzero((con_i != con_m).any(axis=1))[0]
    if len(first_contact):
        lines.append(f"[contact] first mismatch at step {int(first_contact[0])}")

    lines.append(f"[root] Isaac x={root_i[-1, 0]:.3f} z={root_i[-1, 2]:.3f} | "
                 f"MuJoCo x={root_m[-1, 0]:.3f} z={root_m[-1, 2]:.3f}")

    text = "\n".join(lines)
    print(text)
    (out_dir / "diff_report.txt").write_text(text, encoding="utf-8")
    (out_dir / "diff_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote diff_report.txt / diff_report.json / per_step_diff.csv / joint_breakdown.csv in {out_dir}")


if __name__ == "__main__":
    main()
